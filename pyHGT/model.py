import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pack_sequence
import numpy as np
import torch.nn.functional as F
import math
# CUDA = torch.cuda.is_available()

CUDA = torch.cuda.is_available()


class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 4020, gamma=25, dropout = 0.2):
        super(RelTemporalEncoding, self).__init__()
        # self.gamma = gamma
        self.gamma = gamma
        
        # 原有的正弦编码
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        
        # 新增：时间感知权重
        self.time_weight = nn.Linear(1, 1)
        self.lin = nn.Linear(n_hid, n_hid)
        self.fusion = nn.Linear(n_hid * 2, n_hid)
        
    def forward(self, x, t, query_time=None):
        # 检查维度
        # print(f"x shape: {x.shape}")  # 应该是 [batch, seq_len, dim]
        # print(f"t shape: {t.shape}")  # 应该是 [batch, seq_len]
        # print(f"query_time shape: {query_time.shape if query_time is not None else None}")  # 应该是 [batch] 或 [1]
        
        sin_encoding = self.emb(t)
        
        if query_time is not None:
            # 确保维度匹配
            if query_time.dim() == 1 and query_time.shape[0] == 1:
                # 如果query_time是[1]，扩展到batch维度
                query_time = query_time.expand(t.shape[0])
            
            delta_t = query_time.unsqueeze(1) - t
            time_aware = torch.tanh(delta_t.float() ** 2 / self.gamma).unsqueeze(-1)
            
            weighted_sin = sin_encoding * time_aware
            combined = torch.cat([x, weighted_sin], dim=-1)
            return self.fusion(combined)
        else:
            return x + self.lin(sin_encoding)


class TypeGAT(nn.Module):
    def __init__(self, num_e, num_r, relation_embeddings, out_dim, 
                 time_decay_factor=0.1, path_length_penalty=0.1, 
                 hics_neighbor_sample=10):
        super(TypeGAT, self).__init__()
        
        self.num_e = num_e
        self.num_r = num_r
        self.in_dim = relation_embeddings.shape[1]
        self.out_dim = out_dim
        
        # HICS策略参数
        self.time_decay_factor = time_decay_factor
        self.hics_neighbor_sample = hics_neighbor_sample
        
        # Enhanced TopK策略参数
        self.path_length_penalty = path_length_penalty
        
        # 边频率统计（用于权重计算）
        self.register_buffer('edge_frequency', torch.ones(num_r))
        
        self.pad = torch.zeros(1, self.out_dim)
        if CUDA:
            self.pad = self.pad.cuda()
        
        self.relation_embeddings = nn.Parameter(relation_embeddings)
        self.emb = RelTemporalEncoding(self.out_dim)
        # 替换 GRU 为 TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.out_dim,  # 输入和输出维度一致
            nhead=4,               # 多头注意力头数（需整除 out_dim）
            dim_feedforward=self.out_dim * 4,  # 前馈网络维度
            dropout=0.1,           # Dropout 比例
            activation='relu',     # 激活函数
            batch_first=True       # 批次优先
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=2           # 层数
        )
        
        # HICS策略的额外组件
        self.query = nn.Embedding(num_r, out_dim)
        self.hics_fusion = nn.Linear(out_dim * 2, out_dim)
        
        # self.gru.reset_parameters()
    
    
    def hics_strategy(self, history_graph, entity_index, query_embedding, k=None):
        """HICS策略：为没有历史路径的实体生成上下文特征"""
        if k is None:
            k = self.hics_neighbor_sample
        
        device = entity_index.device
        batch_size = entity_index.size(0)
        
        if batch_size == 0:
            return torch.empty(0, query_embedding.size(-1), device=device)
        
        # 安全获取图信息
        try:
            edge_index = torch.stack(history_graph.edges()).to(device)
            edge_types = history_graph.edata['type'].to(device)
            edge_times = history_graph.edata['time'].to(device)
        except:
            return torch.zeros(batch_size, query_embedding.size(-1), device=device)
        
        if edge_index.size(1) == 0:
            return torch.zeros(batch_size, query_embedding.size(-1), device=device)
        
        # 验证实体索引的有效性
        valid_mask = (entity_index >= 0) & (entity_index < history_graph.num_nodes())
        invalid_entities = ~valid_mask
        
        # 预分配输出特征
        dim = query_embedding.size(-1)
        sampled_features = torch.zeros(batch_size, dim, device=device)
        
        # 批量查找入边和出边
        incoming_masks = edge_index[1].unsqueeze(0) == entity_index.unsqueeze(1)  # [batch, num_edges]
        outgoing_masks = edge_index[0].unsqueeze(0) == entity_index.unsqueeze(1)  # [batch, num_edges]
        
        # 为每个实体计算邻居数
        incoming_counts = incoming_masks.sum(dim=1)
        outgoing_counts = outgoing_masks.sum(dim=1)
        
        # 处理有邻居的实体
        has_neighbors = (incoming_counts > 0) | (outgoing_counts > 0)
        if has_neighbors.any():
            for i in range(batch_size):
                if not has_neighbors[i]:
                    continue  # 跳过无邻居的，稍后批量处理
                
                # 提取当前实体的掩码和权重（假设 weights = edge_times，按时间降序采样）
                inc_mask = incoming_masks[i]
                out_mask = outgoing_masks[i]
                weights = edge_times  # 假设基于时间权重；如果有其他权重，请替换
                
                neighbor_embs = []
                
                # 采样入边
                if incoming_counts[i] > 0:
                    inc_indices = inc_mask.nonzero().squeeze(-1)
                    sample_size = min(k // 2, len(inc_indices))
                    if sample_size > 0:
                        if len(inc_indices) > sample_size:
                            _, top_idx = torch.topk(weights[inc_indices], sample_size)
                            inc_indices = inc_indices[top_idx]
                        rel_embs = self.query(edge_types[inc_indices])
                        neighbor_embs.append(rel_embs)
                
                # 采样出边
                if outgoing_counts[i] > 0:
                    out_indices = out_mask.nonzero().squeeze(-1)
                    remaining_k = k - len(neighbor_embs[0]) if neighbor_embs else k  # 注意：neighbor_embs 现在是张量列表
                    sample_size = min(remaining_k, len(out_indices))
                    if sample_size > 0:
                        if len(out_indices) > sample_size:
                            _, top_idx = torch.topk(weights[out_indices], sample_size)
                            out_indices = out_indices[top_idx]
                        rel_embs = self.query(edge_types[out_indices])
                        neighbor_embs.append(rel_embs)
                
                # 聚合
                if neighbor_embs:
                    aggregated = torch.cat(neighbor_embs, dim=0).mean(0)
                    sampled_features[i] = aggregated
        
        # 批量处理无邻居和无效实体：生成虚拟边
        no_neighbor_mask = ~has_neighbors | invalid_entities
        if no_neighbor_mask.any():
            num_no_neighbors = no_neighbor_mask.sum().item()
            max_time = edge_times.max() + 1 if len(edge_times) > 0 else 0
            virtual_rels = torch.randint(0, self.num_r, (num_no_neighbors,), device=device)
            virtual_embs = self.query(virtual_rels)
            virtual_weights = torch.exp(torch.full((num_no_neighbors,), -self.time_decay_factor * 1.0, device=device))
            sampled_features[no_neighbor_mask] = virtual_embs * virtual_weights.unsqueeze(-1)
        
        return sampled_features
    
    
    
    def forward_with_strategies(self, path_index, batch_relation, paths, paths_time, 
                              lengths, path_r, path_neg_index, batch_his_r, 
                              query_time, history_graph=None, entity_without_paths=None):
        """结合HICS和Enhanced TopK策略的前向传播"""
        r_inp = self.relation_embeddings
        
        # 标准路径编码
        pad_r = torch.cat((r_inp, self.pad), dim=0)
        emb = pad_r[paths]
        emb = self.emb(emb, paths_time, query_time)

        # 创建注意力掩码（忽略填充位置）
        max_len = emb.size(1)
        mask = torch.arange(max_len, device=emb.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
        attn_mask = mask.to(emb.device)  # [batch_size, max_len], True 表示忽略

        # Transformer 处理
        # emb: [batch_size, seq_len, out_dim]
        transformer_out = self.transformer(emb, src_key_padding_mask=attn_mask)
        
        # packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, 
        #                              enforce_sorted=False).to(paths.device)
        # _, hidden = self.gru(packed)

        # 聚合序列表示（取最后一个非填充位置的输出）
        batch_size, seq_len, dim = transformer_out.shape
        indices = (lengths - 1).clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, dim)
        path_emb = transformer_out.gather(1, indices).squeeze(1)  # [batch_size, out_dim]
        path_emb = torch.cat((self.pad, path_emb), dim=0)

        del emb, transformer_out, paths
        
        # 如果有无历史路径的实体，使用HICS策略
        if history_graph is not None and entity_without_paths is not None:
            # print('=================================')
            query_emb = pad_r[batch_relation]
            hics_features = self.hics_strategy(history_graph, entity_without_paths, 
                                              query_emb)
            
            # 融合HICS特征
            if hics_features.size(0) > 0:
                combined_features = torch.cat([path_emb[-hics_features.size(0):], 
                                              hics_features], dim=-1)
                path_emb[-hics_features.size(0):] = self.hics_fusion(combined_features)
        
        # 归一化
        pad_r = torch.cat((F.normalize(r_inp, dim=1), self.pad.to(r_inp.device)), dim=0)
        path_emb = F.normalize(path_emb, dim=1)
        
        # 计算评分
        scores = torch.mm(path_emb, pad_r[batch_relation].t()).t()
        mask = torch.zeros((scores.size(0), scores.size(1))).cuda()
        m_index = min(path_index.size(1), mask.size(1))
        mask = mask.scatter(1, path_index[:, 0:m_index], 1)
        
        # 使用Enhanced TopK选择（如果提供了路径时间信息）
        max_score, max_id = torch.max(scores * mask, 1)
        
        # 历史关系评分
        scores_r = torch.mm(pad_r, pad_r.t())[batch_relation]
        his_score = torch.mean(torch.diagonal(scores_r[:, batch_his_r], 
                                             dim1=0, dim2=1).t(), 1)
        
        score = max_score
        
        return score, path_emb[path_neg_index], pad_r[path_r]
    
    # 保留原有的forward和test方法以保持兼容性
    def forward2(self, path_index, batch_relation, paths, paths_time, lengths, 
                path_r, path_neg_index, batch_his_r, query_time):
        return self.forward_with_strategies(path_index, batch_relation, paths, 
                                           paths_time, lengths, path_r, 
                                           path_neg_index, batch_his_r, query_time)
    
    def test(self, path_index, batch_relation, paths, lengths, paths_time, 
             batch_his_r, query_time):
        r_inp = self.relation_embeddings
        pad = torch.zeros(1, self.out_dim)
        
        pad_r = torch.cat((r_inp, pad.to(r_inp.device)), dim=0)
        emb = pad_r[paths]
        emb = self.emb(emb, paths_time, query_time)
        # 创建注意力掩码
        max_len = emb.size(1)
        mask = torch.arange(max_len, device=emb.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
        attn_mask = mask.to(emb.device)

        # Transformer 处理
        transformer_out = self.transformer(emb, src_key_padding_mask=attn_mask)
        batch_size, seq_len, dim = transformer_out.shape
        indices = (lengths - 1).clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, dim)
        path_emb = transformer_out.gather(1, indices).squeeze(1)
        path_emb = torch.cat((self.pad.to(r_inp.device), path_emb), dim=0)

        del emb, transformer_out, paths
        
        pad_r = torch.cat((F.normalize(r_inp, dim=1), pad.to(r_inp.device)), dim=0)
        path_emb = F.normalize(path_emb, dim=1)
        
        scores = torch.mm(path_emb, pad_r[batch_relation[0]].unsqueeze(1)).squeeze(1)
        max_score, max_id = torch.max(scores[path_index], 1)
        
        scores_r = torch.mm(pad_r, pad_r[batch_relation[0]].unsqueeze(1)).squeeze(1)
        his_score = torch.mean(scores_r[batch_his_r], 1)
        
        score = max_score + his_score
        return score