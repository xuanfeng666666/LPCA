import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
import os
import time
import dgl

from pyHGT.model import *
from pyHGT.data import *
from sklearn.utils import shuffle
from operator import itemgetter
from more_itertools import flatten
import networkx as nx
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pack_sequence
from tqdm import tqdm
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data",
                      default="./data/ICEWS18", help="data directory")
    args.add_argument('--dataset', type=str, default='ICEWS18')
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=100, help="Number of epochs")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-6, help="L2 reglarization for conv")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=100, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-4)
    
    # 新增：策略相关参数
    args.add_argument("--use_hics", action='store_true', default=True, help="Use HICS strategy")
    args.add_argument("--use_enhanced_topk", action='store_true', help="Use Enhanced TopK strategy")
    args.add_argument("--time_decay_factor", type=float, default=0.1, help="Time decay factor")
    args.add_argument("--path_length_penalty", type=float, default=0.1, help="Path length penalty")
    args.add_argument("--hics_neighbor_sample", type=int, default=10, help="HICS neighbor sample size")
    args.add_argument("--topk_paths", type=int, default=25, help="Number of paths for TopK selection")
    
    args = args.parse_args()
    return args

def save_model(model, name, folder_name, epoch):
    print("Saving Model")
    torch.save(model.state_dict(),
               (folder_name + "trained" + str(epoch) + ".pth"))
    print("Done saving Model")

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def list_to_array(x, pad):
    dff = pd.concat([pd.DataFrame({'{}'.format(index):labels}) 
                    for index, labels in enumerate(x)], axis=1)
    return dff.fillna(pad).values.T.astype(int)

def build_history_graph(graph_data, t, num_e, num_r):
    """构建历史图用于HICS策略"""
    # 创建一个DGL图
    edges_src = []
    edges_dst = []
    edge_types = []
    edge_times = []
    
    # 从graph_data中提取历史边信息
    # 查看过去的时间步（最多100个）
    start_t = max(0, t - 100)
    
    for past_t in range(start_t, t):
        if hasattr(graph_data, 't_r_id_p_dict') and past_t in graph_data.t_r_id_p_dict:
            for r, id_p in graph_data.t_r_id_p_dict[past_t].items():
                for entity_id, paths in id_p.items():
                    # 从路径中提取边信息
                    for path in paths:
                        if isinstance(path, (list, tuple)) and len(path) >= 2:
                            # 假设路径中包含实体ID
                            # 提取实体对
                            for i in range(len(path) - 1):
                                src = path[i] % num_e if path[i] >= num_e else path[i]
                                dst = path[i+1] % num_e if path[i+1] >= num_e else path[i+1]
                                
                                # 确保实体ID在有效范围内
                                if 0 <= src < num_e and 0 <= dst < num_e:
                                    edges_src.append(src)
                                    edges_dst.append(dst)
                                    edge_types.append(r % (num_r * 2))  # 确保关系类型在范围内
                                    edge_times.append(past_t)
    
    # 创建DGL图
    if len(edges_src) == 0:
        # 创建一个包含所有节点但没有边的图
        g = dgl.graph(([], []), num_nodes=num_e)
        # 添加空的边属性
        g.edata['type'] = torch.LongTensor([])
        g.edata['time'] = torch.LongTensor([])
    else:
        # 创建图
        g = dgl.graph((edges_src, edges_dst), num_nodes=num_e)
        g.edata['type'] = torch.LongTensor(edge_types)
        g.edata['time'] = torch.LongTensor(edge_times)
    
    return g

def find_entities_without_paths(graph_data, t, batch_paths_id):
    """找出没有历史路径的实体"""
    entities_without_paths = []
    
    # 将batch_paths_id转换为numpy数组以便处理
    if isinstance(batch_paths_id, torch.Tensor):
        paths_array = batch_paths_id.cpu().numpy()
    else:
        paths_array = np.array(batch_paths_id)
    
    # 从batch_paths_id中提取所有涉及的实体（去除padding）
    all_entities = set()
    for path in paths_array.flatten():
        if path > 0:  # 忽略padding（通常是0或负数）
            all_entities.add(path)
    
    # 检查哪些实体在当前时间步没有路径
    entities_with_paths = set()
    
    # 检查t_paths中的实体
    if hasattr(graph_data, 't_paths') and t in graph_data.t_paths:
        for path in graph_data.t_paths[t]:
            if isinstance(path, (list, tuple)):
                for entity in path:
                    if entity > 0:
                        entities_with_paths.add(entity)
    
    # 找出没有路径的实体
    for entity in all_entities:
        if entity not in entities_with_paths:
            entities_without_paths.append(entity)
    
    # 限制数量，避免过多的HICS计算
    max_hics_entities = 50  # 可以调整这个参数
    if len(entities_without_paths) > max_hics_entities:
        entities_without_paths = entities_without_paths[:max_hics_entities]
    
    return torch.LongTensor(entities_without_paths) if entities_without_paths else None

def enhanced_topk_path_selection(paths, path_scores, path_times, args):
    """增强的TopK路径选择"""
    if len(paths) == 0:
        return paths, path_scores
    
    enhanced_paths = []
    enhanced_scores = []
    
    for i, (path, score) in enumerate(zip(paths, path_scores)):
        # 获取路径的时间信息
        if i < len(path_times):
            times = path_times[i]
            
            # 检查时间约束（时间戳必须递增）
            if isinstance(times, (list, torch.Tensor)) and len(times) > 1:
                valid_time = all(times[j] <= times[j+1] for j in range(len(times)-1))
                if not valid_time:
                    continue
            
            # 计算路径长度惩罚
            path_length = len(path) if isinstance(path, (list, torch.Tensor)) else 1
            length_penalty = np.exp(-args.path_length_penalty * (path_length - 1))
            
            # 计算时间权重
            if isinstance(times, (list, torch.Tensor)) and len(times) > 0:
                max_time = max(times) if isinstance(times, list) else times.max().item()
                min_time = min(times) if isinstance(times, list) else times.min().item()
                time_span = max_time - min_time + 1
                time_weight = np.exp(-args.time_decay_factor * time_span)
            else:
                time_weight = 1.0
            
            # 综合评分
            enhanced_score = score * length_penalty * time_weight
            
            enhanced_paths.append(path)
            enhanced_scores.append(enhanced_score)
    
    # 排序并选择TopK
    if enhanced_paths:
        sorted_indices = np.argsort(enhanced_scores)[::-1][:args.topk_paths]
        selected_paths = [enhanced_paths[i] for i in sorted_indices]
        selected_scores = [enhanced_scores[i] for i in sorted_indices]
        return selected_paths, selected_scores
    
    return paths[:args.topk_paths], path_scores[:args.topk_paths]

args = parse_args()

mkdirs('./results/bestmodel/{}/conv'.format(args.dataset))
mkdirs('./results/bestmodel/{}/gat'.format(args.dataset))
model_state_file = './results/bestmodel/{}/'.format(args.dataset)

def load_data(args):
    with open(os.path.join('{}'.format(args.data), 'stat.txt'), 'r') as fr:
        for line in fr:
            line_split = line.split()
            num_e, num_r = int(line_split[0]), int(line_split[1])
    
    relation_embeddings = np.random.randn(num_r * 2, args.embedding_size)
    print("Initialised relations and entities randomly")
    return num_e, num_r, torch.FloatTensor(relation_embeddings)

num_e, num_r, relation_embeddings = load_data(args)

print("Initial relation dimensions {}".format(relation_embeddings.size()))

CUDA = torch.cuda.is_available()

def train_conv(args):
    print("Defining model")
    
    # 根据是否使用增强策略选择模型
    if args.use_hics or args.use_enhanced_topk:
        # from enhanced_typegat import Enhanced
        model = TypeGAT(
            num_e, num_r*2, relation_embeddings, args.embedding_size,
            time_decay_factor=args.time_decay_factor,
            path_length_penalty=args.path_length_penalty,
            hics_neighbor_sample=args.hics_neighbor_sample
        )
        print("Using Enhanced TypeGAT with HICS and TopK strategies")
    else:
        model = TypeGAT(num_e, num_r*2, relation_embeddings, args.embedding_size)
        print("Using standard TypeGAT")
    
    if CUDA:
        model.cuda()
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay_conv)
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.5, last_epoch=-1)
    
    margin_loss = torch.nn.SoftMarginLoss()
    cosine_loss = nn.CosineEmbeddingLoss(margin=0.0)
    
    epoch_losses = []
    print("Number of epochs {}".format(args.epochs_conv))
    
    graph_train = renamed_load(open(os.path.join(args.data + '/graph_preprocess_train.pk'), 'rb'))
    
    for epoch in range(args.epochs_conv):
        print("\nepoch-> ", epoch)
        
        model.train()
        start_time = time.time()
        epoch_loss = []
        
        for t, r_dict in tqdm(graph_train.t_r_id_p_dict.items()):
            batch_values = []
            batch_paths_id = []
            batch_relation = []
            batch_his_r = []
            query_time = torch.LongTensor([t])
            
            # for neg paths
            path_r = []
            path_neg_index = []
            
            for r, id_p in r_dict.items():
                len_r = 0
                p_neg_temp = []
                for id, ps in id_p.items():
                    len_r = len_r + len(ps)
                    
                    value = [-1] * len(ps)
                    value[0] = 1
                    batch_values.extend(value)
                    batch_paths_id.extend(ps)
                    
                    batch_his_r.extend(graph_train.r_copy[t][r][id])
                    if len(ps) > 1:
                        p_neg_temp.extend(list(eval('['+str(ps[1:]).replace("[", '').replace("]", '')+']')))
                
                batch_relation.extend([r]*len_r)
                path_r.extend([r]*len(p_neg_temp))
                path_neg_index.extend(p_neg_temp)
            
            # Enhanced TopK路径选择
            if args.use_enhanced_topk and len(batch_paths_id) > 0:
                paths = graph_train.t_paths[t] if t in graph_train.t_paths else []
                paths_time = graph_train.t_paths_time[t] if t in graph_train.t_paths_time else []
                
                # 计算路径分数（这里简化处理，实际应该根据模型输出）
                path_scores = [1.0] * len(batch_paths_id)
                
                selected_paths, selected_scores = enhanced_topk_path_selection(
                    batch_paths_id, path_scores, paths_time, args
                )
                batch_paths_id = selected_paths
            
            path_values = [-1]*len(path_r)
            path_values = torch.FloatTensor(np.expand_dims(np.array(path_values), axis=1))
            path_neg_index = torch.LongTensor(np.array(path_neg_index))
            path_r = torch.LongTensor(np.array(path_r))
            
            batch_paths_id = torch.LongTensor(list_to_array(batch_paths_id, 0))
            batch_relation = torch.LongTensor(np.array(batch_relation))
            batch_values = torch.FloatTensor(np.expand_dims(np.array(batch_values), axis=1))
            batch_his_r = torch.LongTensor(list_to_array(batch_his_r, num_r*2))
            
            paths = graph_train.t_paths[t] if t in graph_train.t_paths else []
            paths_time = graph_train.t_paths_time[t] if t in graph_train.t_paths_time else []
            lengths = graph_train.t_paths_len[t] if t in graph_train.t_paths_len else []
            
            if len(paths) != 0:
                paths = pad_sequence([torch.LongTensor(np.array(p)) for p in paths], 
                                   batch_first=True, padding_value=num_r*2)
                paths_time = pad_sequence([torch.LongTensor(np.array(p)) for p in paths_time], 
                                        batch_first=True, padding_value=0)
            else:
                paths = torch.LongTensor(np.array(paths))
                paths_time = torch.LongTensor(np.array(paths_time))
            lengths = torch.LongTensor(np.array(lengths))
            
            # HICS策略：构建历史图和查找无路径实体
            history_graph = None
            entity_without_paths = None
            if args.use_hics:
                history_graph = build_history_graph(graph_train, t, num_e, num_r)
                entity_without_paths = find_entities_without_paths(graph_train, t, batch_paths_id)
            
            if CUDA:
                batch_paths_id = Variable(batch_paths_id).cuda()
                batch_relation = Variable(batch_relation).cuda()
                batch_his_r = Variable(batch_his_r).cuda()
                paths = Variable(paths).cuda()
                paths_time = Variable(paths_time).cuda()
                lengths = Variable(lengths).cuda()
                path_r = Variable(path_r).cuda()
                path_neg_index = Variable(path_neg_index).cuda()
                query_time = Variable(query_time).cuda()
                batch_values = Variable(batch_values).cuda()
                path_values = Variable(path_values).cuda()
                
                if entity_without_paths is not None:
                    entity_without_paths = entity_without_paths.cuda()
            
            optimizer.zero_grad()
            
            # 使用增强版forward或标准forward
            if args.use_hics or args.use_enhanced_topk:
                preds, p_emb, r_emb = model.forward_with_strategies(
                    batch_paths_id, batch_relation, paths, paths_time, lengths,
                    path_r, path_neg_index, batch_his_r, query_time,
                    history_graph, entity_without_paths
                )
            else:
                preds, p_emb, r_emb = model.forward2(
                    batch_paths_id, batch_relation, paths, paths_time, lengths,
                    path_r, path_neg_index, batch_his_r, query_time
                )
            
            loss_e = margin_loss(preds.view(-1), batch_values.view(-1))
            loss_f = cosine_loss(p_emb, r_emb, path_values.view(-1))
            
            loss = loss_e  # 或 loss = loss_e + loss_f
            loss.backward()
            optimizer.step()
            
            if torch.isnan(torch.FloatTensor([loss.data.item()]))[0]:
                continue
            
            epoch_loss.append(loss.data.item())
        
        scheduler.step()
        
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        if epoch == 99:
            save_model(model, args.data, model_state_file, epoch)
        
        # if epoch == 99:
        #     save_model(model, args.data, model_state_file, epoch)

def evaluate_conv(args):
    # 根据是否使用增强策略选择模型
    if args.use_hics or args.use_enhanced_topk:
        # from enhanced_typegat import EnhancedTypeGAT
        model = TypeGAT(
            num_e, num_r*2, relation_embeddings, args.embedding_size,
            time_decay_factor=args.time_decay_factor,
            path_length_penalty=args.path_length_penalty,
            hics_neighbor_sample=args.hics_neighbor_sample
        )
    else:
        model = TypeGAT(num_e, num_r*2, relation_embeddings, args.embedding_size)
    
    model.load_state_dict(torch.load(
        '{0}/trained99.pth'.format(model_state_file)), strict=False)
    
    if CUDA:
        model.cuda()
    
    model.eval()
    
    with torch.no_grad():
        mr, mrr, hits1, hits3, hits10, hits100 = 0, 0, 0, 0, 0, 0
        test_size = 0
        graph_test = renamed_load(open(os.path.join(args.data + '/graph_preprocess_test.pk'), 'rb'))
        
        for t, r_dict in tqdm(graph_test.t_r_id_p_dict.items()):
            size = 0
            ranks_tail = []
            reciprocal_ranks_tail = []
            hits_at_100_tail = 0
            hits_at_ten_tail = 0
            hits_at_three_tail = 0
            hits_at_one_tail = 0
            
            for r, id_p in r_dict.items():
                for id, ps in id_p.items():
                    len_r = 0
                    batch_paths_id = []
                    batch_relation = []
                    batch_his_r = []
                    query_time = torch.LongTensor([t])
                    
                    size = size + 1
                    len_r = len_r + len(ps)
                    batch_paths_id.extend(ps)
                    batch_relation.extend([r] * len_r)
                    batch_his_r.extend(graph_test.r_copy[t][r][id])
                    
                    batch_paths_id = torch.LongTensor(list_to_array(batch_paths_id, 0))
                    batch_relation = torch.LongTensor(np.array(batch_relation))
                    batch_his_r = torch.LongTensor(list_to_array(batch_his_r, num_r * 2))
                    
                    paths = graph_test.t_paths[t] if t in graph_test.t_paths else []
                    lengths = graph_test.t_paths_len[t] if t in graph_test.t_paths_len else []
                    paths_time = graph_test.t_paths_time[t] if t in graph_test.t_paths_time else []
                    
                    if len(paths) != 0:
                        paths = pad_sequence([torch.LongTensor(np.array(p)) for p in paths],
                                           batch_first=True, padding_value=num_r*2)
                        paths_time = pad_sequence([torch.LongTensor(np.array(p)) for p in paths_time],
                                                batch_first=True, padding_value=0)
                    else:
                        paths = torch.LongTensor(np.array(paths))
                        paths_time = torch.LongTensor(np.array(paths_time))
                    lengths = torch.LongTensor(np.array(lengths))
                    
                    if CUDA:
                        batch_paths_id = Variable(batch_paths_id).cuda()
                        batch_relation = Variable(batch_relation).cuda()
                        paths = Variable(paths).cuda()
                        lengths = Variable(lengths).cuda()
                        paths_time = Variable(paths_time).cuda()
                        batch_his_r = Variable(batch_his_r).cuda()
                        query_time = Variable(query_time).cuda()
                    
                    scores_tail = model.test(batch_paths_id, batch_relation, paths, 
                                            lengths, paths_time, batch_his_r, query_time)
                    
                    sorted_scores_tail, sorted_indices_tail = torch.sort(
                        scores_tail.view(-1), dim=-1, descending=True)
                    
                    ranks_tail.append(
                        np.where(sorted_indices_tail.cpu().numpy() == 0)[0][0] + 1)
                    reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
            
            for i in range(len(ranks_tail)):
                if ranks_tail[i] <= 100:
                    hits_at_100_tail = hits_at_100_tail + 1
                if ranks_tail[i] <= 10:
                    hits_at_ten_tail = hits_at_ten_tail + 1
                if ranks_tail[i] <= 3:
                    hits_at_three_tail = hits_at_three_tail + 1
                if ranks_tail[i] == 1:
                    hits_at_one_tail = hits_at_one_tail + 1
            
            if len(ranks_tail) == 0:
                continue
            
            t_hits100 = hits_at_100_tail / len(ranks_tail)
            t_hits10 = hits_at_ten_tail / len(ranks_tail)
            t_hits3 = hits_at_three_tail / len(ranks_tail)
            t_hits1 = hits_at_one_tail / len(ranks_tail)
            t_mr = sum(ranks_tail) / len(ranks_tail)
            t_mrr = sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)
            
            test_size = test_size + size
            mrr += t_mrr * size
            mr += t_mr * size
            hits1 += t_hits1 * size
            hits3 += t_hits3 * size
            hits10 += t_hits10 * size
            hits100 += t_hits100 * size
        
        mrr = mrr / test_size
        mr = mr / test_size
        hits1 = hits1 / test_size
        hits3 = hits3 / test_size
        hits10 = hits10 / test_size
        hits100 = hits100 / test_size
        
        print("\n=== Final Results ===")
        print("MR : {:.6f}".format(mr))
        print("MRR : {:.6f}".format(mrr))
        print("Hits @ 1: {:.6f}".format(hits1))
        print("Hits @ 3: {:.6f}".format(hits3))
        print("Hits @ 10: {:.6f}".format(hits10))
        print("Hits @ 100: {:.6f}".format(hits100))

if __name__ == "__main__":
    # 训练模型
    train_conv(args)
    
    # 评估模型
    evaluate_conv(args)