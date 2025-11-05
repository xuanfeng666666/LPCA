import json, os
import math, copy, time
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import dill
import torch.nn.functional as F

class ImprovedPathSampler:
    """TPNet风格的路径采样策略"""
    
    def __init__(self, frequency_threshold=5, time_decay=100.0):
        self.frequency_threshold = frequency_threshold
        self.time_decay = time_decay
        self.path_frequency_cache = {}
        
    def compute_path_frequency(self, path):
        """计算路径频率（简化版：使用路径长度倒数）"""
        return len(path) if len(path) > 0 else 1
    
    def sample_paths_with_priority(self, paths, timestamps, lengths, query_time, top_k=10):
        """
        优先采样低频、时间近、短路径
        
        Returns:
            sampled_paths, sampled_times, sampled_lengths, index_mapping
            index_mapping: dict {old_index: new_index} 用于重映射batch_paths_id
        """
        # 边界情况处理
        if len(paths) == 0:
            return [], [], [], {}
            
        if len(paths) <= top_k:
            # 不需要采样，返回恒等映射
            identity_mapping = {i: i for i in range(len(paths))}
            return paths, timestamps, lengths, identity_mapping
        
        # 确保三个列表长度一致
        min_len = min(len(paths), len(timestamps), len(lengths))
        if min_len < len(paths):
            paths = paths[:min_len]
            timestamps = timestamps[:min_len]
            lengths = lengths[:min_len]
        
        scores = []
        for i in range(len(paths)):
            try:
                path = paths[i]
                path_time = timestamps[i]
                path_len = lengths[i]
                
                # 1. 频率得分（低频优先）
                freq = self.compute_path_frequency(path)
                freq_score = 1.0 / (freq + 1.0)
                
                # 2. 时间得分（近期优先）
                if isinstance(path_time, (list, tuple)) and len(path_time) > 0:
                    start_time = path_time[0]
                elif isinstance(path_time, (int, float, np.integer, np.floating)):
                    start_time = int(path_time)
                else:
                    start_time = query_time
                
                time_diff = max(query_time - start_time, 0)
                time_score = np.exp(-float(time_diff) / self.time_decay)
                
                # 3. 长度惩罚（短路径优先）
                length_penalty = 1.0 / max(path_len, 1)
                
                # 综合得分
                total_score = freq_score * time_score * length_penalty
                
                # 检查分数有效性
                if np.isnan(total_score) or np.isinf(total_score):
                    total_score = 0.0
                    
                scores.append((i, total_score))
                
            except Exception as e:
                scores.append((i, 0.0))
        
        # 按得分排序，取top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = min(top_k, len(scores))
        selected_indices = [item[0] for item in scores[:top_k]]
        
        # 创建索引映射：old_index -> new_index
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
        
        # 确保返回的列表长度一致
        sampled_paths = [paths[i] for i in selected_indices]
        sampled_times = [timestamps[i] for i in selected_indices]
        sampled_lengths = [lengths[i] for i in selected_indices]
        
        # 验证
        assert len(sampled_paths) == len(sampled_times) == len(sampled_lengths), \
            f"Sampling error: lengths don't match"
        
        return sampled_paths, sampled_times, sampled_lengths, index_mapping


class Graph():
    def __init__(self, use_sampling=False, max_paths=50, time_decay=100.0):
        super(Graph, self).__init__()
        
        self.t_r_id_p_dict = defaultdict(lambda: {})
        self.t_r_id_target_dict = defaultdict(lambda: {})
        self.r_copy = defaultdict(lambda: {})
        
        self.t_paths = defaultdict(lambda: [])
        self.t_paths_len = defaultdict(lambda: [])
        self.t_paths_time = defaultdict(lambda: [])
        self.t_paths_m_time = defaultdict(lambda: [])
        
        # 新增：采样相关属性
        self.use_sampling = use_sampling
        self.max_paths = max_paths
        if use_sampling:
            self.sampler = ImprovedPathSampler(time_decay=time_decay)
        else:
            self.sampler = None
    
    def get_sampled_paths(self, t):
        """
        获取时间t的采样后路径
        
        Returns:
            paths, paths_time, lengths, index_mapping
        """
        # 如果不使用采样，直接返回原始数据
        if not self.use_sampling or self.sampler is None:
            identity_mapping = {i: i for i in range(len(self.t_paths[t]))}
            return self.t_paths[t], self.t_paths_time[t], self.t_paths_len[t], identity_mapping
        
        original_paths = self.t_paths[t]
        original_times = self.t_paths_time[t]
        original_lengths = self.t_paths_len[t]
        
        # 空数据检查
        if len(original_paths) == 0:
            return [], [], [], {}
        
        # 数据一致性检查和修复
        min_len = min(len(original_paths), len(original_times), len(original_lengths))
        if min_len < len(original_paths):
            original_paths = original_paths[:min_len]
            original_times = original_times[:min_len]
            original_lengths = original_lengths[:min_len]
        
        # 如果路径数量不多，直接返回
        if len(original_paths) <= self.max_paths:
            identity_mapping = {i: i for i in range(len(original_paths))}
            return original_paths, original_times, original_lengths, identity_mapping
        
        # 应用采样
        try:
            sampled_paths, sampled_times, sampled_lengths, index_mapping = self.sampler.sample_paths_with_priority(
                original_paths, original_times, original_lengths, 
                query_time=t, 
                top_k=self.max_paths
            )
            
            # 最终验证
            if len(sampled_paths) != len(sampled_times) or len(sampled_paths) != len(sampled_lengths):
                print(f"Warning: Sampling failed at time {t}, using original data")
                identity_mapping = {i: i for i in range(min(len(original_paths), self.max_paths))}
                return original_paths[:self.max_paths], original_times[:self.max_paths], original_lengths[:self.max_paths], identity_mapping
                
            return sampled_paths, sampled_times, sampled_lengths, index_mapping
            
        except Exception as e:
            print(f"Error in sampling at time {t}: {e}")
            identity_mapping = {i: i for i in range(min(len(original_paths), self.max_paths))}
            return original_paths[:self.max_paths], original_times[:self.max_paths], original_lengths[:self.max_paths], identity_mapping


# ===== 添加这两个函数 =====
class RenameUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "GPT_GNN.data" or module == 'data':
            renamed_module = "pyHGT.data"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()
# ===========================