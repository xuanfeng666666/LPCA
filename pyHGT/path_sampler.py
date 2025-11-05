import numpy as np
import torch

class ImprovedPathSampler:
    """TPNet风格的路径采样策略"""
    
    def __init__(self, frequency_threshold=5, time_decay=100.0):
        self.frequency_threshold = frequency_threshold
        self.time_decay = time_decay
        self.path_frequency_cache = {}  # 缓存路径频率
        
    def compute_path_frequency(self, path, graph):
        """计算路径中边的平均频率"""
        path_key = tuple(path)
        if path_key in self.path_frequency_cache:
            return self.path_frequency_cache[path_key]
        
        # 简化：使用路径长度的倒数作为频率估计
        # 实际使用时可以统计该路径在历史中出现的次数
        freq = len(path) if len(path) > 0 else 1
        self.path_frequency_cache[path_key] = freq
        return freq
    
    def sample_paths_with_priority(self, paths_dict, timestamps_dict, lengths_dict, 
                                   query_time, top_k=10):
        """
        优先采样低频、时间近、短路径
        
        Args:
            paths_dict: {path_id: path_list} 路径字典
            timestamps_dict: {path_id: time_list} 时间戳字典
            lengths_dict: {path_id: length} 长度字典
            query_time: 查询时间
            top_k: 保留的路径数量
        
        Returns:
            采样后的路径、时间戳、长度
        """
        if len(paths_dict) == 0:
            return {}, {}, {}
        
        scores = []
        path_ids = list(paths_dict.keys())
        
        for path_id in path_ids:
            path = paths_dict[path_id]
            path_time = timestamps_dict[path_id]
            path_len = lengths_dict[path_id]
            
            # 1. 频率得分（低频优先，逆序）
            freq = self.compute_path_frequency(path, None)
            freq_score = 1.0 / (freq + 1.0)
            
            # 2. 时间得分（近期优先，指数衰减）
            if len(path_time) > 0:
                start_time = path_time[0]
                time_diff = max(query_time - start_time, 0)
                time_score = np.exp(-time_diff / self.time_decay)
            else:
                time_score = 0.0
            
            # 3. 长度惩罚（短路径优先）
            length_penalty = 1.0 / max(path_len, 1)
            
            # 综合得分
            total_score = freq_score * time_score * length_penalty
            scores.append((path_id, total_score))
        
        # 按得分排序，取top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = min(len(scores), top_k)
        selected_ids = [item[0] for item in scores[:top_k]]
        
        # 返回采样后的路径
        sampled_paths = {pid: paths_dict[pid] for pid in selected_ids}
        sampled_times = {pid: timestamps_dict[pid] for pid in selected_ids}
        sampled_lengths = {pid: lengths_dict[pid] for pid in selected_ids}
        
        return sampled_paths, sampled_times, sampled_lengths