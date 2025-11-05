import torch
import os
import numpy as np
import networkx as nx
from torch.nn.utils.rnn import pack_padded_sequence ,pad_sequence ,pack_sequence
from collections import defaultdict
from more_itertools import flatten
from sklearn.utils import shuffle
from operator import itemgetter
from pyHGT.data import *
import dill

import argparse

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("-data", "--data",
                      default="./data/ICEWS18", help="data directory")
    args.add_argument("-state", "--state",
                      default="test", help="train or test")
    args.add_argument("-neg_ratio", "--ratio",
                      default=1, help="training neg ratio")
    args.add_argument("-his_len", "--his_len",
                      default=50, help="1 hop historial relations")
    args.add_argument("-batch_size", "--batch_size",
                      default=5, type=int, help="batch size for incremental saving")

    args = args.parse_args()
    return args

args = parse_args()


def load_from_batches(filepath):
    """
    从分批保存的文件中加载完整字典
    
    Args:
        filepath: 原始文件路径
        
    Returns:
        完整的字典
    """
    # 加载元数据
    metadata_file = filepath.replace('.pk', '_metadata.pk')
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    metadata = dill.load(open(metadata_file, 'rb'))
    num_batches = metadata['num_batches']
    
    print(f"Loading {metadata['total_items']} items from {num_batches} batches")
    
    # 加载所有批次
    full_dict = {}
    for i in range(num_batches):
        batch_file = filepath.replace('.pk', f'_batch_{i}.pk')
        if not os.path.exists(batch_file):
            print(f"Warning: Batch file not found: {batch_file}")
            continue
        
        batch_data = dill.load(open(batch_file, 'rb'))
        full_dict.update(batch_data)
        print(f"  Loaded batch {i+1}/{num_batches} ({len(batch_data)} items)")
    
    print(f"Total loaded: {len(full_dict)} items")
    return full_dict


def all_simple_edge_paths(G, source, target, cutoff=None):
    if source not in G:
        raise nx.NodeNotFound("source node %s not in graph" % source)
    if target in G:
        targets = {target}
    else:
        try:
            targets = set(target)
        except TypeError:
            raise nx.NodeNotFound("target node %s not in graph" % target)
    if source in targets:
        return []
    if cutoff is None:
        cutoff = len(G) - 1
    if cutoff < 1:
        return []
    if G.is_multigraph():
        for simp_path in _all_simple_edge_paths_multigraph(G, source, targets, cutoff):
            yield simp_path
    else:
        for simp_path in _all_simple_paths_graph(G, source, targets, cutoff):
            yield list(zip(simp_path[:-1], simp_path[1:]))

def _all_simple_edge_paths_multigraph(G, source, targets, cutoff):
    if not cutoff or cutoff < 1:
        return []
    visited = [source]
    stack = [iter(G.edges(source, keys=True))]

    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.pop()
        elif len(visited) < cutoff:
            if child[1] in targets:
                yield visited[1:] + [child]
            if child[1] not in [v[0] for v in visited[1:]]:
                visited.append(child)
                stack.append(iter(G.edges(child[1], keys=True)))
        else:  # len(visited) == cutoff:
            for (u, v, k) in [child] + list(children):
                if v in targets:
                    yield visited[1:] + [(u, v, k)]
            stack.pop()
            visited.pop()

def parse_line(line):
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2

def build_data(path,num_r):

    t_quads = {}
    t_quads_re = {}

    all_triples = set()
    quads_id = {}
    with open(os.path.join(path, 'data.txt'), 'r') as fr:
        times = set()
        for i, line in enumerate(fr):
            line_split = line.split()
            time = int(line_split[3])
            times.add(time)

            e1, relation, e2 = int(line_split[0]), int(line_split[1]), int(line_split[2])

            all_triples.add((e1, relation, e2))
            all_triples.add((e2, relation+num_r, e1))

            t_quads.setdefault(time, []).append((e1, relation, e2))
            t_quads_re.setdefault(time, []).append((e2, relation+num_r, e1))

        all_triples = list(all_triples)
        for i,triple in enumerate(all_triples):
            quads_id[triple]=i


    all_times = list(times)
    all_times.sort()

    t_quadid = {}
    t_quadid_re = {}
    for t in all_times:
        for i in t_quads[t]:
            t_quadid.setdefault(t, []).append(quads_id[i])
        for j in t_quads_re[t]:
            t_quadid_re.setdefault(t, []).append(quads_id[j])
    print("number of triples ->", len(all_triples))


    return t_quadid, t_quadid_re, all_triples, all_times

class Corpus:
    def __init__(self,args, all_triples, num_e, num_r):

        self.all_triples = all_triples
        self.num_e = num_e
        self.num_r = num_r


    def get_neg_triples_incremental(self, args, all_times, t_quads, test_idx, save_dir):
        """
        增量式生成负样本：边生成边保存边删除，避免内存爆炸
        
        Args:
            args: 参数
            all_times: 所有时间戳
            t_quads: 时间戳到quads的映射
            test_idx: 测试集起始索引
            save_dir: 保存目录
            
        Returns:
            saved_indices: 保存的时间戳索引列表
        """
        G = nx.Graph()
        
        times = range(test_idx, len(all_times))
        keys_his = all_times[:test_idx]
        quads_his = list(itemgetter(*keys_his)(t_quads))
        triples_his = np.array(self.all_triples)[list(set(flatten(quads_his)))]
        G.add_edges_from(triples_his[:, [0, 2]])

        # 用于分批保存
        quads_select_batch = {}
        quads_neg_batch = {}
        batch_count = 0
        saved_indices = []  # 记录所有保存的索引
        
        print(f"Starting incremental processing with batch_size={args.batch_size}")

        for idx in tqdm(times):
            quads = t_quads[all_times[idx]]
            quad_id = []
            neg_len = defaultdict(lambda: [])
            pre = 0
            valid_triples = np.array(self.all_triples)[quads].tolist()

            for i, quad in enumerate(quads):
                triple = self.all_triples[quad]

                try:
                    pred = nx.predecessor(G, triple[0], triple[2], 3)
                    if len(pred) > 0:
                        length = nx.shortest_path_length(G, triple[0], triple[2])
                    else:
                        continue
                except:
                    pass
                else:
                    if length < 4:
                        quad_id.append([quad, pre, i, length])
                        pre = i

                        paths_len = nx.single_source_shortest_path_length(G, triple[0], 3)
                        del paths_len[triple[0]]
                        
                        for target, l in paths_len.items():
                            if [triple[0], triple[1], target] not in valid_triples:
                                neg_len[quad].append([target, l])

            # 只保存有效数据
            if len(quad_id) != 0 and len(neg_len) != 0:
                quads_select_batch[idx] = quad_id
                quads_neg_batch[idx] = neg_len
                saved_indices.append(idx)
                
                # 达到批次大小时保存并清空
                if len(quads_select_batch) >= args.batch_size:
                    self._save_batch(quads_select_batch, quads_neg_batch, 
                                   batch_count, save_dir)
                    
                    # 关键：清空当前批次的内存
                    quads_select_batch.clear()
                    quads_neg_batch.clear()
                    batch_count += 1
                    
                    print(f"  Batch {batch_count} saved and memory cleared")

            # 更新图
            triples_his = np.array(self.all_triples)[t_quads[all_times[idx]]]
            G.add_edges_from(triples_his[:, [0, 2]])

        # 保存最后一批（如果有剩余）
        if len(quads_select_batch) > 0:
            self._save_batch(quads_select_batch, quads_neg_batch, 
                           batch_count, save_dir)
            batch_count += 1
            print(f"  Final batch {batch_count} saved")

        # 保存元数据
        metadata = {
            'num_batches': batch_count,
            'batch_size': args.batch_size,
            'total_items': len(saved_indices),
            'saved_indices': saved_indices
        }
        
        metadata_file_select = os.path.join(save_dir, 'quads_select_test_metadata.pk')
        metadata_file_neg = os.path.join(save_dir, 'quads_neg_test_metadata.pk')
        
        dill.dump(metadata, open(metadata_file_select, 'wb'))
        dill.dump(metadata, open(metadata_file_neg, 'wb'))
        
        print(f"\nIncremental saving completed:")
        print(f"  Total batches: {batch_count}")
        print(f"  Total items: {len(saved_indices)}")
        print(f"  Metadata saved")
        
        return saved_indices

    def _save_batch(self, quads_select_batch, quads_neg_batch, batch_count, save_dir):
        """保存单个批次"""
        select_file = os.path.join(save_dir, f'quads_select_test_batch_{batch_count}.pk')
        neg_file = os.path.join(save_dir, f'quads_neg_test_batch_{batch_count}.pk')
        
        dill.dump(dict(quads_select_batch), open(select_file, 'wb'))
        dill.dump(dict(quads_neg_batch), open(neg_file, 'wb'))


    def get_neg_triples(self, args, all_times, t_quads, test_idx):
        """原始方法保持不变，用于训练集等小数据"""
        quads_select = {}
        quads_neg = {}
        G = nx.Graph()
        
        times = range(test_idx, len(all_times))

        keys_his = all_times[:test_idx]

        quads_his = list(itemgetter(*keys_his)(t_quads))
        triples_his = np.array(self.all_triples)[list(set(flatten(quads_his)))]
        G.add_edges_from(triples_his[:, [0, 2]])

        for idx in tqdm(times):

            quads = t_quads[all_times[idx]]
            quad_id = []
            neg_len = defaultdict(lambda: [])
            pre = 0
            valid_triples = np.array(self.all_triples)[quads].tolist()

            for i, quad in enumerate(quads):
                triple = self.all_triples[quad]

                try:
                    pred = nx.predecessor(G, triple[0], triple[2], 3)
                    if len(pred) > 0:
                        length = nx.shortest_path_length(G, triple[0], triple[2])
                    else:
                        continue
                except:
                    pass
                else:
                    if length < 4:
                        quad_id.append([quad, pre, i, length])
                        pre = i

                        paths_len = nx.single_source_shortest_path_length(G, triple[0], 3)
                        del paths_len[triple[0]]
                        
                        for target, l in paths_len.items():
                            if [triple[0], triple[1], target] not in valid_triples:
                                neg_len[quad].append([target, l])

            if len(quad_id)!=0 and len(neg_len)!=0:
                quads_select[idx] = quad_id
                quads_neg[idx] = neg_len

            triples_his = np.array(self.all_triples)[t_quads[all_times[idx]]]
            G.add_edges_from(triples_his[:, [0, 2]])

        return quads_select, quads_neg


    def get_path_test(self,args, G,s,targets, lens, cur_time,num_r):
        target_pid = defaultdict(list)
        target_his_pid = defaultdict(list)


        try:
            paths = []
            for i in range(len(targets)):
                c_graph = G.subgraph([s,targets[i]])
                sG = G.subgraph(c_graph)
                paths.extend(list(all_simple_edge_paths(sG, s, targets[i], 3)))

            path_len = [len(path) for path in paths]
            p_id = np.argsort(path_len)

            tar_dict = {}
        except:
            print('sample error')
        else:
            if len(paths) != 0:
                for id in p_id:
                    path = paths[id]

                    t = np.array(path)[-1][1]
                    pa = np.array(path)[:, 2]
                    pa_t = [cur_time - G.edges[p]['time'] for p in path]



                    if t not in tar_dict.keys():
                        tar_dict[t] = [len(pa),max(pa_t)]
                    elif max(pa_t)<tar_dict[t][1]:
                        tar_dict[t][1] = max(pa_t)
                    elif len(pa)>tar_dict[t][0] and max(pa_t)>tar_dict[t][1]:
                        continue

                    if len(pa) == 3:
                        pl = self.pathlen_3[(pa[0],pa_t[0])][(pa[1],pa_t[1])][(pa[2],pa_t[2])]
                        if pl == 0:
                            self.paths.append(pa)
                            self.paths_time.append(pa_t)
                            self.lengths.append(len(pa))
                            self.paths_m_time.append(max(pa_t))
                            target_pid[t].append(len(self.paths))
                            self.pathlen_3[(pa[0],pa_t[0])][(pa[1],pa_t[1])][(pa[2],pa_t[2])] = len(self.paths)
                        else:
                            if pl not in target_pid[t]:
                                target_pid[t].append(pl)
                    elif len(pa) == 2:
                        pl = self.pathlen_2[(pa[0],pa_t[0])][(pa[1],pa_t[1])]
                        if pl == 0:
                            self.paths.append(pa)
                            self.paths_time.append(pa_t)
                            self.lengths.append(len(pa))
                            self.paths_m_time.append(max(pa_t))
                            target_pid[t].append(len(self.paths))
                            self.pathlen_2[(pa[0],pa_t[0])][(pa[1],pa_t[1])] = len(self.paths)
                        else:
                            if pl not in target_pid[t]:
                                target_pid[t].append(pl)
                    elif len(pa) == 1:
                        pl = self.pathlen_1[(pa[0],pa_t[0])]
                        if pa_t[0] <= args.his_len:
                            target_his_pid[t].append(pa[0])
                        if pl == 0:
                            self.paths.append(pa)
                            self.paths_time.append(pa_t)
                            self.lengths.append(len(pa))
                            self.paths_m_time.append(max(pa_t))
                            target_pid[t].append(len(self.paths))
                            self.pathlen_1[(pa[0],pa_t[0])] = len(self.paths)


                        else:
                            if pl not in target_pid[t]:
                                target_pid[t].append(pl)
        target_pid_sort = defaultdict(list)
        for t in target_pid.keys():
            if t not in target_his_pid.keys():
                target_pid_sort[t] = [num_r * 2]
            else:
                target_pid_sort[t] = target_his_pid[t]

        return target_pid, target_pid_sort


    def get_iteration_batch(self, args, G, batch_quads,negs,quads_cur,cur_time, num_r):

        self.paths = []
        self.lengths = []
        self.paths_time = []
        self.paths_m_time = []

        self.pathlen_1 = defaultdict(int)
        self.pathlen_2 = defaultdict(lambda: defaultdict(int))
        self.pathlen_3 = defaultdict(lambda:
                                defaultdict(lambda:
                                            defaultdict(int)))

        paths_dict =  defaultdict(lambda: defaultdict(list))
        targets_dict = defaultdict(lambda: defaultdict(list))

        paths_dict_copy = defaultdict(lambda:defaultdict(list))


        for quad, pre, pid, length in tqdm(batch_quads):
            target = []
            lens = []
            s, r, o = self.all_triples[quad]

            target.append(o)
            lens.append(length)
            neg = np.array(negs[quad])

            if len(neg) > 0:
                
                
                t_neg = neg[:, [0]]
                target.extend(t_neg.reshape(-1).tolist())

                l_neg = neg[:, [1]]
                lens.extend(l_neg.reshape(-1).tolist())

            subnodes = []
            subnodes.append(s)
            subnodes.extend(target)
            graph1 = G.subgraph(subnodes)
            H = G.subgraph(list(graph1.nodes()))

            target_pid, target_his_pid = self.get_path_test(args, H, s, target, lens, cur_time, num_r)
            if o not in target_pid.keys():
                continue

            # pos triple

            paths_dict[r][quad].append(target_pid[o])
            targets_dict[r][quad].append(o)

            paths_dict_copy[r][quad].append(target_his_pid[o])
            del target_pid[o], target_his_pid[o]

            # neg_triples
            if len(target_pid.keys()) == 0:
                continue

            paths_dict[r][quad].extend(list(target_pid.values()))
            targets_dict[r][quad].extend(list(target_pid.keys()))

            paths_dict_copy[r][quad].extend(list(target_his_pid.values()))


        del self.pathlen_1, self.pathlen_2, self.pathlen_3



        return paths_dict, targets_dict, self.paths, self.lengths, self.paths_time,paths_dict_copy, self.paths_m_time

def main():
    with open(os.path.join('{}'.format(args.data), 'stat.txt'), 'r') as fr:
        for line in fr:
            line_split = line.split()
            num_e, num_r = int(line_split[0]), int(line_split[1])

    t_quads, t_quads_re, all_triples, all_times = build_data(args.data, num_r)

    with open(os.path.join('{}'.format(args.data), 'split.txt'), 'r') as fr:
        for line in fr:
            line_split = line.split()

            valid_start = int(line_split[1].split(',')[0])

            test_start = int(line_split[2].split(',')[0])

    time_list = list(all_times)
    valid_idx = time_list.index(valid_start)
    test_idx = time_list.index(test_start)

    Corpus_ = Corpus(args, all_triples, num_e, num_r)

    print('sample')

    # 训练集和验证集使用原始方法（数据量小）
    if not os.path.exists(os.path.join(args.data + '/quads_select.pk')):
        quads_select, quads_neg = Corpus_.get_neg_triples(args, all_times, t_quads, test_idx)
        dill.dump(quads_select, open(args.data + '/quads_select.pk', 'wb'))
        dill.dump(quads_neg, open(args.data + '/quads_neg.pk', 'wb'))
    else:
        quads_select = renamed_load(open(os.path.join(args.data + '/quads_select.pk'), 'rb'))
        quads_neg = renamed_load(open(os.path.join(args.data + '/quads_neg.pk'), 'rb'))
    
    G = nx.MultiDiGraph()

    print('train')
    
    for idx in range(valid_idx):
        if idx < 1:
            continue
        triples_his = np.array(all_triples)[t_quads[all_times[idx - 1]]]
        G.add_edges_from(triples_his[:, [0, 2, 1]], time=idx-1)

        triples_his_re = np.array(all_triples)[t_quads_re[all_times[idx - 1]]]
        G.add_edges_from(triples_his_re[:, [0, 2, 1]], time=idx - 1)
        quads_cur = t_quads[all_times[idx]]
        
    
    for idx in range(valid_idx, test_idx):
        triples_his = np.array(all_triples)[t_quads[all_times[idx - 1]]]
        G.add_edges_from(triples_his[:, [0, 2, 1]], time=idx - 1)

        triples_his_re = np.array(all_triples)[t_quads_re[all_times[idx - 1]]]
        G.add_edges_from(triples_his_re[:, [0, 2, 1]], time=idx - 1)
        
    graph_test = Graph()
    
    # ============ 关键修改：使用增量式保存 ============
    metadata_file = os.path.join(args.data, 'quads_select_test_metadata.pk')
    
    if not os.path.exists(metadata_file):
        print("Computing negative samples for test set with incremental saving...")
        
        # 使用新的增量式方法：边生成边保存边删除
        saved_indices = Corpus_.get_neg_triples_incremental(
            args, all_times, t_quads, test_idx, args.data
        )
        
        print("Incremental saving completed successfully!")
        print(f"Processed {len(saved_indices)} time steps")
        
    else:
        print("Loading from existing batch files...")
        # 从批次文件加载
        quads_select = load_from_batches(args.data + '/quads_select_test.pk')
        quads_neg = load_from_batches(args.data + '/quads_neg_test.pk')
        print("Batch loading completed successfully!")


    # 测试阶段：需要重新加载数据进行处理
    print('\ntest - loading and processing batches')
    
    # 加载元数据以获取批次信息
    metadata = dill.load(open(metadata_file, 'rb'))
    num_batches = metadata['num_batches']
    
    quads_num = 0
    quads_select_num = 0
    quads_select_neg = 0
    
    # 按批次加载和处理，避免一次性加载所有数据
    for batch_idx in range(num_batches):
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches}")
        
        # 加载当前批次
        select_file = os.path.join(args.data, f'quads_select_test_batch_{batch_idx}.pk')
        neg_file = os.path.join(args.data, f'quads_neg_test_batch_{batch_idx}.pk')
        
        quads_select_batch = dill.load(open(select_file, 'rb'))
        quads_neg_batch = dill.load(open(neg_file, 'rb'))
        
        # 处理当前批次中的每个时间步
        for idx in quads_select_batch.keys():
            # 更新图到当前时间步
            if idx > test_idx:
                for i in range(max(test_idx, list(quads_select_batch.keys())[0]), idx):
                    if i in t_quads:
                        triples_his = np.array(all_triples)[t_quads[all_times[i]]]
                        G.add_edges_from(triples_his[:, [0, 2, 1]], time=i)
                        triples_his_re = np.array(all_triples)[t_quads_re[all_times[i]]]
                        G.add_edges_from(triples_his_re[:, [0, 2, 1]], time=i)
            
            # 更新到当前时间步
            if idx - 1 >= 0 and all_times[idx - 1] in t_quads:
                triples_his = np.array(all_triples)[t_quads[all_times[idx - 1]]]
                G.add_edges_from(triples_his[:, [0, 2, 1]], time=idx - 1)
                triples_his_re = np.array(all_triples)[t_quads_re[all_times[idx - 1]]]
                G.add_edges_from(triples_his_re[:, [0, 2, 1]], time=idx - 1)
            
            quads_cur = t_quads[all_times[idx]]
            quads_num += len(quads_cur)
            
            quads = quads_select_batch[idx]
            negs = quads_neg_batch[idx]
            quads_select_num += len(quads)
            quads_select_neg += len(negs)
            
            paths_dict, targets_dict, paths, lengths, paths_time, paths_dict_copy, paths_m_time = \
                Corpus_.get_iteration_batch(args, G, quads, negs, quads_cur, idx, num_r)
            
            graph_test.t_r_id_p_dict[idx] = paths_dict
            graph_test.t_r_id_target_dict[idx] = targets_dict
            graph_test.t_paths[idx] = paths
            graph_test.t_paths_len[idx] = lengths
            graph_test.t_paths_time[idx] = paths_time
            graph_test.t_paths_m_time[idx] = paths_m_time
            graph_test.r_copy[idx] = paths_dict_copy
            
            print(f"  Time step {idx}: {len(lengths)} paths")
        
        # 清理当前批次的数据
        del quads_select_batch, quads_neg_batch
        print(f"  Batch {batch_idx + 1} processed and memory cleared")

    print("\n" + "="*50)
    print("Summary:")
    print(f"  Select quads: {quads_select_num}")
    print(f"  Select neg: {quads_select_neg}")
    print(f"  All quads: {quads_num}")
    print("="*50)

    print("\nSaving final graph_preprocess_test.pk...")
    dill.dump(graph_test, open(args.data + '/graph_preprocess_test.pk', 'wb'))
    print("All done!")


if __name__ == '__main__':
    main()