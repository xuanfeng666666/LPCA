
import torch


def get_total_rank(scores, targets, all_ans=None, eval_bz=1000):
    n_batch = (len(scores) + eval_bz - 1) // eval_bz
    ranks, filter_ranks = [], []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(len(scores), (idx + 1) * eval_bz)
        score_batch = scores[batch_start:batch_end]
        target_batch = targets[batch_start:batch_end]
        ranks.append(sort_and_rank(score_batch, target_batch))
        if all_ans is not None:
            filter_score_batch = filter_score(score_batch, target_batch, all_ans)
            filter_ranks.append(sort_and_rank(filter_score_batch, target_batch))
    ranks = torch.cat(ranks) + 1
    filter_ranks = torch.cat(filter_ranks) + 1 if all_ans else ranks
    mrr = torch.mean(1.0 / ranks.float()).item()
    filter_mrr = torch.mean(1.0 / filter_ranks.float()).item()
    hits = []
    for k in [1, 3, 10, 100]:
        hits.append(torch.mean((ranks <= k).float()).item())
    return mrr, filter_mrr, hits, ranks, filter_ranks