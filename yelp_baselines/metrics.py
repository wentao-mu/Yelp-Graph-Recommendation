import numpy as np

def recall_at_k(ranked_items, ground_truth_set, k=10):
    if not ground_truth_set:
        return 0.0
    topk = ranked_items[:k]
    hits = sum((1 if i in ground_truth_set else 0) for i in topk)
    return hits / len(ground_truth_set)

def dcg_at_k(ranked_items, ground_truth_set, k=10):
    dcg = 0.0
    for idx, iid in enumerate(ranked_items[:k], start=1):
        if iid in ground_truth_set:
            dcg += 1.0 / np.log2(idx + 1)
    return dcg

def ndcg_at_k(ranked_items, ground_truth_set, k=10):
    dcg = dcg_at_k(ranked_items, ground_truth_set, k)
    ideal_hits = min(len(ground_truth_set), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
