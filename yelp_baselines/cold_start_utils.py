#!/usr/bin/env python


import os
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from metrics import recall_at_k, ndcg_at_k


def load_interactions(data_dir: str) -> pd.DataFrame:

    path = os.path.join(data_dir, "interactions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"interactions.csv not found at {path}. "
            "Make sure you ran the preprocess / split pipeline first."
        )
    inter = pd.read_csv(path)
    if not {"u", "i"}.issubset(inter.columns):
        raise ValueError(f"interactions.csv must contain 'u' and 'i' columns, got {inter.columns}")
    return inter


def compute_item_frequencies(inter: pd.DataFrame) -> pd.Series:
    freq = inter.groupby("i")["u"].size()
    freq.name = "freq"
    return freq


def make_frequency_buckets(
    freq: pd.Series,
    boundaries: Iterable[int] = (1, 3, 10, 10**9),
) -> Dict[str, np.ndarray]:
    boundaries = list(boundaries)
    if len(boundaries) < 2:
        raise ValueError("boundaries must have at least 2 values")

    buckets: Dict[str, np.ndarray] = {}
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        mask = (freq >= lo) & (freq < hi)
        item_ids = freq.index[mask].to_numpy(dtype=np.int64)
        name = f"[{lo},{hi})" if hi < 10**8 else f"[{lo},+inf)"
        buckets[name] = item_ids
    return buckets


def evaluate_cold_start_from_embeddings(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    train_pos: Dict[int, set],
    test_df: pd.DataFrame,
    buckets: Dict[str, np.ndarray],
    Ks: Tuple[int, ...] = (10,),
) -> Dict[str, Dict[str, float]]:
    n_users, dim_u = user_emb.shape
    n_items, dim_i = item_emb.shape
    assert dim_u == dim_i, "user and item embeddings must have same dimension"

    # per-user test ground truth
    gt = test_df.groupby("u")["i"].apply(set).to_dict()
    users = sorted(gt.keys())

    bucket_item_sets = {name: set(ids.tolist()) for name, ids in buckets.items()}

    out: Dict[str, Dict[str, float]] = {}
    bucket_recalls = {name: {k: [] for k in Ks} for name in buckets.keys()}
    bucket_ndcgs = {name: {k: [] for k in Ks} for name in buckets.keys()}
    bucket_num_users = {name: 0 for name in buckets.keys()}

    for u in users:
        if u < 0 or u >= n_users:
            continue

        u_vec = user_emb[u:u+1]                # [1, dim]
        scores = (u_vec @ item_emb.T).reshape(-1)  # [n_items]

        seen = train_pos.get(u, set())
        if seen:
            scores[list(seen)] = -np.inf

        ranked = np.argsort(-scores)  

        for bucket_name, item_ids in buckets.items():
            bucket_gt = gt[u].intersection(bucket_item_sets[bucket_name])
            if not bucket_gt:
                continue

            bucket_num_users[bucket_name] += 1
            for k in Ks:
                r = recall_at_k(ranked, bucket_gt, k)
                nd = ndcg_at_k(ranked, bucket_gt, k)
                bucket_recalls[bucket_name][k].append(r)
                bucket_ndcgs[bucket_name][k].append(nd)

    for bucket_name in buckets.keys():
        res = {}
        for k in Ks:
            recs = bucket_recalls[bucket_name][k]
            nds = bucket_ndcgs[bucket_name][k]
            res[f"Recall@{k}"] = float(np.mean(recs)) if recs else float("nan")
            res[f"NDCG@{k}"] = float(np.mean(nds)) if nds else float("nan")
        res["num_users"] = int(bucket_num_users[bucket_name])
        out[bucket_name] = res

    return out
