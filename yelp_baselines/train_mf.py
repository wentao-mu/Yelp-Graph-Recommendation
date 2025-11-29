#!/usr/bin/env python
import argparse, os, json, pandas as pd, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from utils import set_seed, build_user_pos_items, sample_neg, get_device
from metrics import recall_at_k, ndcg_at_k

from cold_start_utils import (
    load_interactions,
    compute_item_frequencies,
    make_frequency_buckets,
    evaluate_cold_start_from_embeddings,
)

class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=0.01)
        nn.init.normal_(self.item.weight, std=0.01)

    def forward(self, u, i):
        uemb = self.user(u)
        iemb = self.item(i)
        return (uemb * iemb).sum(-1)

def bpr_loss(model, u, i, j, l2=1e-4):
    # maximize log-sigmoid(s_ui - s_uj)
    s_ui = model(u, i)
    s_uj = model(u, j)
    loss = -torch.log(torch.sigmoid(s_ui - s_uj) + 1e-12).mean()
    # L2 on embeddings for involved indices
    reg = (model.user(u).pow(2).sum() + model.item(i).pow(2).sum() + model.item(j).pow(2).sum()) / u.shape[0]
    return loss + l2 * reg

def evaluate(model, train_pos, test_df, n_items, Ks=(10,)):
    model.eval()
    recalls = {k:[] for k in Ks}
    ndcgs = {k:[] for k in Ks}
    # build per-user test GT
    gt = test_df.groupby("u")["i"].apply(set).to_dict()
    users = sorted(gt.keys())
    with torch.no_grad():
        for u in users:
            # score all items (mask train positives)
            u_t = torch.tensor([u], device=model.user.weight.device)
            all_items = torch.arange(n_items, device=u_t.device)
            u_expand = u_t.repeat(n_items)
            scores = model(u_expand, all_items).cpu().numpy()
            # filter items already seen in training
            seen = train_pos.get(u, set())
            scores[list(seen)] = -np.inf
            ranked = np.argsort(-scores)
            for k in Ks:
                recalls[k].append(recall_at_k(ranked, gt[u], k))
                ndcgs[k].append(ndcg_at_k(ranked, gt[u], k))
    out = {f"Recall@{k}": float(np.mean(recalls[k])) for k in Ks}
    out.update({f"NDCG@{k}": float(np.mean(ndcgs[k])) for k in Ks})
    return out

def load_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    with open(os.path.join(data_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    return train, val, test, meta

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--neg_per_pos", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def run_cold_start_eval_mf(args, model, train_pos, test_df):
    inter = load_interactions(args.data_dir)
    freq = compute_item_frequencies(inter)

    buckets = make_frequency_buckets(freq, boundaries=(1, 3, 10, 10**9))

    user_emb = model.user.weight.detach().cpu().numpy()
    item_emb = model.item.weight.detach().cpu().numpy()

    results = evaluate_cold_start_from_embeddings(
        user_emb=user_emb,
        item_emb=item_emb,
        train_pos=train_pos,
        test_df=test_df,
        buckets=buckets,
        Ks=(10,),
    )

    print("=== MF Cold-Start / Long-Tail Analysis (by item freq) ===")
    for name, stats in results.items():
        print(
            f"Bucket {name}: "
            f"num_users={stats['num_users']}, "
            f"Recall@10={stats['Recall@10']:.4f}, "
            f"NDCG@10={stats['NDCG@10']:.4f}"
        )

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train, val, test, meta = load_data(args.data_dir)
    n_users, n_items = meta["n_users"], meta["n_items"]

    model = BPRMF(n_users, n_items, dim=args.dim).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    train_pos = build_user_pos_items(list(zip(train["u"].values, train["i"].values)))

    users = train["u"].values.astype(np.int64)
    items = train["i"].values.astype(np.int64)

    for epoch in range(1, args.epochs+1):
        model.train()
        idx = np.arange(len(users))
        np.random.shuffle(idx)

        total = 0.0
        for start in range(0, len(idx), args.batch_size):
            sl = idx[start:start+args.batch_size]
            u = users[sl]
            i = items[sl]
            j = []
            for uu in u:
                j.extend(sample_neg(train_pos, n_items, uu, num=args.neg_per_pos))
            u = np.repeat(u, args.neg_per_pos)
            i = np.repeat(i, args.neg_per_pos)
            j = np.array(j, dtype=np.int64)

            u = torch.tensor(u, device=device)
            i = torch.tensor(i, device=device)
            j = torch.tensor(j, device=device)

            opt.zero_grad()
            loss = bpr_loss(model, u, i, j, l2=args.l2)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(sl)
        # evaluate
        val_metrics = evaluate(model, train_pos, val, n_items, Ks=(10,))
        print(f"[Epoch {epoch}] train_loss={total/len(idx):.4f} | val {val_metrics}")

    # final test
    test_metrics = evaluate(model, train_pos, test, n_items, Ks=(10,))
    print(f"[TEST] {test_metrics}")
    run_cold_start_eval_mf(args, model, train_pos, test)

if __name__ == "__main__":
    main()
