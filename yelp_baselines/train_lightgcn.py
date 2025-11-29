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

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim=64, n_layers=3):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=0.01)
        nn.init.normal_(self.item.weight, std=0.01)
        self.n_layers = n_layers

    def propagate(self, A_hat):
        # initial embeddings
        E_u = self.user.weight
        E_i = self.item.weight
        E = torch.cat([E_u, E_i], dim=0)
        embs = [E]
        for _ in range(self.n_layers):
            E = torch.sparse.mm(A_hat, E)
            embs.append(E)
        out = torch.stack(embs, dim=0).mean(dim=0)
        return out[:self.n_users], out[self.n_users:]

    def scores(self, U, I, A_hat):
        E_u, E_i = self.propagate(A_hat)
        u = E_u[U]
        i = E_i[I]
        return (u * i).sum(-1)

def build_normalized_adj(n_users, n_items, train_pairs, device):
    # Build symmetric normalized adjacency for bipartite graph
    N = n_users + n_items
    # degrees
    deg = torch.zeros(N, dtype=torch.float32)
    rows, cols, vals = [], [], []
    for (u, i) in train_pairs:
        ui = int(u); ii = int(i) + n_users
        rows += [ui, ii]; cols += [ii, ui]; vals += [1.0, 1.0]
        deg[ui] += 1; deg[ii] += 1
    rows = torch.tensor(rows, dtype=torch.long)
    cols = torch.tensor(cols, dtype=torch.long)
    vals = torch.tensor(vals, dtype=torch.float32)
    deg_inv_sqrt = torch.pow(deg + 1e-10, -0.5)
    norm_vals = deg_inv_sqrt[rows] * vals * deg_inv_sqrt[cols]
    A = torch.sparse_coo_tensor(
        torch.stack([rows, cols], dim=0),
        norm_vals, size=(N, N), device=device
    ).coalesce()
    return A

def bpr_loss(u_scores_pos, u_scores_neg, reg):
    loss = -torch.log(torch.sigmoid(u_scores_pos - u_scores_neg) + 1e-12).mean()
    return loss + reg

def evaluate(model, A_hat, train_pos, test_df, n_items, Ks=(10,)):
    model.eval()
    recalls = {k:[] for k in Ks}
    ndcgs = {k:[] for k in Ks}
    gt = test_df.groupby("u")["i"].apply(set).to_dict()
    users = sorted(gt.keys())
    with torch.no_grad():
        E_u, E_i = model.propagate(A_hat)
        for u in users:
            uemb = E_u[u:u+1]
            scores = torch.matmul(uemb, E_i.t()).squeeze(0).cpu().numpy()
            seen = train_pos.get(u, set())
            if seen:
                scores[list(seen)] = -np.inf
            ranked = np.argsort(-scores)
            for k in Ks:
                recalls[k].append(recall_at_k(ranked, gt[u], k))
                ndcgs[k].append(ndcg_at_k(ranked, gt[u], k))
    out = {f"Recall@{k}": float(np.mean(recalls[k])) for k in Ks}
    out.update({f"NDCG@{k}": float(np.mean(ndcgs[k])) for k in Ks})
    return out

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--neg_per_pos", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def run_cold_start_eval_lightgcn(args, model, A_hat, train_pos, test_df):
    inter = load_interactions(args.data_dir)
    freq = compute_item_frequencies(inter)
    buckets = make_frequency_buckets(freq, boundaries=(1, 3, 10, 10**9))

    with torch.no_grad():
        E_u, E_i = model.propagate(A_hat)
        user_emb = E_u.detach().cpu().numpy()
        item_emb = E_i.detach().cpu().numpy()

    results = evaluate_cold_start_from_embeddings(
        user_emb=user_emb,
        item_emb=item_emb,
        train_pos=train_pos,
        test_df=test_df,
        buckets=buckets,
        Ks=(10,),
    )

    print("=== LightGCN Cold-Start / Long-Tail Analysis (by item freq) ===")
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

    if device.type == "mps":
        device = torch.device("cpu")

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    with open(os.path.join(args.data_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    n_users, n_items = meta["n_users"], meta["n_items"]

    model = LightGCN(n_users, n_items, dim=args.dim, n_layers=args.layers).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    train_pairs = list(zip(train["u"].values.astype(int), train["i"].values.astype(int)))
    train_pos = build_user_pos_items(train_pairs)
    A_hat = build_normalized_adj(n_users, n_items, train_pairs, device)

    user_idx = train["u"].values.astype(np.int64)
    item_idx = train["i"].values.astype(np.int64)

    for epoch in range(1, args.epochs+1):
        model.train()
        idx = np.arange(len(user_idx))
        np.random.shuffle(idx)
        total = 0.0
        for start in range(0, len(idx), args.batch_size):
            sl = idx[start:start+args.batch_size]
            u = user_idx[sl]
            i = item_idx[sl]
            j = []
            for uu in u:
                j.extend(sample_neg(train_pos, n_items, uu, num=args.neg_per_pos))
            u = np.repeat(u, args.neg_per_pos)
            i = np.repeat(i, args.neg_per_pos)
            j = np.array(j, dtype=np.int64)

            u = torch.tensor(u, device=device)
            i = torch.tensor(i, device=device)
            j = torch.tensor(j, device=device)

            # forward with current embeddings (propagated through A_hat)
            E_u, E_i = model.propagate(A_hat)
            s_ui = (E_u[u] * E_i[i]).sum(-1)
            s_uj = (E_u[u] * E_i[j]).sum(-1)
            reg = args.l2 * (E_u[u].pow(2).sum() + E_i[i].pow(2).sum() + E_i[j].pow(2).sum()) / u.shape[0]

            opt.zero_grad()
            loss = bpr_loss(s_ui, s_uj, reg)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(sl)

        val_metrics = evaluate(model, A_hat, train_pos, val, n_items, Ks=(10,))
        print(f"[Epoch {epoch}] train_loss={total/len(idx):.4f} | val {val_metrics}")

    test_metrics = evaluate(model, A_hat, train_pos, test, n_items, Ks=(10,))
    print(f"[TEST] {test_metrics}")
    run_cold_start_eval_lightgcn(args, model, A_hat, train_pos, test)
    
if __name__ == "__main__":
    main()
