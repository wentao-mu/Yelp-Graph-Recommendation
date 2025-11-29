import argparse
import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import set_seed, get_device
from metrics import recall_at_k, ndcg_at_k


def read_meta(data_dir: str) -> Tuple[int, int]:
    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    n_users = int(meta["n_users"])
    n_items = int(meta["n_items"])
    return n_users, n_items


def read_category_count(data_dir: str) -> int:
    cat_map_path = os.path.join(data_dir, "category_map.csv")
    if not os.path.exists(cat_map_path):
        return 0
    df = pd.read_csv(cat_map_path)
    return df.shape[0]


def build_user_pos_items(df: pd.DataFrame) -> Dict[int, set]:
    """user -> set(pos_item)"""
    user_pos: Dict[int, set] = {}
    for u, i in zip(df["u"].values, df["i"].values):
        u = int(u)
        i = int(i)
        user_pos.setdefault(u, set()).add(i)
    return user_pos


def build_sparse_adj(
    num_src: int,
    num_tgt: int,
    edges: List[Tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    """
    构建行归一化的稀疏邻接矩阵 A (num_src x num_tgt)：
    每一行的和为 1，用 torch.sparse.mm 做 message passing。
    """
    if len(edges) == 0:
        return None

    rows = np.array([e[0] for e in edges], dtype=np.int64)
    cols = np.array([e[1] for e in edges], dtype=np.int64)

    indices = torch.tensor(
        np.vstack([rows, cols]),
        dtype=torch.long,
        device=device,
    )
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)

    adj = torch.sparse_coo_tensor(indices, values, size=(num_src, num_tgt))
    adj = adj.coalesce()

    # row-normalize
    row_sum = torch.sparse.sum(adj, dim=1).to_dense()  # (num_src,)
    row_sum[row_sum == 0] = 1.0
    inv_row = 1.0 / row_sum
    inv_row = inv_row[adj.indices()[0]]

    norm_values = adj.values() * inv_row
    norm_adj = torch.sparse_coo_tensor(
        adj.indices(), norm_values, size=adj.shape
    ).coalesce()
    return norm_adj


class SimpleHeteroGNN(nn.Module):
    """
    一个极简的 heterogeneous GNN：
      节点类型：user, item, category
      关系：
        user--item (interactions)
        user--user (friends)
        item--category
    每一层：
        u_{k+1} <- A_ui * i_k + A_uu * u_k
        i_{k+1} <- A_iu * u_k + A_ic * c_k
        c_{k+1} <- A_ci * i_k
    最后把每一层的 embedding 取平均（类似 LightGCN）。
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_cats: int,
        dim: int,
        n_layers: int,
        A_ui: torch.Tensor,
        A_iu: torch.Tensor,
        A_uu: torch.Tensor = None,
        A_ic: torch.Tensor = None,
        A_ci: torch.Tensor = None,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_cats = n_cats
        self.dim = dim
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.cat_emb = nn.Embedding(n_cats, dim) if n_cats > 0 else None

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        if self.cat_emb is not None:
            nn.init.normal_(self.cat_emb.weight, std=0.01)

        self.A_ui = A_ui  # user <- item
        self.A_iu = A_iu  # item <- user
        self.A_uu = A_uu  # user <- user
        self.A_ic = A_ic  # item <- cat
        self.A_ci = A_ci  # cat  <- item

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """整图 propagation，返回最终的 (user_emb, item_emb, cat_emb)。"""
        h_u = self.user_emb.weight
        h_i = self.item_emb.weight
        h_c = self.cat_emb.weight if self.cat_emb is not None else None

        all_user_layers = [h_u]
        all_item_layers = [h_i]

        for _ in range(self.n_layers):
            # user from items
            msg_u_from_i = (
                torch.sparse.mm(self.A_ui, h_i) if self.A_ui is not None else 0
            )
            # item from users
            msg_i_from_u = (
                torch.sparse.mm(self.A_iu, h_u) if self.A_iu is not None else 0
            )
            # user from friends
            msg_u_from_u = (
                torch.sparse.mm(self.A_uu, h_u) if self.A_uu is not None else 0
            )

            # item <-> category
            if h_c is not None and self.A_ic is not None and self.A_ci is not None:
                msg_i_from_c = torch.sparse.mm(self.A_ic, h_c)
                msg_c_from_i = torch.sparse.mm(self.A_ci, h_i)
            else:
                msg_i_from_c = 0
                msg_c_from_i = 0 if h_c is not None else None

            new_h_u = msg_u_from_i
            if isinstance(msg_u_from_u, torch.Tensor):
                new_h_u = new_h_u + msg_u_from_u

            new_h_i = msg_i_from_u
            if isinstance(msg_i_from_c, torch.Tensor):
                new_h_i = new_h_i + msg_i_from_c

            if h_c is not None:
                if isinstance(msg_c_from_i, torch.Tensor):
                    new_h_c = msg_c_from_i
                else:
                    new_h_c = h_c
            else:
                new_h_c = None

            h_u = new_h_u
            h_i = new_h_i
            h_c = new_h_c

            all_user_layers.append(h_u)
            all_item_layers.append(h_i)

        out_u = torch.stack(all_user_layers, dim=0).mean(dim=0)
        out_i = torch.stack(all_item_layers, dim=0).mean(dim=0)
        return out_u, out_i, h_c

    def get_user_item_emb(self) -> Tuple[torch.Tensor, torch.Tensor]:
        u, i, _ = self.propagate()
        return u, i


def bpr_loss(
    u_e: torch.Tensor,
    i_e: torch.Tensor,
    j_e: torch.Tensor,
    l2: float,
) -> torch.Tensor:
    """标准 BPR loss + 简单 L2 正则。"""
    x_ui = (u_e * i_e).sum(dim=1)
    x_uj = (u_e * j_e).sum(dim=1)
    x = x_ui - x_uj
    loss = -F.logsigmoid(x).mean()
    reg = (u_e.pow(2).sum() + i_e.pow(2).sum() + j_e.pow(2).sum()) / u_e.size(0)
    return loss + l2 * reg


def sample_bpr_batch(
    train_df: pd.DataFrame,
    user_pos: Dict[int, set],
    n_items: int,
    batch_size: int,
    neg_per_pos: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从 train.csv 里采样一批 (u, pos_i, neg_j) 三元组。
    """
    idx = np.random.randint(0, len(train_df), size=batch_size)
    users = train_df["u"].values[idx]
    items = train_df["i"].values[idx]

    u_list, pos_list, neg_list = [], [], []
    for u, i in zip(users, items):
        u = int(u)
        i = int(i)
        for _ in range(neg_per_pos):
            cnt = 0
            while True:
                j = np.random.randint(0, n_items)
                if j not in user_pos[u]:
                    break
                cnt += 1
                if cnt > 50:  # 兜底防止极端情况死循环
                    break
            u_list.append(u)
            pos_list.append(i)
            neg_list.append(j)

    u_t = torch.tensor(u_list, dtype=torch.long, device=device)
    p_t = torch.tensor(pos_list, dtype=torch.long, device=device)
    n_t = torch.tensor(neg_list, dtype=torch.long, device=device)
    return u_t, p_t, n_t


def evaluate(
    model: SimpleHeteroGNN,
    train_pos: Dict[int, set],
    eval_df: pd.DataFrame,
    n_items: int,
    device: torch.device,
    Ks=(10,),
) -> Dict[str, float]:
    """
    全量 ranking 评估：对每个有 test 行为的 user，
    在全 item 上打分，过滤掉训练正样本，算 Recall@K / NDCG@K。
    """
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.get_user_item_emb()
        user_emb = user_emb.to("cpu")
        item_emb = item_emb.to("cpu")

    user_test_items = eval_df.groupby("u")["i"].apply(set).to_dict()

    recalls = {f"Recall@{K}": [] for K in Ks}
    ndcgs = {f"NDCG@{K}": [] for K in Ks}

    for u, gt_items in user_test_items.items():
        u = int(u)
        if u not in train_pos:
            continue
        gt_items = list(gt_items)

        u_vec = user_emb[u]  # (dim,)
        scores = torch.matmul(item_emb, u_vec).numpy()

        # 过滤掉训练正样本，避免泄露
        pos_items = train_pos[u]
        if len(pos_items) > 0:
            scores[list(pos_items)] = -1e9

        ranked = np.argsort(-scores)  # 从大到小排序

        for K in Ks:
            rec = recall_at_k(ranked, gt_items, K)
            ndcg = ndcg_at_k(ranked, gt_items, K)
            recalls[f"Recall@{K}"].append(rec)
            ndcgs[f"NDCG@{K}"].append(ndcg)

    metrics = {}
    for K in Ks:
        key_r = f"Recall@{K}"
        key_n = f"NDCG@{K}"
        if len(recalls[key_r]) == 0:
            metrics[key_r] = 0.0
            metrics[key_n] = 0.0
        else:
            metrics[key_r] = float(np.mean(recalls[key_r]))
            metrics[key_n] = float(np.mean(ndcgs[key_n]))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--neg_per_pos", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_friends", type=int, default=1)
    parser.add_argument("--use_categories", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    # M4 上 Torch 的 sparse + MPS 不友好，这里直接 fallback 到 CPU
    if device.type == "mps":
        print("[HeteroGNN] sparse ops not supported on MPS, falling back to CPU.")
        device = torch.device("cpu")

    n_users, n_items = read_meta(args.data_dir)
    n_cats = read_category_count(args.data_dir)
    print(f"#users={n_users}, #items={n_items}, #cats={n_cats}")

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    train_pos = build_user_pos_items(train)

    # ---------- Build heterogeneous relations ----------
    # user–item(interactions) 只用 train，避免泄露
    ub_edges = [(int(u), int(i)) for u, i in zip(train["u"].values, train["i"].values)]
    A_ui = build_sparse_adj(n_users, n_items, ub_edges, device)               # user <- item
    A_iu = build_sparse_adj(n_items, n_users, [(i, u) for (u, i) in ub_edges], device)  # item <- user

    # user–user friends
    friends_path = os.path.join(args.data_dir, "user_friends.csv")
    if args.use_friends and os.path.exists(friends_path):
        fr = pd.read_csv(friends_path)
        if {"u", "v"}.issubset(fr.columns):
            uu = fr["u"].values.astype(int)
            vv = fr["v"].values.astype(int)
        else:
            # 兜底：取前两列
            uu = fr.iloc[:, 0].values.astype(int)
            vv = fr.iloc[:, 1].values.astype(int)
        uu_edges = list(zip(uu, vv)) + list(zip(vv, uu))
        A_uu = build_sparse_adj(n_users, n_users, uu_edges, device)
    else:
        A_uu = None

    # item–category
    bc_path = os.path.join(args.data_dir, "business_categories.csv")
    if args.use_categories and n_cats > 0 and os.path.exists(bc_path):
        bc = pd.read_csv(bc_path)
        if {"i", "c"}.issubset(bc.columns):
            ii = bc["i"].values.astype(int)
            cc = bc["c"].values.astype(int)
        else:
            # 兜底：取前两列
            ii = bc.iloc[:, 0].values.astype(int)
            cc = bc.iloc[:, 1].values.astype(int)
        ic_edges = list(zip(ii, cc))  # item -> cat
        ci_edges = list(zip(cc, ii))  # cat  -> item
        A_ic = build_sparse_adj(n_items, n_cats, ic_edges, device)
        A_ci = build_sparse_adj(n_cats, n_items, ci_edges, device)
    else:
        A_ic = None
        A_ci = None

    # ---------- Model & optimizer ----------
    model = SimpleHeteroGNN(
        n_users=n_users,
        n_items=n_items,
        n_cats=n_cats,
        dim=args.dim,
        n_layers=args.layers,
        A_ui=A_ui,
        A_iu=A_iu,
        A_uu=A_uu,
        A_ic=A_ic,
        A_ci=A_ci,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = 0.0
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        # 每个 epoch 随机采样一批 BPR 三元组
        u, p, n = sample_bpr_batch(
            train_df=train,
            user_pos=train_pos,
            n_items=n_items,
            batch_size=args.batch_size,
            neg_per_pos=args.neg_per_pos,
            device=device,
        )

        # 注意：这里调用 get_user_item_emb()，里面会做整图 propagate，
        # 所以每个 batch 都会重新跑一遍 GNN，比较慢但代码简单。
        user_emb, item_emb = model.get_user_item_emb()
        u_e = user_emb[u]
        p_e = item_emb[p]
        n_e = item_emb[n]
        loss = bpr_loss(u_e, p_e, n_e, l2=args.l2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            val_metrics = evaluate(
                model=model,
                train_pos=train_pos,
                eval_df=val,
                n_items=n_items,
                device=device,
                Ks=(10,),
            )
            cur = val_metrics["Recall@10"]
            print(
                f"[Epoch {epoch}] loss={loss.item():.4f} "
                f"val_Recall@10={cur:.4f} val_NDCG@10={val_metrics['NDCG@10']:.4f}"
            )
            if cur > best_val:
                best_val = cur
                best_metrics = val_metrics

    print("Best val metrics:", best_metrics)
    test_metrics = evaluate(
        model=model,
        train_pos=train_pos,
        eval_df=test,
        n_items=n_items,
        device=device,
        Ks=(10,),
    )
    print("[TEST]", test_metrics)


if __name__ == "__main__":
    main()
