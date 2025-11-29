#!/usr/bin/env python
import argparse, os, json, csv
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory from yelp_sample_filter.py")
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()

def load_csv(path, skip_header=True, dtype=int):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        it = iter(f)
        if skip_header: next(it, None)
        for line in it:
            parts = line.strip().split(",")
            if not parts or len(parts)<2: continue
            data.append([dtype(parts[0]), dtype(parts[1])])
    return np.asarray(data, dtype=np.int64) if data else np.zeros((0,2), dtype=np.int64)

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.in_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    n_users = int(meta["n_users"]); n_items = int(meta["n_items"])
    n_categories = int(meta.get("n_categories", 0))

    # Edges
    ui = load_csv(os.path.join(args.in_dir, "interactions.csv"), skip_header=True, dtype=int)  # [u,i]
    uu = load_csv(os.path.join(args.in_dir, "user_friends.csv"), skip_header=True, dtype=int)   # [u,v]
    ic = load_csv(os.path.join(args.in_dir, "business_categories.csv"), skip_header=True, dtype=int)  # [i,c_idx]

    # Try PyG
    saved = None
    try:
        import torch
        from torch_geometric.data import HeteroData
        data = HeteroData()
        data["user"].num_nodes = n_users
        data["business"].num_nodes = n_items
        data["category"].num_nodes = n_categories

        if ui.size > 0:
            src = torch.tensor(ui[:,0], dtype=torch.long)
            dst = torch.tensor(ui[:,1], dtype=torch.long)
            data["user","reviews","business"].edge_index = torch.stack([src, dst], dim=0)
            # reverse edge
            data["business","rev_by","user"].edge_index = torch.stack([dst, src], dim=0)
        if uu.size > 0:
            uu2 = np.vstack([uu, np.flip(uu, axis=1)])  # undirected as two directed
            src = torch.tensor(uu2[:,0], dtype=torch.long)
            dst = torch.tensor(uu2[:,1], dtype=torch.long)
            data["user","friends","user"].edge_index = torch.stack([src, dst], dim=0)
        if ic.size > 0:
            src = torch.tensor(ic[:,0], dtype=torch.long)
            dst = torch.tensor(ic[:,1], dtype=torch.long)
            data["business","in_category","category"].edge_index = torch.stack([src, dst], dim=0)

        torch.save(data, os.path.join(args.out_dir, "hetero_graph.pt"))
        saved = "hetero_graph.pt"
    except Exception as e:
        # Fallback: save as npz
        np.savez_compressed(os.path.join(args.out_dir, "hetero_graph_npz.npz"),
                            n_users=n_users, n_items=n_items, n_categories=n_categories,
                            user_business=ui, user_user=uu, business_category=ic)
        saved = "hetero_graph_npz.npz"

    # Summary
    summary = {
        "n_users": n_users, "n_items": n_items, "n_categories": n_categories,
        "edges": {
            "user_business": int(ui.shape[0]),
            "user_user": int(uu.shape[0]),
            "business_category": int(ic.shape[0])
        },
        "saved": saved
    }
    with open(os.path.join(args.out_dir, "graph_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Graph saved:", os.path.join(args.out_dir, saved))

if __name__ == "__main__":
    main()
