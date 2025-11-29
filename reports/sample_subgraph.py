#!/usr/bin/env python
import argparse, os, json, random
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_users", type=int, default=100)
    ap.add_argument("--n_items", type=int, default=150)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    inter = pd.read_csv(os.path.join(args.in_dir, "interactions.csv"))
    friends = pd.read_csv(os.path.join(args.in_dir, "user_friends.csv"))
    bcat = pd.read_csv(os.path.join(args.in_dir, "business_categories.csv"))

    # Sample users & items
    users = inter["u"].unique().tolist()
    items = inter["i"].unique().tolist()
    random.shuffle(users); random.shuffle(items)
    users = set(users[:args.n_users])
    items = set(items[:args.n_items])

    inter_s = inter[inter["u"].isin(users) & inter["i"].isin(items)]
    fr_s = friends[(friends["u"].isin(users)) & (friends["v"].isin(users))]
    bc_s = bcat[bcat["i"].isin(items)]

    # Export CSVs
    inter_s.to_csv(os.path.join(args.out_dir, "subgraph_user_business.csv"), index=False)
    fr_s.to_csv(os.path.join(args.out_dir, "subgraph_user_user.csv"), index=False)
    bc_s.to_csv(os.path.join(args.out_dir, "subgraph_business_category.csv"), index=False)

    # Simple networkx visualization (users/items only for clarity)
    G = nx.Graph()
    G.add_nodes_from([f"u_{u}" for u in users], bipartite=0)
    G.add_nodes_from([f"i_{i}" for i in items], bipartite=1)
    G.add_edges_from([(f"u_{row.u}", f"i_{row.i}") for _,row in inter_s.iterrows()])

    # layout
    pos = nx.spring_layout(G, k=0.15, iterations=30, seed=42)
    plt.figure(figsize=(8,6))
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes() if n.startswith("u_")], node_size=10)
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes() if n.startswith("i_")], node_size=10)
    nx.draw_networkx_edges(G, pos, width=0.3)
    plt.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "subgraph_ub.png"), dpi=150)
    plt.close()

    print("Subgraph CSVs/PNG saved to", args.out_dir)

if __name__ == "__main__":
    main()
