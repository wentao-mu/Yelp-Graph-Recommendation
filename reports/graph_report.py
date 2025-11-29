#!/usr/bin/env python
import argparse, os, json, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="dir from yelp_sample_filter.py")
    ap.add_argument("--graph_dir", required=True, help="dir from build_graph.py (for summary)")
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()

def save_hist(data, bins, out_png, title, xlabel):
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    inter = pd.read_csv(os.path.join(args.in_dir, "interactions.csv"))
    umap = pd.read_csv(os.path.join(args.in_dir, "user_map.csv"))
    imap = pd.read_csv(os.path.join(args.in_dir, "item_map.csv"))
    friends = pd.read_csv(os.path.join(args.in_dir, "user_friends.csv"))
    bcat = pd.read_csv(os.path.join(args.in_dir, "business_categories.csv"))
    with open(os.path.join(args.in_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    n_users = int(meta["n_users"]); n_items = int(meta["n_items"])

    # Degrees
    deg_u_inter = Counter(inter["u"].tolist())
    deg_i_inter = Counter(inter["i"].tolist())
    deg_u_friend = Counter(friends["u"].tolist()) + Counter(friends["v"].tolist())

    # Isolation (no interactions & no friends for users; no interactions for items)
    iso_users = sum(1 for u in range(n_users) if deg_u_inter.get(u,0) == 0 and deg_u_friend.get(u,0) == 0)
    iso_items = sum(1 for i in range(n_items) if deg_i_inter.get(i,0) == 0)
    iso_user_rate = iso_users / n_users if n_users>0 else 0.0
    iso_item_rate = iso_items / n_items if n_items>0 else 0.0

    # CSV exports
    pd.DataFrame({"u": list(deg_u_inter.keys()), "deg_inter": list(deg_u_inter.values())}).to_csv(
        os.path.join(args.out_dir, "degrees_user_interactions.csv"), index=False)
    pd.DataFrame({"i": list(deg_i_inter.keys()), "deg_inter": list(deg_i_inter.values())}).to_csv(
        os.path.join(args.out_dir, "degrees_business_interactions.csv"), index=False)
    pd.DataFrame({"u": list(deg_u_friend.keys()), "deg_friend": list(deg_u_friend.values())}).to_csv(
        os.path.join(args.out_dir, "degrees_user_friends.csv"), index=False)

    # Histograms (PNG)
    save_hist(list(deg_u_inter.values()), bins=50,
              out_png=os.path.join(args.out_dir, "hist_user_interactions.png"),
              title="User Interaction Degree", xlabel="deg(u)")
    save_hist(list(deg_i_inter.values()), bins=50,
              out_png=os.path.join(args.out_dir, "hist_business_interactions.png"),
              title="Business Interaction Degree", xlabel="deg(i)")
    save_hist(list(deg_u_friend.values()) if len(deg_u_friend)>0 else [0], bins=50,
              out_png=os.path.join(args.out_dir, "hist_user_friend_degree.png"),
              title="User Friend Degree", xlabel="deg_friend(u)")

    # Summary JSON
    summary = {
        "n_users": n_users, "n_items": n_items,
        "user_interaction_degree_mean": float(np.mean(list(deg_u_inter.values())) if deg_u_inter else 0.0),
        "business_interaction_degree_mean": float(np.mean(list(deg_i_inter.values())) if deg_i_inter else 0.0),
        "user_friend_degree_mean": float(np.mean(list(deg_u_friend.values())) if deg_u_friend else 0.0),
        "user_isolation_rate": iso_user_rate,
        "item_isolation_rate": iso_item_rate
    }
    with open(os.path.join(args.out_dir, "graph_integrity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Also copy build_graph summary if exists
    try:
        with open(os.path.join(args.graph_dir, "graph_summary.json"), "r") as f:
            summ = json.load(f)
        with open(os.path.join(args.out_dir, "graph_summary_copy.json"), "w") as f:
            json.dump(summ, f, indent=2)
    except Exception:
        pass

    print("Graph integrity report written to", args.out_dir)

if __name__ == "__main__":
    main()
