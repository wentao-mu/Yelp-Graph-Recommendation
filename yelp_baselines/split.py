#!/usr/bin/env python
import argparse, os, json, pandas as pd, numpy as np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    return ap.parse_args()

def main():
    args = parse_args()
    inter = pd.read_csv(os.path.join(args.in_dir, "interactions.csv"), parse_dates=["ts"])
    with open(os.path.join(args.in_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    # split chronologically per-user: last x% -> val/test
    def split_user(dfu):
        n = len(dfu)
        n_test = max(1, int(n * args.test_ratio))
        n_val = max(1, int(n * args.val_ratio))
        # chronological: earliest -> train, then val, then test
        # ensure at least one train if possible
        cut2 = n
        cut1 = max(1, n - n_val - n_test)
        val_start = cut1
        test_start = n - n_test
        dfu = dfu.reset_index(drop=True)
        parts = (
            dfu.iloc[:cut1].assign(split="train"),
            dfu.iloc[val_start:test_start].assign(split="val"),
            dfu.iloc[test_start:].assign(split="test"),
        )
        return pd.concat(parts, ignore_index=True)

    spl = inter.groupby("u", group_keys=False).apply(split_user)
    for s in ["train","val","test"]:
        spl[spl["split"]==s][["u","i","ts"]].to_csv(os.path.join(args.in_dir, f"{s}.csv"), index=False)

    # stats
    stats = {k:int(len(spl[spl['split']==k])) for k in ["train","val","test"]}
    with open(os.path.join(args.in_dir, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("Split done:", stats)

if __name__ == "__main__":
    main()
