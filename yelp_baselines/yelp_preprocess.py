#!/usr/bin/env python
import argparse, os, json, pandas as pd, numpy as np
from datetime import datetime
from collections import defaultdict

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--business_json", type=str, required=True)
    ap.add_argument("--review_json", type=str, required=True)
    ap.add_argument("--city", type=str, default=None, help="Filter businesses by this city (optional)")
    ap.add_argument("--min_stars", type=int, default=4, help="Minimum stars to count as implicit positive")
    ap.add_argument("--out_dir", type=str, required=True)
    return ap.parse_args()

def load_business_ids(path, city=None):
    keep = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if (city is None) or (o.get("city") == city):
                keep.add(o["business_id"])
    return keep

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.city:
        keep_biz = load_business_ids(args.business_json, args.city)
    else:
        keep_biz = None

    user2id, item2id = {}, {}
    u_ptr = i_ptr = 0

    rows = []
    with open(args.review_json, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            bid = o["business_id"]
            if keep_biz is not None and bid not in keep_biz:
                continue
            stars = o["stars"]
            if stars < args.min_stars:
                continue
            uid = o["user_id"]
            date = o.get("date")  # YYYY-MM-DD

            if uid not in user2id:
                user2id[uid] = u_ptr; u_ptr += 1
            if bid not in item2id:
                item2id[bid] = i_ptr; i_ptr += 1
            rows.append((user2id[uid], item2id[bid], date))

    if not rows:
        raise RuntimeError("No interactions after filtering; try another city or lower --min_stars.")

    df = pd.DataFrame(rows, columns=["u","i","date"])
    df["ts"] = pd.to_datetime(df["date"])

    # sort chronologically for each user
    df = df.sort_values(["u","ts"]).reset_index(drop=True)

    meta = {
        "n_users": int(df["u"].max()+1),
        "n_items": int(df["i"].max()+1),
        "city": args.city,
        "min_stars": args.min_stars
    }

    os.makedirs(args.out_dir, exist_ok=True)
    df[["u","i","ts"]].to_csv(os.path.join(args.out_dir, "interactions.csv"), index=False)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"Saved {len(df)} interactions to {args.out_dir}. Users={meta['n_users']} Items={meta['n_items']}")

if __name__ == "__main__":
    main()
