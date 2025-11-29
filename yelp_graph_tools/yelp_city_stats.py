#!/usr/bin/env python
import argparse, os, json, csv
from collections import Counter

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--business_json", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--topn", type=int, default=50)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    counter = Counter()
    with open(args.business_json, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            c = (o.get("city") or "").strip()
            if c: counter[c] += 1
    rows = [["rank","city","business_count"]]
    for rank,(city, n) in enumerate(counter.most_common(args.topn), start=1):
        rows.append([rank, city, n])
    with open(os.path.join(args.out_dir, "city_counts.csv"), "w", newline="") as f:
        writer = csv.writer(f); writer.writerows(rows)
    print("Wrote", os.path.join(args.out_dir, "city_counts.csv"))

if __name__ == "__main__":
    main()
