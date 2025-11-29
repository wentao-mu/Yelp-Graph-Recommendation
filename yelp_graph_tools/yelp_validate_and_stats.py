#!/usr/bin/env python
import argparse, os, json, csv
from collections import Counter
from datetime import datetime

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--business_json", required=True, type=str)
    ap.add_argument("--review_json", required=True, type=str)
    ap.add_argument("--user_json", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--max_lines", type=int, default=None)
    return ap.parse_args()

def iter_jsonl(path, max_lines=None):
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if max_lines and idx > max_lines: break
            yield json.loads(line)

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Business
    biz_cnt = 0
    city_counter = Counter()
    cat_counter = Counter()
    for o in iter_jsonl(args.business_json, args.max_lines):
        biz_cnt += 1
        city = (o.get("city") or "").strip()
        if city: city_counter[city] += 1
        cats = o.get("categories")
        if isinstance(cats, str) and cats:
            for c in [c.strip() for c in cats.split(",") if c.strip()]:
                cat_counter[c] += 1

    # Reviews
    rev_cnt = 0
    stars_counter = Counter()
    year_counter = Counter()
    first_dt, last_dt = None, None
    for o in iter_jsonl(args.review_json, args.max_lines):
        rev_cnt += 1
        s = int(o.get("stars", 0))
        stars_counter[s] += 1
        ds = o.get("date")
        if ds:
            try:
                dt = datetime.fromisoformat(ds[:10])
                if first_dt is None or dt < first_dt: first_dt = dt
                if last_dt  is None or dt > last_dt:  last_dt  = dt
                year_counter[dt.year] += 1
            except Exception:
                pass

    # Users
    user_cnt = 0
    friend_edges = 0
    for o in iter_jsonl(args.user_json, args.max_lines):
        user_cnt += 1
        fr = o.get("friends")
        if isinstance(fr, str) and fr:
            n = len([x for x in fr.split(",") if x.strip()])
            friend_edges += n

    summary = {
        "users": user_cnt, "businesses": biz_cnt, "reviews": rev_cnt,
        "friend_edges": friend_edges,
        "review_date_range": {
            "first": first_dt.isoformat() if first_dt else None,
            "last": last_dt.isoformat() if last_dt else None
        },
        "stars_distribution": dict(sorted(stars_counter.items())),
        "yearly_counts": dict(sorted(year_counter.items())),
        "top_cities_sample": dict(city_counter.most_common(20)),
        "top_categories_sample": dict(cat_counter.most_common(30)),
    }

    with open(os.path.join(args.out_dir, "global_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # CSV exports
    with open(os.path.join(args.out_dir, "stars_distribution.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stars","count"])
        for k,v in sorted(stars_counter.items()): w.writerow([k,v])
    with open(os.path.join(args.out_dir, "yearly_counts.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["year","count"])
        for k,v in sorted(year_counter.items()): w.writerow([k,v])
    with open(os.path.join(args.out_dir, "city_sample_top20.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["city","business_count"])
        for k,v in city_counter.most_common(20): w.writerow([k,v])

    print("Wrote global stats to", args.out_dir)

if __name__ == "__main__":
    main()
