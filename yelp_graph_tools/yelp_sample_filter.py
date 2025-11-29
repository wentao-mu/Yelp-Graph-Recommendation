#!/usr/bin/env python
import argparse, os, json, csv, re
from collections import defaultdict

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--business_json", required=True)
    ap.add_argument("--review_json", required=True)
    ap.add_argument("--user_json", required=True)
    ap.add_argument("--city", required=True, help="City filter string (used with --city_mode)")
    ap.add_argument("--city_mode", choices=["exact","icontains","regex","metro"], default="exact",
                    help="How to match city names; metro will expand to a set of known aliases")
    ap.add_argument("--state", type=str, default=None, help="Optional two-letter state to constrain (e.g., NV, AZ)")
    ap.add_argument("--min_stars", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_lines", type=int, default=None)
    return ap.parse_args()

def iter_jsonl(path, max_lines=None):
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if max_lines and idx > max_lines: break
            yield json.loads(line)

def metro_aliases(name):
    n = name.strip().lower()
    if n in {"las vegas","las_vegas"}:
        # Common aliases in the Yelp dataset area
        return {"las vegas","north las vegas","henderson","paradise","spring valley","summerlin"}
    if n in {"phoenix"}:
        return {"phoenix","scottsdale","tempe","mesa","chandler","glendale"}
    return {name}

def city_matcher(mode, pattern, state=None):
    pat_lc = pattern.lower()
    aliases = metro_aliases(pattern) if mode == "metro" else {pattern}
    regex = re.compile(pattern) if mode == "regex" else None
    def _match(o):
        c = (o.get("city") or "").strip()
        s = (o.get("state") or "").strip()
        ok_state = True if state is None else (s.upper() == state.upper())
        if not ok_state:
            return False
        if mode == "exact":
            return c == pattern
        elif mode == "icontains":
            return pat_lc in c.lower()
        elif mode == "regex":
            return bool(regex.search(c))
        elif mode == "metro":
            return c.lower() in aliases
        else:
            return False
    return _match

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    match = city_matcher(args.city_mode, args.city, state=args.state)

    # 1) Businesses in city + categories
    keep_biz = {}
    biz_cats = {}
    for o in iter_jsonl(args.business_json, args.max_lines):
        if match(o):
            bid = o["business_id"]
            keep_biz[bid] = True
            cats = o.get("categories")
            if isinstance(cats, str) and cats:
                biz_cats[bid] = [c.strip() for c in cats.split(",") if c.strip()]
            else:
                biz_cats[bid] = []

    if not keep_biz:
        # Provide a helpful message with suggestions
        raise RuntimeError(f"No businesses found for city='{args.city}' with mode={args.city_mode}. "
                           f"Try --city_mode icontains or --city_mode metro (e.g., 'Las Vegas').")

    # 2) Interactions from reviews (implicit positives)
    u_map, i_map = {}, {}
    u_ptr = i_ptr = 0
    interactions = []
    all_reviews_in_city = 0
    for o in iter_jsonl(args.review_json, args.max_lines):
        bid = o["business_id"]
        if bid not in keep_biz: 
            continue
        all_reviews_in_city += 1
        if int(o.get("stars", 0)) < args.min_stars: 
            continue
        uid = o["user_id"]
        if uid not in u_map: u_map[uid] = u_ptr; u_ptr += 1
        if bid not in i_map: i_map[bid] = i_ptr; i_ptr += 1
        ds = o.get("date")
        ts = (ds[:10] if isinstance(ds, str) and len(ds)>=10 else "")
        interactions.append((u_map[uid], i_map[bid], ts))

    if not interactions:
        raise RuntimeError("No positive interactions after filtering; lower --min_stars or try another city/mode.")

    # 3) Friend edges among filtered users
    filtered_user_ids = set(u_map.keys())
    friend_edges = set()
    for o in iter_jsonl(args.user_json, args.max_lines):
        uid = o["user_id"]
        if uid not in filtered_user_ids: 
            continue
        fr = o.get("friends")
        if isinstance(fr, str) and fr:
            for fid in [x.strip() for x in fr.split(",") if x.strip()]:
                if fid in filtered_user_ids and fid != uid:
                    a = u_map[uid]; b = u_map[fid]
                    if a != b:
                        e = (min(a,b), max(a,b))
                        friend_edges.add(e)

    # 4) Business-category edges + category map
    cat2id = {}
    bcat_edges = []
    for bid, cat_list in biz_cats.items():
        if bid not in i_map: continue
        bi = i_map[bid]
        for c in cat_list:
            if c not in cat2id:
                cat2id[c] = len(cat2id)
            bcat_edges.append((bi, cat2id[c]))

    # 5) Write outputs
    def wcsv(path, header, rows):
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header)
            for r in rows: w.writerow(r)

    wcsv(os.path.join(args.out_dir, "interactions.csv"), ["u","i","ts"], interactions)
    wcsv(os.path.join(args.out_dir, "user_map.csv"), ["user_id","u_idx"], list(u_map.items()))
    wcsv(os.path.join(args.out_dir, "item_map.csv"), ["business_id","i_idx"], list(i_map.items()))
    wcsv(os.path.join(args.out_dir, "user_friends.csv"), ["u","v"], sorted(friend_edges))
    wcsv(os.path.join(args.out_dir, "business_categories.csv"), ["i","c_idx"], bcat_edges)
    wcsv(os.path.join(args.out_dir, "category_map.csv"), ["c_idx","category_name"],
         [(cid, name) for name, cid in sorted(cat2id.items(), key=lambda x: x[1])])

    # 6) Stats & meta
    sample_stats = {
        "city": args.city,
        "city_mode": args.city_mode,
        "state": args.state,
        "min_stars": args.min_stars,
        "reviews_in_city_before_threshold": all_reviews_in_city,
        "positives_after_threshold": len(interactions),
        "n_users": len(u_map),
        "n_items": len(i_map),
        "friend_edges_within_subset": len(friend_edges),
        "n_categories": len(cat2id),
    }
    with open(os.path.join(args.out_dir, "sample_stats.json"), "w") as f:
        json.dump(sample_stats, f, indent=2)

    meta = {"n_users": len(u_map), "n_items": len(i_map), "n_categories": len(cat2id),
            "city": args.city, "city_mode": args.city_mode, "state": args.state, "min_stars": args.min_stars}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Filtered city='{args.city}' mode={args.city_mode} -> users={len(u_map)} items={len(i_map)} positives={len(interactions)}")

if __name__ == "__main__":
    main()
