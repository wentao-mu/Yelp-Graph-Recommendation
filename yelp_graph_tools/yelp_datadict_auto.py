#!/usr/bin/env python
import argparse, os, json, random
from collections import defaultdict, Counter

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--business_json", required=False, type=str)
    ap.add_argument("--review_json", required=False, type=str)
    ap.add_argument("--user_json", required=False, type=str)
    ap.add_argument("--out_path", required=True, type=str, help="Markdown output path")
    ap.add_argument("--sample_lines", type=int, default=2000)
    return ap.parse_args()

def infer_type(v):
    if isinstance(v, bool): return "bool"
    if isinstance(v, int): return "int"
    if isinstance(v, float): return "float"
    if isinstance(v, str): return "str"
    if isinstance(v, list): return "list"
    if isinstance(v, dict): return "dict"
    if v is None: return "null"
    return type(v).__name__

def scan_file(path, sample_lines):
    if not path: return None
    keys = defaultdict(Counter)
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if idx > sample_lines: break
            try:
                o = json.loads(line)
            except Exception:
                continue
            for k, v in o.items():
                keys[k][infer_type(v)] += 1
    return keys

def main():
    args = parse_args()
    sections = []
    for name, path in [("business.json", args.business_json),
                       ("review.json", args.review_json),
                       ("user.json", args.user_json)]:
        if not path: continue
        stats = scan_file(path, args.sample_lines)
        if not stats: continue
        sections.append(f"## {name}\n")
        sections.append("| field | observed types (count in sample) |\n|---|---|\n")
        for k, counter in sorted(stats.items()):
            type_str = ", ".join([f"{t}:{c}" for t,c in counter.most_common()])
            sections.append(f"| `{k}` | {type_str} |\n")
        sections.append("\n")
    md = "# Auto-generated Data Dictionary (sample-based)\n\n" + "".join(sections)
    with open(args.out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print("Wrote", args.out_path)

if __name__ == "__main__":
    main()
