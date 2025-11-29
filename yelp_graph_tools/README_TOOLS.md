# Yelp Data Tools & Hetero Graph Builder

This folder completes **Task 1** (data acquisition, compliance, stats, sampling) and **Task 2** (graph schema & build).
Use together with the baseline package.

## Scripts

1) `yelp_validate_and_stats.py`
   - Validate newline-delimited JSON (business.json, review.json, user.json)
   - Global stats: entity counts, review date range, star distribution, yearly counts, friend-edge counts
   - Outputs: `global_stats.json`, `stars_distribution.csv`, `yearly_counts.csv`, `city_sample_top20.csv`

2) `yelp_city_stats.py`
   - Rank cities by business count (from business.json)
   - Output: `city_counts.csv`

3) `yelp_sample_filter.py`
   - Filter to one city and binarize interactions (`stars >= min_stars`)
   - Outputs (to `--out_dir`):
     - `interactions.csv` (u,i,ts)
     - `user_map.csv`, `item_map.csv`
     - `user_friends.csv` (edges within filtered users)
     - `business_categories.csv` (i,c_idx)
     - `category_map.csv` (c_idx,category_name)
     - `sample_stats.json`, `meta.json`

4) `yelp_datadict_auto.py`
   - Auto-generate a Markdown data dictionary from sampled keys/types
   - Output: `data_dictionary_generated.md`

5) `build_graph.py`
   - Build heterogeneous graph nodes {user, business, category} and edges
     `{user–business(review), user–user(friend), business–category}`
   - Saves PyTorch Geometric `hetero_graph.pt` if PyG is installed;
     otherwise saves `hetero_graph_npz.npz`.
   - Also writes `graph_summary.json` for quick inspection.

## Minimal Workflow

```bash
# 0) (optional) Global validation & stats
python yelp_validate_and_stats.py   --business_json /path/to/business.json   --review_json /path/to/review.json   --user_json /path/to/user.json   --out_dir out/global

# 1) Rank cities to choose a subset
python yelp_city_stats.py --business_json /path/to/business.json --topn 30 --out_dir out/cities

# 2) Filter to e.g., Las Vegas
python yelp_sample_filter.py   --business_json /path/to/business.json   --review_json /path/to/review.json   --user_json /path/to/user.json   --city "Las Vegas"   --min_stars 4   --out_dir data/las_vegas

# 3) Build heterogeneous graph
python build_graph.py --in_dir data/las_vegas --out_dir graphs/las_vegas
```

> Compliance: Use for academic research; follow Yelp Dataset Terms.
