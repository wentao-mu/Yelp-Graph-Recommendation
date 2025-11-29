# Yelp One-Click Pipeline (Graph Construction & Baseline Models)

## 0) Layout & Artifacts

Repo highlights (names may vary slightly in your tree):

- `yelp_graph_tools/`
  - `yelp_validate_and_stats.py`: Global sanity check on raw JSON (dataset overview).
  - `yelp_city_stats.py`: Aggregates `business.json` by **`city`** to produce a ranked city list.
  - `yelp_sample_filter.py`: Samples a **city subset** and exports training-ready CSVs (interactions, maps, friends, categories, etc.).
  - `build_graph.py`: Builds a heterogeneous graph (PyTorch Geometric `HeteroData` or NPZ fallback).
- `reports/`
  - `graph_report.py`: Degree distributions & integrity report (CSV/PNG/JSON).
  - `sample_subgraph.py`: Small subgraph visualization (exports PNG/CSV).
- `yelp_baselines/`
  - `split.py`: Chronological user-wise split into train/val/test.
  - (Optional) `train_mf.py`, `train_lgcn.py`, or Makefile targets for baselines.
- `Makefile`: One-click targets (`all / sample / build_graph / report / viz_small / split / train_*`).

Outputs go to:

```
out/
├── global/                    # Raw data health check
├── cities/                    # Ranked city list (city_counts.csv)
├── data/<CITY_SLUG>/          # Sampled, training-ready data
├── graphs/<CITY_SLUG>/        # Graph artifacts (pt/npz + summary)
└── reports/<CITY_SLUG>/       # Integrity report & visualizations
```

---

## 1) Environment

- **Python**: 3.9+ (3.10/3.11 fine).
- **Deps**: `pandas`, `numpy`, `matplotlib`, `tqdm`, `pyyaml`, etc.
  - For `hetero_graph.pt`: `torch` + **optional** `torch-geometric (PyG)`.
  - Without PyG, the pipeline saves an NPZ fallback.

Install (example):

```bash
# Prefer your repo's requirements if provided
pip install -r requirements.txt  # if present
# otherwise install essentials
pip install pandas numpy matplotlib tqdm pyyaml
# PyTorch & PyG: follow official instructions for your CUDA/CPU
# pip install torch torch-geometric
```

---

## 2) Datasets

Place Yelp Academic Dataset JSONL files locally (or pass absolute paths):

```
./yelp_dataset/
  yelp_academic_dataset_business.json
  yelp_academic_dataset_review.json
  yelp_academic_dataset_user.json
```

> **Compliance**: Ensure usage adheres to Yelp Academic Dataset Terms of Use. No re-identification of individuals, etc.

---

## 3) Quickstart (End-to-End)

### 3.1 Get the City Ranking (pick the exact `CITY` string)

```bash
python yelp_graph_tools/yelp_city_stats.py \
  --business_json ./yelp_dataset/yelp_academic_dataset_business.json \
  --topn 50 \
  --out_dir out/cities

# Peek & search
head -n 20 out/cities/city_counts.csv
grep -i "vegas" out/cities/city_counts.csv | head
```

> `city_counts.csv` is computed from the **`city` field in `business.json`** only. It does **not** look at business names. A shop with name "Las Vegas ..." in city "Tampa" is counted under **Tampa**, not Las Vegas.

### 3.2 Sampling (produce training CSVs)

> **This must succeed** before graph/report/split. All later steps consume files under `out/data/<CITY_SLUG>/`.

**Option A: Makefile (simple)**

```bash
make all \
  BUSINESS_JSON=./yelp_dataset/yelp_academic_dataset_business.json \
  REVIEW_JSON=./yelp_dataset/yelp_academic_dataset_review.json \
  USER_JSON=./yelp_dataset/yelp_academic_dataset_user.json \
  CITY="Hendersonville" CITY_MODE=exact \
  CITY_SLUG=Hendersonville
```

- `CITY_MODE`: `exact | icontains | regex`.
- **Metro/aliases via regex** (use **single quotes** to avoid shell parsing surprises):

```bash
make sample \
  BUSINESS_JSON=./yelp_dataset/yelp_academic_dataset_business.json \
  REVIEW_JSON=./yelp_dataset/yelp_academic_dataset_review.json \
  USER_JSON=./yelp_dataset/yelp_academic_dataset_user.json \
  CITY='las\s*vegas|north\s*las\s*vegas|henderson|paradise|spring\s*valley|enterprise|whitney|winchester|sunrise\s*manor' \
  CITY_MODE=regex \
  CITY_SLUG=VegasMetro
```

> Do **not** wrap city names like `<Las Vegas>` (angle brackets cause shell issues). Characters like `|` also need care inside Makefiles; either pass `CITY_SLUG` explicitly (safe) or see the FAQ for a Make fix.

**Option B: Direct script (most debuggable)**

```bash
python yelp_graph_tools/yelp_sample_filter.py \
  --business_json ./yelp_dataset/yelp_academic_dataset_business.json \
  --review_json   ./yelp_dataset/yelp_academic_dataset_review.json \
  --user_json     ./yelp_dataset/yelp_academic_dataset_user.json \
  --city "Hendersonville" --city_mode exact --min_stars 4 \
  --out_dir "out/data/Hendersonville"
```

- If positives are too few, relax: `--min_stars 3`.
- For quick smoke tests: `--max_lines 200000`.

**Success checklist (after sampling)**:

```
out/data/<CITY_SLUG>/
  interactions.csv  user_map.csv  item_map.csv
  user_friends.csv  business_categories.csv  category_map.csv
  sample_stats.json meta.json
```

Sanity check:

```bash
cat out/data/<CITY_SLUG>/sample_stats.json  # n_users / n_items / positives_after_threshold > 0
```

### 3.3 Graph, Reports, Split, Baselines

```bash
# Build graph
make build_graph  CITY_SLUG=<CITY_SLUG>

# Integrity report + small subgraph viz
make report       CITY_SLUG=<CITY_SLUG>
make viz_small    CITY_SLUG=<CITY_SLUG>

# Chronological split (per-user)
make split        CITY_SLUG=<CITY_SLUG>

# (Optional) Baseline training
make train_mf     CITY_SLUG=<CITY_SLUG>
make train_lgcn   CITY_SLUG=<CITY_SLUG>
```

---

## 4) Outputs by Step

### 4.1 `out/global/` (raw data health)
- `global_stats.json`: counts of users/businesses/reviews, time range, friend edges, etc.
- Additional CSVs (star hist, year hist, etc., depending on your script version).

### 4.2 `out/cities/` (choose a city)
- `city_counts.csv`: ranked `city,business_count` from `business.json`. Use this to pick the **exact** `CITY` string.

### 4.3 `out/data/<CITY_SLUG>/` (training-ready subset)
- `interactions.csv`: positive user–business interactions (`u,i,ts`).
- `user_map.csv` / `item_map.csv`: raw IDs → contiguous indices.
- `user_friends.csv`: user–user friend edges (within subset).
- `business_categories.csv`: business–category edges; `category_map.csv`: category index↔name.
- `sample_stats.json` / `meta.json`: size/metadata of the subset.
- After split: `train.csv` / `val.csv` / `test.csv` + `split_stats.json`.

### 4.4 `out/graphs/<CITY_SLUG>/` (graph artifacts)
- `hetero_graph.pt`: PyG `HeteroData` (if PyG is available).
- `hetero_graph_npz.npz`: NPZ fallback if PyG not installed.
- `graph_summary.json`: node/edge sizes & file pointers.

### 4.5 `out/reports/<CITY_SLUG>/` (integrity & viz)
- `degrees_user_interactions.csv`, `degrees_business_interactions.csv`, `degrees_user_friends.csv`.
- `graph_integrity_summary.json`:
  - `user_isolation_rate` / `item_isolation_rate`;
  - degree means & distribution summaries.
- Histograms: `hist_user_interactions.png`, `hist_business_interactions.png`, `hist_user_friend_degree.png`.
- Subgraph: `subgraph_ub.png` and CSV edge lists (`subgraph_user_business.csv`, etc.).

---

## 5) FAQ

### Q1: `No businesses found for city='Las Vegas' ...`
- Most often: your `business.json` simply has **no rows with `city=Las Vegas`**.
- Fix:
  1) Inspect `out/cities/city_counts.csv` to see the real city strings in **your** dataset.
  2) Use `CITY_MODE=regex` for metro coverage (see examples).
  3) Remember: **business name containing "Las Vegas" ≠ `city` is Las Vegas**. We filter by the `city` field.

### Q2: `FileNotFoundError: out/data/<CITY>/meta.json` or missing `interactions.csv`
- Root cause: **sampling failed**, so downstream steps cannot find inputs.
- Fix: ensure step **3.2** completed and produced the CSVs under `out/data/<CITY_SLUG>/`.

### Q3: `/bin/sh: syntax error near unexpected token '|'` (or empty `--out_dir`)
- Cause: Makefile computes `CITY_SLUG` using a shell pipeline like `$(shell echo $(CITY) | tr ' ' '_')`. If `CITY` contains special chars (e.g., `|` in regex), the shell chokes.
- Fix:
  - **Simple**: pass an explicit safe slug when calling: `CITY_SLUG=VegasMetro`.
  - **Permanent**: change Makefile to pure-Make substitution (no shell):
    ```make
    # Old (problematic with special chars):
    # CITY_SLUG := $(shell echo $(CITY) | tr ' ' '_')
    
    # New (safe for spaces → underscores, avoids shell):
    CITY_SLUG := $(CITY: =_)
    ```

### Q4: Too few positives (dataset too small)
- Lower threshold: `MIN_STARS=3`.
- Widen match: `CITY_MODE=icontains` or a richer `regex`.
- For pipeline validation, start with a top-N city from your `city_counts.csv`.

### Q5: How to verify a city exists in `business.json`?
```bash
grep -n '"city":"Hendersonville"' ./yelp_dataset/yelp_academic_dataset_business.json | head
# or broader search by pattern
grep -nE '"city":\s*"[^\"]*vegas' ./yelp_dataset/yelp_academic_dataset_business.json | head
```

---

## 6) Command Cheatsheet

**Rank cities**
```bash
python yelp_graph_tools/yelp_city_stats.py \
  --business_json ./yelp_dataset/yelp_academic_dataset_business.json \
  --topn 50 --out_dir out/cities
```

**Sampling (exact)**
```bash
make sample BUSINESS_JSON=./yelp_dataset/yelp_academic_dataset_business.json \
  REVIEW_JSON=./yelp_dataset/yelp_academic_dataset_review.json \
  USER_JSON=./yelp_dataset/yelp_academic_dataset_user.json \
  CITY="Hendersonville" CITY_MODE=exact CITY_SLUG=Hendersonville
```

**Sampling (Vegas metro via regex)**
```bash
make sample BUSINESS_JSON=./yelp_dataset/yelp_academic_dataset_business.json \
  REVIEW_JSON=./yelp_dataset/yelp_academic_dataset_review.json \
  USER_JSON=./yelp_dataset/yelp_academic_dataset_user.json \
  CITY='las\s*vegas|north\s*las\s*vegas|henderson|paradise|spring\s*valley|enterprise|whitney|winchester|sunrise\s*manor' \
  CITY_MODE=regex CITY_SLUG=VegasMetro
```

**Graph / Reports / Subgraph / Split**
```bash
make build_graph  CITY_SLUG=<CITY_SLUG>
make report       CITY_SLUG=<CITY_SLUG>
make viz_small    CITY_SLUG=<CITY_SLUG>
make split        CITY_SLUG=<CITY_SLUG>
```

**Baselines (optional)**
```bash
make train_mf   CITY_SLUG=<CITY_SLUG>
make train_lgcn CITY_SLUG=<CITY_SLUG>
```

---

## 7) Module Quick Reference

| Module / Script | Purpose | Typical Inputs | Outputs |
|---|---|---|---|
| `yelp_validate_and_stats.py` | Global dataset sanity | business/review/user JSON | `out/global/*` |
| `yelp_city_stats.py` | City aggregation | business.json | `out/cities/city_counts.csv` |
| `yelp_sample_filter.py` | City subset sampling & CSV export | 3 JSONs + city params | `out/data/<CITY_SLUG>/*` |
| `build_graph.py` | Graph build (PyG HeteroData / NPZ) | `out/data/<CITY_SLUG>/*` | `out/graphs/<CITY_SLUG>/*` |
| `reports/graph_report.py` | Degree & integrity report | interactions/friends/categories CSVs | `out/reports/<CITY_SLUG>/*` |
| `reports/sample_subgraph.py` | Small subgraph sampling & viz | same | PNG/CSV in `out/reports` |
| `yelp_baselines/split.py` | Chronological split | interactions.csv | train/val/test + stats |
| `make train_*` | Baseline training | split/graph artifacts | logs/metrics (path depends on impl) |

---

## 8) Notes & Suggestions

- For rapid iteration, start with `--max_lines` to validate flow end-to-end, then run full-scale.
- Watch `graph_integrity_summary.json` (isolation rates & long tails) to tune sampling (regex coverage, `min_stars`, etc.).
- Prefer passing an explicit `CITY_SLUG` to avoid Makefile shell parsing issues with special characters.
- If you also want **name-based filtering** (e.g., match `name` contains "Las Vegas", regardless of `city`), it’s easy to extend `yelp_sample_filter.py` with `--name_regex` for side-by-side experiments.

---

> Feel free to extend this README with `requirements.txt`, exact training hyperparameters, seed settings, and metrics definitions for full reproducibility.