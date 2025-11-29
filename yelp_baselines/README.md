# Yelp Baselines: MF (BPR) & LightGCN

This package provides **two baselines** for the Yelp recommendation task (implicit feedback; ranking metrics):
- `train_mf.py`: Matrix Factorization with BPR loss
- `train_lightgcn.py`: LightGCN with BPR loss

It also includes:
- `yelp_preprocess.py`: build (user, item) implicit interactions from Yelp `review.json`, optionally filtered to a city using `business.json`
- `split.py`: chronological (time-based) train/val/test split to avoid leakage
- `metrics.py`: Recall@K and NDCG@K
- `utils.py`: helpers (seed, sparse adjacency build, negative sampling)

> Tip: Start with a **single city** to keep experiments fast (e.g., "Las Vegas" or "Phoenix").

## Quickstart

1) Prepare a small subset (city-filtered) interactions:
```bash
python yelp_preprocess.py   --business_json /path/to/business.json   --review_json /path/to/review.json   --city "Las Vegas"   --min_stars 4   --out_dir data/las_vegas
```

2) Create chronological splits:
```bash
python split.py --in_dir data/las_vegas --val_ratio 0.1 --test_ratio 0.1
```

3) Train MF (BPR) baseline:
```bash
python train_mf.py --data_dir data/las_vegas --epochs 10 --dim 64 --lr 5e-3 --l2 1e-4 --neg_per_pos 1
```

4) Train LightGCN baseline:
```bash
python train_lightgcn.py --data_dir data/las_vegas --epochs 10 --dim 64 --layers 3 --lr 1e-3 --l2 1e-4 --neg_per_pos 1
```

Both scripts will print **Recall@10 / NDCG@10** for validation and test.

## Requirements
- Python 3.9+
- PyTorch, NumPy, Pandas

Install minimal deps:
```bash
pip install torch numpy pandas
```

## Notes
- We binarize feedback as `1` if `stars >= min_stars` (default 4), else ignore.
- Cold-start analysis can be added by slicing test users/items with small training-degree.
- Use the **time-based split** from `split.py` for fair evaluation.

## Dataset Terms
Use of Yelp Dataset is governed by Yelp’s Dataset Terms of Use; **this code is for academic use only**.
Do **not** publicly redistribute raw data and follow the restrictions about disclosure and presentation.
Please review the terms in your copy of the dataset or online documentation.
