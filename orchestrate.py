#!/usr/bin/env python
import os, subprocess, sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra

def run(cmd):
    print(">>", " ".join(cmd))
    res = subprocess.run(cmd, check=True)
    return res.returncode

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Directories
    out_global = Path(cfg.out.global_dir); out_global.mkdir(parents=True, exist_ok=True)
    out_cities = Path(cfg.out.cities_dir); out_cities.mkdir(parents=True, exist_ok=True)
    data_dir   = Path(cfg.out.data_dir);   data_dir.mkdir(parents=True, exist_ok=True)
    graph_dir  = Path(cfg.out.graph_dir);  graph_dir.mkdir(parents=True, exist_ok=True)
    reports_dir= Path(cfg.out.reports_dir);reports_dir.mkdir(parents=True, exist_ok=True)

    # 0) Global validation/stats (optional but recommended)
    run([sys.executable, "yelp_graph_tools/yelp_validate_and_stats.py",
        "--business_json", cfg.dataset.business_json,
        "--review_json", cfg.dataset.review_json,
        "--user_json", cfg.dataset.user_json,
        "--out_dir", str(out_global)])

    # 1) City ranking
    run([sys.executable, "yelp_graph_tools/yelp_city_stats.py",
        "--business_json", cfg.dataset.business_json,
        "--topn", "50",
        "--out_dir", str(out_cities)])

    # 2) Filter to target city
    run([sys.executable, "yelp_graph_tools/yelp_sample_filter.py",
        "--business_json", cfg.dataset.business_json,
        "--review_json", cfg.dataset.review_json,
        "--user_json", cfg.dataset.user_json,
        "--city", cfg.dataset.city,
        "--min_stars", str(cfg.dataset.min_stars),
        "--out_dir", str(data_dir)])

    # 3) Build heterogeneous graph
    run([sys.executable, "yelp_graph_tools/build_graph.py",
        "--in_dir", str(data_dir),
        "--out_dir", str(graph_dir)])

    # 4) Graph integrity report + small-sample visualization
    run([sys.executable, "reports/graph_report.py",
        "--in_dir", str(data_dir),
        "--graph_dir", str(graph_dir),
        "--out_dir", str(reports_dir)])

    run([sys.executable, "reports/sample_subgraph.py",
        "--in_dir", str(data_dir),
        "--out_dir", str(reports_dir),
        "--n_users", "80",
        "--n_items", "120"])

    # 5) Chronological split
    run([sys.executable, "yelp_baselines/split.py",
        "--in_dir", str(data_dir),
        "--val_ratio", "0.1",
        "--test_ratio", "0.1"])

    # 6) Train baselines
    run([sys.executable, "yelp_baselines/train_mf.py",
        "--data_dir", str(data_dir),
        "--epochs", str(cfg.train.epochs),
        "--dim", str(cfg.train.dim),
        "--lr", str(cfg.train.lr_mf),
        "--l2", str(cfg.train.l2),
        "--neg_per_pos", str(cfg.train.neg_per_pos)])

    run([sys.executable, "yelp_baselines/train_lightgcn.py",
        "--data_dir", str(data_dir),
        "--epochs", str(cfg.train.epochs),
        "--dim", str(cfg.train.dim),
        "--layers", str(cfg.train.layers),
        "--lr", str(cfg.train.lr_lgcn),
        "--l2", str(cfg.train.l2),
        "--neg_per_pos", str(cfg.train.neg_per_pos)])

    print("Pipeline completed. Outputs under:", cfg.out.root)

if __name__ == "__main__":
    main()
