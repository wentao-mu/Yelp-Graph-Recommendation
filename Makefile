CITY_MODE?=regrex
# One-click Yelp pipeline (Makefile)
PY=python

BUSINESS_JSON?=/path/to/business.json
REVIEW_JSON?=/path/to/review.json
USER_JSON?=/path/to/user.json
CITY?=Las Vegas
MIN_STARS?=4

OUT_ROOT?=out
GLOBAL_DIR=$(OUT_ROOT)/global
CITIES_DIR=$(OUT_ROOT)/cities

# turn "Las Vegas" -> "Las_Vegas" for directory names
CITY_SLUG := $(shell echo $(CITY) | tr ' ' '_')
DATA_DIR=$(OUT_ROOT)/data/$(CITY_SLUG)
GRAPH_DIR=$(OUT_ROOT)/graphs/$(CITY_SLUG)
REPORTS_DIR=$(OUT_ROOT)/reports/$(CITY_SLUG)

.PHONY: all env validate city_stats sample build_graph report viz_small split train_mf train_lgcn clean

all: validate city_stats sample build_graph report viz_small split train_mf train_lgcn

env:
	pip install -r requirements.txt

validate:
	$(PY) yelp_graph_tools/yelp_validate_and_stats.py \
		--business_json "$(BUSINESS_JSON)" \
		--review_json "$(REVIEW_JSON)" \
		--user_json "$(USER_JSON)" \
		--out_dir "$(GLOBAL_DIR)"

city_stats:
	$(PY) yelp_graph_tools/yelp_city_stats.py \
		--business_json "$(BUSINESS_JSON)" \
		--topn 50 \
		--out_dir "$(CITIES_DIR)"

sample:
	$(PY) yelp_graph_tools/yelp_sample_filter.py \
		--business_json "$(BUSINESS_JSON)" \
		--review_json "$(REVIEW_JSON)" \
		--user_json "$(USER_JSON)" \
		--city "$(CITY)" \
		--city_mode "$(CITY_MODE)" \
		--min_stars $(MIN_STARS) \
		--out_dir "$(DATA_DIR)"

build_graph:
	$(PY) yelp_graph_tools/build_graph.py \
		--in_dir "$(DATA_DIR)" \
		--out_dir "$(GRAPH_DIR)"

report:
	$(PY) reports/graph_report.py \
		--in_dir "$(DATA_DIR)" \
		--graph_dir "$(GRAPH_DIR)" \
		--out_dir "$(REPORTS_DIR)"

viz_small:
	$(PY) reports/sample_subgraph.py \
		--in_dir "$(DATA_DIR)" \
		--out_dir "$(REPORTS_DIR)" \
		--n_users 80 --n_items 120

split:
	$(PY) yelp_baselines/split.py \
		--in_dir "$(DATA_DIR)" \
		--val_ratio 0.1 --test_ratio 0.1

train_mf:
	$(PY) yelp_baselines/train_mf.py \
		--data_dir "$(DATA_DIR)" \
		--epochs 10 --dim 64 --lr 5e-3 --l2 1e-4 --neg_per_pos 1

train_lgcn:
	$(PY) yelp_baselines/train_lightgcn.py \
		--data_dir "$(DATA_DIR)" \
		--epochs 10 --dim 64 --layers 3 --lr 1e-3 --l2 1e-4 --neg_per_pos 1

clean:
	rm -rf "$(OUT_ROOT)"