"""
src/models/train_ranker.py
Train LightGBM LambdaRank stage-2 ranker.

How it works:
  1. Load ALS model — use it to generate top-100 candidates per user
  2. Label the LOO held-out item as positive (1), all others as negative (0)
  3. Build 7 ranking features per candidate
  4. Train LightGBM LambdaRank on labelled groups
  5. Save booster to checkpoints/lgbm_ranker.pkl

Usage:
    python -m src.models.train_ranker
"""
from __future__ import annotations

import logging
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.loader import load_interactions
from .als_model import ALSRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINTS = Path("checkpoints")
PROCESSED   = Path("data/processed")
CHECKPOINTS.mkdir(exist_ok=True)

FEATURES = [
    "als_score",
    "item_purchase_count",
    "item_avg_review",
    "log_price",
    "user_avg_price",
    "category_match",
    "user_tx_count",
]


def _build_dataset(
    als: ALSRecommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    item_feat: pd.DataFrame,
    user_feat: pd.DataFrame,
    n_cands: int = 100,
    max_users: int = 3000,
) -> pd.DataFrame:
    """Build labelled ranking dataset for LightGBM."""

    # User category history
    user_cats: dict[str, set] = {}
    for uid, grp in train_df.groupby("user_id"):
        cats = set()
        for pid in grp["product_id"]:
            if pid in item_feat.index:
                cat = item_feat.loc[pid, "product_category_name_english"]
                if pd.notna(cat):
                    cats.add(str(cat))
        user_cats[uid] = cats

    test_gt = dict(zip(test_df["user_id"], test_df["product_id"]))
    eligible = [u for u in test_gt if u in als.user_id_map]
    users    = random.sample(eligible, min(max_users, len(eligible)))
    logger.info("Building ranker dataset from %d users...", len(users))

    rows = []
    for uid in users:
        gt  = test_gt[uid]
        uf  = user_feat.loc[uid] if uid in user_feat.index else None
        idx = als.user_id_map[uid]

        try:
            item_ids, scores = als.model.recommend(
                idx, als.matrix[idx], N=n_cands, filter_already_liked_items=True,
            )
        except Exception:
            continue

        cands  = [als.item_id_reverse.get(i) for i in item_ids]
        cands  = [c for c in cands if c]
        scores = list(scores)

        if gt not in cands:
            cands.append(gt)
            scores.append(0.0)

        ucats = user_cats.get(uid, set())

        for pid, score in zip(cands, scores):
            itf     = item_feat.loc[pid] if pid in item_feat.index else None
            cat     = str(itf.get("product_category_name_english", "unknown")) if itf is not None else "unknown"
            rows.append({
                "user_id":            uid,
                "item_id":            pid,
                "label":              int(pid == gt),
                "als_score":          float(score),
                "item_purchase_count": float(itf.get("purchase_count",  0))   if itf is not None else 0.0,
                "item_avg_review":    float(itf.get("avg_review_score", 3.0)) if itf is not None else 3.0,
                "log_price":          float(itf.get("log_price",        0.0)) if itf is not None else 0.0,
                "user_avg_price":     float(uf.get("avg_price",         0.0)) if uf is not None else 0.0,
                "category_match":     1.0 if cat in ucats else 0.0,
                "user_tx_count":      float(uf.get("purchase_count",    1))   if uf is not None else 1.0,
            })

    df = pd.DataFrame(rows)
    logger.info(
        "Ranker dataset: %d rows | %d positives | %d users",
        len(df), df["label"].sum(), df["user_id"].nunique(),
    )
    return df


def train_ranker(df: pd.DataFrame):
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("pip install lightgbm")

    X      = df[FEATURES]
    y      = df["label"]
    groups = df.groupby("user_id", sort=False).size().values

    params = {
        "objective":         "lambdarank",
        "metric":            "ndcg",
        "ndcg_eval_at":      [5, 10],
        "learning_rate":     0.05,
        "num_leaves":        31,
        "min_child_samples": 5,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "verbose":           -1,
        "n_jobs":            -1,
    }

    dataset = lgb.Dataset(X, label=y, group=groups, free_raw_data=False)
    logger.info("Training LightGBM LambdaRank (200 rounds)...")
    booster = lgb.train(params, dataset, num_boost_round=200,
                        callbacks=[lgb.log_evaluation(50)])

    logger.info("Feature importances (gain):")
    for feat, imp in sorted(
        zip(FEATURES, booster.feature_importance("gain")), key=lambda x: -x[1]
    ):
        logger.info("  %-25s %.1f", feat, imp)

    return booster


def main() -> None:
    train_df, test_df = load_interactions()

    als_path = CHECKPOINTS / "als_model.pkl"
    if not als_path.exists():
        raise FileNotFoundError("ALS model not found. Run: python -m src.models.train_als")

    als       = ALSRecommender.load(als_path)
    item_feat = pd.read_parquet(PROCESSED / "item_features.parquet")
    user_feat = pd.read_parquet(PROCESSED / "user_features.parquet")

    df      = _build_dataset(als, train_df, test_df, item_feat, user_feat)
    booster = train_ranker(df)

    out = CHECKPOINTS / "lgbm_ranker.pkl"
    with open(out, "wb") as f:
        pickle.dump(booster, f)
    logger.info("LightGBM ranker saved → %s", out)


if __name__ == "__main__":
    main()
