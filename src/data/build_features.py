"""
src/data/build_features.py
Builds all features needed for Two-Tower and ALS training.

Steps:
  1. Load Olist CSVs
  2. Build LOO train/test splits
  3. Aggregate user behavioral features
  4. Aggregate item features + category translations
  5. Encode product descriptions with SBERT (all-MiniLM-L6-v2)
  6. Save everything to data/processed/ and data/splits/

Usage:
    python -m src.data.build_features
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import load_olist, build_interactions, save_interactions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED = Path("data/processed")
SPLITS    = Path("data/splits")


# ── User features ──────────────────────────────────────────────────────────────

def build_user_features(data: dict) -> pd.DataFrame:
    df = (
        data["orders"]
        .merge(data["order_items"], on="order_id")
        .merge(data["customers"],   on="customer_id")
        .merge(data["reviews"][["order_id", "review_score"]], on="order_id", how="left")
        .query("order_status == 'delivered'")
    )
    agg = (
        df.groupby("customer_unique_id")
        .agg(
            purchase_count=   ("order_id",     "nunique"),
            avg_price=        ("price",         "mean"),
            avg_review_given= ("review_score",  "mean"),
        )
        .fillna({"avg_review_given": 3.0})
    )
    agg.index.name = "user_id"
    agg["log_purchase_count"] = np.log1p(agg["purchase_count"])
    logger.info("User features: %d rows", len(agg))
    return agg


# ── Item features ──────────────────────────────────────────────────────────────

def build_item_features(data: dict) -> pd.DataFrame:
    items = data["products"].merge(
        data["translations"], on="product_category_name", how="left"
    )

    purchase_agg = (
        data["order_items"]
        .merge(data["orders"][["order_id", "order_status"]], on="order_id")
        .query("order_status == 'delivered'")
        .groupby("product_id")
        .agg(purchase_count=("order_id", "count"), avg_price=("price", "mean"))
    )

    review_agg = (
        data["order_items"][["order_id", "product_id"]]
        .merge(data["reviews"][["order_id", "review_score"]], on="order_id", how="left")
        .groupby("product_id")["review_score"]
        .mean()
        .rename("avg_review_score")
    )

    features = (
        items.set_index("product_id")
        .join(purchase_agg, how="left")
        .join(review_agg,   how="left")
    )
    features["purchase_count"]   = features["purchase_count"].fillna(0).astype(int)
    features["avg_price"]        = features["avg_price"].fillna(
        features["avg_price"].median()
    )
    features["avg_review_score"] = features["avg_review_score"].fillna(3.0)
    features["log_price"]        = np.log1p(features["avg_price"])

    cat_col = "product_category_name_english"
    features[cat_col] = features[cat_col].fillna("unknown")

    # Text description for SBERT
    features["text_description"] = (
        features[cat_col].str.replace("_", " ")
        + ". "
        + features.get(
            "product_description_lenght",
            pd.Series("", index=features.index),
        )
        .fillna("")
        .astype(str)
        .str[:300]
    )

    logger.info("Item features: %d rows", len(features))
    return features


# ── SBERT encoding ─────────────────────────────────────────────────────────────

def encode_item_texts(item_features: pd.DataFrame) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed — using zero embeddings")
        return np.zeros((len(item_features), 384), dtype=np.float32)

    logger.info("Loading SBERT model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = item_features["text_description"].fillna("").tolist()
    logger.info("Encoding %d product descriptions...", len(texts))
    embs = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embs.astype(np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("Building features")
    logger.info("=" * 60)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)

    data = load_olist()

    train, test = build_interactions(data)
    save_interactions(train, test)

    user_features = build_user_features(data)
    item_features = build_item_features(data)

    # SBERT text embeddings
    text_embs = encode_item_texts(item_features)

    # ID maps from train only
    train_users = train["user_id"].unique()
    train_items = train["product_id"].unique()
    user_id_map  = {u: i for i, u in enumerate(train_users)}
    item_id_map  = {p: i for i, p in enumerate(train_items)}

    cats = item_features["product_category_name_english"].dropna().unique().tolist()
    if "unknown" not in cats:
        cats.append("unknown")
    category_id_map = {c: i for i, c in enumerate(cats)}

    # Save
    user_features.to_parquet(PROCESSED / "user_features.parquet")
    item_features.drop(columns=["text_description"], errors="ignore").to_parquet(
        PROCESSED / "item_features.parquet"
    )
    np.save(PROCESSED / "item_text_embs.npy", text_embs)

    with open(PROCESSED / "id_maps.pkl", "wb") as f:
        pickle.dump(
            {
                "user_id_map":     user_id_map,
                "item_id_map":     item_id_map,
                "item_id_reverse": {v: k for k, v in item_id_map.items()},
                "category_id_map": category_id_map,
            },
            f,
        )

    logger.info("=" * 60)
    logger.info("Feature build complete")
    logger.info("  user_features : %s  (%d rows)", PROCESSED / "user_features.parquet", len(user_features))
    logger.info("  item_features : %s  (%d rows)", PROCESSED / "item_features.parquet", len(item_features))
    logger.info("  text_embs     : %s  shape=%s", PROCESSED / "item_text_embs.npy", text_embs.shape)
    logger.info("  id_maps       : %d users, %d items, %d categories",
                len(user_id_map), len(item_id_map), len(category_id_map))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
