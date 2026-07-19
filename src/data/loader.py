"""
src/data/loader.py
Loads Olist CSVs and builds implicit-feedback interactions with Leave-One-Out split.

Design note — why LOO instead of temporal split:
  Olist repeat-purchase rate is ~6% (5,654 of 93K users bought 2+ times).
  Temporal split yields only ~650 usable test interactions regardless of cutoff.
  LOO (He et al., NCF 2017) holds out each multi-purchase user's last item
  as the test target and trains on everything else.
  Result: 5,654 test users each with exactly 1 known ground-truth item.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

DATA_DIR   = Path("data/raw")
SPLITS_DIR = Path("data/splits")


# ── CSV loader ─────────────────────────────────────────────────────────────────

def load_olist() -> dict[str, pd.DataFrame]:
    files = {
        "orders":       "olist_orders_dataset.csv",
        "order_items":  "olist_order_items_dataset.csv",
        "products":     "olist_products_dataset.csv",
        "customers":    "olist_customers_dataset.csv",
        "reviews":      "olist_order_reviews_dataset.csv",
        "translations": "product_category_name_translation.csv",
    }
    data = {}
    for key, fname in files.items():
        path = DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}. Download Olist dataset first.")
        data[key] = pd.read_csv(path)
        logger.info("Loaded %-45s %d rows", fname, len(data[key]))
    return data


# ── Interaction builder ────────────────────────────────────────────────────────

def build_interactions(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (user_id, product_id, implicit_weight, ts) interactions then
    apply Leave-One-Out split.

    Returns
    -------
    train : all interactions minus each multi-purchase user's last item
    test  : one row per multi-purchase user — their last purchased item
    """
    df = (
        data["orders"]
        .merge(data["order_items"], on="order_id")
        .merge(data["customers"],   on="customer_id")
        .merge(
            data["reviews"][["order_id", "review_score"]],
            on="order_id", how="left",
        )
        .query("order_status == 'delivered'")
    )

    df["implicit_weight"] = df["review_score"].fillna(3.0) / 5.0

    interactions = (
        df[["customer_unique_id", "product_id", "implicit_weight",
            "order_purchase_timestamp"]]
        .drop_duplicates(subset=["customer_unique_id", "product_id"])
        .rename(columns={
            "customer_unique_id":       "user_id",
            "order_purchase_timestamp": "ts",
        })
        .copy()
    )
    interactions["ts"] = pd.to_datetime(interactions["ts"])
    interactions = interactions.sort_values(["user_id", "ts"]).reset_index(drop=True)

    # ── LOO split ─────────────────────────────────────────────────────────────
    counts      = interactions.groupby("user_id").size()
    multi_users = set(counts[counts >= 2].index)

    logger.info(
        "Users: %d total | %d multi-purchase (>=2) | %d single-purchase",
        len(counts), len(multi_users), (counts == 1).sum(),
    )

    # Rank interactions per user by timestamp descending (1 = most recent)
    interactions["_rank"] = (
        interactions.groupby("user_id")["ts"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    is_test = interactions["user_id"].isin(multi_users) & (interactions["_rank"] == 1)
    test    = interactions[is_test].drop(columns=["_rank"]).reset_index(drop=True)
    train   = interactions[~is_test].drop(columns=["_rank"]).reset_index(drop=True)

    logger.info(
        "LOO split — train: %d interactions (%d users) | test: %d interactions (%d users)",
        len(train), train["user_id"].nunique(),
        len(test),  test["user_id"].nunique(),
    )
    return train, test


# ── Sparse matrix ──────────────────────────────────────────────────────────────

def build_sparse_matrix(
    interactions: pd.DataFrame,
    user_id_map: dict[str, int] | None = None,
    item_id_map: dict[str, int] | None = None,
) -> tuple[sp.csr_matrix, dict, dict]:
    if user_id_map is None:
        user_id_map = {u: i for i, u in enumerate(interactions["user_id"].unique())}
    if item_id_map is None:
        item_id_map = {p: i for i, p in enumerate(interactions["product_id"].unique())}

    rows = interactions["user_id"].map(user_id_map).dropna().astype(int)
    cols = interactions["product_id"].map(item_id_map).dropna().astype(int)
    vals = interactions.loc[rows.index, "implicit_weight"].values

    matrix = sp.csr_matrix(
        (vals, (rows.values, cols.values)),
        shape=(len(user_id_map), len(item_id_map)),
    )
    return matrix, user_id_map, item_id_map


# ── Persist ────────────────────────────────────────────────────────────────────

def save_interactions(train: pd.DataFrame, test: pd.DataFrame) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(SPLITS_DIR / "train.parquet", index=False)
    test.to_parquet(SPLITS_DIR  / "test.parquet",  index=False)
    logger.info("Saved splits → %s", SPLITS_DIR)


def load_interactions() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    test  = pd.read_parquet(SPLITS_DIR  / "test.parquet")
    return train, test
