"""
src/models/als_model.py
ALS implicit collaborative filtering baseline using the `implicit` library.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


class ALSRecommender:
    def __init__(
        self,
        factors:        int   = 128,
        iterations:     int   = 50,
        regularization: float = 0.01,
        alpha:          float = 40.0,
        random_state:   int   = 42,
    ) -> None:
        self.factors        = factors
        self.iterations     = iterations
        self.regularization = regularization
        self.alpha          = alpha
        self.random_state   = random_state

        self.model:          object | None           = None
        self.user_id_map:    dict[str, int]          = {}
        self.item_id_map:    dict[str, int]          = {}
        self.item_id_reverse: dict[int, str]         = {}
        self.matrix:         sp.csr_matrix | None    = None

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(self, interactions_df: pd.DataFrame) -> "ALSRecommender":
        try:
            from implicit import als
        except ImportError:
            raise ImportError("pip install implicit")

        users = interactions_df["user_id"].unique()
        items = interactions_df["product_id"].unique()
        self.user_id_map     = {u: i for i, u in enumerate(users)}
        self.item_id_map     = {p: i for i, p in enumerate(items)}
        self.item_id_reverse = {v: k for k, v in self.item_id_map.items()}

        rows = interactions_df["user_id"].map(self.user_id_map).values
        cols = interactions_df["product_id"].map(self.item_id_map).values
        data = interactions_df["implicit_weight"].values * self.alpha

        self.matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(users), len(items)),
        )

        logger.info(
            "Fitting ALS: %d users x %d items  (factors=%d, iters=%d, alpha=%.0f)",
            len(users), len(items), self.factors, self.iterations, self.alpha,
        )
        self.model = als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            use_gpu=False,
            random_state=self.random_state,
        )
        self.model.fit(self.matrix)
        logger.info("ALS training complete.")
        return self

    # ── Inference ──────────────────────────────────────────────────────────────

    def recommend(self, user_id: str, n: int = 10) -> list[str]:
        """Return top-n product IDs. Returns [] for unknown users."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        if user_id not in self.user_id_map:
            return []
        uid = self.user_id_map[user_id]
        ids, _ = self.model.recommend(
            uid, self.matrix[uid], N=n, filter_already_liked_items=True
        )
        return [self.item_id_reverse[i] for i in ids if i in self.item_id_reverse]

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("ALS model saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ALSRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)
