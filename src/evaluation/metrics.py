"""
src/evaluation/metrics.py
RecSys evaluation metrics: NDCG@k, Recall@k, Precision@k, MRR, Hit Rate, Coverage.

All metrics are designed for Leave-One-Out evaluation where each test user
has exactly ONE held-out ground-truth item.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


# ── Per-user metrics ───────────────────────────────────────────────────────────

def ndcg_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    dcg  = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    return sum(1 for item in recommended[:k] if item in relevant) / k


def recall_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    if not relevant:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / len(relevant)


def hit_rate_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """1 if any relevant item is in top-k, else 0."""
    return float(any(item in relevant for item in recommended[:k]))


def mrr(recommended: list, relevant: set) -> float:
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ── Catalog metrics ────────────────────────────────────────────────────────────

def catalog_coverage(all_recs: list[list], catalog_size: int) -> float:
    unique = {item for recs in all_recs for item in recs}
    return len(unique) / catalog_size if catalog_size > 0 else 0.0


def popularity_bias(
    all_recs: list[list],
    item_purchase_counts: dict[str, int],
) -> dict[str, float]:
    all_items = [item for recs in all_recs for item in recs]
    counts    = np.array([item_purchase_counts.get(it, 0) for it in all_items], dtype=float)
    avg_pop   = float(counts.mean()) if len(counts) else 0.0

    # Gini coefficient
    sorted_c = np.sort(counts)
    n = len(sorted_c)
    if n == 0 or sorted_c.sum() == 0:
        gini = 0.0
    else:
        idx  = np.arange(1, n + 1)
        gini = float((2 * (idx * sorted_c).sum()) / (n * sorted_c.sum()) - (n + 1) / n)

    return {"avg_recommendation_popularity": avg_pop, "gini_coefficient": gini}


# ── Full model evaluation ──────────────────────────────────────────────────────

def evaluate_model(
    model_fn:          Callable[[str, int], list[str]],
    test_interactions: dict[str, list[str]],
    k:                 int = 10,
    max_users:         int | None = None,
) -> dict[str, float]:
    """
    Evaluate a recommendation model on the test set.

    Parameters
    ----------
    model_fn          : callable(user_id, n) → list of item_ids
    test_interactions : {user_id: [ground_truth_item_ids]}
    k                 : cutoff for metrics
    max_users         : limit to first N users (speed)

    Returns
    -------
    dict of metric_name → mean_value
    """
    ndcgs, precs, recalls, mrrs, hits = [], [], [], [], []
    all_recs = []

    users = list(test_interactions.keys())
    if max_users:
        users = users[:max_users]

    for uid in users:
        relevant = set(test_interactions[uid])
        if not relevant:
            continue
        try:
            recs = model_fn(uid, n=k)
        except Exception:
            continue
        if not recs:
            continue

        ndcgs.append(ndcg_at_k(recs, relevant, k))
        precs.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        mrrs.append(mrr(recs, relevant))
        hits.append(hit_rate_at_k(recs, relevant, k))
        all_recs.append(recs)

    def _mean(lst: list) -> float:
        return float(np.mean(lst)) if lst else 0.0

    return {
        f"ndcg@{k}":       _mean(ndcgs),
        f"precision@{k}":  _mean(precs),
        f"recall@{k}":     _mean(recalls),
        f"hit_rate@{k}":   _mean(hits),
        "mrr":             _mean(mrrs),
        "n_users_evaluated": len(ndcgs),
    }
