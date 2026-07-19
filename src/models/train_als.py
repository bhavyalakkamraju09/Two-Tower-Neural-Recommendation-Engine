"""
src/models/train_als.py
Train ALS baseline and evaluate on LOO test set.

Usage:
    python -m src.models.train_als
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..data.loader import load_interactions
from ..evaluation.metrics import evaluate_model
from .als_model import ALSRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINTS = Path("checkpoints")
CHECKPOINTS.mkdir(exist_ok=True)


def train_als(
    factors:        int   = 128,
    iterations:     int   = 50,
    regularization: float = 0.01,
    alpha:          float = 40.0,
) -> ALSRecommender:

    train_df, test_df = load_interactions()

    model = ALSRecommender(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        alpha=alpha,
    )
    model.fit(train_df)

    # Evaluate on LOO test set
    # test_df has one row per user — their held-out last item
    test_interactions = (
        test_df.groupby("user_id")["product_id"]
        .apply(list)
        .to_dict()
    )

    import random
    sample_users = random.sample(
        list(test_interactions.keys()),
        min(1000, len(test_interactions)),
    )
    test_sample = {u: test_interactions[u] for u in sample_users}

    results = evaluate_model(
        model_fn=lambda uid, n=10: model.recommend(uid, n=n),
        test_interactions=test_sample,
        k=10,
    )
    logger.info("ALS evaluation (%d users): %s", len(sample_users), results)

    # Log to MLflow — sanitise metric names (@ not allowed)
    try:
        import mlflow
        mlflow.set_experiment("als_baseline")
        with mlflow.start_run(run_name="als_training"):
            mlflow.log_params({
                "factors": factors, "iterations": iterations,
                "regularization": regularization, "alpha": alpha,
            })
            clean = {k.replace("@", "_at_"): v for k, v in results.items()}
            mlflow.log_metrics(clean)
    except Exception as e:
        logger.warning("MLflow logging skipped: %s", e)

    model.save(CHECKPOINTS / "als_model.pkl")
    return model


if __name__ == "__main__":
    train_als()
