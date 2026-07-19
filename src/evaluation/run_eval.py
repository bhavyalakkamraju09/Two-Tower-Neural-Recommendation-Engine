"""
src/evaluation/run_eval.py
Full offline evaluation: ALS vs Popularity baseline.
Two-Tower eval requires Pinecone — runs if PINECONE_API_KEY is set.

Usage:
    python -m src.evaluation.run_eval

Results printed to console and saved to data/processed/eval_results.json
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..data.loader import load_interactions
from ..models.als_model import ALSRecommender
from .metrics import evaluate_model, catalog_coverage, popularity_bias

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINTS = Path("checkpoints")
PROCESSED   = Path("data/processed")


def build_popularity_fn(train_df: pd.DataFrame):
    top = (
        train_df.groupby("product_id").size()
        .sort_values(ascending=False)
        .index.tolist()
    )
    return lambda uid, n=10: top[:n]


def ab_test(
    scores_a: list[float],
    scores_b: list[float],
    name_a: str,
    name_b: str,
    k: int = 10,
) -> dict:
    mean_a = float(np.mean(scores_a)) if scores_a else 0.0
    mean_b = float(np.mean(scores_b)) if scores_b else 0.0

    if scores_a and scores_b:
        stat, p = stats.mannwhitneyu(scores_a, scores_b, alternative="greater")
    else:
        stat, p = 0.0, 1.0

    result = {
        f"mean_ndcg@{k}_{name_a}": mean_a,
        f"mean_ndcg@{k}_{name_b}": mean_b,
        "delta":      mean_a - mean_b,
        "p_value":    float(p),
        "significant": bool(p < 0.05),
        "winner":     name_a if (p < 0.05 and mean_a > mean_b) else name_b,
    }
    logger.info(
        "A/B %s vs %s | Δ=%.4f | p=%.4f | sig=%s | winner=%s",
        name_a, name_b, mean_a - mean_b, p, result["significant"], result["winner"],
    )
    return result


def main(max_eval_users: int = 2000) -> None:
    train_df, test_df = load_interactions()

    test_interactions = (
        test_df.groupby("user_id")["product_id"].apply(list).to_dict()
    )
    all_users    = list(test_interactions.keys())
    sample_users = random.sample(all_users, min(max_eval_users, len(all_users)))
    test_sample  = {u: test_interactions[u] for u in sample_users}
    logger.info("Evaluating on %d test users", len(test_sample))

    # Load ALS
    als_path = CHECKPOINTS / "als_model.pkl"
    if not als_path.exists():
        raise FileNotFoundError("Run: python -m src.models.train_als first")
    als   = ALSRecommender.load(als_path)
    als_fn = lambda uid, n=10: als.recommend(uid, n=n)
    pop_fn = build_popularity_fn(train_df)

    catalog_size         = train_df["product_id"].nunique()
    item_purchase_counts = train_df.groupby("product_id").size().to_dict()

    all_results: dict = {}

    # ── Evaluate each model ───────────────────────────────────────────────────
    for name, fn in [("als", als_fn), ("popularity", pop_fn)]:
        logger.info("Evaluating %s...", name)
        results = evaluate_model(fn, test_sample, k=10)
        logger.info("%s: %s", name, {k: round(v, 4) for k, v in results.items()})

        sample_recs = [fn(uid) for uid in list(test_sample.keys())[:500]]
        sample_recs = [r for r in sample_recs if r]
        cov  = catalog_coverage(sample_recs, catalog_size)
        bias = popularity_bias(sample_recs, item_purchase_counts)

        all_results[name] = {
            **results,
            "catalog_coverage":  round(cov, 4),
            "avg_popularity":    round(bias["avg_recommendation_popularity"], 2),
            "gini_coefficient":  round(bias["gini_coefficient"], 4),
        }

    # ── A/B test: ALS vs Popularity ───────────────────────────────────────────
    ndcg_als = []
    ndcg_pop = []
    from .metrics import ndcg_at_k
    for uid, relevant_items in test_sample.items():
        relevant = set(relevant_items)
        r_als = als_fn(uid)
        r_pop = pop_fn(uid)
        if r_als: ndcg_als.append(ndcg_at_k(r_als, relevant))
        if r_pop: ndcg_pop.append(ndcg_at_k(r_pop, relevant))

    all_results["ab_test_als_vs_popularity"] = ab_test(
        ndcg_als, ndcg_pop, "als", "popularity"
    )

    # ── Optional: Two-Tower eval (needs Pinecone) ──────────────────────────────
    if os.getenv("PINECONE_API_KEY"):
        try:
            logger.info("Pinecone key found — evaluating Two-Tower...")
            import pickle, torch
            from ..models.two_tower import TwoTowerModel
            from ..embeddings.user_encoder import get_user_embedding_fn
            from ..index.pinecone_client import query_similar_items

            ckpt_path = CHECKPOINTS / "two_tower_best.pth"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu")
                hp   = ckpt["hparams"]
                id_maps = ckpt.get("id_maps") or pickle.load(
                    open(PROCESSED / "id_maps.pkl", "rb")
                )
                model = TwoTowerModel(
                    n_users=hp["n_users"], n_items=hp["n_items"],
                    n_categories=hp["n_categories"],
                    embed_dim=hp.get("embed_dim", 256),
                    output_dim=hp.get("output_dim", 128),
                )
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()

                user_features = pd.read_parquet(PROCESSED / "user_features.parquet")
                encode_user   = get_user_embedding_fn(model, id_maps, user_features)

                def tt_fn(uid, n=10):
                    emb = encode_user(uid)
                    if emb is None:
                        return []
                    return [c["item_id"] for c in query_similar_items(emb, top_k=n)]

                results_tt = evaluate_model(tt_fn, test_sample, k=10, max_users=500)
                logger.info("Two-Tower: %s", {k: round(v, 4) for k, v in results_tt.items()})
                all_results["two_tower"] = results_tt

                ndcg_tt = []
                for uid, relevant_items in list(test_sample.items())[:500]:
                    r = tt_fn(uid)
                    if r:
                        ndcg_tt.append(ndcg_at_k(r, set(relevant_items)))
                all_results["ab_test_tt_vs_als"] = ab_test(ndcg_tt, ndcg_als[:len(ndcg_tt)], "two_tower", "als")
        except Exception as e:
            logger.warning("Two-Tower eval failed: %s", e)
    else:
        logger.info("PINECONE_API_KEY not set — skipping Two-Tower eval")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = PROCESSED / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved → %s", out_path)

    # ── Print summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    for model_name, metrics in all_results.items():
        if "ab_test" in model_name:
            continue
        logger.info("\n%s:", model_name.upper())
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info("  %-25s %.4f", k, v)
    logger.info("=" * 60)
    logger.info("Full results: %s", out_path)
    logger.info("Run: mlflow ui  to view in MLflow")


if __name__ == "__main__":
    main()
