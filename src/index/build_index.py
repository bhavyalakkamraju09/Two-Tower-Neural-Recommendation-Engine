"""
src/index/build_index.py
Encode all Olist items with the trained item tower and upsert to Pinecone.

Usage:
    export PINECONE_API_KEY=your_key
    python -m src.index.build_index
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..models.two_tower import TwoTowerModel
from .pinecone_client import upsert_item_embeddings, index_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED   = Path("data/processed")
CHECKPOINTS = Path("checkpoints")
BATCH       = 512


def main() -> None:
    # Device
    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    logger.info("Device: %s", device)

    # Load checkpoint
    ckpt_path = CHECKPOINTS / "two_tower_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found. Run train_two_tower first.")

    ckpt    = torch.load(ckpt_path, map_location="cpu")
    hp      = ckpt["hparams"]
    id_maps = ckpt.get("id_maps") or pickle.load(open(PROCESSED / "id_maps.pkl", "rb"))

    model = TwoTowerModel(
        n_users=hp["n_users"], n_items=hp["n_items"], n_categories=hp["n_categories"],
        embed_dim=hp.get("embed_dim", 256), output_dim=hp.get("output_dim", 128),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    item_features  = pd.read_parquet(PROCESSED / "item_features.parquet")
    text_embs      = np.load(PROCESSED / "item_text_embs.npy")
    item_id_map    = id_maps["item_id_map"]
    item_id_rev    = id_maps["item_id_reverse"]
    category_id_map = id_maps["category_id_map"]

    item_ids = list(item_id_map.keys())
    logger.info("Encoding %d items...", len(item_ids))

    all_embeddings: dict[str, np.ndarray] = {}

    for start in tqdm(range(0, len(item_ids), BATCH), desc="Item encoding"):
        batch_pids = item_ids[start:start + BATCH]

        idx_t = torch.tensor([item_id_map[p] for p in batch_pids], dtype=torch.long)
        cat_t = torch.tensor(
            [
                category_id_map.get(
                    str(item_features.loc[p, "product_category_name_english"])
                    if p in item_features.index else "unknown",
                    0,
                )
                for p in batch_pids
            ],
            dtype=torch.long,
        )
        txt_t = torch.tensor(
            np.stack([
                text_embs[item_id_map[p]] if item_id_map[p] < len(text_embs)
                else np.zeros(384, dtype=np.float32)
                for p in batch_pids
            ]),
            dtype=torch.float,
        )
        feat_t = torch.tensor(
            np.array([
                [
                    float(item_features.loc[p, "log_price"])        if p in item_features.index else 0.0,
                    float(item_features.loc[p, "avg_review_score"]) if p in item_features.index else 3.0,
                ]
                for p in batch_pids
            ], dtype=np.float32),
            dtype=torch.float,
        )

        with torch.no_grad():
            embs = model.encode_item(
                idx_t.to(device), cat_t.to(device),
                txt_t.to(device), feat_t.to(device),
            ).cpu().numpy()

        for pid, emb in zip(batch_pids, embs):
            all_embeddings[pid] = emb

    logger.info("Encoded %d item embeddings.", len(all_embeddings))

    # Save locally
    np.save(PROCESSED / "item_embeddings_128d.npy", np.stack(list(all_embeddings.values())))
    with open(PROCESSED / "item_embedding_ids.pkl", "wb") as f:
        pickle.dump(list(all_embeddings.keys()), f)

    # Build metadata
    metadata = {}
    for pid, emb in all_embeddings.items():
        if pid in item_features.index:
            row = item_features.loc[pid]
            metadata[pid] = {
                "category":       str(row.get("product_category_name_english", "unknown")),
                "log_price":      float(row.get("log_price", 0.0)),
                "avg_review":     float(row.get("avg_review_score", 3.0)),
                "purchase_count": int(row.get("purchase_count", 0)),
            }

    upsert_item_embeddings(all_embeddings, metadata)
    stats = index_stats()
    logger.info("Pinecone index stats: %s", stats)


if __name__ == "__main__":
    main()
