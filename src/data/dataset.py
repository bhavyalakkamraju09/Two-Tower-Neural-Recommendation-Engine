"""
src/data/dataset.py
PyTorch Dataset for Two-Tower contrastive training.
Each sample = (user, positive_item) pair.
Negatives are sampled in-batch by the InfoNCE loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TwoTowerDataset(Dataset):
    def __init__(
        self,
        interactions:    pd.DataFrame,
        user_features:   pd.DataFrame,
        item_features:   pd.DataFrame,
        text_embs:       np.ndarray,
        user_id_map:     dict[str, int],
        item_id_map:     dict[str, int],
        category_id_map: dict[str, int],
    ) -> None:
        valid_users = set(user_features.index) & set(user_id_map.keys())
        valid_items = set(item_features.index) & set(item_id_map.keys())
        self.df = interactions[
            interactions["user_id"].isin(valid_users)
            & interactions["product_id"].isin(valid_items)
        ].reset_index(drop=True)

        self.user_features   = user_features
        self.item_features   = item_features
        self.text_embs       = text_embs
        self.user_id_map     = user_id_map
        self.item_id_map     = item_id_map
        self.category_id_map = category_id_map

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        uid = str(row["user_id"])
        pid = str(row["product_id"])

        # User side
        user_idx = self.user_id_map.get(uid, 0)
        uf = self.user_features.loc[uid] if uid in self.user_features.index else None
        user_feats = np.array(
            [
                float(uf.get("log_purchase_count", 0)) if uf is not None else 0.0,
                float(uf.get("avg_price",          0)) if uf is not None else 0.0,
                float(uf.get("avg_review_given", 3.0)) if uf is not None else 3.0,
            ],
            dtype=np.float32,
        )

        # Item side
        item_idx = self.item_id_map.get(pid, 0)
        itf      = self.item_features.loc[pid] if pid in self.item_features.index else None
        cat_name = str(itf.get("product_category_name_english", "unknown")) if itf is not None else "unknown"
        cat_idx  = self.category_id_map.get(cat_name, 0)
        item_feats = np.array(
            [
                float(itf.get("log_price",        0.0)) if itf is not None else 0.0,
                float(itf.get("avg_review_score", 3.0)) if itf is not None else 3.0,
            ],
            dtype=np.float32,
        )

        # SBERT text embedding
        text_emb_idx = self.item_id_map.get(pid, 0)
        text_emb = (
            self.text_embs[text_emb_idx]
            if text_emb_idx < len(self.text_embs)
            else np.zeros(384, dtype=np.float32)
        )

        return {
            "user_id":    torch.tensor(user_idx,  dtype=torch.long),
            "user_feats": torch.tensor(user_feats, dtype=torch.float),
            "item_id":    torch.tensor(item_idx,  dtype=torch.long),
            "cat_id":     torch.tensor(cat_idx,   dtype=torch.long),
            "text_emb":   torch.tensor(text_emb,  dtype=torch.float),
            "item_feats": torch.tensor(item_feats, dtype=torch.float),
            "weight":     torch.tensor(float(row["implicit_weight"]), dtype=torch.float),
        }
