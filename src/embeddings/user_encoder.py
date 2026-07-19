"""
src/embeddings/user_encoder.py
Online user tower inference — given a user_id, return 128-d embedding.
"""
from __future__ import annotations
from typing import Callable

import numpy as np
import pandas as pd
import torch

from ..models.two_tower import TwoTowerModel


def get_user_embedding_fn(
    model: TwoTowerModel,
    id_maps: dict,
    user_features: pd.DataFrame,
) -> Callable[[str], np.ndarray | None]:
    """
    Returns encode(user_id) → 128-d numpy array or None if unknown user.
    """
    user_id_map = id_maps["user_id_map"]
    device      = next(model.parameters()).device

    def encode(user_id: str) -> np.ndarray | None:
        if user_id not in user_id_map:
            return None

        uid_t = torch.tensor([user_id_map[user_id]], dtype=torch.long, device=device)
        uf    = user_features.loc[user_id] if user_id in user_features.index else None
        feats = np.array(
            [
                float(uf.get("log_purchase_count", 0)) if uf is not None else 0.0,
                float(uf.get("avg_price",          0)) if uf is not None else 0.0,
                float(uf.get("avg_review_given", 3.0)) if uf is not None else 3.0,
            ],
            dtype=np.float32,
        )
        feats_t = torch.tensor(feats, device=device).unsqueeze(0)

        with torch.no_grad():
            return model.encode_user(uid_t, feats_t).squeeze(0).cpu().numpy()

    return encode
