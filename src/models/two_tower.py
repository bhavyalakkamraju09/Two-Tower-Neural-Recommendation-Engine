"""
src/models/two_tower.py
Two-Tower neural collaborative filtering model.

UserTower : user_id embedding (256d) + behavioral features → 128d L2-norm
ItemTower : item_id (256d) + category (64d) + SBERT (384d) + price/review → 128d L2-norm

At serving time only the item tower is run offline (batch encode all items).
Online: user tower forward pass → Pinecone ANN query.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dims: list[int], dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.BatchNorm1d(dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UserTower(nn.Module):
    """
    Inputs
    ------
    user_ids   : (B,) long
    user_feats : (B, 3) float  [log_purchase_count, avg_price, avg_review_given]

    Output : (B, output_dim) L2-normalised
    """
    def __init__(
        self,
        n_users: int,
        embed_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_users + 1, embed_dim, padding_idx=0)
        self.mlp   = MLP([embed_dim + 3, 256, 128, output_dim], dropout)

    def forward(self, user_ids: torch.Tensor, user_feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.embed(user_ids), user_feats], dim=1)
        return F.normalize(self.mlp(x), dim=1)


class ItemTower(nn.Module):
    """
    Inputs
    ------
    item_ids   : (B,) long
    cat_ids    : (B,) long
    text_embs  : (B, 384) float  SBERT embeddings
    item_feats : (B, 2)   float  [log_price, avg_review_score]

    Output : (B, output_dim) L2-normalised
    """
    def __init__(
        self,
        n_items: int,
        n_categories: int,
        text_dim: int = 384,
        embed_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.item_embed = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.cat_embed  = nn.Embedding(n_categories + 1, 64, padding_idx=0)
        in_dim = embed_dim + 64 + text_dim + 2   # 256+64+384+2 = 706
        self.mlp = MLP([in_dim, 512, 256, output_dim], dropout)

    def forward(
        self,
        item_ids:   torch.Tensor,
        cat_ids:    torch.Tensor,
        text_embs:  torch.Tensor,
        item_feats: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [self.item_embed(item_ids), self.cat_embed(cat_ids), text_embs, item_feats],
            dim=1,
        )
        return F.normalize(self.mlp(x), dim=1)


class TwoTowerModel(nn.Module):
    """Full Two-Tower model with learnable temperature."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_categories: int,
        text_dim: int = 384,
        embed_dim: int = 256,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.user_tower = UserTower(n_users, embed_dim, output_dim)
        self.item_tower = ItemTower(n_items, n_categories, text_dim, embed_dim, output_dim)
        # Learnable temperature — initialised to ~0.07
        self.log_temperature = nn.Parameter(torch.tensor(-2.659))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)

    def forward(
        self,
        user_ids:   torch.Tensor,
        user_feats: torch.Tensor,
        item_ids:   torch.Tensor,
        cat_ids:    torch.Tensor,
        text_embs:  torch.Tensor,
        item_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.user_tower(user_ids, user_feats)
        v = self.item_tower(item_ids, cat_ids, text_embs, item_feats)
        return u, v

    @torch.no_grad()
    def encode_user(
        self, user_ids: torch.Tensor, user_feats: torch.Tensor
    ) -> torch.Tensor:
        return self.user_tower(user_ids, user_feats)

    @torch.no_grad()
    def encode_item(
        self,
        item_ids:   torch.Tensor,
        cat_ids:    torch.Tensor,
        text_embs:  torch.Tensor,
        item_feats: torch.Tensor,
    ) -> torch.Tensor:
        return self.item_tower(item_ids, cat_ids, text_embs, item_feats)
