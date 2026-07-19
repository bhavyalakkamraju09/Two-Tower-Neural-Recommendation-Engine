"""
src/models/infonce_loss.py
Symmetric InfoNCE (NT-Xent) contrastive loss for Two-Tower training.

For B (user, positive_item) pairs in a batch:
  - Build B×B similarity matrix
  - Row i: user_i vs all B items  → positive = column i
  - Col i: item_i vs all B users  → positive = row i
  - Loss = average of both cross-entropies (symmetric)

In-batch negatives naturally skew popular items (hard negatives) — exactly
what we want to avoid false positives in a long-tail catalog.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def infonce_loss(
    user_embs: torch.Tensor,
    item_embs: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """
    Parameters
    ----------
    user_embs   : (B, D) L2-normalised
    item_embs   : (B, D) L2-normalised
    temperature : scalar > 0

    Returns
    -------
    Scalar loss
    """
    B      = user_embs.size(0)
    logits = torch.matmul(user_embs, item_embs.T) / temperature  # (B, B)
    labels = torch.arange(B, device=user_embs.device)
    loss   = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
    return loss


def weighted_infonce_loss(
    user_embs:  torch.Tensor,
    item_embs:  torch.Tensor,
    temperature: torch.Tensor,
    weights:    torch.Tensor,
) -> torch.Tensor:
    """
    InfoNCE weighted by implicit feedback confidence (review_score / 5).
    weights : (B,) in [0, 1]
    """
    B      = user_embs.size(0)
    logits = torch.matmul(user_embs, item_embs.T) / temperature
    labels = torch.arange(B, device=user_embs.device)

    loss_u = F.cross_entropy(logits,   labels, reduction="none")
    loss_i = F.cross_entropy(logits.T, labels, reduction="none")

    w = weights / (weights.sum() + 1e-8) * B
    return ((loss_u + loss_i) / 2.0 * w).mean()
