"""
src/models/train_two_tower.py
Training loop for the Two-Tower model.

Usage:
    python -m src.models.train_two_tower [--epochs 30] [--batch-size 512] [--lr 1e-3]

Recommended: run on Google Colab T4 GPU (~1 hr for 30 epochs).
On Apple MPS or CPU: ~12-15 min for 30 epochs on Olist dataset.
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..data.dataset import TwoTowerDataset
from ..data.loader import load_interactions
from .infonce_loss import weighted_infonce_loss
from .two_tower import TwoTowerModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED   = Path("data/processed")
CHECKPOINTS = Path("checkpoints")
CHECKPOINTS.mkdir(exist_ok=True)


def train(
    epochs:     int   = 30,
    batch_size: int   = 512,
    lr:         float = 1e-3,
    embed_dim:  int   = 256,
    output_dim: int   = 128,
    dropout:    float = 0.2,
    patience:   int   = 5,
) -> None:

    # Device — prefer MPS (Apple Silicon) then CUDA then CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # Load data
    train_df, _ = load_interactions()
    user_features = pd.read_parquet(PROCESSED / "user_features.parquet")
    item_features = pd.read_parquet(PROCESSED / "item_features.parquet")
    text_embs     = np.load(PROCESSED / "item_text_embs.npy")

    with open(PROCESSED / "id_maps.pkl", "rb") as f:
        id_maps = pickle.load(f)

    dataset = TwoTowerDataset(
        interactions=train_df,
        user_features=user_features,
        item_features=item_features,
        text_embs=text_embs,
        user_id_map=id_maps["user_id_map"],
        item_id_map=id_maps["item_id_map"],
        category_id_map=id_maps["category_id_map"],
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,       # 0 = safer on MPS / Windows
        pin_memory=device.type == "cuda",
        drop_last=True,      # InfoNCE needs full batches
    )
    logger.info("Dataset: %d samples | %d batches/epoch", len(dataset), len(loader))

    n_users = len(id_maps["user_id_map"])
    n_items = len(id_maps["item_id_map"])
    n_cats  = len(id_maps["category_id_map"])

    model = TwoTowerModel(
        n_users=n_users, n_items=n_items, n_categories=n_cats,
        embed_dim=embed_dim, output_dim=output_dim,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss  = float("inf")
    no_improve = 0
    best_ckpt  = CHECKPOINTS / "two_tower_best.pth"

    # MLflow (optional — skip gracefully if not available or file-store blocked)
    try:
        import mlflow
        mlflow.set_experiment("two_tower_training")
        run = mlflow.start_run(run_name=f"e{epochs}_b{batch_size}_lr{lr}")
        mlflow.log_params({
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "embed_dim": embed_dim, "output_dim": output_dim,
            "n_users": n_users, "n_items": n_items, "device": str(device),
        })
        use_mlflow = True
    except Exception:
        use_mlflow = False
        logger.info("MLflow unavailable — skipping experiment tracking")

    logger.info("Starting training: %d epochs, batch=%d, lr=%g", epochs, batch_size, lr)
    logger.info("Model: %d users, %d items, %d categories", n_users, n_items, n_cats)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            u_ids  = batch["user_id"].to(device)
            u_feat = batch["user_feats"].to(device)
            i_ids  = batch["item_id"].to(device)
            c_ids  = batch["cat_id"].to(device)
            t_emb  = batch["text_emb"].to(device)
            i_feat = batch["item_feats"].to(device)
            w      = batch["weight"].to(device)

            u, v = model(u_ids, u_feat, i_ids, c_ids, t_emb, i_feat)
            loss = weighted_infonce_loss(u, v, model.temperature, w)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        temp     = model.temperature.item()

        if use_mlflow:
            try:
                mlflow.log_metrics({"train_loss": avg_loss, "temperature": temp}, step=epoch)
            except Exception:
                pass

        logger.info("Epoch %3d/%d | loss=%.4f | temp=%.4f", epoch, epochs, avg_loss, temp)

        if avg_loss < best_loss - 1e-4:
            best_loss  = avg_loss
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "id_maps": id_maps,
                    "hparams": {
                        "n_users": n_users, "n_items": n_items,
                        "n_categories": n_cats,
                        "embed_dim": embed_dim, "output_dim": output_dim,
                    },
                },
                best_ckpt,
            )
            logger.info("  ✓ Saved best checkpoint (loss=%.4f)", best_loss)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if use_mlflow:
        try:
            mlflow.log_metric("best_loss", best_loss)
            mlflow.end_run()
        except Exception:
            pass

    logger.info("Training complete. Best checkpoint: %s (loss=%.4f)", best_ckpt, best_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--embed-dim",  type=int,   default=256)
    parser.add_argument("--output-dim", type=int,   default=128)
    parser.add_argument("--dropout",    type=float, default=0.2)
    parser.add_argument("--patience",   type=int,   default=5)
    args = parser.parse_args()
    train(**{k.replace("-", "_"): v for k, v in vars(args).items()})
