"""
src/api/main.py
FastAPI recommendation service.

Endpoints:
  POST /recommend       top-N items for a user
  GET  /health          health check + model status
  GET  /item/{item_id}  item metadata
  GET  /stats           Pinecone index stats

Run:
    export PINECONE_API_KEY=your_key
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINTS = Path("checkpoints")
PROCESSED   = Path("data/processed")


# ── App state ──────────────────────────────────────────────────────────────────

class _State:
    two_tower   = None
    als_model   = None
    encode_user = None
    user_features: pd.DataFrame = None
    item_features: pd.DataFrame = None
    id_maps:  dict = {}
    top_popular: list[str] = []
    redis = None

state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models...")

    # ID maps
    maps_path = PROCESSED / "id_maps.pkl"
    if maps_path.exists():
        with open(maps_path, "rb") as f:
            state.id_maps = pickle.load(f)

    # Features
    uf_path = PROCESSED / "user_features.parquet"
    if_path = PROCESSED / "item_features.parquet"
    if uf_path.exists(): state.user_features = pd.read_parquet(uf_path)
    if if_path.exists(): state.item_features = pd.read_parquet(if_path)

    # ALS
    als_path = CHECKPOINTS / "als_model.pkl"
    if als_path.exists():
        from ..models.als_model import ALSRecommender
        state.als_model = ALSRecommender.load(als_path)
        logger.info("ALS model loaded.")
        if state.item_features is not None:
            state.top_popular = (
                state.item_features["purchase_count"]
                .sort_values(ascending=False)
                .index.tolist()[:200]
            )

    # Two-Tower
    tt_path = CHECKPOINTS / "two_tower_best.pth"
    if tt_path.exists() and state.id_maps:
        from ..models.two_tower import TwoTowerModel
        from ..embeddings.user_encoder import get_user_embedding_fn
        ckpt = torch.load(tt_path, map_location="cpu")
        hp   = ckpt["hparams"]
        model = TwoTowerModel(
            n_users=hp["n_users"], n_items=hp["n_items"], n_categories=hp["n_categories"],
            embed_dim=hp.get("embed_dim", 256), output_dim=hp.get("output_dim", 128),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        state.two_tower = model
        if state.user_features is not None:
            state.encode_user = get_user_embedding_fn(model, state.id_maps, state.user_features)
        logger.info("Two-Tower model loaded.")

    # Redis (optional)
    try:
        import redis
        state.redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True,
            socket_connect_timeout=1,
        )
        state.redis.ping()
        logger.info("Redis connected.")
    except Exception:
        state.redis = None
        logger.info("Redis not available — caching disabled.")

    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="E-Commerce Recommendation API",
    description="Two-Tower NCF + ALS + Popularity baseline",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    user_id: str
    n:       int  = Field(default=10, ge=1, le=50)
    model:   str  = Field(default="als", pattern="^(two_tower|als|popular)$")


class RecommendResponse(BaseModel):
    user_id:         str
    recommendations: list[str]
    model:           str
    latency_ms:      float
    from_cache:      bool = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cache_get(user_id: str) -> np.ndarray | None:
    if not state.redis:
        return None
    try:
        raw = state.redis.get(f"uemb:{user_id}")
        return np.array(json.loads(raw), dtype=np.float32) if raw else None
    except Exception:
        return None


def _cache_set(user_id: str, emb: np.ndarray) -> None:
    if not state.redis:
        return
    try:
        state.redis.setex(f"uemb:{user_id}", 3600, json.dumps(emb.tolist()))
    except Exception:
        pass


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    t0         = time.monotonic()
    from_cache = False

    if req.model == "two_tower":
        if state.encode_user is None:
            raise HTTPException(503, "Two-Tower model not loaded.")
        from ..index.pinecone_client import query_similar_items

        emb = _cache_get(req.user_id)
        if emb is None:
            emb = state.encode_user(req.user_id)
            if emb is None:
                raise HTTPException(404, f"User {req.user_id} not in training set.")
            _cache_set(req.user_id, emb)
        else:
            from_cache = True

        candidates = query_similar_items(emb, top_k=req.n)
        recs = [c["item_id"] for c in candidates]

    elif req.model == "als":
        if state.als_model is None:
            raise HTTPException(503, "ALS model not loaded.")
        recs = state.als_model.recommend(req.user_id, n=req.n)
        if not recs:
            raise HTTPException(404, f"User {req.user_id} is a cold-start user for ALS.")

    elif req.model == "popular":
        recs = state.top_popular[:req.n]

    else:
        raise HTTPException(400, f"Unknown model: {req.model}")

    return RecommendResponse(
        user_id=req.user_id,
        recommendations=recs,
        model=req.model,
        latency_ms=round((time.monotonic() - t0) * 1000, 2),
        from_cache=from_cache,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "two_tower": state.two_tower is not None,
            "als":       state.als_model is not None,
            "popular":   len(state.top_popular) > 0,
        },
        "redis": state.redis is not None,
    }


@app.get("/item/{item_id}")
def item_metadata(item_id: str):
    if state.item_features is None:
        raise HTTPException(503, "Item features not loaded.")
    if item_id not in state.item_features.index:
        raise HTTPException(404, f"Item {item_id} not found.")
    row = state.item_features.loc[item_id]
    return {
        "item_id":      item_id,
        "category":     str(row.get("product_category_name_english", "unknown")),
        "avg_price":    float(row.get("avg_price",        0)),
        "avg_review":   float(row.get("avg_review_score", 0)),
        "purchase_count": int(row.get("purchase_count",   0)),
    }


@app.get("/stats")
def pinecone_stats():
    try:
        from ..index.pinecone_client import index_stats
        return index_stats()
    except Exception as e:
        raise HTTPException(503, str(e))
