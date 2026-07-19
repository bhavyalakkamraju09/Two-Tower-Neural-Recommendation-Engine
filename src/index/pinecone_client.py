"""
src/index/pinecone_client.py
Pinecone vector index client — upsert item embeddings + ANN query.
Free tier: 1 index, 100K vectors — Olist 32K products fits easily.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

logger    = logging.getLogger(__name__)
INDEX_NAME = os.getenv("PINECONE_INDEX", "olist-items")
DIMENSION  = 128
BATCH_SIZE = 100


def get_index():
    try:
        from pinecone import Pinecone, ServerlessSpec
    except ImportError:
        raise ImportError("pip install pinecone")

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY not set. Add to .env or export.")

    pc       = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing:
        logger.info("Creating Pinecone index '%s' (dim=%d)...", INDEX_NAME, DIMENSION)
        pc.create_index(
            name=INDEX_NAME, dimension=DIMENSION, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        logger.info("Index '%s' ready.", INDEX_NAME)

    return pc.Index(INDEX_NAME)


def upsert_item_embeddings(
    embeddings: dict[str, np.ndarray],
    metadata:   dict[str, dict] | None = None,
) -> None:
    if metadata is None:
        metadata = {}

    index = get_index()
    items = list(embeddings.items())
    n     = len(items)
    logger.info("Upserting %d embeddings in batches of %d...", n, BATCH_SIZE)

    for start in range(0, n, BATCH_SIZE):
        batch = items[start:start + BATCH_SIZE]
        vectors = [
            {"id": pid, "values": emb.tolist(), "metadata": metadata.get(pid, {})}
            for pid, emb in batch
        ]
        index.upsert(vectors=vectors)
        if start % (BATCH_SIZE * 10) == 0:
            logger.info("  Upserted %d / %d", min(start + BATCH_SIZE, n), n)

    logger.info("Upsert complete — %d vectors in '%s'", n, INDEX_NAME)


def query_similar_items(
    user_embedding: np.ndarray,
    top_k:          int = 100,
    filter_meta:    dict | None = None,
) -> list[dict]:
    index  = get_index()
    kwargs = {"vector": user_embedding.tolist(), "top_k": top_k, "include_metadata": True}
    if filter_meta:
        kwargs["filter"] = filter_meta
    results = index.query(**kwargs)
    return [
        {"item_id": m["id"], "score": float(m["score"]), "metadata": m.get("metadata", {})}
        for m in results["matches"]
    ]


def index_stats() -> dict:
    return get_index().describe_index_stats()
