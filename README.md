# E-Commerce Recommendation Engine

**Two-Tower NCF · InfoNCE Loss · Pinecone ANN · ALS Baseline · LightGBM Ranker · LOO Evaluation**

> Portfolio Project 04 — Bhavya Lakkamraju · [Portfolio](https://bhavyalakkamraju09.github.io) · [GitHub](https://github.com/bhavyalakkamraju09)

---

## Results

| Metric | ALS Baseline | Popularity | Two-Tower |
|--------|-------------|------------|-----------|
| **NDCG@10** | 0.035 | — | pending |
| **Hit Rate@10** | 0.042 | — | pending |
| **Recall@10** | 0.042 | — | pending |
| **MRR** | 0.033 | — | pending |
| **Catalog Coverage** | — | — | — |

*Evaluated on 5,654 held-out users via Leave-One-Out protocol (NCF 2017).*
*Two-Tower metrics pending Pinecone eval — run `python -m src.evaluation.run_eval`.*

### Why LOO and not temporal split?

Olist has a 94% single-purchase rate — 87,704 of 93,358 users bought exactly once.
Temporal split (any cutoff) yields only ~650 usable test interactions.
Leave-One-Out holds out each multi-purchase user's last item as the test target,
giving 5,654 test users with known ground truth — the standard approach for sparse
e-commerce datasets (He et al., NCF 2017).

---

## Architecture

```
User Tower                              Item Tower
  user_id  →  Embedding (256d)            product_id  →  Embedding (256d)
  + [log_purchase_count,      128-d       + category_id  →  Embedding (64d)
     avg_price,               L2-norm     + SBERT description  (384d)
     avg_review_given]  (3d)              + [log_price, avg_review]  (2d)
  →  MLP [259 → 256 → 128]               →  MLP [706 → 512 → 256 → 128]
  →  L2-normalize                         →  L2-normalize

Training:  InfoNCE contrastive loss (symmetric, in-batch negatives B=512)
           Learnable temperature τ: 0.07 → 0.031  ·  30 epochs
           AdamW + CosineAnnealingLR  ·  Early stopping (patience=5)

Serving:   User embedding → Pinecone ANN (top-100) → LightGBM LambdaRank → top-10
           Redis user embedding cache (1h TTL) · FastAPI · p99 latency <50ms
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Two-Tower model | PyTorch |
| Text embeddings | SBERT (all-MiniLM-L6-v2) |
| ALS baseline | implicit (Ben Frederickson) |
| ANN index | Pinecone Serverless (free tier) |
| Stage-2 ranker | LightGBM LambdaRank |
| Evaluation | NDCG@k, HR@k, Recall@k, MRR, LOO |
| Experiment tracking | MLflow |
| API | FastAPI + asyncio |
| Caching | Redis (1h TTL) |
| Demo | Streamlit |
| Data transforms | pandas + dbt |

---

## Dataset

**Olist Brazilian E-Commerce** — free Kaggle download.

| Metric | Value |
|--------|-------|
| Total orders | 99,441 |
| Total interactions | 100,380 |
| Training interactions | 94,726 |
| Users | 93,358 |
| Products | 32,951 |
| Products in training | 30,928 |
| Test users (LOO) | 5,654 |
| Single-purchase users | 87,704 (94%) |
| Date range | Jan 2017 – Aug 2018 |

---

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
cp .env.example .env
# Add your PINECONE_API_KEY to .env (free at pinecone.io)
```

### 2. Download dataset
```bash
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip
```

### 3. Build features
```bash
python -m src.data.build_features
# ~2 min: LOO split + SBERT encoding of 32K products
```

### 4. Train ALS baseline
```bash
export MLFLOW_ALLOW_FILE_STORE=true
python -m src.models.train_als
# ~1 min on CPU
```

### 5. Train Two-Tower
```bash
python -m src.models.train_two_tower
# ~12 min on Apple MPS / ~1 hr on CPU / ~1 hr on Colab T4
```

### 6. Build Pinecone index
```bash
export PINECONE_API_KEY=your_key
python -m src.index.build_index
# Encodes 32K items → upserts to Pinecone
```

### 7. Train LightGBM ranker
```bash
python -m src.models.train_ranker
# ~5 min — builds ALS candidates → trains re-ranker
```

### 8. Evaluate (get real metrics)
```bash
python -m src.evaluation.run_eval
# Saves results to data/processed/eval_results.json
```

### 9. Run demo
```bash
streamlit run app/streamlit_app.py
# Streamlit auto-loads real metrics from eval_results.json
```

### 10. Serve API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Repository Structure

```
ecommerce-recsys/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── raw/              # Olist CSVs (gitignored)
│   ├── processed/        # features, embeddings, id maps, eval results
│   └── splits/           # LOO train/test parquet files
│
├── src/
│   ├── data/
│   │   ├── loader.py         # LOO split logic + sparse matrix builder
│   │   ├── dataset.py        # TwoTowerDataset (PyTorch)
│   │   └── build_features.py # SBERT encoding + feature tables
│   │
│   ├── models/
│   │   ├── two_tower.py        # UserTower + ItemTower + TwoTowerModel
│   │   ├── infonce_loss.py     # Symmetric InfoNCE + weighted variant
│   │   ├── als_model.py        # ALS implicit CF wrapper
│   │   ├── lgbm_ranker.py      # LightGBM LambdaRank features
│   │   ├── train_two_tower.py  # Training loop + early stopping
│   │   ├── train_als.py        # ALS training + LOO eval
│   │   └── train_ranker.py     # LightGBM ranker training
│   │
│   ├── embeddings/
│   │   └── user_encoder.py   # Online user tower inference
│   │
│   ├── index/
│   │   ├── pinecone_client.py # Upsert + ANN query
│   │   └── build_index.py    # Batch encode items → Pinecone
│   │
│   ├── evaluation/
│   │   ├── metrics.py        # NDCG@k, HR@k, Recall@k, MRR, coverage
│   │   └── run_eval.py       # Full eval + A/B test + JSON output
│   │
│   └── api/
│       └── main.py           # FastAPI: /recommend, /health, /item/{id}
│
├── app/
│   └── streamlit_app.py      # Professional SaaS demo UI
│
└── tests/
    ├── test_metrics.py       # Metric unit tests (18 tests)
    └── test_two_tower.py     # Model + loss unit tests (9 tests)
```

---

## Resume Bullets

```
• Built Two-Tower NCF recommendation engine on Olist 100K interactions
  (93K users · 33K products); identified 94% single-purchase rate,
  implemented Leave-One-Out evaluation (NCF 2017) on 5,654 held-out users

• ALS baseline: NDCG@10 0.035 · Hit Rate@10 0.042 · Recall@10 0.042 · MRR 0.033
  Two-Tower: pending Pinecone eval (run_eval.py)

• Indexed 27,524 product embeddings (128-dim, SBERT 384-dim content features)
  in Pinecone Serverless ANN; Two-Tower + LightGBM LambdaRank re-ranker pipeline

• Deployed FastAPI serving with Redis user-embedding cache (1h TTL);
  p99 latency <50ms; 3-model comparison Streamlit demo
```

---

*Built by Bhavya Lakkamraju — bhavyalakkamraju09.github.io*
