# Two-Tower Neural Recommendation Engine

**Two-Tower NCF · InfoNCE Loss · Pinecone ANN · ALS Baseline · LightGBM Ranker · LOO Evaluation**

> Portfolio Project 04 — Bhavya Lakkamraju · [Portfolio](https://bhavyalakkamraju09.github.io) · [GitHub](https://github.com/bhavyalakkamraju09)

---

## Results

| Metric | Two-Tower | ALS Baseline | Popularity |
|--------|-----------|-------------|------------|
| **NDCG@10** | 0.003 | **0.038** | 0.014 |
| **Hit Rate@10** | 0.004 | **0.048** | 0.028 |
| **Recall@10** | 0.004 | **0.038** | 0.026 |
| **MRR** | 0.002 | **0.033** | 0.010 |
| **Catalog Coverage** | — | 2.57% | 0.03% |
| **A/B vs Popularity** | — | p=0.0002 ✅ | baseline |

*Evaluated on 2,000 sampled users from 5,334 LOO held-out users (NCF 2017 protocol).*

### Why ALS outperforms Two-Tower here

ALS directly factorizes the interaction matrix it is evaluated on — a structural advantage on LOO tasks. Two-Tower trained with in-batch InfoNCE negatives optimizes for embedding space geometry rather than exact item retrieval, and benefits from longer training and richer content signals. On this dataset (94% single-purchase, sparse catalog), CF has a known advantage over neural retrieval. Two-Tower advantage emerges on cold-start users, larger catalogs, and when content features carry strong signal.

### Why LOO and not temporal split

Olist has a 94% single-purchase rate — 88,024 of 93,358 users bought exactly once. Temporal split yields only ~650 usable test interactions regardless of cutoff. Leave-One-Out holds out each multi-purchase user's last item as the test target, giving 5,334 test users with known ground truth — the standard approach for sparse e-commerce datasets (He et al., NCF 2017).

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
           Learnable temperature τ: 0.067 → 0.029  ·  30 epochs
           AdamW + CosineAnnealingLR  ·  Early stopping (patience=5)
           Final loss: 3.24 (from 5.30)

Serving:   User embedding → Pinecone ANN (top-100 cosine) → LightGBM LambdaRank → top-10
           Redis user embedding cache (1h TTL) · FastAPI · p99 latency <50ms

Ranker:    302,809 training rows · 3,000 users · top feature: als_score (gain 273K)
           Features: als_score, category_match, item_popularity, avg_review,
                     log_price, user_avg_price, user_tx_count
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Two-Tower model | PyTorch |
| Text embeddings | SBERT (all-MiniLM-L6-v2, 384d) |
| ALS baseline | implicit (Ben Frederickson) |
| ANN index | Pinecone Serverless (free tier, 30,838 vectors) |
| Stage-2 ranker | LightGBM LambdaRank |
| Evaluation | NDCG@10, HR@10, Recall@10, MRR, LOO protocol |
| API | FastAPI + asyncio |
| Caching | Redis (1h TTL) |
| Demo | Streamlit (auto-loads real metrics) |
| Data transforms | pandas |

---

## Dataset

**Olist Brazilian E-Commerce** — free Kaggle download, no registration required.

| Metric | Value |
|--------|-------|
| Total orders | 99,441 |
| Total interactions | 100,380 |
| Training interactions | 94,451 |
| Users | 93,358 |
| Products | 32,951 |
| Products in training | 30,838 |
| Test users (LOO) | 5,334 |
| Single-purchase users | 88,024 (94%) |
| Multi-purchase users | 5,334 (6%) |
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
# Output: 94,451 train interactions | 5,334 test users
```

### 4. Train ALS baseline
```bash
export MLFLOW_ALLOW_FILE_STORE=true
python -m src.models.train_als
# ~1 min on CPU
# Output: NDCG@10 0.038 · HR@10 0.048 · Recall@10 0.048
```

### 5. Train Two-Tower
```bash
python -m src.models.train_two_tower
# ~12 min on Apple MPS / ~1 hr on Colab T4
# Output: loss 5.30 → 3.24 over 30 epochs
```

### 6. Build Pinecone index
```bash
export PINECONE_API_KEY=your_key
python -m src.index.build_index
# Encodes 30,838 items → upserts 128-dim embeddings to Pinecone
```

### 7. Train LightGBM ranker
```bash
python -m src.models.train_ranker
# ~5 min: 302K training rows from 3,000 users
```

### 8. Run full evaluation
```bash
export PINECONE_API_KEY=your_key
python -m src.evaluation.run_eval
# Evaluates ALS + Popularity + Two-Tower on 2,000 test users
# Saves results to data/processed/eval_results.json
# A/B test: ALS vs Popularity p=0.0002 (statistically significant)
```

### 9. Launch demo
```bash
streamlit run app/streamlit_app.py
# Auto-loads real metrics from eval_results.json
```

### 10. Serve API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# Endpoints: POST /recommend · GET /health · GET /item/{id}
```

---

## Repository Structure

```
Two-Tower-Neural-Recommendation-Engine/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── data/
│   │   ├── loader.py           # LOO split · sparse matrix builder
│   │   ├── dataset.py          # TwoTowerDataset (PyTorch)
│   │   └── build_features.py   # SBERT encoding · feature tables
│   │
│   ├── models/
│   │   ├── two_tower.py        # UserTower · ItemTower · TwoTowerModel
│   │   ├── infonce_loss.py     # Symmetric InfoNCE · weighted variant
│   │   ├── als_model.py        # ALS implicit CF wrapper
│   │   ├── train_two_tower.py  # Training loop · early stopping · MPS/CUDA
│   │   ├── train_als.py        # ALS training · LOO evaluation
│   │   └── train_ranker.py     # LightGBM LambdaRank training
│   │
│   ├── embeddings/
│   │   └── user_encoder.py     # Online user tower inference
│   │
│   ├── index/
│   │   ├── pinecone_client.py  # Upsert · ANN query
│   │   └── build_index.py      # Batch encode items → Pinecone
│   │
│   ├── evaluation/
│   │   ├── metrics.py          # NDCG@k · HR@k · Recall@k · MRR · coverage
│   │   └── run_eval.py         # Full eval · A/B test · JSON output
│   │
│   └── api/
│       └── main.py             # FastAPI: /recommend · /health · /item/{id}
│
├── app/
│   └── streamlit_app.py        # Professional SaaS demo · 3-model comparison
│
└── tests/
    ├── test_metrics.py         # 18 metric unit tests
    └── test_two_tower.py       # 8 model + loss unit tests
```

---

## Key Design Decisions

**InfoNCE over BPR loss** — BPR samples 1 negative per step. InfoNCE uses all B−1 in-batch items as negatives simultaneously (B=512), giving 511× more signal per step. In-batch negatives naturally skew toward popular items (hard negatives), reducing popularity bias.

**L2-normalised embeddings** — dot product of unit vectors equals cosine similarity. Makes embedding space uniform (no magnitude variation), critical for Pinecone ANN quality, and makes temperature τ interpretable.

**Two-stage retrieval** — end-to-end ranking over 31K items per request is O(N·D). Pinecone ANN retrieval in O(D log N) gets top-100 candidates in <10ms, then LightGBM reranks cheaply. Industry standard at YouTube, Pinterest, Airbnb.

**LOO evaluation** — temporal split is infeasible on Olist (94% single-purchase rate). LOO on 5,334 multi-purchase users follows the NCF 2017 evaluation standard and gives meaningful, comparable metrics.

---

*Built by Bhavya Lakkamraju — [bhavyalakkamraju09.github.io](https://bhavyalakkamraju09.github.io)*
