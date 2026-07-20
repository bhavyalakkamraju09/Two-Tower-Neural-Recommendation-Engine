# Two-Tower Neural Recommendation Engine

Two-Tower NCF · InfoNCE Loss · Pinecone ANN · ALS Baseline · LightGBM Ranker · LOO Evaluation

Portfolio Project 04 — [Bhavya Lakkamraju](https://bhavyalakkamraju09.github.io)

## Results

| Metric | ALS Baseline | Popularity | Two-Tower |
|--------|-------------|------------|-----------|
| NDCG@10 | **0.038** | 0.014 | 0.004 |
| Hit Rate@10 | **0.048** | 0.028 | 0.008 |
| Recall@10 | **0.048** | 0.028 | 0.008 |
| MRR | **0.033** | 0.010 | 0.002 |
| Catalog Coverage | 2.57% | 0.03% | — |

Evaluated on 2,000 sampled users from 5,334 LOO held-out users. A/B test: ALS significantly outperforms popularity baseline (Mann-Whitney U, p=0.0002).

### Why ALS beats Two-Tower here

ALS directly factorizes the interaction matrix it's evaluated on, which gives it a structural advantage on LOO tasks. Olist has a 94% single-purchase rate — most users bought exactly once — so there's very little collaborative signal for the neural model to exploit. Two-Tower shines on denser datasets and cold-start scenarios where content features carry more weight.

### Why LOO and not temporal split

Olist has a 94% single-purchase rate. Temporal split (any cutoff) yields only ~650 usable test interactions, which isn't enough to compute meaningful metrics. Leave-One-Out holds out each multi-purchase user's last item as the test target, giving 5,334 test users with known ground truth — the standard approach for sparse datasets (He et al., NCF 2017).

## Architecture

```
User Tower                              Item Tower
  user_id  ->  Embedding (256d)           product_id  ->  Embedding (256d)
  + [log_purchase_count,      128-d       + category_id  ->  Embedding (64d)
     avg_price,               L2-norm     + SBERT description  (384d)
     avg_review_given]  (3d)              + [log_price, avg_review]  (2d)
  ->  MLP [259 -> 256 -> 128]             ->  MLP [706 -> 512 -> 256 -> 128]
  ->  L2-normalize                        ->  L2-normalize

Training:  InfoNCE contrastive loss (symmetric, in-batch negatives B=1024)
           Learnable temperature: 0.069 -> 0.011  over 100 epochs on Colab T4
           AdamW + CosineAnnealingLR  ·  Early stopping (patience=10)
           Final loss: 2.12 (from 6.18, 65.6% reduction)

Serving:   User embedding -> Pinecone ANN (top-100 cosine) -> LightGBM LambdaRank -> top-10
           Redis user embedding cache (1h TTL)  ·  FastAPI  ·  p99 latency <50ms
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Two-Tower model | PyTorch |
| Text embeddings | SBERT all-MiniLM-L6-v2 (384d) |
| ALS baseline | implicit |
| ANN index | Pinecone Serverless (30,838 vectors, 128-dim) |
| Stage-2 ranker | LightGBM LambdaRank |
| Evaluation | NDCG@10, HR@10, Recall@10, MRR, LOO protocol |
| API | FastAPI + asyncio |
| Caching | Redis (1h TTL) |
| Demo | Streamlit |

## Dataset

Olist Brazilian E-Commerce — free Kaggle download.

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
| Date range | Jan 2017 – Aug 2018 |

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# Add PINECONE_API_KEY to .env (free at pinecone.io)
```

Download dataset:
```bash
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip
```

Build features (runs LOO split + SBERT encoding, ~2 min):
```bash
python -m src.data.build_features
```

Train ALS baseline:
```bash
export MLFLOW_ALLOW_FILE_STORE=true
python -m src.models.train_als
```

Train Two-Tower (use `notebooks/03_two_tower_training.ipynb` on Colab T4 for GPU):
```bash
python -m src.models.train_two_tower
```

Build Pinecone index:
```bash
export PINECONE_API_KEY=your_key
python -m src.index.build_index
```

Train LightGBM ranker:
```bash
python -m src.models.train_ranker
```

Run evaluation:
```bash
python -m src.evaluation.run_eval
```

Launch demo:
```bash
streamlit run app/streamlit_app.py
```

Serve API:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Repository Structure

```
Two-Tower-Neural-Recommendation-Engine/
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── src/
│   ├── data/
│   │   ├── loader.py           # LOO split, sparse matrix builder
│   │   ├── dataset.py          # TwoTowerDataset (PyTorch)
│   │   └── build_features.py   # SBERT encoding, feature tables
│   │
│   ├── models/
│   │   ├── two_tower.py        # UserTower, ItemTower, TwoTowerModel
│   │   ├── infonce_loss.py     # symmetric InfoNCE, weighted variant
│   │   ├── als_model.py        # ALS implicit CF wrapper
│   │   ├── train_two_tower.py  # training loop, early stopping, MPS/CUDA
│   │   ├── train_als.py        # ALS training, LOO evaluation
│   │   └── train_ranker.py     # LightGBM LambdaRank training
│   │
│   ├── embeddings/
│   │   └── user_encoder.py     # online user tower inference
│   │
│   ├── index/
│   │   ├── pinecone_client.py  # upsert, ANN query
│   │   └── build_index.py      # batch encode items -> Pinecone
│   │
│   ├── evaluation/
│   │   ├── metrics.py          # NDCG@k, HR@k, Recall@k, MRR, coverage
│   │   └── run_eval.py         # full eval, A/B test, JSON output
│   │
│   └── api/
│       └── main.py             # FastAPI: /recommend, /health, /item/{id}
│
├── app/
│   └── streamlit_app.py        # 3-model comparison demo
│
├── notebooks/
│   └── 03_two_tower_training.ipynb  # Colab T4 GPU training, Drive-backed
│
└── tests/
    ├── test_metrics.py         # 18 metric unit tests
    └── test_two_tower.py       # 9 model + loss unit tests
```

## Design Notes

**InfoNCE over BPR** — BPR samples one negative per step. InfoNCE uses all B-1 in-batch items as negatives simultaneously (B=1024 on T4), which gives much stronger gradient signal. In-batch negatives also naturally weight popular items more heavily, which is a useful inductive bias for recommendation.

**L2-normalized embeddings** — dot product of unit vectors equals cosine similarity. This keeps the embedding space uniform, makes Pinecone ANN retrieval more reliable, and gives the temperature parameter in InfoNCE a consistent interpretation.

**Two-stage retrieval** — scoring all 31K items per request at inference time is too slow. Pinecone ANN retrieval gets top-100 candidates in milliseconds, then LightGBM does a cheap re-rank. Same pattern used at YouTube, Pinterest, and Airbnb at scale.

**LOO evaluation** — the standard approach for sparse datasets where temporal split doesn't yield enough test interactions. Follows the NCF 2017 paper protocol.

---

Built by [Bhavya Lakkamraju](https://bhavyalakkamraju09.github.io)
