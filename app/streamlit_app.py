"""
app/streamlit_app.py
Professional SaaS-style Recommendation Engine demo.
Loads real eval metrics from data/processed/eval_results.json if available.

Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="RecsysAI · Recommendation Engine",
    page_icon="🎯", layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[data-testid="stAppViewContainer"]{background:#f8f9fc!important;color:#111827!important;font-family:'Inter',sans-serif!important;}
[data-testid="stSidebar"]{display:none!important;}
[data-testid="stAppViewContainer"]>section{padding:0!important;}
.block-container{padding:0!important;max-width:100%!important;}
div[data-testid="stVerticalBlock"]>div{gap:0!important;}
[data-testid="stMarkdownContainer"] p{color:#111827;}
.nav{background:white;border-bottom:1px solid #e5e7eb;padding:0 48px;display:flex;align-items:center;height:60px;}
.nav-logo{font-size:18px;font-weight:800;color:#111827;letter-spacing:-0.03em;}
.nav-logo span{color:#6366f1;}
.nav-links{display:flex;gap:28px;margin-left:40px;font-size:13px;font-weight:500;color:#6b7280;}
.nav-right{margin-left:auto;}
.nav-badge{background:#f0f0ff;color:#6366f1;border:1px solid #e0e0ff;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;}
.hero{background:white;padding:56px 48px 48px;border-bottom:1px solid #e5e7eb;}
.hero-eyebrow{display:inline-flex;align-items:center;gap:6px;background:#f0f0ff;color:#6366f1;border:1px solid #e0e0ff;padding:5px 14px;border-radius:20px;font-size:12px;font-weight:600;margin-bottom:20px;font-family:'JetBrains Mono',monospace;}
.hero-title{font-size:44px;font-weight:800;color:#111827;letter-spacing:-0.03em;line-height:1.1;margin-bottom:14px;max-width:680px;}
.hero-title em{font-style:normal;color:#6366f1;}
.hero-desc{font-size:15px;color:#6b7280;line-height:1.7;max-width:560px;margin-bottom:28px;}
.hero-stats{display:flex;gap:36px;flex-wrap:wrap;}
.hero-stat-val{font-size:26px;font-weight:800;color:#111827;letter-spacing:-0.02em;}
.hero-stat-val span{color:#6366f1;}
.hero-stat-label{font-size:12px;color:#9ca3af;margin-top:2px;}
.strip{background:#6366f1;padding:18px 48px;display:flex;gap:0;}
.strip-item{flex:1;padding:0 24px;border-right:1px solid rgba(255,255,255,0.15);display:flex;flex-direction:column;gap:2px;}
.strip-item:first-child{padding-left:0;}
.strip-item:last-child{border-right:none;}
.strip-label{font-size:10px;font-weight:600;color:rgba(255,255,255,0.6);letter-spacing:0.1em;text-transform:uppercase;}
.strip-value{font-size:20px;font-weight:800;color:white;letter-spacing:-0.02em;}
.strip-sub{font-size:10px;color:rgba(255,255,255,0.55);}
.main{padding:36px 48px;}
.section-hdr{margin-bottom:14px;}
.section-hdr-title{font-size:17px;font-weight:700;color:#111827;}
.section-hdr-sub{font-size:13px;color:#9ca3af;margin-top:2px;}
[data-testid="stSelectbox"]>div>div{background:white!important;border:1px solid #d1d5db!important;border-radius:8px!important;color:#111827!important;font-size:13px!important;}
[data-testid="stTextInput"]>div>div>input{background:white!important;border:1px solid #d1d5db!important;border-radius:8px!important;color:#111827!important;font-size:13px!important;}
[data-testid="stButton"]>button{background:#6366f1!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;font-size:13px!important;padding:10px 20px!important;width:100%!important;}
[data-testid="stButton"]>button:hover{background:#4f46e5!important;box-shadow:0 4px 12px rgba(99,102,241,0.35)!important;}
[data-testid="stToggle"] label{color:#6b7280!important;font-size:13px!important;}
[data-testid="stHorizontalBlock"]{gap:16px!important;}
.user-tag{display:inline-flex;align-items:center;gap:8px;background:#f0fdf4;border:1px solid #bbf7d0;padding:8px 16px;border-radius:8px;margin-bottom:20px;font-size:12px;font-family:'JetBrains Mono',monospace;color:#15803d;}
.model-panel{background:white;border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.05);}
.model-panel-hdr{padding:14px 18px;border-bottom:1px solid #f3f4f6;display:flex;align-items:center;gap:10px;}
.model-icon{width:30px;height:30px;border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:14px;}
.icon-primary{background:#eef2ff;}
.icon-green{background:#f0fdf4;}
.icon-gray{background:#f9fafb;}
.model-panel-title{font-size:13px;font-weight:700;color:#111827;}
.model-panel-sub{font-size:11px;color:#9ca3af;}
.model-panel-badge{margin-left:auto;font-size:10px;font-weight:700;padding:3px 8px;border-radius:20px;text-transform:uppercase;letter-spacing:0.06em;font-family:'JetBrains Mono',monospace;}
.badge-indigo{background:#eef2ff;color:#6366f1;}
.badge-green{background:#f0fdf4;color:#16a34a;}
.badge-gray{background:#f9fafb;color:#9ca3af;border:1px solid #e5e7eb;}
.model-panel-body{padding:14px 18px;}
.model-desc{font-size:11px;color:#9ca3af;line-height:1.6;margin-bottom:12px;font-family:'JetBrains Mono',monospace;}
.rec-item{background:#f8f9fc;border:1px solid #f3f4f6;border-radius:8px;padding:10px 13px;margin-bottom:6px;}
.rec-item:hover{background:white;border-color:#e0e0ff;}
.rec-rank{display:inline-block;width:19px;height:19px;background:#6366f1;color:white;border-radius:5px;font-size:9px;font-weight:700;text-align:center;line-height:19px;margin-right:7px;flex-shrink:0;}
.rec-rank-top{background:#f59e0b!important;}
.rec-id{font-size:11px;font-family:'JetBrains Mono',monospace;color:#6366f1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.rec-tags{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;}
.rec-tag{font-size:10px;padding:2px 6px;border-radius:4px;font-weight:500;}
.tag-cat{background:#eef2ff;color:#6366f1;}
.tag-price{background:#f0fdf4;color:#16a34a;}
.tag-review{background:#fffbeb;color:#d97706;}
.tag-pop{background:#f9fafb;color:#6b7280;border:1px solid #f3f4f6;}
.state-empty{background:#f8f9fc;border:1px dashed #d1d5db;border-radius:8px;padding:20px;text-align:center;color:#9ca3af;font-size:13px;}
.state-warn{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:12px 14px;color:#92400e;font-size:12px;margin-bottom:8px;}
.state-err{background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:12px 14px;color:#991b1b;font-size:12px;}
.latency-pill{display:inline-block;background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0;border-radius:20px;font-size:10px;font-weight:600;padding:2px 8px;font-family:'JetBrains Mono',monospace;margin-bottom:10px;}
.divider{height:1px;background:#f3f4f6;margin:32px 0;}
.arch-section{background:#111827;border-radius:12px;padding:22px 26px;margin-top:20px;}
.arch-pre{font-family:'JetBrains Mono',monospace;font-size:11px;color:#d1d5db;line-height:1.9;margin:0;}
.footer{background:white;border-top:1px solid #e5e7eb;padding:20px 48px;margin-top:40px;display:flex;justify-content:space-between;align-items:center;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PROCESSED   = Path("data/processed")
CHECKPOINTS = Path("checkpoints")
API_URL     = os.getenv("API_URL", "http://localhost:8000")


# ── Load real eval metrics ────────────────────────────────────────────────────
@st.cache_data
def load_eval_metrics() -> dict:
    p = PROCESSED / "eval_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_item_features():
    p = PROCESSED / "item_features.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_resource
def load_als_model():
    p = CHECKPOINTS / "als_model.pkl"
    if not p.exists(): return None
    from src.models.als_model import ALSRecommender
    return ALSRecommender.load(p)


@st.cache_data
def get_sample_users(n=50):
    p = Path("data/splits/test.parquet")
    if not p.exists(): return []
    return pd.read_parquet(p)["user_id"].unique()[:n].tolist()


@st.cache_resource
def get_popularity_list():
    f = load_item_features()
    if f is None: return []
    return f["purchase_count"].sort_values(ascending=False).index.tolist()


def api_recommend(user_id, model, n=10):
    try:
        import httpx
        r = httpx.post(f"{API_URL}/recommend",
                       json={"user_id": user_id, "model": model, "n": n}, timeout=8.0)
        return r.json()
    except Exception as e:
        return {"error": str(e), "recommendations": []}


def rec_card(item_id, rank, item_features):
    cat, price, review, pop = "unknown", "—", "—", "—"
    if item_features is not None and item_id in item_features.index:
        row    = item_features.loc[item_id]
        cat    = str(row.get("product_category_name_english", "unknown")).replace("_", " ")
        price  = "R$ " + str(int(row.get("avg_price", 0)))
        review = str(round(float(row.get("avg_review_score", 0)), 1)) + " \u2605"
        pop    = str(int(row.get("purchase_count", 0))) + " sales"
    sid      = item_id[:22] + "\u2026" if len(item_id) > 22 else item_id
    rank_cls = "rec-rank-top" if rank <= 3 else ""
    html = (
        '<div class="rec-item">'
          '<div style="display:flex;align-items:center;">'
            '<span class="rec-rank ' + rank_cls + '">' + str(rank) + '</span>'
            '<span class="rec-id">' + sid + '</span>'
          '</div>'
          '<div class="rec-tags">'
            '<span class="rec-tag tag-cat">' + cat + '</span>'
            '<span class="rec-tag tag-price">' + price + '</span>'
            '<span class="rec-tag tag-review">' + review + '</span>'
            '<span class="rec-tag tag-pop">' + pop + '</span>'
          '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
eval_metrics  = load_eval_metrics()
item_features = load_item_features()
als_model     = load_als_model()
pop_list      = get_popularity_list()
sample_users  = get_sample_users(50)

# Extract real metrics or fall back to "—"
als_m     = eval_metrics.get("als",        {})
pop_m     = eval_metrics.get("popularity", {})
tt_m      = eval_metrics.get("two_tower",  {})
ab_result = eval_metrics.get("ab_test_als_vs_popularity", {})

def _fmt(val, decimals=3):
    return f"{val:.{decimals}f}" if isinstance(val, float) else "—"

ndcg_als  = _fmt(als_m.get("ndcg@10",     als_m.get("ndcg_at_10")))
hr_als    = _fmt(als_m.get("hit_rate@10", als_m.get("hit_rate_at_10")))
recall_als = _fmt(als_m.get("recall@10",  als_m.get("recall_at_10")))
ndcg_tt   = _fmt(tt_m.get("ndcg@10",      tt_m.get("ndcg_at_10"))) if tt_m else "—"
cov_als   = _fmt(als_m.get("catalog_coverage", None), 3) if "catalog_coverage" in als_m else "—"
n_users   = str(int(als_m.get("n_users_evaluated", 0))) if als_m else "—"

# ── NAV ───────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="nav">'
    '<div class="nav-logo">RecSys<span>AI</span></div>'
    '<div class="nav-links"><span>Models</span><span>Metrics</span><span>Architecture</span></div>'
    '<div class="nav-right"><span class="nav-badge">Portfolio Project 04</span></div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero">'
    '<div class="hero-eyebrow">\u26a1 Two-Tower Neural Collaborative Filtering</div>'
    '<div class="hero-title">Personalized recommendations<br>that <em>actually</em> work</div>'
    '<div class="hero-desc">Neural retrieval with InfoNCE contrastive loss, Pinecone ANN, and '
    'LightGBM LambdaRank re-ranking \u2014 trained on 100K real Brazilian e-commerce orders.</div>'
    '<div class="hero-stats">'
    '<div><div class="hero-stat-val"><span>93K</span></div><div class="hero-stat-label">users</div></div>'
    '<div><div class="hero-stat-val"><span>33K</span></div><div class="hero-stat-label">products</div></div>'
    '<div><div class="hero-stat-val"><span>100K</span></div><div class="hero-stat-label">interactions</div></div>'
    '<div><div class="hero-stat-val">&lt;<span>50</span>ms</div><div class="hero-stat-label">p99 latency</div></div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── STRIP ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="strip">'
    '<div class="strip-item"><div class="strip-label">NDCG@10 · ALS</div>'
    f'<div class="strip-value">{ndcg_als}</div><div class="strip-sub">Collaborative baseline</div></div>'
    '<div class="strip-item"><div class="strip-label">Hit Rate@10 · ALS</div>'
    f'<div class="strip-value">{hr_als}</div><div class="strip-sub">LOO eval · 5,654 users</div></div>'
    '<div class="strip-item"><div class="strip-label">Recall@10 · ALS</div>'
    f'<div class="strip-value">{recall_als}</div><div class="strip-sub">Leave-One-Out</div></div>'
    '<div class="strip-item"><div class="strip-label">NDCG@10 · Two-Tower</div>'
    f'<div class="strip-value">{ndcg_tt}</div><div class="strip-sub">After Pinecone eval</div></div>'
    '<div class="strip-item"><div class="strip-label">Test Users</div>'
    f'<div class="strip-value">5,654</div><div class="strip-sub">LOO held-out set</div></div>'
    '<div class="strip-item"><div class="strip-label">Items in Pinecone</div>'
    '<div class="strip-value">27,524</div><div class="strip-sub">128-dim cosine ANN</div></div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main">', unsafe_allow_html=True)

st.markdown(
    '<div class="section-hdr">'
    '<div class="section-hdr-title">Try it live</div>'
    '<div class="section-hdr-sub">Pick a user from the test set and compare all three models</div>'
    '</div>',
    unsafe_allow_html=True,
)

col_sel, col_manual, col_n, col_api, col_btn = st.columns([3, 2, 1, 1, 1])
with col_sel:
    selected = st.selectbox("User", sample_users, label_visibility="collapsed") if sample_users else ""
with col_manual:
    manual = st.text_input("Manual", placeholder="Or paste a customer ID\u2026", label_visibility="collapsed")
with col_n:
    n_recs = st.selectbox("N", [5, 10, 15, 20], index=1, label_visibility="collapsed")
with col_api:
    use_api = st.toggle("API mode", value=False)
with col_btn:
    run = st.button("\U0001f3af  Get recommendations")

user_id = manual.strip() if manual.strip() else (selected if selected else "")

if user_id:
    st.markdown(
        '<div class="user-tag">\u2713 &nbsp;<strong>Active user</strong>&nbsp;\u00b7&nbsp;' + user_id + '</div>',
        unsafe_allow_html=True,
    )

# ── Model panels ──────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3, gap="medium")
EMPTY = '<div class="state-empty">Select a user and click<br><strong>Get recommendations</strong></div>'

with c1:
    st.markdown(
        '<div class="model-panel">'
        '<div class="model-panel-hdr">'
        '<div class="model-icon icon-primary">\U0001f9e0</div>'
        '<div><div class="model-panel-title">Two-Tower NCF</div>'
        '<div class="model-panel-sub">Neural retrieval + re-ranking</div></div>'
        '<span class="model-panel-badge badge-indigo">Primary</span>'
        '</div><div class="model-panel-body">'
        '<div class="model-desc">128-d L2-norm \u00b7 InfoNCE loss<br>'
        'Pinecone ANN (top-100) \u00b7 LightGBM ranker<br>'
        'SBERT content features (384d)</div>',
        unsafe_allow_html=True,
    )
    if run and user_id:
        t0 = time.monotonic()
        if use_api:
            res  = api_recommend(user_id, "two_tower", n_recs)
            recs = res.get("recommendations", [])
            err  = res.get("error")
        else:
            recs, err = [], None
            st.markdown('<div class="state-warn">\u26a0 Enable API mode + start FastAPI to use Two-Tower + Pinecone.</div>', unsafe_allow_html=True)
        ms = (time.monotonic() - t0) * 1000
        if err:
            st.markdown('<div class="state-err">\u26a0 FastAPI not reachable \u2014 run: <code>uvicorn src.api.main:app --port 8000</code></div>', unsafe_allow_html=True)
        elif recs:
            st.markdown('<span class="latency-pill">\u26a1 ' + str(int(ms)) + 'ms \u00b7 ' + str(len(recs)) + ' results</span>', unsafe_allow_html=True)
            for i, pid in enumerate(recs, 1): rec_card(pid, i, item_features)
    else:
        st.markdown(EMPTY, unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

with c2:
    st.markdown(
        '<div class="model-panel">'
        '<div class="model-panel-hdr">'
        '<div class="model-icon icon-green">\U0001f4ca</div>'
        '<div><div class="model-panel-title">ALS Baseline</div>'
        '<div class="model-panel-sub">Collaborative filtering</div></div>'
        '<span class="model-panel-badge badge-green">Baseline</span>'
        '</div><div class="model-panel-body">'
        '<div class="model-desc">Implicit feedback \u00b7 128 factors \u00b7 \u03b1=40<br>'
        '50 iterations \u00b7 93K\u00d731K matrix<br>'
        f'NDCG@10 {ndcg_als} \u00b7 HR@10 {hr_als}</div>',
        unsafe_allow_html=True,
    )
    if run and user_id:
        t0 = time.monotonic()
        recs = als_model.recommend(user_id, n=n_recs) if als_model else []
        if use_api:
            res  = api_recommend(user_id, "als", n_recs)
            recs = res.get("recommendations", recs)
        ms = (time.monotonic() - t0) * 1000
        if not recs:
            st.markdown('<div class="state-warn">\u26a0 Cold-start user \u2014 ALS has no embedding for this user.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="latency-pill">\u26a1 ' + str(int(ms)) + 'ms \u00b7 ' + str(len(recs)) + ' results</span>', unsafe_allow_html=True)
            for i, pid in enumerate(recs, 1): rec_card(pid, i, item_features)
    else:
        st.markdown(EMPTY, unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

with c3:
    st.markdown(
        '<div class="model-panel">'
        '<div class="model-panel-hdr">'
        '<div class="model-icon icon-gray">\U0001f4c8</div>'
        '<div><div class="model-panel-title">Popularity Baseline</div>'
        '<div class="model-panel-sub">Non-personalised lower bound</div></div>'
        '<span class="model-panel-badge badge-gray">Lower bound</span>'
        '</div><div class="model-panel-body">'
        '<div class="model-desc">Top-N most purchased globally<br>'
        'Zero personalisation \u00b7 &lt;1ms<br>'
        'Benchmark for A/B significance</div>',
        unsafe_allow_html=True,
    )
    if run and user_id:
        t0   = time.monotonic()
        recs = pop_list[:n_recs]
        ms   = (time.monotonic() - t0) * 1000
        st.markdown('<span class="latency-pill">\u26a1 ' + str(round(ms, 2)) + 'ms \u00b7 ' + str(len(recs)) + ' results</span>', unsafe_allow_html=True)
        for i, pid in enumerate(recs, 1): rec_card(pid, i, item_features)
    else:
        st.markdown(EMPTY, unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

# ── CHARTS ────────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-hdr">'
    '<div class="section-hdr-title">Evaluation &amp; Bias Audit</div>'
    '<div class="section-hdr-sub">Real metrics from Leave-One-Out evaluation on 5,654 held-out users</div>'
    '</div>',
    unsafe_allow_html=True,
)

if item_features is not None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(16, 3.4))
    fig.patch.set_facecolor("white")

    # NDCG comparison
    ax = axes[0]
    models_  = ["ALS\nBaseline", "Popularity\nBaseline"]
    vals_    = [
        als_m.get("ndcg@10", als_m.get("ndcg_at_10", 0)) if als_m else 0,
        pop_m.get("ndcg@10", pop_m.get("ndcg_at_10", 0)) if pop_m else 0,
    ]
    if tt_m:
        models_.insert(0, "Two-Tower\nNCF")
        vals_.insert(0, tt_m.get("ndcg@10", tt_m.get("ndcg_at_10", 0)))
    colors_ = ["#6366f1", "#10b981", "#9ca3af"][:len(models_)]
    bars = ax.bar(models_, vals_, color=colors_, width=0.5, edgecolor="none", zorder=3)
    ax.set_facecolor("white"); ax.set_ylim(0, max(vals_) * 1.3 if vals_ else 0.1)
    ax.yaxis.grid(True, color="#f3f4f6", linewidth=1, zorder=0); ax.set_axisbelow(True)
    for s in ["top", "right", "left", "bottom"]: ax.spines[s].set_visible(False)
    ax.tick_params(colors="#9ca3af", labelsize=9)
    ax.set_title("NDCG@10 (real LOO eval)", fontsize=11, fontweight="700", color="#111827", pad=10)
    for bar, val in zip(bars, vals_):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="700", color="#111827")

    # Long tail
    ax2 = axes[1]
    top60 = item_features["purchase_count"].sort_values(ascending=False).head(60).values
    bar_c = ["#6366f1" if i < 5 else "#e0e7ff" for i in range(len(top60))]
    ax2.bar(range(len(top60)), top60, color=bar_c, width=0.85, edgecolor="none", zorder=3)
    ax2.set_facecolor("white"); ax2.yaxis.grid(True, color="#f3f4f6", linewidth=1, zorder=0)
    ax2.set_axisbelow(True)
    for s in ["top", "right", "left", "bottom"]: ax2.spines[s].set_visible(False)
    ax2.tick_params(colors="#9ca3af", labelsize=9)
    ax2.set_xlabel("Item rank", color="#9ca3af", fontsize=9)
    ax2.set_title("Long-tail Distribution", fontsize=11, fontweight="700", color="#111827", pad=10)
    ax2.axvline(4.5, color="#6366f1", linewidth=1.5, linestyle="--", alpha=0.6)
    ax2.text(6, top60[0] * 0.88, "Top 5", color="#6366f1", fontsize=8, fontweight="600")

    # Hit rate comparison
    ax3 = axes[2]
    hr_vals = [
        als_m.get("hit_rate@10", als_m.get("hit_rate_at_10", 0)) if als_m else 0,
        pop_m.get("hit_rate@10", pop_m.get("hit_rate_at_10", 0)) if pop_m else 0,
    ]
    hr_models = ["ALS", "Popularity"]
    hr_colors = ["#10b981", "#9ca3af"]
    if tt_m:
        hr_vals.insert(0, tt_m.get("hit_rate@10", tt_m.get("hit_rate_at_10", 0)))
        hr_models.insert(0, "Two-Tower")
        hr_colors.insert(0, "#6366f1")
    bars3 = ax3.bar(hr_models, hr_vals, color=hr_colors[:len(hr_models)], width=0.5, edgecolor="none", zorder=3)
    ax3.set_facecolor("white"); ax3.set_ylim(0, max(hr_vals) * 1.3 if hr_vals else 0.1)
    ax3.yaxis.grid(True, color="#f3f4f6", linewidth=1, zorder=0); ax3.set_axisbelow(True)
    for s in ["top", "right", "left", "bottom"]: ax3.spines[s].set_visible(False)
    ax3.tick_params(colors="#9ca3af", labelsize=9)
    ax3.set_title("Hit Rate@10 (real LOO eval)", fontsize=11, fontweight="700", color="#111827", pad=10)
    for bar, val in zip(bars3, hr_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="700", color="#111827")

    plt.tight_layout(pad=2.0)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    if eval_metrics:
        st.markdown(
            '<div style="font-size:12px;color:#9ca3af;margin-top:8px;line-height:1.7;">'
            '<strong style="color:#111827;">LOO Evaluation Protocol:</strong> '
            '5,654 multi-purchase users \u2014 last item held out as ground truth. '
            'Olist has 94% single-purchase users, making temporal split infeasible. '
            'LOO follows the NCF 2017 evaluation standard for sparse e-commerce datasets. '
            'Metrics computed on 1,000-user random sample from the 5,654 test users. '
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:12px;color:#f59e0b;margin-top:8px;">'
            '\u26a0 Run <code>python -m src.evaluation.run_eval</code> to populate real metrics.'
            '</div>',
            unsafe_allow_html=True,
        )

# ── ARCHITECTURE ──────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-hdr">'
    '<div class="section-hdr-title">System Architecture</div>'
    '<div class="section-hdr-sub">Two-stage retrieval + neural re-ranking pipeline</div>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="arch-section"><pre class="arch-pre">'
    'User Tower                              Item Tower\n'
    '  user_id  \u2192  Embedding (256d)             product_id \u2192 Embedding (256d)\n'
    '  + [log_purchase_count,      128-d    + category_id  \u2192 Embedding (64d)\n'
    '     avg_price,               L2-norm  + SBERT text description  (384d)\n'
    '     avg_review_given]  (3d)           + [log_price, avg_review_score]  (2d)\n'
    '  \u2192  MLP [259 \u2192 256 \u2192 128]               \u2192  MLP [706 \u2192 512 \u2192 256 \u2192 128]\n'
    '  \u2192  L2-normalize                       \u2192  L2-normalize\n\n'
    'Training:  InfoNCE contrastive loss (symmetric, in-batch negatives B=512)\n'
    '           Learnable temperature \u03c4: 0.07 \u2192 0.031  \u00b7  30 epochs  \u00b7  AdamW + CosineAnnealingLR\n\n'
    'Serving:   User embedding \u2192 Pinecone ANN (top-100 cosine) \u2192 LightGBM LambdaRank \u2192 top-10\n'
    '           Redis user embedding cache (1h TTL)  \u00b7  FastAPI  \u00b7  p99 latency &lt;50ms\n\n'
    'Evaluation: Leave-One-Out on 5,654 multi-purchase users (NCF 2017 protocol)\n'
    '            Olist: 94% single-purchase rate \u2014 temporal split yields only 650 test rows'
    '</pre></div>',
    unsafe_allow_html=True,
)

st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    '<div style="font-size:14px;font-weight:800;color:#111827;">RecSys<span style="color:#6366f1">AI</span></div>'
    '<div style="font-size:12px;color:#9ca3af;">Built by '
    '<strong style="color:#111827;">Bhavya Lakkamraju</strong> \u00b7 '
    '<a href="https://bhavyalakkamraju09.github.io" style="color:#6366f1;text-decoration:none;">Portfolio</a> \u00b7 '
    '<a href="https://github.com/bhavyalakkamraju09" style="color:#6366f1;text-decoration:none;">GitHub</a> \u00b7 '
    'Olist dataset \u00b7 PyTorch \u00b7 Pinecone \u00b7 MLflow</div>'
    '</div>',
    unsafe_allow_html=True,
)
