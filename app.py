# -*- coding: utf-8 -*-
"""
Media Intelligence Command Center – Sri Lanka News (Python-only, Streamlit + Plotly)

This app builds a complete media-intelligence layer on top of the Sri Lankan news dataset:
- Story clustering across Sinhala/Tamil/English (multilingual embeddings, with TF-IDF fallback)
- First-to-Break Score (FBS), Time-to-Pickup (TTP), Break/ Pickup Rates
- Story diffusion graphs + Source Influence Rank (SIR via PageRank)
- Coverage Health: Share of Voice (SoV), Topic Breadth, Avg TTP
- Outlet Similarity & Beat Overlap (UMAP of topic/angle vectors)
- Framing Divergence across si/ta/en (lexical divergence; optional translations from ext_articles)
- "Who covers next?" predictor (pairwise conditional probabilities with momentum features)

DATA SOURCES (structure and field assumptions):
- Tidy dataset (Parquet/Arrow) on Hugging Face: `nuuuwan/lk-news-docs`
  Fields include: doc_id, date_str, lang (si/ta/en), newspaper_id, description (title/summary), url_metadata, time_ut
  See dataset viewer for schema and time span (2021-09-12 → 2025-10-01).  # [HF Dataset]
  Source: https://huggingface.co/datasets/nuuuwan/lk-news-docs  (accessed 2025-10-02)

- Data-only GitHub repo `news_lk3_data` (articles + ext_articles with translations & NER).
  README documents counts (~62k–65k), sources, and that ~95%+ have extensions.
  Use `ext_articles/` for optional per-article translations/NER.                     # [GH Data Repo]
  Sources:
  - https://github.com/nuuuwan/news_lk3_data
  - Example ext JSON structure: https://github.com/nuuuwan/news_lk3_data/blob/main/ext_articles/689cd094.ext.json
  - Repo README with dataset stats: https://github.com/nuuuwan/news_lk3_data/blob/main/README.md

NOTES:
- For cross-lingual similarity we default to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
  If it cannot be downloaded, the app falls back to TF-IDF (works but less robust).
- Clustering uses HDBSCAN if available; else KMeans on embeddings as a fallback.
- To keep performance reasonable in a browser, the app defaults to a selectable date window (e.g., last 60 days).
- "Who covers next?" is an interpretable baseline (pairwise conditional probabilities + momentum), not a heavy ML model.

CREDITS:
- Dataset and structure per the sources above. Always attribute original publishers when referencing or exporting stories.
"""

import os
import io
import json
import math
import time
import random
import string
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from datasets import load_dataset

# Optional heavy libs (guarded imports)
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:
    HAS_ST = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ----------------------------
# CONFIG
# ----------------------------

DEFAULT_DAYS = 60               # default date window
MAX_DOCS = 25000                # safety cap for viewport (speed)
EMB_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
N_TOPICS = 40                   # for topic modeling (NMF)
CLUSTER_WINDOW_HOURS = 72       # story window assumption (rolling)
PRED_PICKUP_HOURS = 24          # "who covers next" prediction horizon

st.set_page_config(page_title="Sri Lanka Media Intelligence", layout="wide")

# ----------------------------
# UTILITIES
# ----------------------------

@st.cache_data(show_spinner=False)
def load_docs_from_hf(date_from: datetime, date_to: datetime) -> pd.DataFrame:
    """
    Load docs from Hugging Face dataset `nuuuwan/lk-news-docs` and filter by date range.

    HF Dataset reference:
    - https://huggingface.co/datasets/nuuuwan/lk-news-docs  [Schema/fields/date range]
    """
    ds = load_dataset("nuuuwan/lk-news-docs", split="train")
    df = ds.to_pandas()
    # Expected fields: doc_id, date_str, lang, newspaper_id, description, url_metadata, time_ut
    df["ts"] = pd.to_datetime(df["date_str"], errors="coerce")
    df = df.dropna(subset=["ts"])
    mask = (df["ts"] >= pd.Timestamp(date_from)) & (df["ts"] <= pd.Timestamp(date_to))
    df = df.loc[mask].copy()
    if len(df) > MAX_DOCS:
        df = df.sort_values("ts", ascending=False).head(MAX_DOCS).copy()
    # Normalize some fields
    df["lang"] = df["lang"].fillna("unk")
    df["newspaper_id"] = df["newspaper_id"].fillna("unknown")
    # Make a short title from description
    df["title"] = df["description"].astype(str).str.strip()
    return df


def safe_text(x, maxlen=280):
    if not isinstance(x, str):
        return ""
    x = x.replace("\n", " ").strip()
    return x[:maxlen]


def extract_domain(url_metadata: str) -> str:
    """Extract 'domain' from the url_metadata JSON-like string if present."""
    if not isinstance(url_metadata, str):
        return ""
    try:
        # Try loading as JSON first
        meta = json.loads(url_metadata)
        if isinstance(meta, dict) and "domain" in meta:
            return str(meta["domain"])
    except Exception:
        pass
    # Fallback regex
    import re
    m = re.search(r'"domain"\s*:\s*"([^"]+)"', url_metadata)
    return m.group(1) if m else ""


@st.cache_resource(show_spinner=False)
def load_embedder(model_name=EMB_MODEL):
    """Try to load multilingual SentenceTransformer; fallback handled by caller."""
    if not HAS_ST:
        return None
    model = SentenceTransformer(model_name)
    return model


def build_embeddings(texts, use_st=True, st_model=None):
    """
    Build embeddings for a list of texts.
    - If SentenceTransformer available: use it (cross-lingual).
    - Else: TF-IDF vectors (sparse).
    """
    if use_st and HAS_ST and st_model is not None:
        emb = st_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return emb, "st"
    # TF-IDF fallback
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2)
    X = vec.fit_transform(texts)
    return X, "tfidf"


def cluster_stories(df: pd.DataFrame, text_col="title"):
    """
    Cluster documents into stories. Use HDBSCAN on embeddings if available; else KMeans.
    We intentionally do NOT hard-slice by time but compute clusters and later constrain
    diffusion windows by CLUSTER_WINDOW_HOURS.
    """
    st_model = load_embedder()
    emb, mode = build_embeddings(df[text_col].fillna("").tolist(), use_st=True, st_model=st_model)

    # Clustering
    if HAS_HDBSCAN and mode == "st":
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, metric="euclidean")
        labels = clusterer.fit_predict(emb)
    else:
        # Fallback: KMeans with heuristic k
        k = max(10, min(400, df.shape[0] // 200))  # heuristic
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        # For TF-IDF sparse, KMeans still OK
        labels = km.fit_predict(emb)

    df["cluster_id"] = labels
    # drop noise cluster (-1) if present
    if (df["cluster_id"] == -1).any():
        # Keep noise as separate small "singleton" clusters or drop?
        # We'll keep as -1 (can be filtered in UI).
        pass
    return df, mode


def compute_story_metrics(df: pd.DataFrame):
    """
    Compute First-to-Break, Time-to-Pickup, chain sizes, etc., within a diffusion window.
    """
    # Extract domain for reference
    if "domain" not in df.columns:
        df["domain"] = df["url_metadata"].apply(extract_domain)

    # Group by cluster
    records = []
    for cid, g in df.groupby("cluster_id"):
        if cid == -1:
            continue
        g = g.sort_values("ts")
        first_ts = g["ts"].min()
        first_row = g.iloc[0]
        origin_outlet = first_row["newspaper_id"]

        # Compute TTP for others within window
        g["ttp_mins"] = (g["ts"] - first_ts).dt.total_seconds() / 60.0
        within = g[g["ts"] <= first_ts + pd.Timedelta(hours=CLUSTER_WINDOW_HOURS)]
        # FBS: 1 for first publisher(s)
        within["is_first"] = (within["ts"] == first_ts).astype(int)

        n_outlets = within["newspaper_id"].nunique()
        median_ttp = within.loc[within["is_first"] == 0, "ttp_mins"].median()
        records.append({
            "cluster_id": cid,
            "origin_outlet": origin_outlet,
            "origin_time": first_ts,
            "n_articles_within_window": len(within),
            "n_outlets_within_window": n_outlets,
            "median_ttp_mins": float(median_ttp) if pd.notnull(median_ttp) else np.nan,
            "title_sample": safe_text(first_row["title"], 160)
        })
    story_df = pd.DataFrame.from_records(records).sort_values("origin_time", ascending=False)

    # Per-outlet PR/BR/TTP
    # Break Rate (BR): share of clusters where outlet is origin
    origin_counts = story_df["origin_outlet"].value_counts().rename("breaks").to_frame()
    # Pickup Rate (PR): out of all clusters originated by others, how often outlet covered within window
    pickups = []
    for cid, g in df.groupby("cluster_id"):
        if cid == -1: 
            continue
        g = g.sort_values("ts")
        first_ts = g["ts"].min()
        origin = g.iloc[0]["newspaper_id"]
        within = g[g["ts"] <= first_ts + pd.Timedelta(hours=CLUSTER_WINDOW_HOURS)]
        for outlet, gg in within.groupby("newspaper_id"):
            pickups.append({"cluster_id": cid, "origin_outlet": origin, "outlet": outlet,
                            "is_origin": int(outlet == origin),
                            "ttp_mins": float((gg["ts"].min() - first_ts).total_seconds()/60.0)})

    pr_df = pd.DataFrame(pickups)
    # Exclude self-origin for PR
    pr_df_pick = pr_df[pr_df["is_origin"] == 0]
    outlet_pr = pr_df_pick.groupby("outlet").agg(
        pickups=("cluster_id","nunique"),
        avg_ttp=("ttp_mins","mean")
    )
    # Denominator for PR: clusters originated by others (approx = total clusters - breaks by outlet)
    total_clusters = story_df["cluster_id"].nunique()
    outlet_pr["pr_denom"] = total_clusters - origin_counts.reindex(outlet_pr.index).fillna(0)["breaks"]
    outlet_pr["pickup_rate"] = (outlet_pr["pickups"] / outlet_pr["pr_denom"]).replace([np.inf, -np.inf], np.nan)

    # Assemble outlet KPIs
    outlet_kpi = origin_counts.join(outlet_pr, how="outer").fillna(0)
    outlet_kpi["break_rate"] = outlet_kpi["breaks"] / total_clusters
    return story_df, pr_df, outlet_kpi


def build_diffusion_graph(df: pd.DataFrame):
    """
    Build a global diffusion graph (origin→follower edges) across clusters within window.
    Compute Source Influence Rank (PageRank).
    """
    G = nx.DiGraph()
    # Add nodes for all outlets
    for outlet in df["newspaper_id"].unique():
        G.add_node(outlet)

    for cid, g in df.groupby("cluster_id"):
        if cid == -1: 
            continue
        g = g.sort_values("ts")
        first_ts = g["ts"].min()
        origin = g.iloc[0]["newspaper_id"]
        within = g[g["ts"] <= first_ts + pd.Timedelta(hours=CLUSTER_WINDOW_HOURS)]
        followers = within["newspaper_id"].unique().tolist()
        for f in followers:
            if f == origin:
                continue
            # edge origin -> follower (one per cluster; weight accumulate)
            if G.has_edge(origin, f):
                G[origin][f]["weight"] += 1
            else:
                G.add_edge(origin, f, weight=1)

    # PageRank (weighted)
    if G.number_of_edges() > 0:
        pr = nx.pagerank(G, weight="weight")
    else:
        pr = {n: 0.0 for n in G.nodes}

    pr_df = pd.DataFrame({"newspaper_id": list(pr.keys()), "sir_pagerank": list(pr.values())}) \
              .sort_values("sir_pagerank", ascending=False)
    return G, pr_df


def nmf_topics(df: pd.DataFrame, text_col="title", n_topics=N_TOPICS):
    """
    Lightweight topic model with TF-IDF + NMF (fast, language-agnostic).
    Returns (topic_assignments, components, vectorizer, nmf_model).
    """
    texts = df[text_col].fillna("").tolist()
    vec = TfidfVectorizer(max_features=20000, min_df=3, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    nmf = NMF(n_components=n_topics, random_state=42, init="nndsvd", max_iter=400)
    W = nmf.fit_transform(X)    # doc-topic
    H = nmf.components_         # topic-term
    topics = W.argmax(axis=1)
    return topics, H, vec, nmf


def topic_labels(H, vec, topn=10):
    idx2term = np.array(vec.get_feature_names_out())
    labels = []
    for k in range(H.shape[0]):
        top_terms = idx2term[np.argsort(H[k])[::-1][:topn]]
        labels.append(", ".join(top_terms))
    return labels


def outlet_similarity(df: pd.DataFrame, topics: np.ndarray):
    """
    Build per-outlet topic distributions and embed to 2D via UMAP (if available).
    """
    dt = pd.DataFrame({"newspaper_id": df["newspaper_id"].values,
                       "topic": topics})
    pivot = dt.pivot_table(index="newspaper_id", columns="topic", aggfunc=len, fill_value=0)
    # Normalize rows
    pivot = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0)
    if HAS_UMAP and pivot.shape[0] > 2:
        reducer = umap.UMAP(n_neighbors=8, min_dist=0.3, random_state=42)
        X2 = reducer.fit_transform(pivot.values)
    else:
        # PCA-ish fallback using first two components of SVD via NMF trick
        nmf = NMF(n_components=min(2, max(1, pivot.shape[1]-1)), random_state=42)
        X2 = nmf.fit_transform(pivot.values)
        if X2.shape[1] == 1:
            X2 = np.hstack([X2, np.zeros_like(X2)])
    sim_df = pd.DataFrame({"newspaper_id": pivot.index,
                           "x": X2[:,0], "y": X2[:,1]})
    return sim_df, pivot


def jaccard_keywords(a: str, b: str) -> float:
    tok = lambda s: set([t for t in "".join(ch if ch.isalnum() else " " for ch in s.lower()).split() if len(t)>2])
    A, B = tok(a), tok(b)
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)


def framing_divergence(df: pd.DataFrame, cluster_id: int) -> pd.DataFrame:
    """
    Compare language-specific framing within a single cluster via lexical divergence.

    Robust against clusters with <2 languages or empty pair generation.
    Returns a DataFrame with columns ["pair", "jaccard", "len_diff"] or empty with same schema.
    """
    # Always return a DF with the right columns if we have nothing to compare
    EMPTY = pd.DataFrame(columns=["pair", "jaccard", "len_diff"])

    # Guard: cluster subset
    g = df[df.get("cluster_id") == cluster_id]
    if g is None or g.empty:
        return EMPTY

    # Guard: ensure we actually have language info and >1 language
    if "lang" not in g.columns:
        return EMPTY
    langs = g["lang"].dropna().unique().tolist()
    if len(langs) < 2:
        return EMPTY

    # Build language-pair comparisons
    pairs = []
    # Group once to avoid recomputation
    grouped = {la: gb for la, gb in g.groupby("lang")}
    for i, la in enumerate(langs):
        for lb in langs[i+1:]:
            gb = grouped.get(la)
            gc = grouped.get(lb)
            if gb is None or gc is None or gb.empty or gc.empty:
                continue
            a = " ".join(gb["title"].astype(str).tolist())
            b = " ".join(gc["title"].astype(str).tolist())
            jac = jaccard_keywords(a, b)  # 0..1 (higher=more similar)
            len_diff = abs(len(a) - len(b)) / (1 + (len(a)+len(b))/2)
            pairs.append({"pair": f"{la} vs {lb}", "jaccard": float(jac), "len_diff": float(len_diff)})

    if not pairs:
        return EMPTY

    df_pairs = pd.DataFrame(pairs)
    # Extra guard: only sort if column exists
    if "jaccard" in df_pairs.columns:
        return df_pairs.sort_values("jaccard", ascending=True, kind="mergesort").reset_index(drop=True)
    return df_pairs.reset_index(drop=True)


def who_covers_next_baseline(df: pd.DataFrame, horizon_hours=PRED_PICKUP_HOURS):
    """
    Baseline predictor of 'who covers next' using conditional probabilities:
    P(follower | origin, lang) measured from historical clusters within `horizon_hours`.
    Adds a simple momentum feature: recent growth rate in last 2 hours.
    """
    # Build historical edges with small time threshold
    edges = []
    for cid, g in df.groupby("cluster_id"):
        if cid == -1: 
            continue
        g = g.sort_values("ts")
        first_ts = g["ts"].min()
        origin = g.iloc[0]["newspaper_id"]
        lang = g.iloc[0]["lang"]
        within = g[g["ts"] <= first_ts + pd.Timedelta(hours=horizon_hours)]
        followers = within["newspaper_id"].unique().tolist()
        for f in followers:
            if f == origin: 
                continue
            edges.append((origin, f, lang))
    if not edges:
        return {}

    edge_df = pd.DataFrame(edges, columns=["origin","follower","lang"])
    cond = edge_df.groupby(["origin","lang","follower"]).size().rename("cnt").reset_index()
    totals = cond.groupby(["origin","lang"])["cnt"].sum().rename("tot").reset_index()
    probs = cond.merge(totals, on=["origin","lang"], how="left")
    probs["p"] = probs["cnt"] / probs["tot"]
    # organize into dict
    table = {}
    for _, r in probs.iterrows():
        table.setdefault((r["origin"], r["lang"]), []).append((r["follower"], float(r["p"])))
    # sort
    for k in table:
        table[k] = sorted(table[k], key=lambda x: x[1], reverse=True)
    return table


def compute_cluster_momentum(df: pd.DataFrame, cluster_id: int, window_minutes=120):
    g = df[df["cluster_id"] == cluster_id].sort_values("ts")
    if g.empty: 
        return 0.0
    last_ts = g["ts"].max()
    recent = g[g["ts"] >= last_ts - pd.Timedelta(minutes=window_minutes)]
    prev = g[g["ts"] < last_ts - pd.Timedelta(minutes=window_minutes)]
    rate = (len(recent) - len(prev)) / (1 + len(prev))
    return rate


# ----------------------------
# STREAMLIT UI
# ----------------------------

st.title("🇱🇰 Media Intelligence – Sri Lanka News (Python)")

with st.sidebar:
    st.header("Controls")
    today = datetime.now()
    date_to = st.date_input("End date", value=today.date())
    default_from = today - timedelta(days=DEFAULT_DAYS)
    date_from = st.date_input("Start date", value=default_from.date())
    date_from_dt = datetime.combine(date_from, datetime.min.time())
    date_to_dt = datetime.combine(date_to, datetime.max.time())

    st.caption("💡 Keep the date window manageable (e.g., 30–120 days) for best performance.")
    do_filter_noise = st.checkbox("Hide noise cluster (-1)", value=True)
    show_records = st.slider("Max rows to show in tables", 100, 5000, 1000, step=100)
    st.markdown("---")
    st.subheader("Clustering")
    st.caption("Attempts multilingual sentence embeddings; falls back to TF‑IDF if unavailable.")
    st.write(f"Cluster window: {CLUSTER_WINDOW_HOURS} hours")

    st.subheader("Predictor")
    st.caption(f"'Who covers next' horizon: {PRED_PICKUP_HOURS} hours")

    st.markdown("---")
    st.subheader("Data Sources")
    st.markdown(
        "- HF dataset: [`nuuuwan/lk-news-docs`](https://huggingface.co/datasets/nuuuwan/lk-news-docs)  \n"
        "- GitHub data-only repo: [`news_lk3_data`](https://github.com/nuuuwan/news_lk3_data)  \n"
        "- Example ext JSON: [`ext_articles/689cd094.ext.json`](https://github.com/nuuuwan/news_lk3_data/blob/main/ext_articles/689cd094.ext.json)  \n"
        "- Repo README (stats/coverage): [`README.md`](https://github.com/nuuuwan/news_lk3_data/blob/main/README.md)"
    )

# Load
with st.spinner("Loading documents from Hugging Face…"):
    df = load_docs_from_hf(date_from_dt, date_to_dt)
st.success(f"Loaded {len(df):,} documents.")

# Cluster
with st.spinner("Clustering stories (multilingual)…"):
    df, emb_mode = cluster_stories(df, text_col="title")
st.success(f"Clustering complete. Embedding mode: **{emb_mode.upper()}**. "
           f"Clusters: {df['cluster_id'].nunique():,} "
           f"(noise: {int((df['cluster_id']==-1).sum())} docs).")

if do_filter_noise:
    df = df[df["cluster_id"] != -1].copy()

# Metrics
with st.spinner("Computing story metrics & outlet KPIs…"):
    story_df, pr_df, outlet_kpi = compute_story_metrics(df)
st.success("Metrics computed.")

# Diffusion graph + SIR
with st.spinner("Building diffusion graph & Source Influence Rank (PageRank)…"):
    G, sir_df = build_diffusion_graph(df)
st.success("Diffusion graph ready.")

# Topics
with st.spinner("Deriving lightweight topics (TF‑IDF + NMF)…"):
    topics, H, vec, nmf = nmf_topics(df, text_col="title", n_topics=N_TOPICS)
    df["topic_id"] = topics
    topic_names = topic_labels(H, vec, topn=8)
    df["topic_label"] = df["topic_id"].apply(lambda k: f"T{k:02d}: {topic_names[k]}" if 0 <= k < len(topic_names) else f"T{k}")
st.success(f"Topic model built with {N_TOPICS} topics.")

# Outlet similarity
with st.spinner("Computing outlet similarity map…"):
    sim_df, outlet_topic_dist = outlet_similarity(df, topics)
st.success("Outlet similarity map ready.")

# Who covers next baseline
with st.spinner("Training baseline 'who covers next' table…"):
    next_table = who_covers_next_baseline(df, horizon_hours=PRED_PICKUP_HOURS)
st.success("Baseline next‑outlet table computed.")

# -------------
# TABS
# -------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚨 Breaking Radar",
    "🌐 Diffusion Map & SIR",
    "📊 Coverage Health",
    "🧭 Framing Lens",
    "🗺️ Outlet Similarity"
])

# ---- TAB 1: BREAKING RADAR ----

with tab1:
    st.subheader("Breaking Radar (newest clusters)")

    # Show the N newest clusters by origin_time
    n_show = st.slider("How many recent clusters?", 10, 200, 50, step=10, key="tab1_n_show"
    radar = story_df.head(n_show).copy()

    # Predict who covers next for each cluster, using origin outlet + lang + momentum
    preds = []
    for _, r in radar.iterrows():
        cid = r["cluster_id"]
        g = df[df["cluster_id"] == cid].sort_values("ts")
        if g.empty:
            continue
        origin = r["origin_outlet"]
        lang = g.iloc[0]["lang"]
        candidates = next_table.get((origin, lang), [])
        # momentum score
        mom = compute_cluster_momentum(df, cid, window_minutes=120)
        # Adjust probabilities with momentum (simple scaling)
        adjusted = [(o, min(0.99, p*(1 + 0.25*np.tanh(mom)))) for o,p in candidates]
        top5 = adjusted[:5]
        preds.append({"cluster_id": cid, "predicted_next": ", ".join([f"{o} ({p:.2f})" for o,p in top5])})

    pred_df = pd.DataFrame(preds)
    radar = radar.merge(pred_df, on="cluster_id", how="left")

    st.dataframe(radar.head(show_records), use_container_width=True)

    # Quick bar: clusters by number of outlets (spread)
    fig = px.histogram(radar, x="n_outlets_within_window", nbins=20,
                       title="Distribution: Number of Outlets per Recent Story (within window)",
                       labels={"n_outlets_within_window":"#Outlets in 72h window"})
    st.plotly_chart(fig, use_container_width=True)


# ---- TAB 2: DIFFUSION MAP & SIR ----
with tab2:
    st.subheader("Source Influence Rank (SIR) – PageRank on story diffusion")
    st.caption("Edge origin→follower if follower covered the story within the window; weights = number of such stories.")

    # SIR leaderboard
    st.dataframe(sir_df.head(50), use_container_width=True)

    # Interactive cluster diffusion map
    st.markdown("### Story Diffusion Graph (select a cluster)")
    cluster_choices = story_df["cluster_id"].tolist()
    sel_cid = st.selectbox(
        "Cluster",
        cluster_choices[:500] if len(cluster_choices) > 500 else cluster_choices,
        key="tab2_cluster"  # <-- unique key
    )

    g = df[df["cluster_id"] == sel_cid].sort_values("ts")

    if g.empty:
        st.info("No data for selected cluster.")
    else:
        first_ts = g["ts"].min()
        origin = g.iloc[0]["newspaper_id"]
        within = g[g["ts"] <= first_ts + pd.Timedelta(hours=CLUSTER_WINDOW_HOURS)]
        nodes = within["newspaper_id"].unique().tolist()
        # positions via spring layout
        Hgraph = nx.DiGraph()
        for n in nodes:
            Hgraph.add_node(n)
        for n in nodes:
            if n == origin: 
                continue
            Hgraph.add_edge(origin, n)
        pos = nx.spring_layout(Hgraph, seed=42, k=0.6)

        # Build Plotly scatter
        x, y, text, color = [], [], [], []
        for n in Hgraph.nodes:
            x.append(pos[n][0]); y.append(pos[n][1])
            text.append(n)
            color.append("crimson" if n==origin else "royalblue")
        node_fig = go.Figure()
        # edges
        edge_x, edge_y = [], []
        for u,v in Hgraph.edges:
            edge_x += [pos[u][0], pos[v][0], None]
            edge_y += [pos[u][1], pos[v][1], None]
        node_fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='#aaa', width=1),
                                      hoverinfo='none', showlegend=False))
        node_fig.add_trace(go.Scatter(x=x, y=y, mode='markers+text',
                                      marker=dict(color=color, size=14, opacity=0.9),
                                      text=text, textposition='top center', hoverinfo='text',
                                      showlegend=False))
        node_fig.update_layout(title=f"Cluster {sel_cid} Diffusion – Origin: {origin}  |  Articles: {len(within)}  |  Outlets: {len(nodes)}",
                               xaxis=dict(visible=False), yaxis=dict(visible=False),
                               height=500)
        st.plotly_chart(node_fig, use_container_width=True)


# ---- TAB 3: COVERAGE HEALTH ----
with tab3:
    st.subheader("Outlet Coverage Health")

    # Merge SIR with KPIs
    health = outlet_kpi.reset_index().rename(columns={"index":"newspaper_id"})
    if "newspaper_id" not in health.columns:
        health = health.reset_index().rename(columns={"index":"newspaper_id"})
    health = pd.merge(health, sir_df, on="newspaper_id", how="outer").fillna(0)

    # Show table
    disp = health.sort_values("sir_pagerank", ascending=False)
    st.dataframe(disp.head(show_records), use_container_width=True)

    # Scatter: Pickup Rate vs SIR
    fig = px.scatter(disp, x="sir_pagerank", y="pickup_rate", size="breaks", color="avg_ttp",
                     hover_name="newspaper_id",
                     title="Pickup Rate vs Source Influence Rank (size = breaks; color = Avg TTP mins)",
                     labels={"sir_pagerank":"SIR (PageRank)", "pickup_rate":"Pickup Rate", "avg_ttp":"Avg TTP (mins)"})
    st.plotly_chart(fig, use_container_width=True)

    # Share of Voice by topic
    st.markdown("### Share of Voice by Topic")
    sov = df.groupby(["newspaper_id","topic_label"]).size().rename("count").reset_index()
    top_topics = sov.groupby("topic_label")["count"].sum().sort_values(ascending=False).head(15).index
    sov_top = sov[sov["topic_label"].isin(top_topics)]
    fig = px.treemap(sov_top, path=["topic_label","newspaper_id"], values="count",
                     title="Share of Voice (Top 15 Topics)")
    st.plotly_chart(fig, use_container_width=True)


# ---- TAB 4: FRAMING LENS ----
# ---- TAB 4: FRAMING LENS ----
with tab4:
    st.subheader("Framing Divergence across Languages")
    st.caption("Lexical Jaccard similarity and length difference across si/ta/en within a story cluster.")

    if story_df.empty:
        st.info("No clusters available in the selected date range.")
    else:
        framing_choices = story_df["cluster_id"].dropna().unique().tolist()
        framing_choices = framing_choices[:500]
        sel_cid2 = st.selectbox(
            "Select cluster for framing",
            framing_choices,
            key="tab4_cluster"  # <-- unique key
        )

        fd = framing_divergence(df, sel_cid2)
        if fd.empty:
            st.info("Not enough language variation in this cluster.")
        else:
            st.dataframe(fd, use_container_width=True)
            fig = px.bar(
                fd, x="pair", y="jaccard",
                title="Jaccard similarity (higher → more similar)",
                range_y=[0, 1]
            )
            st.plotly_chart(fig, use_container_width=True)

        # (Optional) show languages present in this cluster
        langs_in_cluster = df.loc[df["cluster_id"] == sel_cid2, "lang"].dropna().unique().tolist()




# ---- TAB 5: OUTLET SIMILARITY ----
with tab5:
    st.subheader("Outlet Similarity & Beat Overlap")
    if sim_df.empty:
        st.info("Not enough outlets to plot.")
    else:
        fig = px.scatter(sim_df, x="x", y="y", color="newspaper_id", hover_name="newspaper_id",
                         title="Outlets in 2D (UMAP/NMF of topic distributions)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Tip: outlets close together tend to cover similar topics/angles in this period.")

# Footer notes & citations
st.markdown("---")
st.markdown(
    "**Citations & data structure references**  \n"
    "• Hugging Face dataset (schema, date range, fields): "
    "[nuuuwan/lk-news-docs](https://huggingface.co/datasets/nuuuwan/lk-news-docs)  \n"
    "• Data‑only repo, counts, extensions (translations/NER): "
    "[news_lk3_data](https://github.com/nuuuwan/news_lk3_data), "
    "[README.md (stats)](https://github.com/nuuuwan/news_lk3_data/blob/main/README.md)  \n"
    "• Example of `ext_articles` translation JSON structure: "
    "[689cd094.ext.json](https://github.com/nuuuwan/news_lk3_data/blob/main/ext_articles/689cd094.ext.json)"
)


