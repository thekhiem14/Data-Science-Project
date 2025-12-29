from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

import joblib
# Optional but recommended: robust image loading for MAL CDN
import requests


# ----------------------------
# Paths
# ----------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR
ART_DIR = ROOT_DIR / "artifacts"
FIG_DIR = ART_DIR / "figures"
HISTORY_PATH = ART_DIR / "user_history.json"
OLD_CSV_PATH = ART_DIR / "anime_clean_old.csv"


# ----------------------------
# Utilities
# ----------------------------
def safe_http_url(x) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip().strip('"').strip("'")
    if s.startswith("//"):
        s = "https:" + s
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return None


def explode_genres(s: str) -> List[str]:
    if not s:
        return []
    s = str(s)
    if "|" in s:
        parts = s.split("|")
    elif ";" in s:
        parts = s.split(";")
    else:
        parts = s.split(",")
    return [p.strip() for p in parts if p.strip()]


def now_bucket() -> str:
    import datetime
    h = datetime.datetime.now().hour
    if 5 <= h < 11:
        return "morning"
    if 11 <= h < 17:
        return "afternoon"
    if 17 <= h < 23:
        return "evening"
    return "late_night"


def load_history() -> Dict:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_history(h: Dict) -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(h, ensure_ascii=False, indent=2), encoding="utf-8")


def add_history_event(user_key: str, event: str, title: str):
    h = load_history()
    u = h.get(user_key, {"events": [], "context": {}})

    is_duplicate = any(e.get("event") == event and e.get("title") == title for e in u["events"])
    if not is_duplicate:
        u["events"].append({"event": event, "title": title})
        h[user_key] = u
        save_history(h)


def get_user_events(user_key: str) -> List[Dict]:
    h = load_history()
    return h.get(user_key, {}).get("events", [])


def get_user_titles(user_key: str, event_name: str) -> List[str]:
    ev = get_user_events(user_key)
    return [e.get("title") for e in ev if e.get("event") == event_name and e.get("title")]


@st.cache_data(show_spinner=False)
def fetch_image_bytes(url: str) -> Optional[bytes]:
    """Fetch image with a browser-like UA (MAL CDN often blocks hotlinking)."""
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code == 200:
            return r.content
    except Exception:
        return None
    return None


@st.cache_data(show_spinner=False)
def old_image_lookup(anime_id: int) -> Optional[str]:
    """Fallback to old csv if provided (optional). Only used if current image_url missing."""
    if not OLD_CSV_PATH.exists():
        return None
    try:
        old = pd.read_csv(OLD_CSV_PATH, usecols=["anime_id", "image_url"])
        old = old.dropna(subset=["anime_id"])
        old["anime_id"] = old["anime_id"].astype(int)
        row = old.loc[old["anime_id"] == int(anime_id)]
        if row.empty:
            return None
        return safe_http_url(row.iloc[0]["image_url"])
    except Exception:
        return None


# ----------------------------
# Load artifacts
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(ART_DIR / "anime_clean.csv")

    # Ensure minimal columns for UI (do NOT auto-create image_url if missing in real pipeline)
    for c in ["title", "genres", "synopsis", "combined_text", "anime_url"]:
        if c not in df.columns:
            df[c] = ""

    if "image_url" not in df.columns:
        df["image_url"] = ""

    # numeric columns used in ranking/filters (create if missing)
    if "score_filled" not in df.columns:
        if "score" in df.columns:
            df["score_filled"] = pd.to_numeric(df["score"], errors="coerce")
        else:
            df["score_filled"] = np.nan
        df["score_filled"] = df["score_filled"].fillna(
            df["score_filled"].median() if df["score_filled"].notna().any() else 0.0
        )

    if "weighted_score" not in df.columns:
        df["weighted_score"] = df["score_filled"]

    if "anime_id" not in df.columns:
        df["anime_id"] = np.arange(len(df))

    # sanitize urls
    df["image_url"] = df["image_url"].astype(str).str.strip().str.replace(r"^//", "https://", regex=True)
    df["anime_url"] = df["anime_url"].astype(str).str.strip()

    return df

def _parse_at_k_block(block: dict):
    """
    block dạng:
    {"tfidf":{"5":0.7,"10":0.6}, "sbert":{"5":0.72,"10":0.64}}
    -> DataFrame index=K, cols=[TF-IDF, SBERT, ...]
    """
    if not isinstance(block, dict) or not block:
        return None

    # Thu tất cả model keys: tfidf/sbert/matrix_cf/...
    model_keys = [k for k in block.keys() if isinstance(block.get(k), dict)]
    if not model_keys:
        return None

    # Collect all K
    ks = set()
    for mk in model_keys:
        for k in (block.get(mk) or {}).keys():
            try:
                ks.add(int(k))
            except Exception:
                pass
    if not ks:
        return None
    ks = sorted(ks)

    # Build rows
    rows = []
    for k in ks:
        row = {"K": k}
        for mk in model_keys:
            val = block.get(mk, {}).get(str(k), None)
            row[mk.upper()] = val
        rows.append(row)

    df = pd.DataFrame(rows).set_index("K")
    return df

def _pretty_metric(value, digits=4):
    try:
        if value is None:
            return "N/A"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"
@st.cache_resource(show_spinner=False)
def load_models():
    tfidf = None
    tfidf_matrix = None
    tfidf_vec_path = ART_DIR / "tfidf_vectorizer.joblib"
    tfidf_mat_path = ART_DIR / "tfidf_matrix.npz"
    if tfidf_vec_path.exists() and tfidf_mat_path.exists():
        tfidf = joblib.load(tfidf_vec_path)
        tfidf_matrix = sparse.load_npz(tfidf_mat_path)

    emb = None
    emb_meta = None
    emb_path = ART_DIR / "sbert_embeddings.npy"
    meta_path = ART_DIR / "embedding_meta.json"
    if emb_path.exists():
        emb = np.load(emb_path)
    if meta_path.exists():
        emb_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return tfidf, tfidf_matrix, emb, emb_meta


@st.cache_resource(show_spinner=False)
def load_matrix_bundle():
    m_path = ART_DIR / "user_item_matrix.npz"
    if not m_path.exists():
        return None, None, None

    R = sparse.load_npz(m_path).tocsr()
    user_norm = np.sqrt(R.power(2).sum(axis=1)).A1
    item_norm = np.sqrt(R.power(2).sum(axis=0)).A1
    return R, user_norm, item_norm


@st.cache_data(show_spinner=False)
def load_metrics():
    p = ART_DIR / "metrics.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


# ----------------------------
# Helpers
# ----------------------------
def find_title_idx(df: pd.DataFrame, title: str) -> Optional[int]:
    if not title:
        return None
    m = df.index[df["title"].astype(str) == str(title)].tolist()
    if m:
        return int(m[0])
    t = str(title).strip().lower()
    m2 = df.index[df["title"].astype(str).str.lower() == t].tolist()
    if m2:
        return int(m2[0])
    return None


def apply_filters(x: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if not filters:
        return x
    y = x.copy()

    genres = filters.get("genres", [])
    if genres and "genres" in y.columns:
        y = y[y["genres"].fillna("").apply(lambda g: any(t in g for t in genres))]

    types = filters.get("type", [])
    if types and "type" in y.columns:
        y = y[y["type"].isin(types)]

    score_range = filters.get("score_range")
    if score_range and "score" in y.columns:
        lo, hi = score_range
        y = y[pd.to_numeric(y["score"], errors="coerce").fillna(-1).between(lo, hi)]

    return y


def _align_matrix_to_df(df: pd.DataFrame, R: sparse.csr_matrix, item_norm: np.ndarray):
    if R is None:
        return df, None, None
    n = min(len(df), R.shape[1])
    df0 = df.iloc[:n].reset_index(drop=True)
    R0 = R[:, :n]
    norm0 = item_norm[:n] if item_norm is not None and len(item_norm) >= n else None
    return df0, R0, norm0


# ----------------------------
# Recommenders
# ----------------------------
def recommend_tfidf(df: pd.DataFrame, tfidf_matrix, title: str, top_n: int = 12, filters=None) -> pd.DataFrame:
    if tfidf_matrix is None:
        return pd.DataFrame()

    n = min(len(df), tfidf_matrix.shape[0])
    df0 = df.iloc[:n].reset_index(drop=True)
    M = tfidf_matrix[:n]

    idx = find_title_idx(df0, title)
    if idx is None:
        return pd.DataFrame()

    q = M[idx]
    sims = (M @ q.T).toarray().ravel()
    sims[idx] = -1
    order = np.argsort(-sims)

    cand = df0.iloc[order].copy()
    cand["sim"] = sims[order]
    if filters:
        cand = apply_filters(cand, filters)

    out = cand.head(top_n).copy()
    out["reason"] = "TF-IDF similarity"
    return out


def recommend_sbert(df: pd.DataFrame, emb, title: str, top_n: int = 12, filters=None) -> pd.DataFrame:
    if emb is None:
        return pd.DataFrame()

    n = min(len(df), emb.shape[0])
    df0 = df.iloc[:n].reset_index(drop=True)
    E = emb[:n]

    idx = find_title_idx(df0, title)
    if idx is None:
        return pd.DataFrame()

    q = E[idx]
    sims = (E @ q).ravel()
    sims[idx] = -1
    order = np.argsort(-sims)

    cand = df0.iloc[order].copy()
    cand["sim"] = sims[order]
    if filters:
        cand = apply_filters(cand, filters)

    out = cand.head(top_n).copy()
    out["reason"] = "SBERT similarity"
    return out


def recommend_matrix_item(df: pd.DataFrame, R: sparse.csr_matrix, item_norm: np.ndarray, title: str,
                          top_n: int = 12, filters=None) -> pd.DataFrame:
    if R is None:
        return pd.DataFrame()

    df0, R0, norm0 = _align_matrix_to_df(df, R, item_norm)
    if R0 is None or norm0 is None:
        return pd.DataFrame()

    idx = find_title_idx(df0, title)
    if idx is None or idx >= R0.shape[1]:
        return pd.DataFrame()

    col = R0[:, idx]
    sims = (col.T @ R0).toarray().ravel().astype(np.float32)

    denom = float(norm0[idx]) * norm0
    sims = np.divide(sims, denom, out=np.zeros_like(sims), where=denom != 0)

    sims[idx] = -1
    order = np.argsort(-sims)

    cand = df0.iloc[order].copy()
    cand["sim"] = sims[order]
    if filters:
        cand = apply_filters(cand, filters)

    out = cand.head(top_n).copy()
    out["reason"] = "Matrix CF (co-watch cosine)"
    return out


def recommend_matrix_user(df: pd.DataFrame, R: sparse.csr_matrix, user_norm: np.ndarray, item_norm: np.ndarray,
                          seed_titles: List[str], top_n: int = 12, filters=None,
                          exclude_titles: Optional[set] = None) -> pd.DataFrame:
    if R is None:
        return pd.DataFrame()

    df0, R0, norm0 = _align_matrix_to_df(df, R, item_norm)
    if R0 is None:
        return pd.DataFrame()

    seed_idx = []
    for t in seed_titles:
        i = find_title_idx(df0, t)
        if i is not None:
            seed_idx.append(i)
    seed_idx = sorted(set(seed_idx))
    if not seed_idx:
        return pd.DataFrame()

    sim_users = R0[:, seed_idx].sum(axis=1).A1.astype(np.float32)

    if user_norm is not None and len(user_norm) >= R0.shape[0]:
        denom = user_norm[:R0.shape[0]] * float(np.sqrt(len(seed_idx)))
        sim_users = np.divide(sim_users, denom, out=np.zeros_like(sim_users), where=denom != 0)

    scores = (sim_users @ R0).astype(np.float32)

    if norm0 is not None:
        scores = np.divide(scores, norm0, out=scores, where=norm0 != 0)

    scores[seed_idx] = -1
    if exclude_titles:
        for t in exclude_titles:
            i = find_title_idx(df0, t)
            if i is not None:
                scores[i] = -1

    order = np.argsort(-scores)
    cand = df0.iloc[order].copy()
    cand["sim"] = scores[order]
    if filters:
        cand = apply_filters(cand, filters)

    out = cand.head(top_n).copy()
    out["reason"] = "Matrix CF (personalized from your history)"
    return out


def recommend_matrix(df: pd.DataFrame, R, user_norm, item_norm, base_title: str,
                     seed_titles: List[str], top_n: int = 12, filters=None,
                     exclude_titles: Optional[set] = None) -> pd.DataFrame:
    """Unified Matrix CF: personalized if seeds exist, else item-based from base_title."""
    if seed_titles:
        rec = recommend_matrix_user(df, R, user_norm, item_norm, seed_titles, top_n=top_n,
                                    filters=filters, exclude_titles=exclude_titles)
        if not rec.empty:
            return rec
    return recommend_matrix_item(df, R, item_norm, base_title, top_n=top_n, filters=filters)


def hybrid_rank(df: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Hybrid rank = normalize(sim_TFIDF) + normalize(sim_SBERT). (No Popularity/Weighted mixing)"""
    if (a is None or a.empty) and (b is None or b.empty):
        return pd.DataFrame()

    base = df.copy()
    score = np.zeros(len(base), dtype=np.float32)

    def add_sim(d: pd.DataFrame, w: float) -> None:
        if d is None or d.empty or "sim" not in d.columns:
            return
        s = d.set_index("title")["sim"].copy()
        v = s.values
        if len(v) > 0:
            mn, mx = float(np.min(v)), float(np.max(v))
            if mx > mn:
                s = (s - mn) / (mx - mn)
            else:
                s = s * 0.0

        for t, val in s.items():
            idxs = base.index[base["title"] == t].tolist()
            if idxs:
                score[idxs[0]] += w * float(val)

    add_sim(a, 0.5)
    add_sim(b, 0.5)

    base = base.copy()
    base["sim"] = score
    base = base.sort_values("sim", ascending=False).head(top_n)
    base["reason"] = "Hybrid (TF-IDF + SBERT)"
    return base



# ----------------------------
# UI components
# ----------------------------
def movie_card(row: pd.Series, show_actions: bool, user_key: str):
    cols = st.columns([1, 2], gap="small")

    with cols[0]:
        url = safe_http_url(row.get("image_url"))
        if not url:
            # optional fallback to old file, if present
            url = old_image_lookup(int(row.get("anime_id", -1))) if OLD_CSV_PATH.exists() else None

        if url:
            b = fetch_image_bytes(url)
            if b:
                st.image(b, use_container_width=True)
            else:
                st.info("Image blocked/unreachable")
        else:
            st.info("No image")

    with cols[1]:
        st.subheader(row.get("title", "Untitled"))

        anime_url = safe_http_url(row.get("anime_url"))
        if anime_url:
            st.caption(anime_url)

        g = row.get("genres", "")
        st.write(f"**Genres:** {g if g else 'N/A'}")
        if "type" in row:
            st.write(f"**Type:** {row.get('type','')}")
        st.write(f"**Score:** {row.get('score_filled','')}  |  **Weighted:** {row.get('weighted_score','')}")
        if "sim" in row and not pd.isna(row["sim"]):
            st.write(f"**Similarity:** {float(row['sim']):.4f}")
        st.write(f"**Why:** {row.get('reason','')}")

        syn = str(row.get("synopsis", "")).strip()
        if syn:
            with st.expander("Synopsis"):
                st.write(syn)

        if show_actions:
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("👍 Like", key=f"like_{user_key}_{row.get('anime_id','')}", use_container_width=True):
                    add_history_event(user_key, "like", row.get("title", ""))
                    st.rerun()
            with c2:
                if st.button("👀 Watched", key=f"watched_{user_key}_{row.get('anime_id','')}", use_container_width=True):
                    add_history_event(user_key, "watched", row.get("title", ""))
                    st.session_state["page"] = "Detail"
                    st.session_state["selected_item"] = row.get("anime_id")
                    st.rerun()
            with c3:
                if st.button("👎 Dislike", key=f"dislike_{user_key}_{row.get('anime_id','')}", use_container_width=True):
                    add_history_event(user_key, "dislike", row.get("title", ""))
                    st.rerun()


# ----------------------------
# Pages
# ----------------------------
def sidebar_filters(df: pd.DataFrame) -> dict:
    st.sidebar.header("🔎 Filters")
    filters: dict = {}

    # Search moved to Home, so keep sidebar simple
    if "genres" in df.columns:
        all_genres = set()
        for g in df["genres"].dropna().astype(str):
            for x in g.split(","):
                all_genres.add(x.strip())
        filters["genres"] = st.sidebar.multiselect("Genres", sorted(all_genres))

    if "type" in df.columns:
        all_types = sorted(set(df["type"].fillna("").astype(str).unique()))
        filters["type"] = st.sidebar.multiselect("Type", all_types)
    else:
        filters["type"] = []

    if "score" in df.columns:
        s = pd.to_numeric(df["score"], errors="coerce")
        min_score = float(np.nanmin(s)) if np.isfinite(np.nanmin(s)) else 0.0
        max_score = float(np.nanmax(s)) if np.isfinite(np.nanmax(s)) else 10.0
        filters["score_range"] = st.sidebar.slider(
            "Score range", min_value=min_score, max_value=max_score, value=(min_score, max_score)
        )

    filters["exclude_watched"] = st.sidebar.checkbox("Exclude watched items", value=True)

    return filters


def page_home(df: pd.DataFrame, user_key: str, filters: dict):
    st.title("🎌 Anime Recommender (Top 15,000)")
    st.caption("Search + Recommend: TF-IDF, SBERT, Hybrid, and Matrix CF (user–item).")

    # Search is merged into Home
    st.subheader("🔍 Search")
    q = st.text_input("Search by title", "", placeholder="e.g., Frieren, Steins;Gate, Gintama...")
    if q.strip():
        x = df[df["title"].astype(str).str.contains(q.strip(), case=False, na=False)].copy()
        x = apply_filters(x, filters)
        if filters.get("exclude_watched", False):
            watched = set(get_user_titles(user_key, "watched"))
            if watched:
                x = x[~x["title"].isin(watched)]
        st.write(f"Results: {len(x)}")
        show = x.sort_values("weighted_score", ascending=False).head(12)
        for _, r in show.iterrows():
            st.divider()
            movie_card(r, show_actions=True, user_key=user_key)

        st.subheader("🎲 Discover (Random)")

    cand = df.copy()
    if filters:
        cand = apply_filters(cand, filters)

    if filters.get("exclude_watched", False):
        watched = set(get_user_titles(user_key, "watched"))
        if watched:
            cand = cand[~cand["title"].isin(watched)]

    if cand.empty:
        st.warning("No items to show (filters too strict).")
        return

    if "discover_seed" not in st.session_state:
        st.session_state["discover_seed"] = 0

    if st.button("🔄 Refresh suggestions", use_container_width=True):
        st.session_state["discover_seed"] += 1

    n = min(12, len(cand))
    # sample with a stable seed so the UI doesn't shuffle on every rerun
    show = cand.sample(n=n, random_state=st.session_state["discover_seed"]).copy()
    show["reason"] = "Random discovery"
    show["sim"] = np.nan

    for _, r in show.iterrows():
        st.divider()
        movie_card(r, show_actions=True, user_key=user_key)



def page_detail(df, tfidf_matrix, emb, R, user_norm, item_norm):
    anime_id = st.session_state.get("selected_item")
    if anime_id is None:
        st.warning("No item selected.")
        return

    row_df = df[df["anime_id"].astype(int) == int(anime_id)]
    if row_df.empty:
        st.warning("Selected item not found.")
        return
    row = row_df.iloc[0]

    st.title(str(row.get("title", "Untitled")))
    url = safe_http_url(row.get("image_url")) or old_image_lookup(int(row.get("anime_id", -1)))
    if url:
        b = fetch_image_bytes(url)
        if b:
            st.image(b, width=280)

    st.markdown(f"**Score:** {row.get('score_filled', '')}")
    st.markdown(f"**Genres:** {row.get('genres', '')}")
    if row.get("synopsis", ""):
        st.write(row.get("synopsis", ""))

    st.markdown("---")
    st.subheader("🎯 Recommended based on this anime")

    algo = st.radio(
        "Similarity source",
        ["TF-IDF", "SBERT", "Matrix CF"],
        horizontal=True,
    )
    base_title = str(row.get("title", ""))

    if algo == "TF-IDF":
        if tfidf_matrix is None:
            st.warning("TF-IDF artifacts not found.")
            return
        recs = recommend_tfidf(df, tfidf_matrix, base_title, top_n=12)

    elif algo == "SBERT":
        if emb is None:
            st.warning("SBERT embeddings not found.")
            return
        recs = recommend_sbert(df, emb, base_title, top_n=12)

    else:
        if R is None:
            st.warning("user_item_matrix.npz not found.")
            return
        recs = recommend_matrix_item(df, R, item_norm, base_title, top_n=12)

    if recs is None or recs.empty:
        st.info("No recommendations found.")
        return

    cols = st.columns(4, gap="medium")
    for i, (_, r) in enumerate(recs.iterrows()):
        with cols[i % 4]:
            url2 = safe_http_url(r.get("image_url")) or old_image_lookup(int(r.get("anime_id", -1)))
            if url2:
                b2 = fetch_image_bytes(url2)
                if b2:
                    st.image(b2, use_container_width=True)
            st.caption(str(r.get("title", "")))

            if st.button("View Detail", key=f"detail_{int(r.get('anime_id', i))}"):
                st.session_state["selected_item"] = r.get("anime_id")
                st.session_state["page"] = "Detail"
                st.rerun()


def page_recommend(df: pd.DataFrame, tfidf_matrix, emb, R, user_norm, item_norm, user_key: str, filters: dict):
    st.header("✨ Recommend")

    titles = df["title"].astype(str).tolist()
    base = st.selectbox("Pick a base anime", options=titles, index=0)

    algo = st.radio("Algorithm", ["Hybrid", "SBERT", "TF-IDF", "Matrix CF"], horizontal=True)
    top_n = st.slider("Top N", 5, 30, 12)

    exclude_titles = set()
    if filters.get("exclude_watched", False):
        exclude_titles |= set(get_user_titles(user_key, "watched"))
    if algo == "TF-IDF":
        rec = recommend_tfidf(df, tfidf_matrix, base, top_n=top_n, filters=filters)

    elif algo == "SBERT":
        rec = recommend_sbert(df, emb, base, top_n=top_n, filters=filters)

    elif algo == "Matrix CF":
        if R is None:
            st.warning("user_item_matrix.npz not found.")
            return
        liked = get_user_titles(user_key, "like")
        watched = get_user_titles(user_key, "watched")
        seed_titles = list(dict.fromkeys(liked + watched))
        rec = recommend_matrix(
            df, R, user_norm, item_norm,
            base_title=base,
            seed_titles=seed_titles,
            top_n=top_n,
            filters=filters,
            exclude_titles=exclude_titles if exclude_titles else None,
        )

    else:  # Hybrid
        a = recommend_tfidf(df, tfidf_matrix, base, top_n=200, filters=filters) if tfidf_matrix is not None else pd.DataFrame()
        b = recommend_sbert(df, emb, base, top_n=200, filters=filters) if emb is not None else pd.DataFrame()
        rec = hybrid_rank(df, a, b, top_n=top_n)

    if exclude_titles and rec is not None and (not rec.empty) and "title" in rec.columns and algo not in ("Matrix CF",):
        rec = rec[~rec["title"].isin(exclude_titles)]

    if rec is None or rec.empty:
        st.warning("No recommendations (check filters / artifacts).")
        return

    st.subheader("Results")
    for _, r in rec.iterrows():
        st.divider()
        movie_card(r, show_actions=True, user_key=user_key)


def page_realtime_profile(df: pd.DataFrame, user_key: str):
    st.header("👤 Profile")
    st.caption("Thông tin, thẻ sở thích, và lịch sử tương tác (Like / Watched / Dislike).")

    ev = get_user_events(user_key)
    liked = get_user_titles(user_key, "like")
    watched = get_user_titles(user_key, "watched")
    disliked = get_user_titles(user_key, "dislike")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Likes", len(liked))
    c2.metric("Watched", len(watched))
    c3.metric("Dislikes", len(disliked))
    c4.metric("Context", now_bucket())

    # -----------------
    # Preference cards
    # -----------------
    st.subheader("Your top genres")
    if liked or watched:
        seed_titles = list(dict.fromkeys(liked + watched))
        seed_rows = df[df["title"].isin(seed_titles)].copy()

        if seed_rows.empty:
            st.info("Không tìm thấy các title trong dataset (có thể do khác tên).")
        else:
            freq: Dict[str, int] = {}
            for g in seed_rows["genres"].fillna("").astype(str):
                for tag in explode_genres(g):
                    freq[tag] = freq.get(tag, 0) + 1

            top = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:12]
            if top:
                pills = " ".join(
                    [
                        f"<span style='padding:6px 10px;border-radius:999px;background:#eef2ff;margin:4px;display:inline-block'>{k} · {v}</span>"
                        for k, v in top
                    ]
                )
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.info("Không đủ genre để tạo thẻ.")

        with st.expander("Liked / Watched titles"):
            st.write(", ".join(seed_titles[:50]) + (" ..." if len(seed_titles) > 50 else ""))
    else:
        st.info("Chưa có lịch sử. Hãy Like/Watched/Dislike vài anime ở Home/Recommend.")

    # -----------------
    # History (merged)
    # -----------------
    st.markdown("---")
    st.subheader("🧾 Interaction history")

    if not ev:
        st.info("Chưa có lịch sử. Hãy Like/Watched/Dislike vài anime.")
    else:
        hdf = pd.DataFrame(ev)
        st.dataframe(hdf, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🧹 Clear my history", use_container_width=True):
                h = load_history()
                if user_key in h:
                    del h[user_key]
                    save_history(h)
                st.rerun()
        with c2:
            st.write("Context now:", now_bucket())



def page_eda():
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.caption("Các biểu đồ từ notebook Kaggle (quality/popularity + matrix sparsity).")

    figs = [
        # Score distribution
        ("Histogram: Score distribution", FIG_DIR / "hist_score.png"),
        ("Boxplot: Score", FIG_DIR / "box_score.png"),
        ("Violin: Score distribution", FIG_DIR / "violin_score.png"),

        # Popularity vs score
        ("Scatter + Regression: Score vs Popularity", FIG_DIR / "scatter_reg_score_members.png"),

        # Ratings & implicit feedback
        ("Rating Distribution (MyAnimeList)", FIG_DIR / "rating_distribution.png"),
        ("Implicit Feedback Composition (Watched vs Liked)", FIG_DIR / "watched_liked.png"),

        # Correlation
        ("Correlation Heatmap", FIG_DIR / "heatmap.png"),

        # Matrix behavior
        ("User Activity Distribution", FIG_DIR / "user_activity_distribution.png"),
        ("Item Popularity (Interactions per Item)", FIG_DIR / "item_popularity_distribution.png"),
        ("User–Item Interaction Matrix (Sample)", FIG_DIR / "sparsity_pattern.png"),
    ]

    for i in range(0, len(figs), 2):
        cols = st.columns(2, gap="large")
        for col, item in zip(cols, figs[i:i + 2]):
            name, p = item
            with col:
                st.subheader(name)
                if p.exists():
                    st.image(str(p), caption=name, use_container_width=True)
                else:
                    st.info(f"Missing: {p.name}")



def page_evaluation(R: Optional[sparse.csr_matrix]):
    st.header("📈 Evaluation")

    m = load_metrics()
    if not m or not isinstance(m, dict):
        st.error("Không đọc được metrics.json hoặc file rỗng.")
        return

    # --- Top metrics cards ---
    rmse = m.get("rmse", None)
    mae  = m.get("mae", None)

    c1, c2, c3 = st.columns([1, 1, 1])
    c1.metric("RMSE", _pretty_metric(rmse))
    c2.metric("MAE", _pretty_metric(mae))

    # Optional: show dataset sparsity if R exists
    if R is not None:
        n_users, n_items = R.shape
        density = (R.nnz / (n_users * n_items)) if (n_users and n_items) else 0.0
        c3.metric("Interaction Density", f"{density:.6f}")
    else:
        c3.metric("Interaction Density", "N/A")

    st.divider()

    # --- Tabs for Ranking Metrics ---
    tab1, tab2 = st.tabs(["Top-K Metrics", "Summary"])

    with tab1:
        p_block = m.get("precision_at_k", None)
        r_block = m.get("recall_at_k", None) or m.get("recall@k", None) or m.get("recall_at_K", None)

        df_p = _parse_at_k_block(p_block) if p_block else None
        df_r = _parse_at_k_block(r_block) if r_block else None

        colA, colB = st.columns(2)

        with colA:
            st.subheader("Precision@K")
            if df_p is None:
                st.info("Không có dữ liệu Precision@K trong metrics.json.")
            else:
                st.dataframe(df_p.style.format("{:.4f}"), use_container_width=True)

        with colB:
            st.subheader("Recall@K")
            if df_r is None:
                st.info("Không có dữ liệu Recall@K trong metrics.json.")
            else:
                st.dataframe(df_r.style.format("{:.4f}"), use_container_width=True)

    with tab2:

        st.subheader("Metrics Overview")


        keys = sorted(list(m.keys()))

        st.write("**Available keys:**", ", ".join(keys) if keys else "None")


        with st.expander("Show raw metrics.json (debug)"):

            st.json(m)
        def _models_in(block):
            if not isinstance(block, dict):
                return []
            return sorted([k for k, v in block.items() if isinstance(v, dict)])

        p_models = _models_in(m.get("precision_at_k", {}))
        r_models = _models_in(m.get("recall_at_k", {})) or _models_in(m.get("recall@k", {})) or _models_in(m.get("recall_at_K", {}))

        s1, s2 = st.columns(2)

        # Raw JSON only in expander (so UI clean)
        with st.expander("Show raw metrics.json (debug)"):
            st.json(m)



def main():
    st.set_page_config(page_title="Anime Recommender", layout="wide")

    if "page" not in st.session_state:
        st.session_state["page"] = "Home"
    if "selected_item" not in st.session_state:
        st.session_state["selected_item"] = None

    df = load_data()
    _, tfidf_matrix, emb, emb_meta = load_models()
    R, user_norm, item_norm = load_matrix_bundle()

    st.sidebar.title("🎛️ Navigation")
    user_key = st.sidebar.text_input("User ID", value="khiem", help="Dùng để lưu lịch sử (JSON).")
    filters = sidebar_filters(df)

    pages = ["Home", "Recommend", "Profile", "EDA", "Evaluation"]
    nav = st.sidebar.radio("Go to", pages, index=pages.index(st.session_state.get("page", "Home")) if st.session_state.get("page") in pages else 0)

    if st.session_state.get("page") != "Detail":
        st.session_state["page"] = nav

    if st.session_state.get("page") == "Detail":
        if st.sidebar.button("⬅ Back"):
            st.session_state["page"] = "Recommend"
            st.session_state["selected_item"] = None
            st.rerun()
        page_detail(df, tfidf_matrix, emb, R, user_norm, item_norm)
        return

    if nav == "Home":
        page_home(df, user_key, filters)
        with st.expander("Artifacts info"):
            st.write("Emb meta:", emb_meta if emb_meta else "No embedding_meta.json")
            st.write("Matrix:", "loaded" if R is not None else "missing")

    elif nav == "Recommend":
        page_recommend(df, tfidf_matrix, emb, R, user_norm, item_norm, user_key, filters)

    elif nav == "Profile":
        page_realtime_profile(df, user_key)

    elif nav == "EDA":
        page_eda()

    else:
        page_evaluation(R)


if __name__ == "__main__":
    main()
