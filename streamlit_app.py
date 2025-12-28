from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Paths (robust, no need to change)
# ----------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR
ART_DIR = ROOT_DIR / "artifacts"
FIG_DIR = ART_DIR / "figures"
HISTORY_PATH = ART_DIR / "user_history.json"


# ----------------------------
# Utilities
# ----------------------------
def safe_http_url(x) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
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
    # simple context from time-of-day (realtime context)
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


# ----------------------------
# Load artifacts
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(ART_DIR / "anime_clean.csv")
    # ensure required columns exist
    for c in ["title", "genres", "synopsis", "image_url", "weighted_score", "score_filled"]:
        if c not in df.columns:
            df[c] = ""
    if "score_filled" not in df.columns and "score" in df.columns:
        df["score_filled"] = df["score"].fillna(df["score"].median())
    return df


@st.cache_resource(show_spinner=False)
def load_models():
    tfidf = joblib.load(ART_DIR / "tfidf_vectorizer.joblib")
    tfidf_matrix = sparse.load_npz(ART_DIR / "tfidf_matrix.npz")

    emb = None
    emb_meta = None
    emb_path = ART_DIR / "sbert_embeddings.npy"
    meta_path = ART_DIR / "embedding_meta.json"
    if emb_path.exists():
        emb = np.load(emb_path)
    if meta_path.exists():
        emb_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return tfidf, tfidf_matrix, emb, emb_meta


@st.cache_data(show_spinner=False)
def load_metrics():
    p = ART_DIR / "metrics.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


# ----------------------------
# Recommenders
# ----------------------------
def recommend_popular(df: pd.DataFrame, top_n: int = 12, filters=None) -> pd.DataFrame:
    x = df.copy()
    if filters:
        x = apply_filters(x, filters)
    x = x.sort_values("weighted_score", ascending=False).head(top_n)
    x["reason"] = "Top by weighted_score"
    x["sim"] = np.nan
    return x


def recommend_tfidf(
    df: pd.DataFrame,
    tfidf_matrix,
    title: str,
    top_n: int = 12,
    filters=None
) -> pd.DataFrame:
    # ✅ align df & matrix để tránh out-of-bounds
    n = min(len(df), tfidf_matrix.shape[0])
    df0 = df.iloc[:n].reset_index(drop=True)
    M = tfidf_matrix[:n]

    idx = find_title_idx(df0, title)
    if idx is None or idx >= n:
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



def recommend_sbert(
    df: pd.DataFrame,
    emb,
    title: str,
    top_n: int = 12,
    filters=None
) -> pd.DataFrame:
    # ✅ align df & embedding để tránh out-of-bounds
    n = min(len(df), emb.shape[0])
    df0 = df.iloc[:n].reset_index(drop=True)
    E = emb[:n]

    idx = find_title_idx(df0, title)
    if idx is None or idx >= n:
        return pd.DataFrame()

    q = E[idx]
    # cosine sim (giả định E đã normalized; nếu chưa thì vẫn chạy ổn cho ranking)
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



def hybrid_rank(df: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    # combine by normalized sim + popularity/weighted_score
    if a.empty and b.empty:
        return pd.DataFrame()
    base = df.copy()

    score = np.zeros(len(base), dtype=np.float32)

    def add_sim(d, w):
        if d is None or d.empty or "sim" not in d.columns:
            return
        s = d.set_index("title")["sim"]
        # normalize
        v = s.values
        if len(v) > 0:
            mn, mx = float(np.min(v)), float(np.max(v))
            if mx > mn:
                s = (s - mn) / (mx - mn)
            else:
                s = s * 0.0
        for t, val in s.items():
            # match first occurrence
            idxs = base.index[base["title"] == t].tolist()
            if idxs:
                score[idxs[0]] += w * float(val)

    add_sim(a, 0.45)
    add_sim(b, 0.45)

    # add popularity/quality signal
    if "weighted_score" in base.columns:
        ws = base["weighted_score"].astype(float).fillna(0.0).values
        ws = (ws - ws.min()) / (ws.max() - ws.min() + 1e-9)
        score += 0.10 * ws

    base = base.copy()
    base["sim"] = score
    base = base.sort_values("sim", ascending=False).head(top_n)
    base["reason"] = "Hybrid (TF-IDF + SBERT + weighted_score)"
    return base


# ----------------------------
# Context-aware + realtime user profile
# ----------------------------
@dataclass
class Filters:
    genre_include: List[str]
    type_include: List[str]
    min_score: float
    exclude_watched: bool


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    x = df.copy()

    if f.min_score is not None and "score_filled" in x.columns:
        x = x[x["score_filled"].astype(float) >= float(f.min_score)]

    if f.type_include:
        if "type" in x.columns:
            x = x[x["type"].fillna("").astype(str).isin(f.type_include)]

    if f.genre_include:
        want = set(f.genre_include)
        x = x[x["genres"].fillna("").apply(lambda s: len(want & set(explode_genres(s))) > 0)]

    # exclude watched handled outside (needs user history)
    return x


def build_user_profile_vector(df: pd.DataFrame, emb: Optional[np.ndarray], liked_titles: List[str]) -> Optional[np.ndarray]:
    if emb is None:
        return None
    idxs = []
    for t in liked_titles:
        i = find_title_idx(df, t)
        if i is not None:
            idxs.append(i)
    if not idxs:
        return None
    v = emb[idxs].mean(axis=0)
    # already normalized embeddings -> renormalize
    n = np.linalg.norm(v) + 1e-9
    return (v / n).astype(np.float32)


def recommend_from_profile_realtime(
    df: pd.DataFrame,
    emb: Optional[np.ndarray],
    profile_vec: Optional[np.ndarray],
    top_n: int,
    filters: Optional[Filters],
    exclude_titles: Optional[set]
) -> pd.DataFrame:
    if emb is None or profile_vec is None:
        return pd.DataFrame()

    sims = emb @ profile_vec
    order = np.argsort(-sims)
    cand = df.iloc[order].copy()
    cand["sim"] = sims[order]
    cand["reason"] = "Realtime profile (mean of liked items embeddings)"

    if exclude_titles:
        cand = cand[~cand["title"].isin(exclude_titles)]

    if filters:
        cand = apply_filters(cand, filters)

    return cand.head(top_n).copy()


def explain_row(base_title: str, row: pd.Series) -> str:
    bg = set(explode_genres(row.get("genres", "")))
    bt = set(explode_genres(base_title)) if False else set()
    # keep simple & safe
    g = ", ".join(list(bg)[:3]) if bg else "N/A"
    return f"Genres: {g}"


def find_title_idx(df: pd.DataFrame, title: str) -> Optional[int]:
    if not title:
        return None
    # exact first
    m = df.index[df["title"].astype(str) == str(title)].tolist()
    if m:
        return int(m[0])
    # case-insensitive fallback
    t = str(title).strip().lower()
    m2 = df.index[df["title"].astype(str).str.lower() == t].tolist()
    if m2:
        return int(m2[0])
    return None


# ----------------------------
# UI components
# ----------------------------
def movie_card(row: pd.Series, show_actions: bool, user_key: str):
    cols = st.columns([1, 2], gap="small")
    with cols[0]:
        url = safe_http_url(row.get("image_url", None))
        if url:
            st.image(url, use_container_width=True)
        else:
            st.info("No image")

    with cols[1]:
        st.subheader(row.get("title", "Untitled"))
        st.caption(row.get("url", ""))
        g = row.get("genres", "")
        st.write(f"**Genres:** {g if g else 'N/A'}")
        st.write(f"**Type:** {row.get('type','')} ")
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
                if st.button("👍 Like", key=f"like_{user_key}_{row.get('title','')}", use_container_width=True):
                    add_history_event(user_key, "like", row.get("title",""))
                    st.rerun()
            with c2:
                if st.button("👀 Watched", key=f"watched_{user_key}_{row.get('title','')}", use_container_width=True):
                    add_history_event(user_key, "watched", row.get("title",""))
                    st.session_state["page"] = "Detail"
                    st.session_state["selected_item"] = row["anime_id"]
                    st.rerun()
            with c3:
                if st.button("👎 Dislike", key=f"dislike_{user_key}_{row.get('title','')}", use_container_width=True):
                    add_history_event(user_key, "dislike", row.get("title",""))
                    st.rerun()

def page_detail(df, tfidf_matrix, tfidf, emb):
    anime_id = st.session_state.get("selected_item")
    if anime_id is None:
        st.warning("No item selected.")
        return

    row = df[df["anime_id"] == anime_id].iloc[0]

    # --------- Header ----------
    st.title(row["title"])
    st.image(row["image_url"], width=300)

    st.markdown(f"**Score:** {row['score_filled']:.2f}")
    st.markdown(f"**Genres:** {row['genres']}")
    st.markdown(row["synopsis"])

    st.markdown("---")
    st.subheader("🎯 Recommended based on this anime")

    # --------- Recommendation ----------
    recs = recommend_from_item(
        df=df,
        item_id=anime_id,
        tfidf=tfidf,
        tfidf_matrix=tfidf_matrix,
        emb=emb,
        top_k=12
    )

    if recs.empty:
        st.info("No recommendations found.")
        return

    cols = st.columns(4)
    for i, (_, r) in enumerate(recs.iterrows()):
        with cols[i % 4]:
            st.image(r["image_url"], use_container_width=True)
            st.caption(r["title"])

def recommend_from_item(df, item_id, tfidf, tfidf_matrix, emb, top_k=12):
    if item_id not in df["anime_id"].values:
        return pd.DataFrame()

    idx = df.index[df["anime_id"] == item_id][0]

    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sims[idx] = 0  # không recommend chính nó

    top_idx = sims.argsort()[::-1][:top_k]
    return df.iloc[top_idx]

def add_history_event(user_key: str, event: str, title: str):
    h = load_history()
    u = h.get(user_key, {"events": [], "context": {}})
    u["events"].append({"event": event, "title": title})
    h[user_key] = u
    save_history(h)


def get_user_events(user_key: str) -> List[Dict]:
    h = load_history()
    return h.get(user_key, {}).get("events", [])


def get_user_titles(user_key: str, event_name: str) -> List[str]:
    ev = get_user_events(user_key)
    return [e["title"] for e in ev if e.get("event") == event_name and e.get("title")]


# ----------------------------
# App
# ----------------------------
def sidebar_filters(df: pd.DataFrame) -> Filters:
    st.sidebar.markdown("### Filters (Context-aware)")
    all_genres = sorted({g for s in df["genres"].fillna("") for g in explode_genres(s)})
    all_types = sorted(set(df["type"].fillna("").astype(str).unique()))

    genre_include = st.sidebar.multiselect("Preferred genres", all_genres, default=[])
    type_include = st.sidebar.multiselect("Type", all_types, default=[])

    min_score = st.sidebar.slider("Min score", 0.0, 10.0, 7.0, 0.1)

    exclude_watched = st.sidebar.checkbox("Exclude watched", value=True)

    return Filters(
        genre_include=genre_include,
        type_include=type_include,
        min_score=min_score,
        exclude_watched=exclude_watched
    )


def page_home(df: pd.DataFrame, user_key: str, filters: Filters):
    st.title("🎌 Anime Recommender (Top 15,000)")
    st.write(
        "Models: **Popularity/Weighted**, **TF-IDF**, **SBERT Embeddings**, **Hybrid**. "
        "Có **Realtime profile**, **User history**, **Context-aware**."
    )
    # SECTION 1: Top picks (always show something)
    st.subheader("🔥 Top picks (Weighted Score)")
    top = recommend_popular(df, top_n=12, filters=filters)

    # Exclude watched if bật
    if filters.exclude_watched:
        watched = set(get_user_titles(user_key, "watched"))
        if watched:
            top = top[~top["title"].isin(watched)]

    if top.empty:
        st.warning("No items to show (filters too strict). Try relaxing filters in sidebar.")
    else:
        for _, r in top.iterrows():
            st.divider()
            movie_card(r, show_actions=True, user_key=user_key)

    m = load_metrics()
    if m:
        with st.expander("Offline Evaluation Metrics (from Kaggle)"):
            st.json(m)


def page_explore(df: pd.DataFrame, user_key: str, filters: Filters):
    st.header("🔎 Explore")
    q = st.text_input("Search title contains", "")
    x = df.copy()
    if q.strip():
        x = x[x["title"].astype(str).str.contains(q.strip(), case=False, na=False)]

    x = apply_filters(x, filters)

    st.write(f"Found: {len(x)} items")
    show = x.sort_values("weighted_score", ascending=False).head(12)
    for _, r in show.iterrows():
        st.divider()
        movie_card(r, show_actions=True, user_key=user_key)


def page_recommend(df: pd.DataFrame, tfidf, tfidf_matrix, emb, user_key: str, filters: Filters):
    st.header("✨ Recommend by a title")
    titles = df["title"].astype(str).tolist()
    base = st.selectbox("Pick a base anime", options=titles, index=0)

    algo = st.radio("Algorithm", ["Hybrid", "SBERT", "TF-IDF", "Popular"], horizontal=True)
    top_n = st.slider("Top N", 5, 30, 12)

    exclude_titles = set()
    if filters.exclude_watched:
        exclude_titles |= set(get_user_titles(user_key, "watched"))

    if algo == "Popular":
        rec = recommend_popular(df, top_n=top_n, filters=filters)
        # sau khi có rec
        if rec is None or rec.empty:
            st.warning("No recommendations (check filters / embeddings availability).")
            return

        # lọc watched (an toàn)
        if exclude_titles and "title" in rec.columns:
            rec = rec[~rec["title"].isin(exclude_titles)]
    elif algo == "TF-IDF":
        rec = recommend_tfidf(df, tfidf_matrix, base, top_n=top_n, filters=filters)
        # sau khi có rec
        if rec is None or rec.empty:
            st.warning("No recommendations (check filters / embeddings availability).")
            return

        # lọc watched (an toàn)
        if exclude_titles and "title" in rec.columns:
            rec = rec[~rec["title"].isin(exclude_titles)]
    elif algo == "SBERT":
        rec = recommend_sbert(df, emb, base, top_n=top_n, filters=filters)
        # sau khi có rec
        if rec is None or rec.empty:
            st.warning("No recommendations (check filters / embeddings availability).")
            return

        # lọc watched (an toàn)
        if exclude_titles and "title" in rec.columns:
            rec = rec[~rec["title"].isin(exclude_titles)]
    else:
        a = recommend_tfidf(df, tfidf_matrix, base, top_n=200, filters=filters)
        b = recommend_sbert(df, emb, base, top_n=200, filters=filters)
        rec = hybrid_rank(df, a, b, top_n=top_n)
        # sau khi có rec
        if rec is None or rec.empty:
            st.warning("No recommendations (check filters / embeddings availability).")
            return

        # lọc watched (an toàn)
        if exclude_titles and "title" in rec.columns:
            rec = rec[~rec["title"].isin(exclude_titles)]

    if rec.empty:
        st.warning("No recommendations (check filters / embeddings availability).")
        return

    st.subheader("Results")
    for _, r in rec.iterrows():
        st.divider()
        movie_card(r, show_actions=True, user_key=user_key)


def page_realtime_profile(df: pd.DataFrame, emb, user_key: str, filters: Filters):
    st.header("⚡ Realtime Recommendation (User Profile)")
    st.caption("Profile vector = mean embedding của các anime bạn Like (cập nhật realtime).")

    liked = get_user_titles(user_key, "like")
    watched = set(get_user_titles(user_key, "watched")) if filters.exclude_watched else set()

    st.write(f"Liked items: {len(liked)}")
    if liked:
        st.write(", ".join(liked[:10]) + (" ..." if len(liked) > 10 else ""))

    if emb is None:
        st.warning("SBERT embeddings not found. Hãy chạy notebook Kaggle và copy artifacts/sbert_embeddings.npy về local.")
        return

    profile_vec = build_user_profile_vector(df, emb, liked)
    if profile_vec is None:
        st.info("Bạn chưa Like gì. Hãy Like vài anime ở tab Explore/Recommend để hệ realtime hoạt động.")
        return

    top_n = st.slider("Top N", 5, 30, 12, key="rt_topn")
    rec = recommend_from_profile_realtime(df, emb, profile_vec, top_n, filters, exclude_titles=watched)

    if rec.empty:
        st.warning("No results (filters too strict).")
        return

    for _, r in rec.iterrows():
        st.divider()
        movie_card(r, show_actions=True, user_key=user_key)


def page_history(df: pd.DataFrame, user_key: str):
    st.header("🧾 My History (Saved)")
    ev = get_user_events(user_key)
    if not ev:
        st.info("Chưa có lịch sử. Hãy Like/Watched/Dislike vài anime.")
        return

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
    st.header("📊 EDA Figures (from Kaggle)")
    figs = [
        ("Score Distribution", FIG_DIR / "score_distribution.png"),
        ("Top Genres", FIG_DIR / "top_genres.png"),
        ("Top Weighted Score", FIG_DIR / "top_weighted_score.png"),
        ("Correlation Heatmap", FIG_DIR / "correlation_heatmap.png"),
    ]
    for name, p in figs:
        st.subheader(name)
        if p.exists():
            st.image(str(p), use_container_width=True)
        else:
            st.warning(f"Missing: {p.name} (hãy chạy notebook Kaggle rồi copy artifacts/figures về local)")


def main():
    st.set_page_config(page_title="Anime Recommender", layout="wide")

    # --- init session state ---
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"
    if "selected_item" not in st.session_state:
        st.session_state["selected_item"] = None

    # --- load resources ---
    df = load_data()
    tfidf, tfidf_matrix, emb, emb_meta = load_models()

    # --- sidebar ---
    st.sidebar.title("🎛️ Navigation")
    user_key = st.sidebar.text_input("User ID", value="khiem", help="Dùng để lưu lịch sử (JSON).")

    filters = sidebar_filters(df)

    nav = st.sidebar.radio(
        "Go to",
        ["Home", "Explore", "Recommend", "Realtime Profile", "My History", "EDA"],
        index=["Home", "Explore", "Recommend", "Realtime Profile", "My History", "EDA"].index(
            st.session_state.get("page", "Home") if st.session_state.get("page") != "Detail" else "Home"
        )
    )

    # keep session page in sync with sidebar (unless we're in Detail)
    if st.session_state.get("page") != "Detail":
        st.session_state["page"] = nav

    # --- DETAIL has priority ---
    if st.session_state.get("page") == "Detail":
        # optional: a quick back button
        if st.sidebar.button("⬅ Back"):
            st.session_state["page"] = "Recommend"
            st.session_state["selected_item"] = None
            st.rerun()

        page_detail(df, tfidf_matrix, tfidf, emb)
        return

    # --- normal pages ---
    if nav == "Home":
        page_home(df, user_key, filters)
        with st.expander("SBERT embedding info"):
            st.write(emb_meta if emb_meta else "No embedding_meta.json")

    elif nav == "Explore":
        page_explore(df, user_key, filters)

    elif nav == "Recommend":
        page_recommend(df, tfidf, tfidf_matrix, emb, user_key, filters)

    elif nav == "Realtime Profile":
        page_realtime_profile(df, emb, user_key, filters)

    elif nav == "My History":
        page_history(df, user_key)

    else:
        page_eda()



if __name__ == "__main__":
    main()