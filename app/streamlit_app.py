"""Streamlit demo: streamlit run app/streamlit_app.py  (run `python -m recsys.cli all` first)."""
import numpy as np
import pandas as pd
import streamlit as st

from recsys.models.hybrid import DynamicHybridRecommender
from recsys.service import RecommenderService

st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_service() -> RecommenderService:
    return RecommenderService()


try:
    svc = load_service()
except FileNotFoundError:
    st.error("Artifacts not found. Run `python -m recsys.cli all` first.")
    st.stop()

# ---- sidebar: blend controls -------------------------------------------------------------------
st.sidebar.header("Blend settings")
k = st.sidebar.slider("Top-K", 5, 30, 10)
alpha_max = st.sidebar.slider("α_max (collaborative ceiling)", 0.0, 1.0, float(svc.params["alpha_max"]), 0.05)
gamma = st.sidebar.slider("γ (collaborative ramp-up)", 0.01, 1.0, float(svc.params["gamma"]), 0.01)
pop_eta = st.sidebar.slider("η (popularity fade-out)", 0.05, 1.0, float(svc.params.get("pop_eta") or 1.0), 0.05)
pop_penalty = st.sidebar.slider("Discovery (popularity penalty λ)", 0.0, 0.4, float(svc.params.get("pop_penalty", 0.0)),
                                0.05, help="Pushes down blockbusters. 0.1–0.3 widens catalogue coverage by 10–40% "
                                           "for a 0–4% drop in offline accuracy; beyond ~0.4 accuracy collapses.")
st.sidebar.caption("Score = β·popular + (1 − β)·[α·collab + (1 − α)·content]")
st.sidebar.caption("α(u) = α_max·(1 − e^(−γ·N_u)),  β(u) = e^(−η·N_u)")
n_axis = np.arange(0, 41)
_h = DynamicHybridRecommender(alpha_max, gamma, pop_eta)
_a, _b = _h.calculate_alpha(n_axis), _h.calculate_beta(n_axis)
st.sidebar.line_chart(pd.DataFrame({"popular": _b, "collab": (1 - _b) * _a, "content": (1 - _b) * (1 - _a)},
                                   index=pd.Index(n_axis, name="N_u")), height=200)

SCORE_COLS = {
    "title": st.column_config.TextColumn("Title", width="large"),
    "genres": st.column_config.TextColumn("Genres"),
    "s_hybrid": st.column_config.ProgressColumn("Hybrid", min_value=0.0, max_value=1.0, format="%.3f"),
    "s_collab": st.column_config.ProgressColumn("Collab (SVD+iALS)", min_value=0.0, max_value=1.0, format="%.3f"),
    "s_content": st.column_config.ProgressColumn("Content", min_value=0.0, max_value=1.0, format="%.3f"),
    "s_popular": st.column_config.ProgressColumn("Popular", min_value=0.0, max_value=1.0, format="%.3f"),
}


def show_recs(recs: pd.DataFrame, w: dict, n_u: int) -> None:
    a, b = w["alpha"], w["beta"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ratings (N_u)", n_u)
    c2.metric("Popular share β", f"{b:.2f}")
    c3.metric("Collab share", f"{(1 - b) * a:.2f}")
    c4.metric("Content share", f"{(1 - b) * (1 - a):.2f}")
    if n_u == 0:
        st.info("No ratings yet: showing the most popular movies, boosted by your genre picks (if any).")
    elif b >= 0.2:
        st.caption("Few ratings so far, so popularity (and your genre picks) still carry a large share.")
    st.dataframe(recs[list(SCORE_COLS)], column_config=SCORE_COLS, hide_index=True, use_container_width=True)
    with st.expander("Plot summaries"):
        for _, r in recs.iterrows():
            plot = r["plot_summary"]
            snippet = (plot[:400] + "…") if isinstance(plot, str) and len(plot) > 400 else (plot or "_No CMU plot available._")
            st.markdown(f"**{r['title']}** — {snippet}")


st.title("🎬 Hybrid Movie Recommender")
st.caption("MovieLens 1M ratings + CMU Wikipedia plots · popularity-gated content (TF-IDF metadata + Sentence-BERT "
           "plots) and collaborative filtering (mean-centred SVD + implicit ALS), blended by interaction-driven weights; "
           "a popularity prior carries new users and fades out as they rate.")

tab_existing, tab_new = st.tabs(["Existing user", "New user (cold start)"])

with tab_existing:
    user_id = st.number_input("MovieLens user ID", min_value=int(svc.index.user_ids.min()),
                              max_value=int(svc.index.user_ids.max()), value=1, step=1)
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Top-rated history")
        st.dataframe(svc.user_history(int(user_id))[["title", "rating"]].head(15), hide_index=True,
                     use_container_width=True)
    with right:
        st.subheader("Recommendations")
        recs, weights = svc.recommend_existing(int(user_id), k, alpha_max, gamma, pop_eta, pop_penalty)
        show_recs(recs, weights, svc.R[svc.index.uidx(int(user_id))].nnz)

with tab_new:
    ratings: dict = st.session_state.setdefault("new_ratings", {})
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Rate some movies")
        query = st.text_input("Search title", placeholder="e.g. matrix, toy story, godfather")
        hits = svc.search(query) if query else pd.DataFrame(columns=["movieId", "title"])
        if len(hits):
            choice = st.selectbox("Movie", hits["movieId"], format_func=dict(zip(hits["movieId"], hits["title"])).get)
            stars = st.slider("Your rating", 1, 5, 5)
            if st.button("Add rating", type="primary"):
                ratings[int(choice)] = float(stars)
                st.rerun()
        elif query:
            st.caption("No matches.")

        genres = st.multiselect("Favourite genres (shape picks while you have few ratings)", svc.genres)

        if ratings:
            titles = svc.items.set_index("movieId")["title"]
            st.markdown("**Your ratings**")
            for mid, r in list(ratings.items()):
                c1, c2 = st.columns([5, 1])
                c1.write(f"{'★' * int(r)} {titles[mid]}")
                if c2.button("", icon=":material/close:", key=f"rm-{mid}", help="Remove rating"):
                    del ratings[mid]
                    st.rerun()
            if st.button("Clear all"):
                ratings.clear()
                st.rerun()
    with right:
        st.subheader("Recommendations")
        recs, weights = svc.recommend_new(ratings, genres, k, alpha_max, gamma, pop_eta, pop_penalty)
        show_recs(recs, weights, len(ratings))
