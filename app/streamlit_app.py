"""Streamlit UI for the hybrid recommender. Text and tables only, no icons.

Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from hybridrec.config import PRESETS, get_config
from hybridrec.eval.metrics import user_metrics
from hybridrec.eval.protocol import latest_results
from hybridrec.retrieval.merge import SOURCES
from hybridrec.serving.recommend import SOURCE_LABELS, Recommender
from hybridrec.vectordb import qdrant_store as qs

st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")


@st.cache_resource
def qdrant():
    # Local mode allows one client per storage folder, so every dataset shares this one.
    return qs.get_client()


@st.cache_resource(show_spinner="Loading the models for this dataset (first time only)...")
def recommender(name: str) -> Recommender:
    return Recommender(get_config(name), client=qdrant())


@st.cache_data
def movie_options(name: str, limit: int = 5000) -> pd.DataFrame:
    rec = recommender(name)
    counts = rec.hist_all.item_counts
    order = np.argsort(-counts)[:limit]
    m = rec.ds.movies.iloc[order]
    year = m["year"].astype("Int64").astype(str).replace("<NA>", "?")
    labels = m["title"].fillna("Untitled") + " (" + year + ")"
    # A few MovieLens ids share one TMDB entry; keep the most-rated one per label.
    return pd.DataFrame({"movie_id": m["movie_id"].to_numpy(), "label": labels.to_numpy()}).drop_duplicates("label")


@st.cache_data
def test_users(name: str) -> list[int]:
    rec = recommender(name)
    liked = rec.test[rec.test["rating"] >= rec.cfg.like_threshold]
    return sorted(liked["user_id"].unique().tolist())


def available_datasets() -> list[str]:
    return [n for n, c in PRESETS.items() if any((c.artifacts_dir / f"reranker_{k}.pt").exists() for k in ("v1", "v2"))]


def show_table(df: pd.DataFrame) -> None:
    st.table(df.reset_index(drop=True).set_index(df.columns[0]))


def recommendations_tab(rec: Recommender, mode: str, user_id: int | None, picks: list[int], details: bool,
                        history: bool) -> None:
    if mode == "Existing user":
        hist = rec.user_history(user_id)
        st.subheader(f"Top 10 for user {user_id}")
    else:
        if len(picks) < 3:
            st.write("Pick at least 3 movies you like in the sidebar.")
            return
        hist = rec.picks_history(picks)
        st.subheader("Top 10 for your picks")

    result = rec.recommend(hist)
    top = result.top.copy()
    liked_test = rec.test_liked(user_id) if mode == "Existing user" else set()
    if mode == "Existing user":
        top["liked in test"] = ["yes" if m in liked_test else "" for m in top["movie_id"]]
    show_table(top.drop(columns=["movie_id"]))
    st.caption("Score is the reranker's raw output: higher is better, and it is only meaningful within this list. "
               "Found by: the candidate sources (content-based, collaborative, embedding) that proposed the movie.")

    if mode == "Existing user":
        st.subheader("Held-out check")
        if liked_test:
            p, r, _, ndcg = user_metrics(top["movie_id"].tolist(), liked_test, 10)
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision@10", f"{p:.2f}")
            c2.metric("Recall@10", f"{r:.2f}")
            c3.metric("NDCG@10", f"{ndcg:.2f}")
            held = rec.describe_movies(sorted(liked_test)).reset_index()
            held["recommended"] = ["yes" if m in set(top["movie_id"]) else "" for m in held["movie_id"]]
            st.write(f"Movies this user rated 4 or higher in the test block ({len(held)}):")
            show_table(held.drop(columns=["movie_id"]))
        else:
            st.write("This user liked nothing in the test block, so there is nothing to check against.")

    if history:
        st.subheader("Watch history")
        items, ratings, _ = hist.get(0)
        h = rec.describe_movies(rec.ds.items.movie_ids[items]).assign(rating=ratings).reset_index(drop=True)
        st.write(f"{len(h)} movies; the 50 most recent are shown, newest first.")
        show_table(h.iloc[::-1].head(50))

    if details:
        st.subheader("Candidate stage")
        cands = result.candidates
        rows = [{"source": SOURCE_LABELS[s], "candidates": int((cands[f"{s}_rank"] > 0).sum()),
                 "found only by this source": int(((cands[f"{s}_rank"] > 0) & (cands["n_sources"] == 1)).sum())}
                for s in SOURCES]
        show_table(pd.DataFrame(rows))
        st.write(f"Union after removing duplicates and watched movies: {len(cands)} candidates. "
                 f"Found by all three sources: {int((cands['n_sources'] == 3).sum())}.")
        if liked_test:
            found = len(liked_test & set(cands["movie_id"]))
            st.write(f"Liked test movies present among the candidates: {found} of {len(liked_test)}.")
        st.write(f"Reranker: {rec.kind}.")


def similar_tab(rec: Recommender, name: str) -> None:
    opts = movie_options(name)
    label = st.selectbox("Movie", opts["label"].tolist(), key="similar_movie")
    space = st.radio("Similarity", ["Story text", "Viewing behaviour (two-tower)"], horizontal=True)
    st.caption("Story text compares plot, tagline and genres. Viewing behaviour compares the two-tower model's "
               "movie vectors, which group movies the same users watched, often from the same era.")
    movie_id = int(opts.loc[opts["label"] == label, "movie_id"].iloc[0])
    if space == "Story text":
        coll = qs.collection_name(name, "movie_text")
    else:
        coll = qs.collection_name(name, "twotower", "s3")
    hits = qs.similar_movies(qdrant(), coll, movie_id, limit=11)
    rows = [{"title": p["title"], "year": p["year"] or "", "genres": ", ".join(p["genres"]),
             "similarity": round(s, 3)} for _, s, p in hits[:10]]
    show_table(pd.DataFrame(rows))


def evaluation_tab(rec: Recommender) -> None:
    res = latest_results(rec.cfg)
    models = ["source:popularity", "source:cb", "source:cf", "source:text", "source:twotower", "rrf",
              "reranker_v1", "reranker_v2"]
    metrics = {"precision@10": "P@10", "recall@10": "R@10", "ndcg@10": "NDCG@10", "coverage": "coverage"}
    for stage, title in (("s3", "Test (s3)"), ("s2", "Validation (s2)")):
        sub = res[(res["stage"] == stage) & res["model"].isin(models) & res["metric"].isin(metrics)]
        if sub.empty:
            continue
        t = sub.pivot(index="model", columns="metric", values="value").reindex(models).dropna(how="all")
        t = t[[m for m in metrics if m in t]].rename(columns=metrics).round(4).reset_index()
        st.subheader(title)
        show_table(t)
    cand = res[res["model"] == "candidates"]
    if not cand.empty:
        st.subheader("Candidate stage")
        show_table(cand.pivot(index="stage", columns="metric", values="value").round(4).reset_index())


def main() -> None:
    st.title("Hybrid Movie Recommender")
    names = available_datasets()
    if not names:
        st.write("No trained pipeline found. Run scripts/run_all.py first.")
        return
    with st.sidebar:
        name = st.selectbox("Dataset", names)
        rec = recommender(name)
        mode = st.radio("Mode", ["Existing user", "New user"])
        user_id, picks = None, []
        if mode == "Existing user":
            user_id = st.selectbox("Test user", test_users(name))
        else:
            opts = movie_options(name)
            chosen = st.multiselect("Movies you like (at least 3)", opts["label"].tolist())
            picks = opts.set_index("label").loc[chosen, "movie_id"].tolist() if chosen else []
        st.write(f"Candidates per source: CB {rec.cfg.k_cb}, CF {rec.cfg.k_cf}, EMB {rec.cfg.k_emb} "
                 f"(fixed: the reranker was trained at these values).")
        history = st.checkbox("Show watch history")
        details = st.checkbox("Show pipeline details")

    tab1, tab2, tab3 = st.tabs(["Recommendations", "Similar movies", "Evaluation"])
    with tab1:
        recommendations_tab(rec, mode, user_id, picks, details, history)
    with tab2:
        similar_tab(rec, name)
    with tab3:
        evaluation_tab(rec)


main()
