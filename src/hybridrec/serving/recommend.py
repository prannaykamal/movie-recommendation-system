"""One entry point for the UI: a user (or picked movies) -> top 10 with explanations.

Uses the s3 artifacts (every model fit on R+L+V). The content and iALS sources
score in memory; the embedding source searches Qdrant for its top-k and fetches
candidate vectors from Qdrant to score the other candidates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hybridrec.config import Config
from hybridrec.data.dataset import Dataset, Histories, genre_matrix, mask_seen, topk
from hybridrec.data.split import stage_frames
from hybridrec.rerank import train as rt
from hybridrec.rerank.features import ItemStats, ItemTable, build_features, last_liked
from hybridrec.retrieval.merge import SOURCES
from hybridrec.retrieval.registry import load_sources, text_vectors
from hybridrec.vectordb import qdrant_store as qs

STAGE = "s3"
SOURCE_LABELS = {"cb": "CB", "cf": "CF", "emb": "EMB"}


@dataclass
class Result:
    top: pd.DataFrame  # rank, title, year, genres, score, found_by
    candidates: pd.DataFrame  # every candidate with source scores/ranks and rerank score


class Recommender:
    def __init__(self, cfg: Config, client=None):
        self.cfg = cfg
        self.ds = Dataset.load(cfg)
        fit, self.test = stage_frames(self.ds.ratings, STAGE)
        self.fit = fit
        self.hist_all = Histories(fit, self.ds.items)
        vectors = text_vectors(self.ds)
        srcs = load_sources(self.ds, STAGE, self.hist_all, ("cb", "cf", cfg.emb_source))
        self.cb, self.cf, self.emb = srcs["cb"], srcs["cf"], srcs[cfg.emb_source]
        self.items_table = ItemTable.build(self.ds, vectors)
        self.stats = ItemStats.from_hist(self.hist_all)
        tt_path = cfg.stage_dir(STAGE) / "twotower_items.npy"
        self.tt_vecs = np.load(tt_path) if tt_path.exists() else None
        self.kind = rt.best_kind(cfg)
        self.model, self.cols, self.norm = rt.load(cfg.artifacts_dir / f"reranker_{self.kind}.pt", cfg, vectors,
                                                   genre_matrix(self.ds.movies))
        self.client = client or qs.get_client()
        kind = "movie_text" if cfg.emb_source == "text" else "twotower"
        self.emb_collection = qs.collection_name(cfg.name, kind, None if kind == "movie_text" else STAGE)
        self.movies = self.ds.movies.set_index("movie_id")

    # ---- histories ------------------------------------------------------
    def user_history(self, user_id: int) -> Histories:
        return Histories(self.fit[self.fit["user_id"] == user_id], self.ds.items)

    def picks_history(self, movie_ids) -> Histories:
        return Histories.from_picks(movie_ids, self.ds.items)

    def test_liked(self, user_id: int) -> set[int]:
        t = self.test[(self.test["user_id"] == user_id) & (self.test["rating"] >= self.cfg.like_threshold)]
        return set(t["movie_id"].tolist())

    # ---- candidate stage ------------------------------------------------
    def _emb_centroids(self, hist: Histories) -> np.ndarray:
        if self.cfg.emb_source == "text":
            return self.emb.user_centroids(hist, 0)
        return self.emb.user_vectors(hist, np.array([0]))

    def candidates(self, hist: Histories) -> pd.DataFrame:
        cfg, rows = self.cfg, np.array([0])
        seen_ids = self.ds.items.movie_ids[hist.get(0)[0]]
        raw = {"cb": self.cb.score(hist, rows)[0], "cf": self.cf.score(hist, rows)[0]}
        top = {}
        for src in ("cb", "cf"):
            idx, vals = topk(mask_seen(raw[src][None, :].copy(), hist, rows), cfg.k_for(src))
            top[src] = self.ds.items.movie_ids[idx[0][np.isfinite(vals[0])]]

        # Embedding source through Qdrant: one query per profile vector, best score wins.
        cents = self._emb_centroids(hist)
        best: dict[int, float] = {}
        for c in cents:
            for mid, score, _ in qs.search(self.client, self.emb_collection, c, cfg.k_emb, seen_ids):
                best[mid] = max(best.get(mid, -np.inf), score)
        top["emb"] = np.array(sorted(best, key=lambda m: -best[m])[: cfg.k_emb], dtype=np.int64)

        union = np.unique(np.concatenate([top[s] for s in SOURCES]))
        idx = self.ds.items.index(union)
        vecs = qs.fetch_vectors(self.client, self.emb_collection, union)
        V = np.stack([vecs[m] for m in union])
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        out = pd.DataFrame({"user_id": hist.users[0], "movie_id": union, "item_idx": idx})
        out["cb_score"], out["cf_score"] = raw["cb"][idx], raw["cf"][idx]
        out["emb_score"] = (V @ cents.T).max(axis=1)
        for src in SOURCES:
            rank = {m: r + 1 for r, m in enumerate(top[src])}
            out[f"{src}_rank"] = np.array([rank.get(m, 0) for m in union], dtype=np.int16)
        out["n_sources"] = sum((out[f"{s}_rank"] > 0).astype(np.int8) for s in SOURCES)
        return out

    # ---- reranking ------------------------------------------------------
    def recommend(self, hist: Histories, n: int = 10) -> Result:
        import torch

        cands = self.candidates(hist).assign(label=0, rated=0)
        feats = build_features(cands, hist, self.items_table, self.stats, self.tt_vecs,
                               self.cfg.like_threshold, self.cfg.rr_recent_sims)
        X = self.norm(feats[self.cols].to_numpy(np.float32))
        seq = last_liked(hist, np.array([0]), self.cfg.like_threshold, self.cfg.rr_history)
        with torch.no_grad():
            scores = self.model(torch.as_tensor(X), torch.as_tensor(feats["item_idx"].to_numpy()),
                                torch.zeros(len(X), dtype=torch.long), torch.as_tensor(seq)).numpy()
        cands = cands.merge(feats[["movie_id"]].assign(rerank_score=scores), on="movie_id")
        cands = cands.sort_values("rerank_score", ascending=False).reset_index(drop=True)
        cands["final_rank"] = np.arange(1, len(cands) + 1)
        return Result(self._describe(cands.head(n)), cands)

    def _describe(self, top: pd.DataFrame) -> pd.DataFrame:
        m = self.movies.loc[top["movie_id"]]
        found = [", ".join(SOURCE_LABELS[s] for s in SOURCES if r[f"{s}_rank"] > 0) for _, r in top.iterrows()]
        return pd.DataFrame({
            "rank": top["final_rank"].to_numpy(),
            "movie_id": top["movie_id"].to_numpy(),
            "title": m["title"].to_numpy(),
            "year": m["year"].astype("Int64").astype(str).replace("<NA>", "").to_numpy(),
            "genres": [", ".join(g) for g in m["genres"]],
            "score": top["rerank_score"].round(3).to_numpy(),
            "found by": found,
        })

    def describe_movies(self, movie_ids) -> pd.DataFrame:
        m = self.movies.loc[list(movie_ids)]
        return pd.DataFrame({"title": m["title"].to_numpy(),
                             "year": m["year"].astype("Int64").astype(str).replace("<NA>", "").to_numpy(),
                             "genres": [", ".join(g) for g in m["genres"]]}, index=m.index)
