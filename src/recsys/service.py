"""Load trained artifacts and serve recommendations for existing and new users."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from recsys.config import Paths
from recsys.data.interactions import Index, minmax_rows
from recsys.models.content import ContentEngine
from recsys.models.hybrid import DynamicHybridRecommender


class RecommenderService:
    def __init__(self, artifacts: Path | None = None):
        a = artifacts or Paths().artifacts
        self.params = json.loads((a / "model_params.json").read_text())
        self.items = pd.read_parquet(a / "items.parquet")
        idx = np.load(a / "index.npz")
        self.index = Index(idx["user_ids"], idx["movie_ids"])
        self.R = sparse.load_npz(a / "ratings.npz").tocsr()
        self.collab = joblib.load(a / "collab.joblib")
        self.ials = joblib.load(a / "ials.joblib") if (a / "ials.joblib").exists() else None
        self.ials_weight = self.params.get("ials_weight", 0.0) if self.ials is not None else 0.0
        counts = np.load(a / "item_counts.npy")
        self.content = ContentEngine(sparse.load_npz(a / "meta_vecs.npz"), np.load(a / "plot_emb.npy"),
                                     self.items["has_plot"].to_numpy(), self.params["w_meta"],
                                     self.params["w_plot"], self.params["pos_threshold"],
                                     item_counts=counts if self.params.get("content_pop_gate", True) else None)
        self.popularity = minmax_rows(counts[None, :].astype(np.float32))[0]
        self.genres = sorted({g for gs in self.items["genres"] for g in gs.split("|")})

    def _hybrid(self, alpha_max, gamma, pop_eta) -> DynamicHybridRecommender:
        p = self.params
        return DynamicHybridRecommender(p["alpha_max"] if alpha_max is None else alpha_max,
                                        p["gamma"] if gamma is None else gamma,
                                        p.get("pop_eta") if pop_eta is None else pop_eta)

    def _collab(self, svd: np.ndarray, ials) -> np.ndarray:
        """S_collab = w * iALS + (1 - w) * SVD; `ials` is a zero-arg callable so it is only computed if used."""
        w = self.ials_weight
        return svd if w == 0 else w * ials() + (1 - w) * svd

    def popularity_prior(self, genres: list[str] | None = None) -> np.ndarray:
        """S_popular: most-popular, boosted by selected genres as pop * (1 + genre_match)
        (+13% NDCG@10 over popularity alone in the onboarding simulation)."""
        if not genres:
            return self.popularity
        match = self.items["genres"].str.split("|").apply(lambda gs: len(set(gs) & set(genres)) / len(genres))
        return minmax_rows((self.popularity * (1.0 + match.to_numpy(np.float32)))[None])[0]

    def _recommend(self, row: sparse.csr_matrix, content, collab, popularity, k, alpha_max, gamma, pop_eta):
        """Returns (recommendations with per-engine scores, {"alpha", "beta"}) for one user row."""
        n_u = row.nnz
        h = self._hybrid(alpha_max, gamma, pop_eta)
        scores = h.blend(np.array([n_u]), content[None], collab[None], popularity)[0]
        top = DynamicHybridRecommender.top_k(scores[None, :], [row.indices], k)[0]
        expl = h.explain(n_u, content, collab, top, popularity)
        out = expl.merge(self.items[["idx", "movieId", "title", "genres", "plot_summary"]], on="idx")
        return out, {"alpha": float(h.calculate_alpha(n_u)), "beta": float(h.calculate_beta(n_u))}

    def user_history(self, user_id: int) -> pd.DataFrame:
        row = self.R[self.index.uidx(user_id)]
        hist = pd.DataFrame({"idx": row.indices, "rating": row.data})
        return hist.merge(self.items[["idx", "title", "genres"]], on="idx").sort_values("rating", ascending=False)

    def recommend_existing(self, user_id: int, k: int = 10, alpha_max=None, gamma=None, pop_eta=None):
        u = self.index.uidx(user_id)
        row = self.R[u]
        content = self.content.score(row)[0]
        collab = self._collab(self.collab.score(np.array([u]))[0], lambda: self.ials.score(np.array([u]))[0])
        return self._recommend(row, content, collab, self.popularity, k, alpha_max, gamma, pop_eta)

    def recommend_new(self, ratings: dict[int, float], genres: list[str] | None = None, k: int = 10,
                      alpha_max=None, gamma=None, pop_eta=None):
        """Cold-start path for a user outside the training matrix. `ratings` maps movieId -> stars.
        Collaborative scores come from folding the ratings into the learned SVD and iALS item
        factors. The popularity prior (boosted by any selected genres) is the whole score with no
        ratings and fades out as exp(-pop_eta * N_u)."""
        cols = [self.index.iidx(m) for m in ratings]
        row = sparse.csr_matrix((np.array(list(ratings.values()), dtype=np.float32),
                                 (np.zeros(len(cols), dtype=int), cols)), shape=(1, self.index.n_items))
        if row.nnz:
            content = self.content.score(row)[0]
            collab = self._collab(self.collab.score_new(row)[0], lambda: self.ials.score_new(row)[0])
        else:
            content = collab = np.zeros(self.index.n_items, dtype=np.float32)
        return self._recommend(row, content, collab, self.popularity_prior(genres), k, alpha_max, gamma, pop_eta)

    def search(self, query: str, limit: int = 20) -> pd.DataFrame:
        hits = self.items[self.items["title"].str.contains(query, case=False, regex=False)]
        return hits[["movieId", "title"]].head(limit)
