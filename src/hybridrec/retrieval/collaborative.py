"""Collaborative source: implicit ALS on 'watched' signals weighted by rating.

Every rating counts as an interaction with confidence 1 + alpha * rating / 5.
Only movies with at least ``min_item_ratings`` ratings in the fit window get
factors; other movies score NaN (missing) so the reranker can tell them apart.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

from hybridrec.data.dataset import Histories

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


class ImplicitALS:
    name = "cf"

    def __init__(self, factors=64, regularization=0.05, alpha=10.0, iterations=15, min_item_ratings=5, seed=42):
        self.params = dict(factors=factors, regularization=regularization, alpha=alpha,
                           iterations=iterations, min_item_ratings=min_item_ratings, seed=seed)
        self.user_ids: np.ndarray | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None  # (n_items_universe, f); zero rows where missing
        self.has_item: np.ndarray | None = None
        self._model = None

    def _confidence(self, ratings: np.ndarray) -> np.ndarray:
        return 1.0 + self.params["alpha"] * ratings / 5.0

    def fit(self, hist: Histories) -> "ImplicitALS":
        from implicit.cpu.als import AlternatingLeastSquares

        n_items = len(hist.items)
        self.has_item = hist.item_counts >= self.params["min_item_ratings"]
        keep = self.has_item[hist.item_idx]
        C = hist.csr(np.where(keep, self._confidence(hist.rating), 0.0))
        C.eliminate_zeros()
        model = AlternatingLeastSquares(
            factors=self.params["factors"], regularization=self.params["regularization"], alpha=1.0,
            iterations=self.params["iterations"], random_state=self.params["seed"],
        )
        with threadpool_limits(1, "blas"):
            model.fit(C, show_progress=False)
        self._model = model
        self.user_ids = hist.users.copy()
        self.user_factors = np.asarray(model.user_factors, dtype=np.float32)
        items = np.asarray(model.item_factors, dtype=np.float32)
        self.item_factors = np.where(self.has_item[:, None], items[:n_items], 0.0).astype(np.float32)
        return self

    def user_vectors(self, hist: Histories, rows: np.ndarray) -> np.ndarray:
        """Learned factors for users seen in fit; solved on the fly for anyone else."""
        out = np.zeros((len(rows), self.item_factors.shape[1]), dtype=np.float32)
        uids = np.where(rows >= 0, hist.users[np.maximum(rows, 0)], -1)
        known = np.clip(np.searchsorted(self.user_ids, uids), 0, len(self.user_ids) - 1)
        is_known = (self.user_ids[known] == uids) & (rows >= 0)
        for b, p in enumerate(rows):
            if is_known[b]:
                out[b] = self.user_factors[known[b]]
            elif p >= 0:
                out[b] = self._fold_in(hist, p)
        return out

    def _fold_in(self, hist: Histories, p: int) -> np.ndarray:
        """Least-squares user factors from one history (the iALS user step)."""
        items, ratings, _ = hist.get(p)
        keep = self.has_item[items]
        items, conf = items[keep], self._confidence(ratings[keep])
        if len(items) == 0:
            return np.zeros(self.item_factors.shape[1], dtype=np.float32)
        Y = self.item_factors
        reg = self.params["regularization"]
        A = Y.T @ Y + reg * np.eye(Y.shape[1], dtype=np.float32)
        Yi = Y[items]
        A += (Yi * (conf - 1.0)[:, None]).T @ Yi
        b = (Yi * conf[:, None]).sum(axis=0)
        return np.linalg.solve(A, b).astype(np.float32)

    def score(self, hist: Histories, rows: np.ndarray) -> np.ndarray:
        S = self.user_vectors(hist, rows) @ self.item_factors.T
        S[:, ~self.has_item] = np.nan
        return S.astype(np.float32)

    def save(self, path) -> None:
        np.savez(path, user_ids=self.user_ids, user_factors=self.user_factors,
                 item_factors=self.item_factors, has_item=self.has_item, **{f"p_{k}": v for k, v in self.params.items()})

    @classmethod
    def load(cls, path) -> "ImplicitALS":
        z = np.load(path)
        obj = cls(**{k[2:]: z[k].item() for k in z.files if k.startswith("p_")})
        obj.user_ids, obj.user_factors = z["user_ids"], z["user_factors"]
        obj.item_factors, obj.has_item = z["item_factors"], z["has_item"]
        return obj
