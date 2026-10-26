"""Embedding source v1 (no training): compare the user's taste with movie text vectors.

The user profile comes from their last ``recent`` liked movies, with more weight
on recent ones. For users with many liked movies, the liked vectors are grouped
into a few clusters (spherical k-means) and each movie is scored by its best
cluster, so a horror + animation fan isn't averaged into neither.
"""
from __future__ import annotations

import numpy as np

from hybridrec.data.dataset import Histories


def _recent_liked(hist: Histories, p: int, threshold: float, recent: int) -> tuple[np.ndarray, np.ndarray]:
    items, ratings, _ = hist.get(p)
    liked = ratings >= threshold
    if liked.any():
        items, ratings = items[liked], np.ones(liked.sum(), dtype=np.float32)
    else:
        ratings = ratings / 5.0  # no liked movies yet: fall back to everything, weighted by rating
    return items[-recent:], ratings[-recent:].astype(np.float32)


def spherical_kmeans(V: np.ndarray, w: np.ndarray, k: int, iters: int = 10, seed: int = 0) -> np.ndarray:
    """Weighted k-means on unit vectors; returns (k, d) unit centroids."""
    rng = np.random.default_rng(seed)
    # Farthest-point initialisation, starting from the most recent item.
    centers = [V[-1]]
    for _ in range(1, k):
        sims = np.max(np.stack([V @ c for c in centers]), axis=0)
        centers.append(V[int(np.argmin(sims + 1e-6 * rng.random(len(V))))])
    C = np.stack(centers)
    for _ in range(iters):
        assign = np.argmax(V @ C.T, axis=1)
        for j in range(k):
            m = assign == j
            if m.any():
                c = (V[m] * w[m, None]).sum(axis=0)
                C[j] = c / (np.linalg.norm(c) + 1e-9)
    return C


class TextEmbeddingV1:
    name = "text"

    def __init__(self, item_vectors: np.ndarray, like_threshold=4.0, recent=50, half_life=20.0,
                 clusters=3, min_for_clusters=12):
        self.E = item_vectors.astype(np.float32)
        self.like_threshold, self.recent, self.half_life = like_threshold, recent, half_life
        self.clusters, self.min_for_clusters = clusters, min_for_clusters

    def fit(self, hist: Histories) -> "TextEmbeddingV1":
        return self  # stateless: item vectors are fixed

    def user_centroids(self, hist: Histories, p: int) -> np.ndarray:
        if p < 0:
            return np.zeros((1, self.E.shape[1]), dtype=np.float32)
        items, w = _recent_liked(hist, p, self.like_threshold, self.recent)
        if len(items) == 0:
            return np.zeros((1, self.E.shape[1]), dtype=np.float32)
        age = np.arange(len(items))[::-1]  # 0 = most recent
        w = w * 0.5 ** (age / self.half_life)
        V = self.E[items]
        if self.clusters > 1 and len(items) >= self.min_for_clusters:
            return spherical_kmeans(V, w, self.clusters)
        c = (V * w[:, None]).sum(axis=0)
        return (c / (np.linalg.norm(c) + 1e-9))[None, :]

    def user_vector(self, hist: Histories, p: int) -> np.ndarray:
        """Single profile vector (weighted mean), used for the vector-DB query."""
        old = self.clusters
        self.clusters = 1
        try:
            return self.user_centroids(hist, p)[0]
        finally:
            self.clusters = old

    def score(self, hist: Histories, rows: np.ndarray) -> np.ndarray:
        cents = [self.user_centroids(hist, p) for p in rows]
        sizes = np.array([len(c) for c in cents])
        S = np.vstack(cents) @ self.E.T
        out = np.maximum.reduceat(S, np.r_[0, np.cumsum(sizes)[:-1]], axis=0)
        return out.astype(np.float32)
