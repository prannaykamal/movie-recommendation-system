"""Dynamic interaction-driven score interpolation:

    S_hybrid = alpha(u) * S_collab + (1 - alpha(u)) * S_content,  alpha(u) = alpha_max * (1 - exp(-gamma * N_u))
    S_final  = beta(u) * S_popular + (1 - beta(u)) * S_hybrid,    beta(u)  = exp(-pop_eta * N_u)

The fading popularity prior beta(u) covers new users: at N_u = 0 the score is pure popularity, and
for a handful of ratings (where the user profile is still noise) popularity keeps a large share.
With pop_eta=None, beta is 1 at N_u = 0 and 0 otherwise (popularity is only a cold fallback).
"""
import numpy as np
import pandas as pd


class DynamicHybridRecommender:
    def __init__(self, alpha_max: float = 0.80, gamma: float = 0.15, pop_eta: float | None = None):
        self.alpha_max = alpha_max
        self.gamma = gamma
        self.pop_eta = pop_eta

    def calculate_alpha(self, n_user_interactions):
        """Works on scalars and arrays."""
        return self.alpha_max * (1.0 - np.exp(-self.gamma * np.asarray(n_user_interactions, dtype=np.float64)))

    def calculate_beta(self, n_user_interactions):
        """Weight of the popularity prior. Works on scalars and arrays."""
        n = np.asarray(n_user_interactions, dtype=np.float64)
        if self.pop_eta is None:
            return (n == 0).astype(np.float64)
        return np.exp(-self.pop_eta * n)

    def blend(self, n_u: np.ndarray, content: np.ndarray, collab: np.ndarray,
              popularity: np.ndarray | None = None) -> np.ndarray:
        """Scores for a batch of users (rows). `popularity` is a per-item (or per-row) prior in [0, 1];
        without it, the plain alpha blend is returned."""
        alpha = self.calculate_alpha(n_u).astype(np.float32)[:, None]
        scores = alpha * collab + (1.0 - alpha) * content
        if popularity is None:
            return scores
        beta = self.calculate_beta(n_u).astype(np.float32)[:, None]
        return beta * popularity + (1.0 - beta) * scores

    @staticmethod
    def top_k(scores: np.ndarray, seen: list[np.ndarray] | None, k: int) -> np.ndarray:
        """Indices of the top-k items per row, excluding each row's seen items."""
        scores = scores.astype(np.float32, copy=True)
        if seen is not None:
            for row, s in enumerate(seen):
                scores[row, s] = -np.inf
        part = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
        order = np.take_along_axis(scores, part, axis=1).argsort(axis=1)[:, ::-1]
        return np.take_along_axis(part, order, axis=1)

    def recommend(self, user_interaction_count: int, content_scores: np.ndarray, collab_scores: np.ndarray,
                  seen_indices, top_k: int = 10, popularity: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """Single-user API mirroring the architecture doc."""
        n_u = np.array([user_interaction_count])
        scores = self.blend(n_u, content_scores[None, :], collab_scores[None, :],
                            None if popularity is None else popularity[None, :])
        top = self.top_k(scores, [np.asarray(seen_indices, dtype=int)], top_k)[0]
        return top, float(self.calculate_alpha(user_interaction_count))

    def explain(self, user_interaction_count: int, content_scores: np.ndarray, collab_scores: np.ndarray,
                indices: np.ndarray, popularity: np.ndarray | None = None) -> pd.DataFrame:
        alpha = float(self.calculate_alpha(user_interaction_count))
        beta = float(self.calculate_beta(user_interaction_count)) if popularity is not None else 0.0
        c, f = content_scores[indices], collab_scores[indices]
        p = popularity[indices] if popularity is not None else np.zeros(len(indices), dtype=np.float32)
        return pd.DataFrame({"idx": indices, "alpha": alpha, "beta": beta, "s_collab": f, "s_content": c,
                             "s_popular": p, "s_hybrid": beta * p + (1 - beta) * (alpha * f + (1 - alpha) * c)})
