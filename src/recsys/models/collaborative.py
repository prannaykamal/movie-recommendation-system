"""Collaborative engine: user-mean-centred TruncatedSVD."""
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from recsys.data.interactions import minmax_rows


class CollaborativeEngine:
    def __init__(self, n_components: int = 64, random_state: int = 42, mean_shrinkage: float = 5.0):
        self.n_components = n_components
        self.mean_shrinkage = mean_shrinkage
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state, algorithm="randomized",
                                n_iter=7)

    def _center(self, R: sparse.csr_matrix) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Subtract each user's mean from their *observed* ratings only; unobserved stay 0.
        The mean is damped toward the global mean with `mean_shrinkage` pseudo-ratings, so a
        short uniform history (e.g. three 5-star ratings) still yields a non-zero centred row."""
        R = R.tocsr().astype(np.float32)
        counts = np.diff(R.indptr)
        sums = np.asarray(R.sum(axis=1)).ravel()
        b = self.mean_shrinkage
        means = ((sums + b * self.global_mean) / np.maximum(counts + b, 1e-9)).astype(np.float32)
        means[counts == 0] = self.global_mean
        C = R.copy()
        C.data -= np.repeat(means, counts)
        return C, means

    def fit(self, R: sparse.csr_matrix) -> "CollaborativeEngine":
        self.global_mean = float(R.data.mean())
        C, self.user_means = self._center(R)
        self.user_factors = self.svd.fit_transform(C).astype(np.float32)  # P = U * Sigma
        self.item_factors = self.svd.components_.T.astype(np.float32)     # Q
        return self

    def predict_raw(self, uidx: np.ndarray) -> np.ndarray:
        """r_hat = mu_u + p_u . q_i for the given training users."""
        return self.user_means[uidx, None] + self.user_factors[uidx] @ self.item_factors.T

    def fold_in(self, R_new: sparse.csr_matrix) -> np.ndarray:
        """r_hat for users not in the training matrix: p_u = centred_row . Q."""
        C, means = self._center(R_new)
        P = (C @ self.item_factors).astype(np.float32)
        return means[:, None] + P @ self.item_factors.T

    def score(self, uidx: np.ndarray) -> np.ndarray:
        return minmax_rows(self.predict_raw(uidx))

    def score_new(self, R_new: sparse.csr_matrix) -> np.ndarray:
        return minmax_rows(self.fold_in(R_new))
