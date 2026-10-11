"""Non-personalised baselines. Each returns one score per item, computed on training data."""
import numpy as np
from scipy import sparse


def _counts_sums(R: sparse.csr_matrix):
    R = R.tocsc()
    return np.diff(R.indptr).astype(np.float64), np.asarray(R.sum(axis=0)).ravel()


def most_popular(R: sparse.csr_matrix) -> np.ndarray:
    return _counts_sums(R)[0]


def average_rating(R: sparse.csr_matrix) -> np.ndarray:
    n, s = _counts_sums(R)
    return np.divide(s, n, out=np.zeros_like(s), where=n > 0)


def bayesian_average(R: sparse.csr_matrix, prior_count: int = 50) -> np.ndarray:
    """(C*m + sum r) / (C + n): shrinks low-vote items toward the global mean m."""
    n, s = _counts_sums(R)
    m = R.data.mean()
    return (prior_count * m + s) / (prior_count + n)


def random_scores(n_users: int, n_items: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((n_users, n_items), dtype=np.float32)
