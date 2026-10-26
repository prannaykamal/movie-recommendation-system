"""Ranking metrics with binary relevance.

All averages are over users with at least one relevant (liked) movie in the
target block. A user with no recommendations counts as zero.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

_DISCOUNT = 1.0 / np.log2(np.arange(2, 1002))


def user_metrics(ranked: Sequence[int], relevant: set[int], k: int) -> tuple[float, float, float, float]:
    """(precision@k, recall@k, truncated recall@k, ndcg@k) for one user."""
    top = list(ranked)[:k]
    hits = np.array([item in relevant for item in top], dtype=bool)
    n_hits = int(hits.sum())
    n_rel = len(relevant)
    dcg = float(_DISCOUNT[: len(top)][hits].sum())
    idcg = float(_DISCOUNT[: min(k, n_rel)].sum())
    return n_hits / k, n_hits / n_rel, n_hits / min(k, n_rel), dcg / idcg


def evaluate_topk(recs: Mapping[int, Sequence[int]], truth: Mapping[int, set[int]], k: int = 10) -> dict[str, float]:
    rows = [user_metrics(recs.get(u, ()), rel, k) for u, rel in truth.items() if rel]
    p, r, rt, n = np.mean(rows, axis=0) if rows else (0.0, 0.0, 0.0, 0.0)
    return {
        f"precision@{k}": float(p),
        f"recall@{k}": float(r),
        f"recall_trunc@{k}": float(rt),
        f"ndcg@{k}": float(n),
        "n_users": float(len(rows)),
    }


def candidate_recall(cands: Mapping[int, Iterable[int]], truth: Mapping[int, set[int]]) -> float:
    """Mean share of each user's liked movies found anywhere in their candidate set."""
    vals = [len(rel & set(cands.get(u, ()))) / len(rel) for u, rel in truth.items() if rel]
    return float(np.mean(vals)) if vals else 0.0


def beyond_accuracy(recs: Mapping[int, Sequence[int]], movie_ids: np.ndarray, log_count: np.ndarray) -> dict[str, float]:
    """Catalogue coverage (share of movies recommended to anyone) and the mean
    log rating count of recommended movies (higher = more popular)."""
    rec = np.concatenate([np.asarray(r, dtype=np.int64) for r in recs.values()]) if recs else np.array([], np.int64)
    idx = np.searchsorted(movie_ids, rec)
    return {"coverage": len(np.unique(rec)) / len(movie_ids),
            "avg_log_popularity": float(log_count[idx].mean()) if len(idx) else 0.0}
