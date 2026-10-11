"""Top-K ranking metrics with binary relevance."""
import numpy as np


def per_user_metrics(recs: np.ndarray, relevant: list[set], k: int) -> dict[str, np.ndarray]:
    """Precision/Recall/NDCG/AP/RR@k for each user (row). recs: (n_users, >=k) item indices.
    Users with no relevant items must be filtered out by the caller."""
    recs = recs[:, :k]
    n = len(relevant)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    out = {m: np.zeros(n) for m in ("precision", "recall", "ndcg", "map", "mrr")}
    for u, (row, rel) in enumerate(zip(recs, relevant)):
        hits = np.fromiter((i in rel for i in row), dtype=bool, count=len(row))
        n_hit = hits.sum()
        out["precision"][u] = n_hit / k
        out["recall"][u] = n_hit / len(rel)
        out["ndcg"][u] = discounts[hits].sum() / discounts[:min(len(rel), k)].sum()
        if n_hit:
            prec_at = np.cumsum(hits) / np.arange(1, k + 1)
            out["map"][u] = (prec_at * hits).sum() / min(len(rel), k)
            out["mrr"][u] = 1.0 / (np.argmax(hits) + 1)
    return {f"{m}@{k}": v for m, v in out.items()}


def ranking_metrics(recs: np.ndarray, relevant: list[set], k: int) -> dict[str, float]:
    """Mean Precision/Recall/NDCG/MAP/MRR@k over users."""
    return {m: float(v.mean()) for m, v in per_user_metrics(recs, relevant, k).items()}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Mean per-user difference a - b with a 95% percentile bootstrap CI (users resampled, paired)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi)}
