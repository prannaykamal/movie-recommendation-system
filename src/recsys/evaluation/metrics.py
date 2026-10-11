"""Top-K ranking metrics with binary relevance."""
import numpy as np


def ranking_metrics(recs: np.ndarray, relevant: list[set], k: int) -> dict[str, float]:
    """Mean Precision/Recall/NDCG/MAP/MRR@k over users. recs: (n_users, >=k) item indices.
    Users with no relevant items must be filtered out by the caller."""
    recs = recs[:, :k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    p = r = ndcg = ap = rr = 0.0
    for row, rel in zip(recs, relevant):
        hits = np.fromiter((i in rel for i in row), dtype=bool, count=len(row))
        n_hit = hits.sum()
        p += n_hit / k
        r += n_hit / len(rel)
        ndcg += (discounts[hits]).sum() / discounts[:min(len(rel), k)].sum()
        if n_hit:
            prec_at = np.cumsum(hits) / np.arange(1, k + 1)
            ap += (prec_at * hits).sum() / min(len(rel), k)
            rr += 1.0 / (np.argmax(hits) + 1)
    n = len(relevant)
    return {f"precision@{k}": p / n, f"recall@{k}": r / n, f"ndcg@{k}": ndcg / n,
            f"map@{k}": ap / n, f"mrr@{k}": rr / n}
