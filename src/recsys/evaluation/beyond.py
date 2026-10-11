"""Beyond-accuracy metrics for top-K lists: how much of the catalogue is exposed, how popular,
novel and diverse the recommendations are, and how different they are across users.

All popularity quantities use training interaction counts."""
import numpy as np


def gini(x: np.ndarray) -> float:
    """Gini coefficient of non-negative values (0 = perfectly even, 1 = all mass on one item)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def beyond_accuracy(recs: np.ndarray, item_counts: np.ndarray, n_users_train: int, item_genres: np.ndarray,
                    k: int = 10, short_head_frac: float = 0.2, n_pairs: int = 5000, seed: int = 0) -> dict[str, float]:
    """recs: (n_users, >=k) item indices. item_genres: (n_items, n_genres) 0/1 matrix.

    coverage      share of the catalogue recommended to at least one user
    gini          concentration of recommendation slots across items
    long_tail     share of slots filled by items outside the short head (top `short_head_frac`
                  of items by popularity)
    pop_pct       mean popularity percentile of recommended items (0.5 = median item)
    novelty       mean self-information -log2(count / n_users) in bits; higher = less mainstream
    ild           intra-list diversity: 1 - mean pairwise genre cosine within a list
    personal      1 - mean overlap (Jaccard) between the lists of random user pairs
    """
    recs = recs[:, :k]
    n_items = len(item_counts)
    slots = np.bincount(recs.ravel(), minlength=n_items)
    order = np.argsort(-item_counts, kind="stable")
    head = np.zeros(n_items, dtype=bool)
    head[order[:int(round(short_head_frac * n_items))]] = True
    pct = np.empty(n_items)
    pct[np.argsort(item_counts, kind="stable")] = np.arange(n_items) / (n_items - 1)
    info = -np.log2(np.maximum(item_counts, 1) / n_users_train)

    G = item_genres.astype(np.float64)
    G = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12)
    L = G[recs]                                          # (users, k, genres)
    sims = np.einsum("ukg,ujg->ukj", L, L)
    off = ~np.eye(k, dtype=bool)
    ild = 1.0 - sims[:, off].mean(axis=1)

    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, len(recs), (2, n_pairs))
    keep = a != b
    sets = [set(r) for r in recs]
    overlap = np.array([len(sets[i] & sets[j]) / len(sets[i] | sets[j]) for i, j in zip(a[keep], b[keep])])

    return {
        f"coverage@{k}": float((slots > 0).mean()),
        f"gini@{k}": gini(slots),
        f"long_tail@{k}": float(1.0 - head[recs].mean()),
        f"pop_pct@{k}": float(pct[recs].mean()),
        f"novelty@{k}": float(info[recs].mean()),
        f"ild@{k}": float(ild.mean()),
        f"personal@{k}": float(1.0 - overlap.mean()),
    }
