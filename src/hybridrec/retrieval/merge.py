"""Merge the three sources' top-k lists into one candidate set per user.

For every candidate we keep all three sources' scores (NaN where a source can't
score the movie), each source's rank in its own top-k (0 = not in its top-k),
and how many sources found it. Already-rated movies are excluded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hybridrec.data.dataset import Histories, mask_seen, topk

SOURCES = ("cb", "cf", "emb")


def merge_candidates(scorers: dict, hist: Histories, user_ids: np.ndarray, ks: dict[str, int],
                     batch: int = 256) -> pd.DataFrame:
    n_items = len(hist.items)
    parts = []
    for s in range(0, len(user_ids), batch):
        users = user_ids[s : s + batch]
        rows = hist.positions(users)
        raw = {src: scorers[src].score(hist, rows) for src in SOURCES}
        ranks = {}
        for src in SOURCES:
            idx, vals = topk(mask_seen(raw[src].copy(), hist, rows), ks[src])
            R = np.zeros((len(users), n_items), dtype=np.int16)
            ok = np.isfinite(vals)
            b_idx = np.repeat(np.arange(len(users)), idx.shape[1])[ok.ravel()]
            R[b_idx, idx[ok]] = np.tile(np.arange(1, idx.shape[1] + 1), len(users))[ok.ravel()]
            ranks[src] = R
        in_any = (ranks["cb"] > 0) | (ranks["cf"] > 0) | (ranks["emb"] > 0)
        b, item = np.nonzero(in_any)
        part = {"user_id": users[b].astype(np.int64), "item_idx": item.astype(np.int64)}
        for src in SOURCES:
            part[f"{src}_score"] = raw[src][b, item].astype(np.float32)
            part[f"{src}_rank"] = ranks[src][b, item].astype(np.int16)
        parts.append(pd.DataFrame(part))
    out = pd.concat(parts, ignore_index=True)
    out.insert(1, "movie_id", hist.items.movie_ids[out["item_idx"].to_numpy()])
    out["n_sources"] = sum((out[f"{src}_rank"] > 0).astype(np.int8) for src in SOURCES)
    return out


def add_labels(cands: pd.DataFrame, target: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """label = liked in the target block; rated = rated at all in the target block."""
    t = target[["user_id", "movie_id", "rating"]]
    out = cands.merge(t, on=["user_id", "movie_id"], how="left")
    out["label"] = (out["rating"] >= threshold).astype(np.int8)
    out["rated"] = out["rating"].notna().astype(np.int8)
    return out.drop(columns=["rating"])


def topn_by_score(cands: pd.DataFrame, scores: np.ndarray, n: int = 10) -> dict[int, list[int]]:
    """Top-n movie_ids per user ranked by ``scores`` (aligned with ``cands`` rows)."""
    users = cands["user_id"].to_numpy()
    order = np.lexsort((-scores, users))
    u_sorted = users[order]
    starts = np.flatnonzero(np.r_[True, u_sorted[1:] != u_sorted[:-1]])
    ends = np.r_[starts[1:], len(order)]
    mids = cands["movie_id"].to_numpy()
    return {int(u_sorted[s]): mids[order[s : min(e, s + n)]].tolist() for s, e in zip(starts, ends)}
