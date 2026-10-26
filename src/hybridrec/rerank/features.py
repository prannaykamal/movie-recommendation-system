"""Reranker features for (user, candidate) rows, computed from the stage's fit window only.

Groups:
- candidate: each source's score, rank discount, in-top-k flag and within-user
  percentile; iALS missing flag; number of sources; RRF score
- user: rating count, mean/spread of ratings, share liked, days active, mean
  year, how mainstream their taste is, genre preferences
- item: genres, year, runtime, language, collection flag, rating count and
  smoothed average rating in the fit window
- cross: genre match, shared director/cast/collection counts with liked
  movies, year and popularity gaps, text and two-tower similarity to the
  last liked movies
Plus, per user, the last liked movies (item indices) for the v2 attention model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from hybridrec.data.dataset import GENRES, Dataset, Histories, genre_matrix
from hybridrec.rerank.rrf import rrf_scores
from hybridrec.retrieval.merge import SOURCES

GENRE_COLS = [f"g_{g.lower().replace(' ', '_')}" for g in GENRES]
USER_GENRE_COLS = [f"u_{c}" for c in GENRE_COLS]

FEATURE_GROUPS = {
    "candidate": [f"{s}_{f}" for s in SOURCES for f in ("score", "rankdisc", "intop", "pct")]
                 + ["cf_missing", "n_sources", "rrf"],
    "user": ["u_n_ratings", "u_mean", "u_std", "u_frac_liked", "u_days_active", "u_mean_year", "u_mainstream"]
            + USER_GENRE_COLS,
    "item": GENRE_COLS + ["i_year", "i_year_missing", "i_runtime", "i_lang_en", "i_collection",
                          "i_log_count", "i_avg_rating"],
    "cross": ["x_genre_match", "x_dir_shared", "x_cast_shared", "x_coll_shared", "x_year_gap", "x_pop_gap",
              "x_text_max", "x_text_mean", "x_tt_max", "x_tt_mean"],
}
FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]
ID_COLS = ["user_id", "movie_id", "item_idx", "label", "rated"]
TOP_CAST = 3


def _incidence(lists: pd.Series) -> sp.csr_matrix:
    """(N, n_values) 0/1 matrix from a column of id lists."""
    rows = np.repeat(np.arange(len(lists)), [len(v) for v in lists])
    vals = np.concatenate([np.asarray(v, dtype=np.int64) for v in lists]) if len(rows) else np.array([], np.int64)
    uniq, cols = np.unique(vals, return_inverse=True)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(lists), max(len(uniq), 1)))


@dataclass
class ItemTable:
    """Static item features aligned with the dataset's item index."""
    genres: np.ndarray
    year: np.ndarray
    runtime: np.ndarray
    lang_en: np.ndarray
    has_collection: np.ndarray
    directors: sp.csr_matrix
    cast: sp.csr_matrix
    collections: sp.csr_matrix
    text: np.ndarray

    @classmethod
    def build(cls, ds: Dataset, text_vectors: np.ndarray) -> "ItemTable":
        m = ds.movies
        coll = m["collection_id"].astype("Float64").fillna(-1).astype(np.int64)
        return cls(
            genres=genre_matrix(m),
            year=pd.to_numeric(m["year"], errors="coerce").to_numpy(dtype=np.float32),
            runtime=pd.to_numeric(m["runtime"], errors="coerce").to_numpy(dtype=np.float32),
            lang_en=(m["original_language"] == "en").to_numpy(dtype=np.float32),
            has_collection=(coll >= 0).to_numpy(dtype=np.float32),
            directors=_incidence(m["director_ids"]),
            cast=_incidence(m["cast_ids"].map(lambda v: list(v)[:TOP_CAST])),
            collections=_incidence(coll.map(lambda c: [c] if c >= 0 else [])),
            text=text_vectors.astype(np.float32),
        )


@dataclass
class ItemStats:
    """Per-stage item statistics from the fit window."""
    log_count: np.ndarray
    avg_rating: np.ndarray

    @classmethod
    def from_hist(cls, hist: Histories, prior: float = 10.0) -> "ItemStats":
        n = len(hist.items)
        counts = np.bincount(hist.item_idx, minlength=n).astype(np.float64)
        sums = np.bincount(hist.item_idx, weights=hist.rating, minlength=n)
        mu = float(hist.rating.mean())
        return cls(np.log1p(counts).astype(np.float32), ((sums + prior * mu) / (counts + prior)).astype(np.float32))


def last_liked(hist: Histories, rows: np.ndarray, threshold: float, length: int) -> np.ndarray:
    """(len(rows), length) right-aligned item indices of the last liked movies; -1 = padding."""
    out = np.full((len(rows), length), -1, dtype=np.int64)
    for b, p in enumerate(rows):
        if p < 0:
            continue
        items, ratings, _ = hist.get(p)
        liked = items[ratings >= threshold][-length:]
        if len(liked):
            out[b, length - len(liked):] = liked
    return out


def _user_block(hist: Histories, rows: np.ndarray, it: ItemTable, st: ItemStats, threshold: float) -> dict:
    B, N = len(rows), len(hist.items)
    r_data, r_cols, l_data, l_cols, indptr_r, indptr_l = [], [], [], [], [0], [0]
    stats = np.zeros((B, 7), dtype=np.float32)
    for b, p in enumerate(rows):
        items, ratings, ts = hist.get(p) if p >= 0 else (np.array([], np.int64), np.array([], np.float32), np.array([0]))
        liked = ratings >= threshold
        r_cols.append(items); r_data.append(np.ones(len(items), np.float32)); indptr_r.append(indptr_r[-1] + len(items))
        l_cols.append(items[liked]); l_data.append(np.ones(int(liked.sum()), np.float32))
        indptr_l.append(indptr_l[-1] + int(liked.sum()))
        if len(items):
            years = it.year[items]
            stats[b] = (np.log1p(len(items)), ratings.mean(), ratings.std(), liked.mean(),
                        np.log1p((ts.max() - ts.min()) / 86400.0),
                        np.nanmean(years) if np.isfinite(years).any() else np.nan,
                        st.log_count[items].mean())
    rated = sp.csr_matrix((np.concatenate(r_data), np.concatenate(r_cols), indptr_r), shape=(B, N))
    liked = sp.csr_matrix((np.concatenate(l_data), np.concatenate(l_cols), indptr_l), shape=(B, N))
    n_liked = np.asarray(liked.sum(axis=1)).ravel()
    basis = sp.vstack([liked[b] if n_liked[b] > 0 else rated[b] for b in range(B)]).tocsr()
    genre_pref = (basis @ it.genres) / np.maximum(np.asarray(basis.sum(axis=1)), 1.0)
    return {"stats": stats, "genre_pref": genre_pref.astype(np.float32), "liked": liked}


def _shared(incidence: sp.csr_matrix, liked: sp.csr_matrix, b_rows: np.ndarray, items: np.ndarray) -> np.ndarray:
    """For each row: how many of the user's liked movies share a value (director, ...) with the candidate."""
    user_vals = (liked @ incidence).tocsr()  # (B, n_values) counts
    return np.asarray(incidence[items].multiply(user_vals[b_rows]).sum(axis=1)).ravel()


def _sims(vecs: np.ndarray, hist_items: np.ndarray, b_rows: np.ndarray, items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Max and mean cosine between each candidate and the user's recent liked movies."""
    mx, mean = np.zeros(len(items), np.float32), np.zeros(len(items), np.float32)
    for b in np.unique(b_rows):
        sel = np.flatnonzero(b_rows == b)
        h = hist_items[b][hist_items[b] >= 0]
        if len(h) == 0:
            continue
        S = vecs[items[sel]] @ vecs[h].T
        mx[sel], mean[sel] = S.max(axis=1), S.mean(axis=1)
    return mx, mean


def build_features(cands: pd.DataFrame, hist: Histories, it: ItemTable, st: ItemStats,
                   tt_vecs: np.ndarray | None, threshold: float, recent: int, batch_users: int = 512) -> pd.DataFrame:
    """One feature row per candidate, sorted by (user_id, item_idx)."""
    cands = cands.sort_values(["user_id", "item_idx"], kind="stable").reset_index(drop=True)
    cols: dict[str, np.ndarray] = {c: cands[c].to_numpy() for c in ID_COLS if c in cands}

    # Candidate-stage features.
    for src in SOURCES:
        score = cands[f"{src}_score"].astype(np.float32)
        rank = cands[f"{src}_rank"].to_numpy().astype(np.float32)
        cols[f"{src}_score"] = score.fillna(0.0).to_numpy()
        cols[f"{src}_rankdisc"] = np.where(rank > 0, 1.0 / np.log2(1.0 + np.maximum(rank, 1)), 0.0)
        cols[f"{src}_intop"] = (rank > 0).astype(np.float32)
        cols[f"{src}_pct"] = score.groupby(cands["user_id"]).rank(pct=True).fillna(0.0).to_numpy()
    cols["cf_missing"] = cands["cf_score"].isna().to_numpy(np.float32)
    cols["n_sources"] = cands["n_sources"].to_numpy(np.float32)
    cols["rrf"] = rrf_scores(cands) * 60.0

    items = cands["item_idx"].to_numpy()
    # Item features.
    for j, c in enumerate(GENRE_COLS):
        cols[c] = it.genres[items, j]
    year = it.year[items]
    cols["i_year"] = np.nan_to_num((year - 1990.0) / 20.0)
    cols["i_year_missing"] = np.isnan(year).astype(np.float32)
    cols["i_runtime"] = np.nan_to_num(it.runtime[items] / 100.0, nan=1.0)
    cols["i_lang_en"] = it.lang_en[items]
    cols["i_collection"] = it.has_collection[items]
    cols["i_log_count"] = st.log_count[items]
    cols["i_avg_rating"] = st.avg_rating[items]

    # User and cross features, by batches of users.
    for c in FEATURE_GROUPS["user"] + FEATURE_GROUPS["cross"]:
        cols[c] = np.zeros(len(cands), np.float32)
    users = cands["user_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    ends = np.r_[starts[1:], len(users)]
    rows_all = hist.positions(users[starts])
    global_year = float(np.nanmean(it.year))
    stat_names = ["u_n_ratings", "u_mean", "u_std", "u_frac_liked", "u_days_active", "u_mean_year", "u_mainstream"]
    for s in range(0, len(starts), batch_users):
        rows = rows_all[s : s + batch_users]
        b_starts, b_ends = starts[s : s + batch_users], ends[s : s + batch_users]
        sl = slice(b_starts[0], b_ends[-1])
        b_rows = np.repeat(np.arange(len(rows)), b_ends - b_starts)
        cand_items = items[sl]
        ub = _user_block(hist, rows, it, st, threshold)
        stats = ub["stats"]
        stats[:, 5] = np.where(np.isnan(stats[:, 5]), global_year, stats[:, 5])
        for j, c in enumerate(stat_names):
            cols[c][sl] = stats[b_rows, j]
        cols["u_mean_year"][sl] = (stats[b_rows, 5] - 1990.0) / 20.0
        for j, c in enumerate(USER_GENRE_COLS):
            cols[c][sl] = ub["genre_pref"][b_rows, j]

        g = it.genres[cand_items]
        cols["x_genre_match"][sl] = (ub["genre_pref"][b_rows] * g).sum(axis=1) / np.maximum(g.sum(axis=1), 1.0)
        cols["x_dir_shared"][sl] = np.log1p(_shared(it.directors, ub["liked"], b_rows, cand_items))
        cols["x_cast_shared"][sl] = np.log1p(_shared(it.cast, ub["liked"], b_rows, cand_items))
        cols["x_coll_shared"][sl] = np.log1p(_shared(it.collections, ub["liked"], b_rows, cand_items))
        cols["x_year_gap"][sl] = np.nan_to_num(np.abs(it.year[cand_items] - stats[b_rows, 5]) / 10.0)
        cols["x_pop_gap"][sl] = st.log_count[cand_items] - stats[b_rows, 6]
        recent_items = last_liked(hist, rows, threshold, recent)
        cols["x_text_max"][sl], cols["x_text_mean"][sl] = _sims(it.text, recent_items, b_rows, cand_items)
        if tt_vecs is not None:
            cols["x_tt_max"][sl], cols["x_tt_mean"][sl] = _sims(tt_vecs, recent_items, b_rows, cand_items)
    feature_set = set(FEATURES)
    return pd.DataFrame({k: (v.astype(np.float32, copy=False) if k in feature_set else v) for k, v in cols.items()})


def stage_features(ds: Dataset, stage: str, cands: pd.DataFrame, it: ItemTable,
                   tt_vecs: np.ndarray | None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Features for a stage's candidates using only that stage's fit window.

    Returns (feature rows, user_ids, (n_users, rr_history) last-liked item indices).
    """
    from hybridrec.data.split import stage_frames

    fit, _ = stage_frames(ds.ratings, stage)
    hist = Histories(fit, ds.items)
    feats = build_features(cands, hist, it, ItemStats.from_hist(hist), tt_vecs,
                           ds.cfg.like_threshold, ds.cfg.rr_recent_sims)
    users = np.unique(feats["user_id"].to_numpy())
    seq = last_liked(hist, hist.positions(users), ds.cfg.like_threshold, ds.cfg.rr_history)
    return feats, users, seq
