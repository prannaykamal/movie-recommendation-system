"""Training and scoring for the reranker.

Rows are grouped by user. v1 trains with BCE on every positive, every movie
rated below the like threshold, and ``neg_per_pos`` sampled never-rated
candidates per positive (resampled each epoch). v2 trains with a listwise
softmax over each user's list (positives plus sampled negatives). Early stopping
uses s2 NDCG@10 over the full candidate lists.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from hybridrec.config import Config
from hybridrec.eval.metrics import evaluate_topk
from hybridrec.rerank.model import AttnDCNReranker, MLPReranker
from hybridrec.retrieval.merge import topn_by_score

log = logging.getLogger("hybridrec")
CHUNK = 262_144
ID_COLS = ["user_id", "movie_id", "item_idx", "label", "rated"]


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, *Xs: np.ndarray) -> "Normalizer":
        """Mean/std over the rows of several matrices, in chunks, without stacking them."""
        chunks = [X[i : i + CHUNK] for X in Xs for i in range(0, len(X), CHUNK)]
        n = sum(len(c) for c in chunks)
        mean = sum(c.sum(axis=0, dtype=np.float64) for c in chunks) / n
        var = sum(((c.astype(np.float64) - mean) ** 2).sum(axis=0) for c in chunks) / n
        return cls(mean.astype(np.float32), np.maximum(np.sqrt(var), 1e-6).astype(np.float32))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.apply_(X.astype(np.float32, copy=True))

    def apply_(self, X: np.ndarray) -> np.ndarray:
        """Normalize in place, chunk by chunk (X may be float16 or float32)."""
        for i in range(0, len(X), CHUNK):
            blk = X[i : i + CHUNK].astype(np.float32)
            blk -= self.mean
            blk /= self.std
            X[i : i + CHUNK] = np.clip(blk, -5.0, 5.0)
        return X


@dataclass
class ListData:
    """Feature rows grouped by user, plus each user's last liked movies."""
    X: np.ndarray
    y: np.ndarray
    rated: np.ndarray
    item_idx: np.ndarray
    user_ids: np.ndarray  # per row
    starts: np.ndarray  # per user
    ends: np.ndarray
    users: np.ndarray  # unique users, aligned with starts
    hist: np.ndarray  # (n_users, L)
    frame: pd.DataFrame  # user_id, movie_id (for metrics)

    @classmethod
    def from_parts(cls, feats: pd.DataFrame, cols: list[str], users: np.ndarray, seq: np.ndarray,
                   norm: Normalizer | None = None) -> "ListData":
        """From an in-memory feature frame (serving, tests)."""
        feats = feats.sort_values(["user_id", "item_idx"], kind="stable").reset_index(drop=True)
        X = feats[cols].to_numpy(np.float32)
        if norm is not None:
            norm.apply_(X)
        return cls._build(X, {c: feats[c].to_numpy() for c in ID_COLS}, users, seq)

    @classmethod
    def from_parquet(cls, path, cols: list[str], users: np.ndarray, seq: np.ndarray,
                     norm: Normalizer | None = None) -> "ListData":
        """Stream a features.parquet (sorted by user_id, item_idx, as build_features
        writes it) into a float16 matrix: a full20k stage is ~5M rows x 79 features,
        which as pandas + float32 would not fit in 8 GB alongside the others."""
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
        X = np.empty((n, len(cols)), dtype=np.float16)
        ids = {"user_id": np.empty(n, np.int64), "movie_id": np.empty(n, np.int64),
               "item_idx": np.empty(n, np.int64), "label": np.empty(n, np.int8), "rated": np.empty(n, np.int8)}
        pos = 0
        for batch in pf.iter_batches(batch_size=CHUNK, columns=ID_COLS + cols):
            m = batch.num_rows
            for c in ID_COLS:
                ids[c][pos : pos + m] = batch.column(c).to_numpy()
            blk = np.column_stack([batch.column(c).to_numpy() for c in cols]).astype(np.float32)
            X[pos : pos + m] = norm.apply_(blk) if norm is not None else blk
            pos += m
        assert (np.diff(ids["user_id"]) >= 0).all(), "features must be sorted by user"
        return cls._build(X, ids, users, seq)

    @classmethod
    def _build(cls, X, ids: dict, users: np.ndarray, seq: np.ndarray) -> "ListData":
        uid = ids["user_id"]
        starts = np.flatnonzero(np.r_[True, uid[1:] != uid[:-1]])
        ends = np.r_[starts[1:], len(uid)]
        row_users = uid[starts]
        pos = np.searchsorted(users, row_users)
        frame = pd.DataFrame({"user_id": uid, "movie_id": ids["movie_id"]})
        return cls(X, ids["label"].astype(np.int8), ids["rated"].astype(np.int8), ids["item_idx"].astype(np.int64),
                   uid, starts, ends, row_users, seq[pos], frame)

    def positives_per_user(self) -> np.ndarray:
        return np.add.reduceat(self.y.astype(np.int64), self.starts)


class MaskedInput(torch.nn.Module):
    """Zero out some input features (ablation). For these models a constant-zero
    input carries no information, so this equals dropping the feature group."""

    def __init__(self, model: torch.nn.Module, keep: np.ndarray):
        super().__init__()
        self.model = model
        self.register_buffer("keep", torch.as_tensor(keep, dtype=torch.float32))

    def forward(self, x, *args):
        return self.model(x * self.keep, *args)


def build_model(kind: str, n_features: int, cfg: Config, item_text, item_genres):
    if kind == "v1":
        return MLPReranker(n_features, cfg.rr_dropout)
    return AttnDCNReranker(n_features, item_text, item_genres, dropout=cfg.rr_dropout)


def _forward(model, data: ListData, user_sel: np.ndarray, row_sel: list[np.ndarray]):
    rows = np.concatenate(row_sel)
    row_user = np.repeat(np.arange(len(user_sel)), [len(r) for r in row_sel])
    return model(torch.as_tensor(data.X[rows], dtype=torch.float32), torch.as_tensor(data.item_idx[rows]),
                 torch.as_tensor(row_user), torch.as_tensor(data.hist[user_sel])), rows


@torch.no_grad()
def score_lists(model, data: ListData, users_per_batch: int = 64) -> np.ndarray:
    model.eval()
    out = np.zeros(len(data.X), dtype=np.float32)
    for s in range(0, len(data.users), users_per_batch):
        sel = np.arange(s, min(s + users_per_batch, len(data.users)))
        scores, rows = _forward(model, data, sel, [np.arange(data.starts[u], data.ends[u]) for u in sel])
        out[rows] = scores.numpy()
    return out


def evaluate_lists(model, data: ListData, truth: dict[int, set[int]]) -> dict[str, float]:
    scores = score_lists(model, data)
    recs = topn_by_score(data.frame, scores)
    m = evaluate_topk(recs, truth)
    # Reranker-only view: users whose candidates contain at least one liked movie.
    has_pos = set(data.users[data.positives_per_user() > 0].tolist())
    cond = evaluate_topk(recs, {u: r for u, r in truth.items() if u in has_pos})
    m["ndcg@10_with_positive"] = cond["ndcg@10"]
    m["n_users_with_positive"] = cond["n_users"]
    return m


def _sample_rows(data: ListData, u: int, rng, cfg: Config, listwise: bool) -> np.ndarray:
    s, e = data.starts[u], data.ends[u]
    idx = np.arange(s, e)
    y, rated = data.y[s:e], data.rated[s:e]
    pos, explicit, unrated = idx[y == 1], idx[(y == 0) & (rated == 1)], idx[(y == 0) & (rated == 0)]
    if listwise:
        n_neg = max(cfg.rr_list_negatives - len(explicit), 0)
        neg = rng.choice(unrated, size=min(n_neg, len(unrated)), replace=False)
    else:
        neg = rng.choice(unrated, size=min(cfg.rr_neg_per_pos * len(pos), len(unrated)), replace=False)
    return np.concatenate([pos, explicit, neg])


def train(kind: str, train_sets: list[ListData], val: ListData, val_truth: dict, cfg: Config,
          item_text, item_genres, log_prefix: str = "", keep: np.ndarray | None = None
          ) -> tuple[torch.nn.Module, list[dict]]:
    """``keep``: optional 0/1 per feature; zeros are masked out (ablations)."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = build_model(kind, train_sets[0].X.shape[1], cfg, item_text, item_genres)
    if keep is not None:
        model = MaskedInput(model, keep)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.rr_lr, weight_decay=1e-5)
    # Training users: those whose candidates contain at least one liked movie.
    groups = [(d, np.flatnonzero(d.positives_per_user() > 0)) for d in train_sets]
    history, best, best_state, bad = [], -1.0, None, 0
    for epoch in range(cfg.rr_epochs):
        model.train()
        for _, users in groups:
            rng.shuffle(users)
        total, n = 0.0, 0
        if kind == "v1":
            # Pointwise: pool the sampled rows of every user, then batch by rows.
            batches = []
            for d, users in groups:
                rows = np.concatenate([_sample_rows(d, u, rng, cfg, listwise=False) for u in users])
                rng.shuffle(rows)
                batches += [(d, rows[i : i + cfg.rr_batch_rows]) for i in range(0, len(rows), cfg.rr_batch_rows)]
            rng.shuffle(batches)
            for d, rows in batches:
                scores = model(torch.as_tensor(d.X[rows], dtype=torch.float32))
                loss = F.binary_cross_entropy_with_logits(scores, torch.as_tensor(d.y[rows], dtype=torch.float32))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total, n = total + loss.item(), n + 1
        else:
            # Listwise: each batch holds whole (sampled) lists of a few users.
            batches = [(d, users[i : i + cfg.rr_users_per_batch])
                       for d, users in groups for i in range(0, len(users), cfg.rr_users_per_batch)]
            rng.shuffle(batches)
            for d, users in batches:
                row_sel = [_sample_rows(d, u, rng, cfg, listwise=True) for u in users]
                scores, rows = _forward(model, d, users, row_sel)
                loss = _listwise_loss(scores, torch.as_tensor(d.y[rows], dtype=torch.float32), [len(r) for r in row_sel])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total, n = total + loss.item(), n + 1
        m = evaluate_lists(model, val, val_truth)
        history.append({"epoch": epoch + 1, "loss": total / max(n, 1), **m})
        log.info("%sreranker %s epoch %d: loss %.4f | s2 P@10 %.4f R@10 %.4f NDCG@10 %.4f", log_prefix, kind,
                 epoch + 1, total / max(n, 1), m["precision@10"], m["recall@10"], m["ndcg@10"])
        if m["ndcg@10"] > best:
            best, bad = m["ndcg@10"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.rr_patience:
                break
    model.load_state_dict(best_state)
    return model, history


def _listwise_loss(scores: torch.Tensor, y: torch.Tensor, sizes: list[int]) -> torch.Tensor:
    """Mean over users of -mean_{positives} log softmax(list scores)."""
    losses = []
    for s_u, y_u in zip(torch.split(scores, sizes), torch.split(y, sizes)):
        logp = torch.log_softmax(s_u, dim=0)
        losses.append(-(logp * y_u).sum() / y_u.sum().clamp(min=1))
    return torch.stack(losses).mean()


def save(path: Path, model, kind: str, cols: list[str], norm: Normalizer, history: list[dict]) -> None:
    torch.save({"state": model.state_dict(), "kind": kind, "cols": cols,
                "mean": norm.mean, "std": norm.std}, path)
    path.with_suffix(".json").write_text(json.dumps(history, indent=1), encoding="utf-8")


def load(path: Path, cfg: Config, item_text, item_genres):
    ckpt = torch.load(path, weights_only=False)
    model = build_model(ckpt["kind"], len(ckpt["cols"]), cfg, item_text, item_genres)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt["cols"], Normalizer(ckpt["mean"], ckpt["std"])


def best_kind(cfg: Config) -> str:
    """The trained reranker with the higher s2 NDCG@10."""
    scores = {}
    for kind in ("v1", "v2"):
        path = cfg.artifacts_dir / f"reranker_{kind}.json"
        if path.exists():
            scores[kind] = max(h["ndcg@10"] for h in json.loads(path.read_text(encoding="utf-8")))
    return max(scores, key=scores.get)
