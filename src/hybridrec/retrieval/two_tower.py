"""Embedding source v2: a two-tower model trained on viewing order.

Movie side: learned ID embedding (rarely-rated movies share an "unknown" ID),
the frozen text vector, genres and a year bucket -> MLP -> unit vector.
User side: attention pooling over the movie-side vectors of the last liked
movies -> MLP -> unit vector in the same space.

Training example: a liked target movie at position t, with the liked movies
before t as history. Loss: in-batch softmax with a temperature and a
popularity (logQ) correction. Early stopping uses each user's last liked movie
inside the fit window, so L/V/T are never touched.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from hybridrec.config import Config
from hybridrec.data.dataset import Histories, genre_matrix, topk

log = logging.getLogger("hybridrec")
N_YEAR_BUCKETS = 27  # 0 = unknown, 1..26 = 5-year buckets from 1900


def year_buckets(movies: pd.DataFrame) -> np.ndarray:
    y = pd.to_numeric(movies["year"], errors="coerce").to_numpy(dtype=float)
    b = np.clip((y - 1900) // 5, 0, 25) + 1
    return np.where(np.isnan(y), 0, b).astype(np.int64)


class ItemTower(nn.Module):
    def __init__(self, n_ids: int, dim: int, text_dim: int, n_genres: int):
        super().__init__()
        self.id_emb = nn.Embedding(n_ids + 1, 32)
        self.text = nn.Linear(text_dim, 64)
        self.genre = nn.Linear(n_genres, 16)
        self.year = nn.Embedding(N_YEAR_BUCKETS, 8)
        self.mlp = nn.Sequential(nn.Linear(32 + 64 + 16 + 8, 128), nn.ReLU(), nn.Linear(128, dim))

    def forward(self, ids, text, genres, years):
        x = torch.cat([self.id_emb(ids), self.text(text), self.genre(genres), self.year(years)], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class UserTower(nn.Module):
    def __init__(self, dim: int, history: int):
        super().__init__()
        self.pos = nn.Embedding(history, dim)
        self.att = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))
        self.mlp = nn.Sequential(nn.Linear(2 * dim, 128), nn.ReLU(), nn.Linear(128, dim))

    def forward(self, H, mask):
        """H: (B, L, d) history vectors, right-aligned (last column = most recent)."""
        L = H.shape[1]
        H = H * mask.unsqueeze(-1)
        pos = torch.arange(L - 1, -1, -1, device=H.device)  # 0 = most recent
        a = self.att(H + self.pos(pos)).squeeze(-1).masked_fill(~mask, -1e9)
        a = torch.softmax(a, dim=1) * mask
        attn = (a.unsqueeze(-1) * H).sum(1)
        mean = H.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return F.normalize(self.mlp(torch.cat([attn, mean], dim=-1)), dim=-1)


class TwoTowerNet(nn.Module):
    def __init__(self, id_index, text, genres, years, dim: int, history: int):
        super().__init__()
        self.register_buffer("id_index", torch.as_tensor(id_index, dtype=torch.long))
        self.register_buffer("text_vec", torch.as_tensor(text, dtype=torch.float32))
        self.register_buffer("genre_vec", torch.as_tensor(genres, dtype=torch.float32))
        self.register_buffer("year_idx", torch.as_tensor(years, dtype=torch.long))
        self.items = ItemTower(int(id_index.max()), dim, text.shape[1], genres.shape[1])
        self.users = UserTower(dim, history)
        self.id_dropout = 0.0

    def encode_items(self, idx):
        ids = self.id_index[idx]
        if self.training and self.id_dropout > 0:
            ids = torch.where(torch.rand(ids.shape) < self.id_dropout, torch.zeros_like(ids), ids)
        return self.items(ids, self.text_vec[idx], self.genre_vec[idx], self.year_idx[idx])

    def encode_users(self, hist_idx):
        """hist_idx: (B, L) item indices, -1 = padding."""
        mask = hist_idx >= 0
        uniq, inv = torch.unique(hist_idx.clamp(min=0), return_inverse=True)
        H = self.encode_items(uniq)[inv]
        return self.users(H, mask)


def _right_align(seqs: list[np.ndarray], L: int) -> np.ndarray:
    out = np.full((len(seqs), L), -1, dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-L:]
        if len(s):
            out[i, L - len(s):] = s
    return out


def history_for_scoring(hist: Histories, p: int, threshold: float) -> np.ndarray:
    """Liked movies in time order; all rated movies if none are liked yet."""
    if p < 0:
        return np.array([], dtype=np.int64)
    items, ratings, _ = hist.get(p)
    liked = items[ratings >= threshold]
    return liked if len(liked) else items


class TwoTower:
    name = "twotower"

    def __init__(self, cfg: Config, movies: pd.DataFrame, text_vectors: np.ndarray):
        self.cfg = cfg
        self.text = text_vectors.astype(np.float32)
        self.genres = genre_matrix(movies)
        self.years = year_buckets(movies)
        self.net: TwoTowerNet | None = None
        self.item_vecs: np.ndarray | None = None

    # ---- training -------------------------------------------------------
    def fit(self, hist: Histories) -> "TwoTower":
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        id_index = np.zeros(len(hist.items), dtype=np.int64)
        frequent = np.flatnonzero(hist.item_counts >= cfg.tt_min_item_ratings)
        id_index[frequent] = np.arange(1, len(frequent) + 1)
        self.net = TwoTowerNet(id_index, self.text, self.genres, self.years, cfg.tt_dim, cfg.tt_history)
        self.net.id_dropout = 0.1

        liked = [hist.liked(p, cfg.like_threshold) for p in range(len(hist))]
        users_ok = [p for p, s in enumerate(liked) if len(s) >= 3]
        hold_users = rng.choice(users_ok, size=min(2000, len(users_ok)), replace=False)
        hold_hist = _right_align([liked[p][:-1] for p in hold_users], cfg.tt_history)
        hold_target = np.array([liked[p][-1] for p in hold_users])

        target_counts = np.bincount(np.concatenate([s[1:-1] for s in liked if len(s) >= 3] or [np.array([0])]),
                                    minlength=len(hist.items)).astype(np.float64)
        logq = torch.as_tensor(np.log((target_counts + 1) / (target_counts + 1).sum()), dtype=torch.float32)

        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.tt_lr)
        best, best_state, bad = -1.0, None, 0
        for epoch in range(cfg.tt_epochs):
            H, T = self._epoch_examples(liked, rng)
            self.net.train()
            order = rng.permutation(len(T))
            total = 0.0
            for s in range(0, len(order), cfg.tt_batch):
                b = order[s : s + cfg.tt_batch]
                hb, tb = torch.as_tensor(H[b]), torch.as_tensor(T[b])
                u, v = self.net.encode_users(hb), self.net.encode_items(tb)
                logits = u @ v.T / cfg.tt_temperature - logq[tb][None, :]
                dup = (tb[None, :] == tb[:, None]) & ~torch.eye(len(tb), dtype=torch.bool)
                logits = logits.masked_fill(dup, -1e9)
                loss = F.cross_entropy(logits, torch.arange(len(tb)))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            rec = self._holdout_recall(hist, hold_users, hold_hist, hold_target)
            log.info("two-tower epoch %d: loss %.4f, holdout recall@100 %.4f", epoch + 1, total / len(T), rec)
            if rec > best:
                best, bad = rec, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.tt_patience:
                    break
        self.net.load_state_dict(best_state)
        self._refresh_item_vectors()
        return self

    def _epoch_examples(self, liked: list[np.ndarray], rng) -> tuple[np.ndarray, np.ndarray]:
        seqs, targets = [], []
        for s in liked:
            if len(s) < 3:
                continue
            positions = np.arange(1, len(s) - 1)  # last liked movie is the holdout
            if len(positions) > self.cfg.tt_targets_per_user:
                positions = rng.choice(positions, self.cfg.tt_targets_per_user, replace=False)
            for t in positions:
                seqs.append(s[max(0, t - self.cfg.tt_history) : t])
                targets.append(s[t])
        return _right_align(seqs, self.cfg.tt_history), np.array(targets, dtype=np.int64)

    @torch.no_grad()
    def _holdout_recall(self, hist, users, hold_hist, hold_target, k: int = 100) -> float:
        self.net.eval()
        I = self._all_item_vectors()
        hits = 0
        for s in range(0, len(users), 512):
            U = self.net.encode_users(torch.as_tensor(hold_hist[s : s + 512])).numpy()
            S = U @ I.T
            for b, p in enumerate(users[s : s + 512]):
                seen = hist.get(p)[0]
                S[b, seen[seen != hold_target[s + b]]] = -np.inf
            idx, _ = topk(S, k)
            hits += int((idx == hold_target[s : s + 512, None]).any(axis=1).sum())
        return hits / len(users)

    @torch.no_grad()
    def _all_item_vectors(self) -> np.ndarray:
        self.net.eval()
        n = len(self.text)
        return np.vstack([self.net.encode_items(torch.arange(s, min(s + 8192, n))).numpy()
                          for s in range(0, n, 8192)])

    def _refresh_item_vectors(self) -> None:
        self.item_vecs = self._all_item_vectors().astype(np.float32)

    # ---- scoring --------------------------------------------------------
    @torch.no_grad()
    def user_vectors(self, hist: Histories, rows: np.ndarray) -> np.ndarray:
        self.net.eval()
        seqs = [history_for_scoring(hist, p, self.cfg.like_threshold) for p in rows]
        H = torch.as_tensor(_right_align(seqs, self.cfg.tt_history))
        return self.net.encode_users(H).numpy().astype(np.float32)

    def score(self, hist: Histories, rows: np.ndarray) -> np.ndarray:
        return self.user_vectors(hist, rows) @ self.item_vecs.T

    # ---- persistence ----------------------------------------------------
    def save(self, directory: Path) -> None:
        torch.save({"state": self.net.state_dict(), "id_index": self.net.id_index.numpy()}, directory / "twotower.pt")
        np.save(directory / "twotower_items.npy", self.item_vecs)

    @classmethod
    def load(cls, directory: Path, cfg: Config, movies: pd.DataFrame, text_vectors: np.ndarray) -> "TwoTower":
        obj = cls(cfg, movies, text_vectors)
        ckpt = torch.load(directory / "twotower.pt", weights_only=False)
        obj.net = TwoTowerNet(ckpt["id_index"], obj.text, obj.genres, obj.years, cfg.tt_dim, cfg.tt_history)
        obj.net.load_state_dict(ckpt["state"])
        obj.net.eval()
        obj.item_vecs = np.load(directory / "twotower_items.npy")
        return obj
