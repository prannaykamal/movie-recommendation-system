"""Reranker networks.

v1: MLP over the flat (normalized) features.
v2: v1's features plus a candidate vector, an attention summary of the user's
    last liked movies weighted by similarity to the candidate, and 2 DCN-v2
    cross layers before the MLP.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def mlp(n_in: int, hidden=(256, 128, 64), dropout: float = 0.25) -> nn.Sequential:
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


class MLPReranker(nn.Module):
    kind = "v1"

    def __init__(self, n_features: int, dropout: float = 0.25):
        super().__init__()
        self.net = mlp(n_features, dropout=dropout)

    def forward(self, x, item_idx=None, row_user=None, user_hist=None):
        return self.net(x).squeeze(-1)


class CrossLayer(nn.Module):
    """DCN-v2: x_{l+1} = x0 * (W x_l + b) + x_l."""

    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x0, x):
        return x0 * self.lin(x) + x


class AttnDCNReranker(nn.Module):
    kind = "v2"

    def __init__(self, n_features: int, text, genres, d: int = 32, n_cross: int = 2, dropout: float = 0.25):
        super().__init__()
        # Movie vectors use only stage-independent inputs (text, genres): each stage's
        # two-tower model has its own vector space, so those vectors can't be inputs here.
        self.register_buffer("text", torch.as_tensor(text, dtype=torch.float32), persistent=False)
        self.register_buffer("genres", torch.as_tensor(genres, dtype=torch.float32), persistent=False)
        self.text_proj = nn.Linear(text.shape[1], 32)
        self.item_proj = nn.Sequential(nn.Linear(32 + genres.shape[1], d), nn.ReLU(), nn.Linear(d, d))
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        n_in = n_features + 3 * d
        self.cross = nn.ModuleList(CrossLayer(n_in) for _ in range(n_cross))
        self.deep = mlp(n_in, dropout=dropout)
        self.d = d

    def item_repr(self, idx):
        return self.item_proj(torch.cat([self.text_proj(self.text[idx]), self.genres[idx]], dim=-1))

    def forward(self, x, item_idx, row_user, user_hist):
        """x: (R, F); item_idx: (R,); row_user: (R,) index into user_hist (U, L), -1 padded."""
        c = self.item_repr(item_idx)
        mask = user_hist >= 0
        uniq, inv = torch.unique(user_hist.clamp(min=0), return_inverse=True)
        H = self.item_repr(uniq)[inv] * mask.unsqueeze(-1)  # (U, L, d)
        K = self.k(H)[row_user]  # (R, L, d)
        att = (K * self.q(c).unsqueeze(1)).sum(-1) / self.d ** 0.5
        m = mask[row_user]
        att = torch.softmax(att.masked_fill(~m, -1e9), dim=1) * m
        interest = (att.unsqueeze(-1) * H[row_user]).sum(1)
        x0 = torch.cat([x, c, interest, c * interest], dim=-1)
        z = x0
        for layer in self.cross:
            z = layer(x0, z)
        return self.deep(z).squeeze(-1)
