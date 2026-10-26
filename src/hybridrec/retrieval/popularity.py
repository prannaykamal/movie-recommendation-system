"""Most-liked movies in the fit window: the baseline every source must beat."""
from __future__ import annotations

import numpy as np

from hybridrec.data.dataset import Histories


class Popularity:
    name = "popularity"

    def __init__(self, like_threshold: float = 4.0):
        self.like_threshold = like_threshold
        self.scores: np.ndarray | None = None

    def fit(self, hist: Histories) -> "Popularity":
        liked = hist.item_idx[hist.rating >= self.like_threshold]
        counts = np.bincount(liked, minlength=len(hist.items))
        self.scores = np.log1p(counts).astype(np.float32)
        return self

    def score(self, hist: Histories, rows: np.ndarray) -> np.ndarray:
        return np.tile(self.scores, (len(rows), 1))
