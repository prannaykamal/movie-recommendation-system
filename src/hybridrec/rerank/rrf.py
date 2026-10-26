"""Reciprocal rank fusion: the no-learning baseline the reranker must beat."""
from __future__ import annotations

import numpy as np
import pandas as pd

from hybridrec.retrieval.merge import SOURCES

RRF_K = 60


def rrf_scores(cands: pd.DataFrame, k0: int = RRF_K) -> np.ndarray:
    total = np.zeros(len(cands), dtype=np.float64)
    for src in SOURCES:
        r = cands[f"{src}_rank"].to_numpy().astype(np.float64)
        total += np.where(r > 0, 1.0 / (k0 + r), 0.0)
    return total
