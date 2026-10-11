"""Shared harness for evaluating the suggestions.md proposals under the project's protocol.

Run any script from the repository root, e.g. `python experiments/exp1_popgate.py`, after
`python -m recsys.cli extract align embed` (the scripts need items.parquet and the plot embeddings).

CFG is pinned to the *baseline* the experiments were run against (SVD-only collaborative engine,
ungated content, the first tuning round's parameters), not to the current project defaults, which
already include the accepted changes."""
import os
import sys
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # silences a joblib core-count probe on Windows
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from recsys.config import Config
from recsys.data.interactions import minmax_rows, rating_matrix
from recsys.evaluation.evaluate import fit_models, load_data, relevant_sets
from recsys.evaluation.metrics import ranking_metrics
from recsys.evaluation.split import random_split, temporal_split, truncate_history
from recsys.models.hybrid import DynamicHybridRecommender

pd.set_option("display.width", 220)
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
CFG = replace(Config(), n_components=32, w_meta=0.5, w_plot=0.5, alpha_max=0.8, gamma=1.0,
              content_pop_gate=False, ials_weight=0.0, pop_eta=None, center_plots=False, pop_penalty=0.0)
DATA = load_data(CFG)
IDX = DATA.index


@lru_cache(None)
def split(name):
    if name == "random":
        return random_split(DATA.ratings, CFG.test_frac, CFG.seed)
    if name == "temporal":
        return temporal_split(DATA.ratings, CFG.test_frac)
    if name == "val":  # validation carved from the random-split train (test untouched)
        tr, _ = split("random")
        return random_split(tr, CFG.test_frac, CFG.seed + 1)
    raise ValueError(name)


def matrices(train):
    enc = IDX.encode(train)
    R = rating_matrix(enc, IDX.n_users, IDX.n_items)
    T = rating_matrix(enc.assign(rating=enc["timestamp"].astype(np.float64)), IDX.n_users, IDX.n_items)
    return R, T


def rel_and_users(test, users=None):
    rel = relevant_sets(DATA, test, CFG.pos_threshold)
    uidx = np.array(sorted(rel) if users is None else [u for u in users if u in rel], dtype=np.int64)
    return rel, uidx


def evaluate(score_fn, R, rel, uidx, k=10, batch=1000):
    recs = []
    for s in range(0, len(uidx), batch):
        b = uidx[s:s + batch]
        recs.append(DynamicHybridRecommender.top_k(score_fn(b), [R[u].indices for u in b], k))
    return ranking_metrics(np.vstack(recs), [rel[u] for u in uidx], k)


def sample_users(n, seed=None):
    rng = np.random.default_rng(CFG.seed if seed is None else seed)
    users = rng.choice(IDX.user_ids, size=min(n, IDX.n_users), replace=False)
    return users, np.array([IDX.uidx(u) for u in users])


def item_counts(R):
    return np.diff(R.tocsc().indptr).astype(np.float32)


def hybrid(n_u, content, collab, alpha_max=None, gamma=None):
    h = DynamicHybridRecommender(CFG.alpha_max if alpha_max is None else alpha_max,
                                 CFG.gamma if gamma is None else gamma)
    return h.blend(n_u, content, collab)


class Timer:
    def __enter__(self):
        self.t = time.time(); return self

    def __exit__(self, *a):
        print(f"  [{time.time() - self.t:.1f}s]")


def fmt(m):
    return {k.split("@")[0]: round(v, 4) for k, v in m.items()}
