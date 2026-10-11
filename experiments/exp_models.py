"""Base engines fitted on one training frame: SVD, iALS, popularity-gated content, popularity."""
from dataclasses import replace

import numpy as np
from exp_common import *
from exp2_ranking_mf import fit_als
from recsys.evaluation.split import temporal_split

NEW = replace(CFG, w_meta=0.8, w_plot=0.2)
ALS = dict(factors=32, reg=3.0, mode="all", conf=3, iters=30)


class Base:
    def __init__(self, train, cfg=NEW, als=ALS):
        self.f = fit_models(DATA, train, cfg)
        self.R = self.f.R
        self.U, self.V = fit_als(self.R, **als)
        self.counts = item_counts(self.R)
        self.logpop = np.log1p(self.counts)
        self.pop = minmax_rows(self.counts[None])[0]

    def content(self, b):
        return minmax_rows(self.f.content.score(self.R[b]) * self.logpop)

    def svd(self, b):
        return self.f.collab.score(b)

    def ials(self, b):
        return minmax_rows(self.U[b] @ self.V.T)

    def hybrid(self, b, beta=0.5, alpha_max=0.8, gamma=0.3, content=None):
        collab = beta * self.ials(b) + (1 - beta) * self.svd(b)
        c = self.content(b) if content is None else content
        return DynamicHybridRecommender(alpha_max, gamma).blend(self.f.n_u[b], c, collab, self.pop)


def val_splits():
    """random-style and temporal-style validation, both carved from the corresponding train split."""
    r_fit, r_val = split("val")
    t_train, _ = split("temporal")
    t_fit, t_val = temporal_split(t_train, CFG.test_frac)
    return {"random": (r_fit, r_val), "temporal": (t_fit, t_val)}
