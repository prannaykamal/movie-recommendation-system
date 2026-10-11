"""Suggestion 2: ranking-oriented MF (iALS / BPR via `implicit`) instead of TruncatedSVD.
Hyperparameters tuned on the validation split (carved from train); test is only used at the end."""
import itertools
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from implicit.cpu.bpr import BayesianPersonalizedRanking
from exp_common import *


def interactions(R, mode, conf):
    """mode=all: every rating is a positive (confidence grows with stars); mode=pos: only r >= 4."""
    X = R.copy().astype(np.float32)
    if mode == "pos":
        X.data = (X.data >= 4).astype(np.float32)
        X.eliminate_zeros()
        X.data *= conf
    else:
        X.data = 1.0 + conf * (X.data - 1) / 4  # 1..(1+conf)
    return X.tocsr()


def fit_als(R, factors, reg, mode, conf, iters=15):
    m = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iters, random_state=CFG.seed,
                                calculate_training_loss=False)
    m.fit(interactions(R, mode, conf), show_progress=False)
    return m.user_factors, m.item_factors


def fit_bpr(R, factors, lr, reg, mode, iters=100):
    m = BayesianPersonalizedRanking(factors=factors, learning_rate=lr, regularization=reg, iterations=iters,
                                    random_state=CFG.seed)
    m.fit(interactions(R, mode, 1.0), show_progress=False)
    return m.user_factors, m.item_factors


def mf_scorer(U, V):
    return lambda b: minmax_rows(U[b] @ V.T)


stage = sys.argv[1] if len(sys.argv) > 1 else "tune"
if __name__ == "__main__" and stage == "tune":
    fit_tr, val = split("val")
    users, uidx_s = sample_users(1500)
    R, _ = matrices(fit_tr)
    rel, uidx = rel_and_users(val, uidx_s)
    rows = []
    for factors, reg, mode, conf in itertools.product((64, 128), (0.01, 0.1, 1.0), ("all", "pos"), (1, 5, 20)):
        with Timer():
            U, V = fit_als(R, factors, reg, mode, conf)
            m = evaluate(mf_scorer(U, V), R, rel, uidx)
        rows.append({"model": "ials", "factors": factors, "reg": reg, "mode": mode, "conf": conf, "ndcg@10": m["ndcg@10"]})
        print(rows[-1], flush=True)
    for factors, lr, reg, mode in itertools.product((64, 128), (0.01, 0.05), (0.001, 0.01), ("all", "pos")):
        with Timer():
            U, V = fit_bpr(R, factors, lr, reg, mode)
            m = evaluate(mf_scorer(U, V), R, rel, uidx)
        rows.append({"model": "bpr", "factors": factors, "lr": lr, "reg": reg, "mode": mode, "ndcg@10": m["ndcg@10"]})
        print(rows[-1], flush=True)
    f = fit_models(DATA, fit_tr, CFG)
    m = evaluate(lambda b: f.collab.score(b), R, rel, uidx)
    rows.append({"model": "svd(k=32)", "ndcg@10": m["ndcg@10"]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "exp2_tune.csv", index=False)
    print(df.sort_values("ndcg@10", ascending=False).groupby("model").head(3).to_string(index=False))

if __name__ == "__main__" and stage == "refine":
    # second round around the stage-1 optimum (factors=64, conf=5 sat on grid edges), then smaller models
    fit_tr, val = split("val")
    _, uidx_s = sample_users(1500)
    R, _ = matrices(fit_tr)
    rel, uidx = rel_and_users(val, uidx_s)
    rows = []
    grid = list(itertools.product((32, 64, 96), (1.0, 3.0, 10.0), (3, 5, 8), (15, 30))) + \
        [(f_, 3.0, c, 30) for f_, c in itertools.product((16, 24, 48), (1, 2, 3))]
    for factors, reg, conf, it in grid:
        U, V = fit_als(R, factors, reg, "all", conf, it)
        rows.append({"factors": factors, "reg": reg, "conf": conf, "iters": it,
                     "ndcg@10": evaluate(mf_scorer(U, V), R, rel, uidx)["ndcg@10"]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "exp2_refine.csv", index=False)
    print(df.sort_values("ndcg@10", ascending=False).head(10).round(4).to_string(index=False))
