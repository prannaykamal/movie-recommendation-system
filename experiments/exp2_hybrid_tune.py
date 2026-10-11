"""Tune the hybrid blend (alpha_max, w_meta, gamma) on validation with popularity-gated content,
for both collaborative backends (SVD k=32 and iALS)."""
import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
from exp_common import *
from exp2_ranking_mf import fit_als

ALS = dict(factors=32, reg=3.0, mode="all", conf=3, iters=30)

fit_tr, val = split("val")
users, uidx_s = sample_users(1500)
rel, uidx = rel_and_users(val, uidx_s)


def engines(train, cfg):
    f = fit_models(DATA, train, cfg)
    U, V = fit_als(f.R, **ALS)
    return f, {"svd": lambda b: f.collab.score(b), "ials": lambda b: minmax_rows(U[b] @ V.T)}


rows = []
for w in (0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0):
    cfg = replace(CFG, w_meta=w, w_plot=1 - w)
    f, collab = engines(fit_tr, cfg)
    logpop = np.log1p(item_counts(f.R))
    content = {}
    for backend, cfn in collab.items():
        for a in (0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
            def score(b):
                if (b[0], len(b)) not in content:
                    content.clear(); content[(b[0], len(b))] = minmax_rows(f.content.score(f.R[b]) * logpop)
                c = content[(b[0], len(b))]
                return a * cfn(b) + (1 - a) * c
            m = evaluate(score, f.R, rel, uidx)
            rows.append({"backend": backend, "w_meta": w, "alpha_max": a, "ndcg@10": m["ndcg@10"]})
    print(pd.DataFrame(rows).tail(12).round(4).to_string(index=False), flush=True)
s1 = pd.DataFrame(rows)
best = s1.loc[s1.groupby("backend")["ndcg@10"].idxmax()]
print("stage1 best:\n", best.round(4).to_string(index=False))

rows2 = []
for _, b in best.iterrows():
    cfg = replace(CFG, w_meta=b.w_meta, w_plot=1 - b.w_meta)
    per_n = {}
    for n in (3, 5, 10, 20):
        f, collab = engines(truncate_history(fit_tr, users, n), cfg)
        logpop = np.log1p(item_counts(f.R))
        c_all = minmax_rows(f.content.score(f.R[uidx]) * logpop)
        f_all = collab[b.backend](uidx)
        pos = {u: r for r, u in enumerate(uidx)}
        for g in (0.05, 0.15, 0.3, 0.6, 1.0, 100.0):
            fn = lambda bb: hybrid(f.n_u[bb], c_all[[pos[u] for u in bb]], f_all[[pos[u] for u in bb]],
                                   alpha_max=b.alpha_max, gamma=g)
            per_n.setdefault(g, []).append(evaluate(fn, f.R, rel, uidx)["ndcg@10"])
    for g, v in per_n.items():
        rows2.append({"backend": b.backend, "gamma": g, **{f"N={n}": x for n, x in zip((3, 5, 10, 20), v)},
                      "mean": np.mean(v)})
s2 = pd.DataFrame(rows2)
print(s2.round(4).to_string(index=False))
s1.to_csv(RESULTS / "exp2_hybrid_s1.csv", index=False)
s2.to_csv(RESULTS / "exp2_hybrid_s2.csv", index=False)
