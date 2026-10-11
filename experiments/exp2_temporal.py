"""Is iALS's temporal-split loss a tuning artefact? Tune on a temporal validation split
(last 20% of each user's temporal-train), then test; also try an iALS+SVD collaborative ensemble."""
import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
from exp_common import *
from exp2_ranking_mf import fit_als
from recsys.evaluation.split import temporal_split

NEW = replace(CFG, w_meta=0.8, w_plot=0.2)
t_train, t_test = split("temporal")
v_train, v_val = temporal_split(t_train, CFG.test_frac)
_, uidx_s = sample_users(1500)
R, _ = matrices(v_train)
rel, uidx = rel_and_users(v_val, uidx_s)

rows = []
for factors, reg, conf in itertools.product((16, 32, 64), (3.0, 30.0, 100.0), (1, 3, 10)):
    U, V = fit_als(R, factors, reg, "all", conf, 30)
    rows.append({"factors": factors, "reg": reg, "conf": conf,
                 "ndcg": evaluate(lambda b: U[b] @ V.T, R, rel, uidx)["ndcg@10"]})
g = pd.DataFrame(rows).sort_values("ndcg", ascending=False)
print(g.head(5).round(4).to_string(index=False))
f = fit_models(DATA, v_train, NEW)
print("svd on temporal-val:", round(evaluate(lambda b: f.collab.score(b), R, rel, uidx)["ndcg@10"], 4))
best = g.iloc[0]
ALS_T = dict(factors=int(best.factors), reg=float(best.reg), mode="all", conf=float(best.conf), iters=30)

# blend weights on temporal-val
U, V = fit_als(R, **ALS_T)
logpop = np.log1p(item_counts(R))
C = minmax_rows(f.content.score(R[uidx]) * logpop)
I_ = minmax_rows(U[uidx] @ V.T)
S_ = f.collab.score(uidx)
pos = {u: r for r, u in enumerate(uidx)}
res = []
for beta, a in itertools.product((0.0, 0.25, 0.5, 0.75, 1.0), (0.6, 0.7, 0.8, 0.9)):
    fn = lambda b: a * (beta * I_[[pos[u] for u in b]] + (1 - beta) * S_[[pos[u] for u in b]]) + (1 - a) * C[[pos[u] for u in b]]
    res.append({"beta_ials": beta, "alpha_max": a, "ndcg": evaluate(fn, R, rel, uidx)["ndcg@10"]})
res = pd.DataFrame(res).sort_values("ndcg", ascending=False)
print(res.head(6).round(4).to_string(index=False))

# temporal TEST with temporal-tuned params
print("ALS_T", ALS_T)
f = fit_models(DATA, t_train, NEW)
Rt = f.R
U, V = fit_als(Rt, **ALS_T)
U0, V0 = fit_als(Rt, factors=32, reg=3.0, mode="all", conf=3, iters=30)
rel, uidx = rel_and_users(t_test)
logpop = np.log1p(item_counts(Rt))
for name, coll in {"svd": lambda b: f.collab.score(b),
                   "ials (random-val params)": lambda b: minmax_rows(U0[b] @ V0.T),
                   "ials (temporal-val params)": lambda b: minmax_rows(U[b] @ V.T),
                   "0.5 ials(rv) + 0.5 svd": lambda b: 0.5 * minmax_rows(U0[b] @ V0.T) + 0.5 * f.collab.score(b)}.items():
    for a in (0.7, 0.8):
        m = evaluate(lambda b: a * coll(b) + (1 - a) * minmax_rows(f.content.score(Rt[b]) * logpop), Rt, rel, uidx)
        print(f"temporal test | {name:28s} α={a} ndcg@10={m['ndcg@10']:.4f}", flush=True)
