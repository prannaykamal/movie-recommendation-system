"""Final test-set comparison: current system vs accepted changes (params fixed from validation)."""
from dataclasses import replace

import numpy as np
import pandas as pd
from exp_common import *
from exp2_ranking_mf import fit_als

ALS = dict(factors=32, reg=3.0, mode="all", conf=3, iters=30)
NEW = replace(CFG, w_meta=0.8, w_plot=0.2)
rows = []


def run(label, train, test, users=None):
    f_old = fit_models(DATA, train, CFG)
    f_new = fit_models(DATA, train, NEW)
    R = f_old.R
    U, V = fit_als(R, **ALS)
    rel, uidx = rel_and_users(test, users)
    logpop = np.log1p(item_counts(R))
    cnt = minmax_rows(item_counts(R)[None])[0]
    bayes = minmax_rows(f_old.item_scores["bayesian_average"][None].astype(np.float32))[0]
    variants = {
        "current: svd hybrid (α_max .8, γ 1)": lambda b: DynamicHybridRecommender(CFG.alpha_max, CFG.gamma).blend(
            f_old.n_u[b], f_old.content.score(R[b]), f_old.collab.score(b), bayes),
        "+ pop-gated content (svd)": lambda b: DynamicHybridRecommender(CFG.alpha_max, CFG.gamma).blend(
            f_old.n_u[b], minmax_rows(f_old.content.score(R[b]) * logpop), f_old.collab.score(b), cnt),
        "iALS alone": lambda b: minmax_rows(U[b] @ V.T),
        "+ iALS + pop-gated content (α_max .7, γ .3)": lambda b: DynamicHybridRecommender(0.7, 0.3).blend(
            f_new.n_u[b], minmax_rows(f_new.content.score(R[b]) * logpop), minmax_rows(U[b] @ V.T), cnt),
    }
    for name, fn in variants.items():
        m = evaluate(fn, R, rel, uidx)
        rows.append({"setting": label, "model": name, **fmt(m)})
    print(pd.DataFrame(rows[-len(variants):]).to_string(index=False), flush=True)


for s in ("random", "temporal"):
    run(s, *split(s))
train, test = split("random")
users, _ = sample_users(1000)
for n in (0, 3, 5, 10, 20):
    run(f"cold N={n}", truncate_history(train, users, n), test, [IDX.uidx(u) for u in users])
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "final_test.csv", index=False)
order = ["random", "temporal"] + [f"cold N={n}" for n in (0, 3, 5, 10, 20)]
print(df.pivot(index="model", columns="setting", values="ndcg")[order].round(4).to_string())
