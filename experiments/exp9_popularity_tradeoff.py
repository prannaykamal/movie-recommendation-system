"""Accuracy vs exposure trade-off for the FINAL system: re-rank with a popularity penalty

    S'(u, i) = S_final(u, i) - lambda * P(i),   P(i) = min-max scaled log(1 + training count)

and report NDCG@10 against coverage / long-tail share / novelty / Gini for a grid of lambda.
This is a product decision (how much accuracy to trade for exposure), so the script reports the
whole frontier on validation and test instead of picking a value."""
import numpy as np
import pandas as pd
from exp_common import *
from exp8_robustness import FINAL, GENRES
from recsys.evaluation.beyond import beyond_accuracy
from recsys.evaluation.evaluate import collab_scores, fit_models, popularity_prior
from recsys.evaluation.metrics import ranking_metrics
from recsys.evaluation.split import temporal_split

LAMBDAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)


def frontier(train, test, label):
    f = fit_models(DATA, train, FINAL)
    rel, uidx = rel_and_users(test)
    counts = f.item_scores["most_popular"]
    P = minmax_rows(np.log1p(counts)[None].astype(np.float32))[0]
    pop = popularity_prior(f)
    h = DynamicHybridRecommender(FINAL.alpha_max, FINAL.gamma, FINAL.pop_eta)
    recs = {lam: [] for lam in LAMBDAS}
    for s in range(0, len(uidx), 1000):
        b = uidx[s:s + 1000]
        S = h.blend(f.n_u[b], f.content.score(f.R[b]), collab_scores(f, b, FINAL), pop)
        seen = [f.R[u].indices for u in b]
        for lam in LAMBDAS:
            recs[lam].append(DynamicHybridRecommender.top_k(S - lam * P, seen, 10))
    rows = []
    for lam, r in recs.items():
        r = np.vstack(r)
        rows.append({"setting": label, "lambda": lam,
                     **{k: v for k, v in ranking_metrics(r, [rel[u] for u in uidx], 10).items() if k in ("ndcg@10", "recall@10")},
                     **beyond_accuracy(r, counts, int((f.n_u > 0).sum()), GENRES, 10)})
    df = pd.DataFrame(rows)
    df["ndcg_change"] = df["ndcg@10"] / df["ndcg@10"].iloc[0] - 1
    print(df.round(4).to_string(index=False), flush=True)
    return df


r_fit, r_val = split("val")
t_train, t_test = split("temporal")
t_fit, t_val = temporal_split(t_train, CFG.test_frac)
out = pd.concat([frontier(r_fit, r_val, "val random"), frontier(t_fit, t_val, "val temporal"),
                 frontier(*split("random"), "test random"), frontier(t_train, t_test, "test temporal")])
out.to_csv(RESULTS / "exp9_popularity_tradeoff.csv", index=False)
