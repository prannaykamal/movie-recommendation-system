"""Suggestion 3: recency decay w(t) on the content profile (and optionally the SVD input).
Time-based (half-life in days) and order-based (half-life in number of ratings) variants."""
import numpy as np
import pandas as pd
from scipy import sparse
from exp_common import *
from recsys.models.collaborative import CollaborativeEngine


def decay_matrix(R, T, kind, half_life):
    """Same sparsity as R; weight in (0, 1] by recency relative to the user's latest rating."""
    D = R.copy().astype(np.float32)
    for u in range(R.shape[0]):
        s, e = T.indptr[u], T.indptr[u + 1]
        if s == e:
            continue
        t = T.data[s:e]
        if kind == "days":
            age = (t.max() - t) / 86400.0
        else:  # order: 0 = most recent rating
            age = (-t).argsort(kind="stable").argsort().astype(np.float64)
        D.data[s:e] = 0.5 ** (age / half_life)
    return D


rows = []
for split_name in ("temporal", "random"):
    train, test = split(split_name)
    f = fit_models(DATA, train, CFG)
    R = f.R
    _, T = matrices(train)
    rel, uidx = rel_and_users(test)
    logpop = np.log1p(item_counts(R))
    collab = {"svd": f.collab}

    def content_scores(b, D):
        eng = f.content
        orig = eng.profile_weights
        if D is not None:
            eng.profile_weights = lambda Rb, _b=b: orig(Rb).multiply(D[_b]).tocsr()
        try:
            return minmax_rows(eng.score(R[b]) * logpop)
        finally:
            eng.profile_weights = orig

    configs = [("none", None, None)] + [("days", h, None) for h in (7, 30, 180)] + \
              [("order", h, None) for h in (10, 25, 50, 100)]
    for kind, h, _ in configs:
        D = None if kind == "none" else decay_matrix(R, T, kind, h)
        m = evaluate(lambda b: hybrid(f.n_u[b], content_scores(b, D), f.collab.score(b)), R, rel, uidx)
        rows.append({"split": split_name, "target": "content", "decay": f"{kind}:{h}", "ndcg@10": m["ndcg@10"]})
        print(rows[-1])
    # decay applied to SVD input as well (weights on centred ratings)
    for kind, h in (("order", 25), ("order", 50), ("days", 30)):
        D = decay_matrix(R, T, kind, h)
        eng = CollaborativeEngine(CFG.n_components, CFG.seed, CFG.mean_shrinkage)
        eng.global_mean = float(R.data.mean())
        C, eng.user_means = eng._center(R)
        C = C.multiply(D).tocsr()
        eng.user_factors = eng.svd.fit_transform(C).astype(np.float32)
        eng.item_factors = eng.svd.components_.T.astype(np.float32)
        m = evaluate(lambda b: hybrid(f.n_u[b], content_scores(b, D), eng.score(b)), R, rel, uidx)
        rows.append({"split": split_name, "target": "content+svd", "decay": f"{kind}:{h}", "ndcg@10": m["ndcg@10"]})
        print(rows[-1])

df = pd.DataFrame(rows)
print(df.pivot_table(index=["target", "decay"], columns="split", values="ndcg@10").round(4).to_string())
df.to_csv(RESULTS / "exp3.csv", index=False)
