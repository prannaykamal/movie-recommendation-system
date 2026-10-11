"""Suggestion 4: two-stage pipeline. Stage 1 = top-N candidates from the (accepted) hybrid;
stage 2 = LightGBM LambdaRank over per-(user,item) features.
Training: base models on <fit> (validation carve-out), labels from <val>.
Testing: base models refitted on the full train, labels from test (never seen in training)."""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from exp_models import *

N_CAND = 100
genres = DATA.items["genres"].str.split("|")
ALL_G = sorted({g for gs in genres for g in gs})
G = np.array([[g in gs for g in ALL_G] for gs in genres], dtype=np.float32)
YEAR = DATA.items["year"].fillna(1990).to_numpy(np.float32)
HAS_PLOT = DATA.items["has_plot"].to_numpy(np.float32)


def candidates_and_features(B: Base, uidx, batch=500):
    feats, keys = [], []
    bayes = B.f.item_scores["bayesian_average"].astype(np.float32)
    avg = B.f.item_scores["average_rating"].astype(np.float32)
    for s in range(0, len(uidx), batch):
        b = uidx[s:s + batch]
        c_meta, c_plot = B.f.content.channel_scores(B.R[b])
        content = B.content(b)
        svd, ials = B.svd(b), B.ials(b)
        svd_raw = B.f.collab.predict_raw(b)
        ials_raw = B.U[b] @ B.V.T
        hyb = B.hybrid(b, content=content)
        top = DynamicHybridRecommender.top_k(hyb, [B.R[u].indices for u in b], N_CAND)
        Wpos = B.f.content.profile_weights(B.R[b])
        ugenre = np.asarray(Wpos @ G)
        ugenre = ugenre / np.maximum(ugenre.sum(1, keepdims=True), 1e-9)
        n_u = B.f.n_u[b].astype(np.float32)
        for r, u in enumerate(b):
            it = top[r]
            take = lambda M: M[r, it]
            feats.append(np.column_stack([
                take(hyb), np.arange(N_CAND, dtype=np.float32), take(svd), take(svd_raw), take(ials), take(ials_raw),
                take(content), take(c_meta), np.nan_to_num(take(c_plot), nan=-1), B.logpop[it], bayes[it], avg[it],
                np.full(N_CAND, np.log1p(n_u[r])), np.full(N_CAND, B.f.collab.user_means[u]),
                YEAR[it], HAS_PLOT[it], G[it] @ ugenre[r], svd_raw[r, it] - B.f.collab.user_means[u],
            ]))
            keys.append(np.column_stack([np.full(N_CAND, u), it]))
    cols = ["hybrid", "hyb_rank", "svd", "svd_raw", "ials", "ials_raw", "content", "meta_cos", "plot_cos", "logpop",
            "bayes", "avg_rating", "log_n_u", "user_mean", "year", "has_plot", "genre_affinity", "svd_dev"]
    return np.vstack(keys), pd.DataFrame(np.vstack(feats), columns=cols)


def labels(keys, rel):
    return np.array([i in rel.get(u, ()) for u, i in keys], dtype=np.int32)


def train_ranker(split_name, n_users=3000):
    fit, val = val_splits()[split_name]
    B = Base(fit)
    rel, _ = rel_and_users(val)
    _, uidx = sample_users(n_users, seed=7)
    uidx = np.array([u for u in uidx if u in rel])
    keys, X = candidates_and_features(B, uidx)
    y = labels(keys, rel)
    print(f"[{split_name}] train rows {len(y)}, positives {y.mean():.3f}, candidate recall "
          f"{y.sum() / sum(len(rel[u]) for u in uidx):.3f}", flush=True)
    # hold out 20% of the training users for early stopping
    users = np.unique(keys[:, 0])
    rng = np.random.default_rng(0)
    es_users = set(rng.choice(users, size=len(users) // 5, replace=False))
    m_es = np.array([u in es_users for u in keys[:, 0]])
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=1000, learning_rate=0.05, num_leaves=31,
                           min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           verbose=-1, random_state=0)
    model.fit(X[~m_es], y[~m_es], group=[N_CAND] * (int((~m_es).sum()) // N_CAND),
              eval_set=[(X[m_es], y[m_es])], eval_group=[[N_CAND] * (int(m_es.sum()) // N_CAND)],
              eval_at=[10], callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f"[{split_name}] best iteration {model.best_iteration_}", flush=True)
    return model


def test_ranker(model, train, test, users=None, label=""):
    B = Base(train)
    rel, uidx = rel_and_users(test, users)
    keys, X = candidates_and_features(B, uidx)
    pred = model.predict(X).reshape(len(uidx), N_CAND)
    items = keys[:, 1].reshape(len(uidx), N_CAND)
    reranked = np.take_along_axis(items, np.argsort(-pred, axis=1), axis=1)
    from recsys.evaluation.metrics import ranking_metrics
    relevant = [rel[u] for u in uidx]
    out = {"setting": label,
           "hybrid (stage 1 only)": ranking_metrics(items, relevant, 10)["ndcg@10"],
           "hybrid + LightGBM rerank": ranking_metrics(reranked, relevant, 10)["ndcg@10"]}
    print(out, flush=True)
    return out, model


if __name__ == "__main__":
    # Each ranker is tested on its own protocol AND transferred to the other settings, to check
    # whether its gains are real or specific to the split it was trained on.
    rows = []
    train, test = split("random")
    users, _ = sample_users(1000)
    for ranker_split in ("random", "temporal"):
        model = train_ranker(ranker_split)
        imp = pd.Series(model.booster_.feature_importance("gain"), index=model.booster_.feature_name())
        print((imp / imp.sum()).sort_values(ascending=False).head(8).round(3).to_string(), flush=True)
        for split_name in ("random", "temporal"):
            out = test_ranker(model, *split(split_name), label=split_name)[0]
            rows.append({"ranker_trained_on": ranker_split, **out})
        for n in (3, 5, 10, 20):
            out = test_ranker(model, truncate_history(train, users, n), test, [IDX.uidx(u) for u in users],
                              f"cold N={n}")[0]
            rows.append({"ranker_trained_on": ranker_split, **out})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    pd.DataFrame(rows).to_csv(RESULTS / "exp4.csv", index=False)
