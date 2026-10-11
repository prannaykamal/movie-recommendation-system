"""Suggestion 5: onboarding for N_u = 0. Simulate signup answers from each sampled user's (hidden)
training history: their 3 most-liked genres, or their 3 favourite movies."""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
from exp_common import *
from recsys.features.metadata import build_metadata_matrix
from recsys.models import baselines

train, test = split("random")
users, uidx_s = sample_users(1000)
cold_train = truncate_history(train, users, 0)            # sampled users have no history
f = fit_models(DATA, cold_train, CFG)
R0 = f.R
rel, uidx = rel_and_users(test, uidx_s)
counts = item_counts(R0)
logpop = np.log1p(counts)
bayes = minmax_rows(f.item_scores["bayesian_average"][None].astype(np.float32))[0]
mostpop = minmax_rows(counts[None])[0]

# simulated signup answers, from the hidden full training history
R_full, _ = matrices(train)
genres = DATA.items["genres"].str.split("|")
all_genres = sorted({g for gs in genres for g in gs})
G = sparse.csr_matrix(np.array([[g in gs for g in all_genres] for gs in genres], dtype=np.float32))
rng = np.random.default_rng(0)
top_genres, seeds = {}, {}
for u in uidx:
    row = R_full[u]
    pos = row.indices[row.data >= 4]
    gcount = np.asarray(G[pos].sum(axis=0)).ravel() / np.maximum(np.asarray(G.sum(axis=0)).ravel(), 1) ** 0.5
    top_genres[u] = np.argsort(-gcount)[:3]
    fav = row.indices[row.data == row.data.max()]
    seeds[u] = rng.choice(fav, size=min(3, len(fav)), replace=False)

_, vec = build_metadata_matrix(DATA.items)
vocab = vec.vocabulary_
meta = f.content.meta_vecs


def genre_match(b):
    out = np.zeros((len(b), IDX.n_items), np.float32)
    for r, u in enumerate(b):
        out[r] = np.asarray(G[:, top_genres[u]].sum(axis=1)).ravel() / 3
    return out


def genre_profile(b):
    """Pseudo-user vector U_seed in TF-IDF space from the selected genre tokens."""
    rows = []
    for u in b:
        q = np.zeros(meta.shape[1], np.float32)
        for gi in top_genres[u]:
            tok = "g_" + all_genres[gi].lower().replace("'", "_").replace("-", "_")
            j = vocab.get(tok)
            if j is not None:
                q[j] = 1
        rows.append(q)
    U = normalize(np.vstack(rows))
    return np.asarray(meta @ U.T).T.astype(np.float32)


def seed_hybrid(b):
    data, ri, ci = [], [], []
    for r, u in enumerate(b):
        for i in seeds[u]:
            data.append(5.0); ri.append(r); ci.append(i)
    Rs = sparse.csr_matrix((np.array(data, np.float32), (ri, ci)), shape=(len(b), IDX.n_items))
    content = minmax_rows(f.content.score(Rs) * logpop)
    collab = f.collab.score_new(Rs)
    n = np.diff(Rs.indptr)
    return hybrid(n, content, collab), Rs


variants = {
    "bayes_avg (current N=0 fallback)": lambda b: np.broadcast_to(bayes, (len(b), IDX.n_items)),
    "most_popular": lambda b: np.broadcast_to(mostpop, (len(b), IDX.n_items)),
    "genres: 0.6 match + 0.4 bayes (current app)": lambda b: 0.6 * genre_match(b) + 0.4 * bayes,
    "genres: 0.6 match + 0.4 mostpop": lambda b: 0.6 * genre_match(b) + 0.4 * mostpop,
    "genres: TF-IDF U_seed x log-pop": lambda b: minmax_rows(genre_profile(b) * logpop),
}
rows = []
for name, fn in variants.items():
    m = evaluate(fn, R0, rel, uidx)
    rows.append({"variant": name, **fmt(m)})

# seed movies: exclude the seeds themselves from recommendations
recs = []
for s in range(0, len(uidx), 1000):
    b = uidx[s:s + 1000]
    S, Rs = seed_hybrid(b)
    recs.append(DynamicHybridRecommender.top_k(S, [Rs[r].indices for r in range(len(b))], 10))
from recsys.evaluation.metrics import ranking_metrics
m = ranking_metrics(np.vstack(recs), [rel[u] - set(seeds[u]) or rel[u] for u in uidx], 10)
rows.append({"variant": "3 seed favourites -> hybrid (fold-in)", **fmt(m)})
print(pd.DataFrame(rows).to_string(index=False))

# blend weight sweep for genre onboarding (additive vs multiplicative boost on most-popular)
sweep = []
for w in (0.0, 0.1, 0.2, 0.3, 0.45):
    sweep.append({"blend": f"additive w={w}", "ndcg@10": evaluate(lambda b: w * genre_match(b) + (1 - w) * mostpop,
                                                                   R0, rel, uidx)["ndcg@10"]})
for beta in (0.25, 0.5, 1.0, 2.0):
    sweep.append({"blend": f"pop*(1+{beta}*match)", "ndcg@10": evaluate(lambda b: mostpop * (1 + beta * genre_match(b)),
                                                                         R0, rel, uidx)["ndcg@10"]})
print(pd.DataFrame(sweep).round(4).to_string(index=False))
pd.concat([pd.DataFrame(rows), pd.DataFrame(sweep)]).to_csv(RESULTS / "exp5.csv", index=False)
