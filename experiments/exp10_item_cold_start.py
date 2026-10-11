"""Item cold start: are the Sentence-BERT plot embeddings worth their cost for NEW movies?

Protocol (per seed): hold out 10% of the movies with >= 20 ratings and remove 100% of their ratings
from training (N_i = 0 for every held-out movie). Users keep their full history on the other
movies. Each user then ranks ONLY the held-out movies; relevant = held-out movies they rated >= 4.
Collaborative filtering is blind here (no interactions), so the question is which content
representation ranks unseen movies best: metadata TF-IDF, SBERT plots, or a mix.

To give the plots a fair chance, two standard fixes for sentence embeddings are also tried:
mean-centring (SBERT similarities are compressed into a narrow cone, so the average profile is
close to every movie) and max-similarity scoring (each candidate is scored by its closest liked
movie rather than by the averaged profile).

Also measured: how often held-out movies reach the deployed system's full-catalogue top-10 at all,
since popularity gating and the popularity prior give a zero-interaction movie zero weight.
"""
from dataclasses import replace

import numpy as np
import pandas as pd
from exp_common import *
from exp8_robustness import FINAL, plot_only
from recsys.evaluation.evaluate import collab_scores, fit_models, popularity_penalty, popularity_prior
from recsys.evaluation.metrics import paired_bootstrap, per_user_metrics
from recsys.models.content import ContentEngine

SEEDS = (42, 43, 44, 45, 46)
COLD_FRAC = 0.10
MIN_RATINGS = 20
MIXES = {"meta only": (1.0, 0.0), "meta .65 / plot .35 (final)": (0.65, 0.35), "meta .5 / plot .5": (0.5, 0.5),
         "meta .35 / plot .65 (design doc)": (0.35, 0.65)}


def centred_plots() -> np.ndarray:
    E = DATA.plot_vecs.copy()
    has = DATA.items["has_plot"].to_numpy()
    E[has] -= E[has].mean(axis=0)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    return np.divide(E, norms, out=np.zeros_like(E), where=norms > 0) * has[:, None]


E_C = centred_plots()


def max_sim(R, cold, E) -> np.ndarray:
    """Score = cosine to the closest positively rated movie (>= 4) with a plot; cold columns only."""
    out = np.zeros((R.shape[0], IDX.n_items), dtype=np.float32)
    Ec = E[cold]
    for r in range(R.shape[0]):
        row = R[r]
        liked = row.indices[row.data >= CFG.pos_threshold]
        liked = liked[np.abs(E[liked]).sum(axis=1) > 0]
        if len(liked):
            out[r, cold] = (Ec @ E[liked].T).max(axis=1)
    return out


def run(seed):
    rng = np.random.default_rng(seed)
    counts_all = DATA.ratings.groupby("movieId").size()
    eligible = counts_all[counts_all >= MIN_RATINGS].index.to_numpy()
    cold_movies = rng.choice(eligible, size=int(COLD_FRAC * len(eligible)), replace=False)
    is_cold = DATA.ratings["movieId"].isin(cold_movies)
    train, test = DATA.ratings[~is_cold], DATA.ratings[is_cold]
    cold = np.sort([IDX.iidx(m) for m in cold_movies])
    pos = {i: c for c, i in enumerate(cold)}

    R, _ = matrices(train)
    rel, uidx = rel_and_users(test)
    rel_local = [{pos[i] for i in rel[u]} for u in uidx]
    has_plot = DATA.items["has_plot"].to_numpy()
    counts = item_counts(R)

    def engine(wm, wp, gate, plots=None):
        return ContentEngine(DATA.meta_vecs, DATA.plot_vecs if plots is None else plots, has_plot, wm, wp,
                             CFG.pos_threshold, item_counts=counts if gate else None)

    variants = {f"content: {k}": (lambda b, wm=wm, wp=wp: engine(wm, wp, False).score(R[b]))
                for k, (wm, wp) in MIXES.items()}
    variants["content: plot only"] = lambda b: plot_only(engine(0.0, 1.0, False), R[b])
    variants["content: plot only, centred"] = lambda b: plot_only(engine(0.0, 1.0, False, E_C), R[b])
    variants["content: plot only, centred, max-sim"] = lambda b: max_sim(R[b], cold, E_C)
    variants["content: meta .65 / centred plot .35"] = lambda b: engine(0.65, 0.35, False, E_C).score(R[b])
    variants["content: meta .5 / centred plot .5"] = lambda b: engine(0.5, 0.5, False, E_C).score(R[b])
    variants["content: final mix, popularity-gated (as deployed)"] = lambda b: engine(0.65, 0.35, True).score(R[b])

    f = fit_models(DATA, train, replace(FINAL, seed=seed))
    h = DynamicHybridRecommender(FINAL.alpha_max, FINAL.gamma, FINAL.pop_eta)
    final_full = lambda b: h.blend(f.n_u[b], f.content.score(f.R[b]), collab_scores(f, b, FINAL), popularity_prior(f))
    variants["final system (as deployed)"] = final_full
    variants["random"] = lambda b: np.random.default_rng(seed).random((len(b), IDX.n_items), dtype=np.float32)

    rows, per_user, exposure = [], {}, {}
    for name, fn in variants.items():
        recs = []
        for s in range(0, len(uidx), 1000):
            b = uidx[s:s + 1000]
            S = fn(b)
            recs.append(DynamicHybridRecommender.top_k(S[:, cold], None, 10))
            if name == "final system (as deployed)":
                for lam in (0.0, 0.2):
                    top = DynamicHybridRecommender.top_k(S - lam * popularity_penalty(f), [f.R[u].indices for u in b], 10)
                    exposure.setdefault(lam, []).append(np.isin(top, cold).mean(axis=1))
        m = per_user_metrics(np.vstack(recs), rel_local, 10)
        per_user[name] = m["ndcg@10"]
        rows.append({"seed": seed, "variant": name, "ndcg@10": m["ndcg@10"].mean(), "recall@10": m["recall@10"].mean()})
    exp_rows = [{"seed": seed, "lambda": lam, "cold_share_of_top10": np.concatenate(v).mean(),
                 "cold_share_of_catalogue": len(cold) / IDX.n_items} for lam, v in exposure.items()]
    cold_plot_share = has_plot[cold].mean()
    print(f"seed {seed}: {len(cold)} cold movies ({cold_plot_share:.0%} with plots), {len(uidx)} users", flush=True)
    return rows, per_user, exp_rows


all_rows, comps, exp_all = [], [], []
pairs = [("content: meta .65 / plot .35 (final)", "content: meta only"),
         ("content: meta .5 / plot .5", "content: meta only"),
         ("content: meta .35 / plot .65 (design doc)", "content: meta only"),
         ("content: plot only", "content: meta only"),
         ("content: plot only, centred", "content: plot only"),
         ("content: plot only, centred, max-sim", "content: meta only"),
         ("content: meta .65 / centred plot .35", "content: meta only"),
         ("content: meta .5 / centred plot .5", "content: meta only"),
         ("content: meta .65 / plot .35 (final)", "content: final mix, popularity-gated (as deployed)"),
         ("content: meta .65 / plot .35 (final)", "final system (as deployed)")]
for seed in SEEDS:
    rows, pu, ex = run(seed)
    all_rows += rows
    exp_all += ex
    for a, b in pairs:
        comps.append({"seed": seed, "variant": a, "reference": b, **paired_bootstrap(pu[a], pu[b], seed=seed)})

res = pd.DataFrame(all_rows)
summary = res.groupby("variant")[["ndcg@10", "recall@10"]].agg(["mean", "std"]).sort_values(("ndcg@10", "mean"),
                                                                                          ascending=False)
c = pd.DataFrame(comps)
base = res.groupby("variant")["ndcg@10"].mean()
cs = c.groupby(["variant", "reference"]).agg(diff=("diff", "mean"),
                                             seeds_sig_better=("ci_low", lambda x: int((x > 0).sum())),
                                             seeds_sig_worse=("ci_high", lambda x: int((x < 0).sum()))).reset_index()
cs["rel_diff_%"] = [100 * d / base[r] for d, r in zip(cs["diff"], cs["reference"])]
ex = pd.DataFrame(exp_all).groupby("lambda").mean(numeric_only=True).drop(columns="seed")

res.to_csv(RESULTS / "exp10_item_cold_start.csv", index=False)
cs.to_csv(RESULTS / "exp10_comparisons.csv", index=False)
ex.to_csv(RESULTS / "exp10_exposure.csv")
pd.set_option("display.width", 220)
print("\nRanking held-out (zero-rating) movies, NDCG@10 / Recall@10 over 5 seeds:")
print(summary.round(4).to_string())
print("\nPaired comparisons (mean over seeds; sig = seeds whose 95% CI excludes 0):")
print(cs.round(4).to_string(index=False))
print("\nExposure of zero-rating movies in the deployed system's full-catalogue top-10:")
print(ex.round(4).to_string())
