"""How should the "Fresh Discoveries" row score zero-rating movies that have NO plot?

The real fresh catalogue (ML-1M movies with no ratings) is 177 movies, of which only 25% have a CMU
plot. Without a plot, the content score is metadata alone, and a perfect genre match reaches 1.0;
with a plot it is a blend of two scaled channels that rarely does. So plot-less movies tend to win
the row by skipping a channel. The candidate fix is to blend plot-less movies with the user's
median plot score instead.

Protocol (per seed, as in exp10, but with the fresh row's composition): hold out ~300 movies with
>= 20 ratings, 25% with a plot and 75% without, removing 100% of their ratings from training.
Users rank only the held-out movies with the production content engine (centred plots, the
deployed w_meta/w_plot, ungated, i.e. ContentEngine.score_items). Reported: NDCG@10, and the share
of top-10 slots that go to plot-bearing movies vs their share of relevant movies.
"""
import numpy as np
import pandas as pd
from exp_common import *
from recsys.evaluation.metrics import paired_bootstrap, per_user_metrics
from recsys.models.content import ContentEngine
from recsys.pipeline import load_best_params

SEEDS = (42, 43, 44, 45, 46)
N_COLD, PLOT_SHARE, MIN_RATINGS = 300, 0.25, 20
DEPLOYED = load_best_params(Config())
HAS_PLOT = DATA.items["has_plot"].to_numpy()

rows, comps = [], []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    counts_all = DATA.ratings.groupby("movieId").size()
    eligible = np.array([IDX.iidx(m) for m in counts_all[counts_all >= MIN_RATINGS].index])
    with_plot, without = eligible[HAS_PLOT[eligible]], eligible[~HAS_PLOT[eligible]]
    n_plot = int(PLOT_SHARE * N_COLD)
    cold = np.sort(np.concatenate([rng.choice(with_plot, n_plot, replace=False),
                                   rng.choice(without, min(N_COLD - n_plot, len(without)), replace=False)]))
    cold_movies = IDX.movie_ids[cold]
    is_cold = DATA.ratings["movieId"].isin(cold_movies)
    train, test = DATA.ratings[~is_cold], DATA.ratings[is_cold]
    R, _ = matrices(train)
    rel, uidx = rel_and_users(test)
    pos = {i: c for c, i in enumerate(cold)}
    rel_local = [{pos[i] for i in rel[u]} for u in uidx]
    # the production engine, scoring the held-out movies as external items exactly like the fresh row
    eng = ContentEngine(DATA.meta_vecs, DATA.plot_vecs, HAS_PLOT, DEPLOYED.w_meta, DEPLOYED.w_plot,
                        DEPLOYED.pos_threshold, center_plots=True)
    meta_c, plot_c, has_c = DATA.meta_vecs[cold], DATA.plot_vecs[cold], HAS_PLOT[cold]
    variants = {
        "metadata only": lambda b: ContentEngine(DATA.meta_vecs, DATA.plot_vecs, HAS_PLOT, 1.0, 0.0,
                                                 center_plots=True).score_items(R[b], meta_c, plot_c, has_c),
        "meta + centred plots, plot-less = metadata only": lambda b: eng.score_items(R[b], meta_c, plot_c, has_c),
        "meta + centred plots, plot-less = median plot imputed": lambda b: eng.score_items(R[b], meta_c, plot_c, has_c,
                                                                                            impute_missing_plot=True),
        "random": lambda b: np.random.default_rng(seed).random((len(b), len(cold)), dtype=np.float32),
    }
    pu = {}
    for name, fn in variants.items():
        recs = np.vstack([DynamicHybridRecommender.top_k(fn(uidx[s:s + 1000]), None, 10)
                          for s in range(0, len(uidx), 1000)])
        pu[name] = per_user_metrics(recs, rel_local, 10)["ndcg@10"]
        rel_plot_share = np.mean([has_c[list(r)].mean() for r in rel_local])
        rows.append({"seed": seed, "variant": name, "ndcg@10": pu[name].mean(),
                     "plot_share_top10": has_c[recs].mean(), "plot_share_relevant": rel_plot_share})
    for a in ("meta + centred plots, plot-less = metadata only", "meta + centred plots, plot-less = median plot imputed"):
        comps.append({"seed": seed, "variant": a, "reference": "metadata only",
                      **paired_bootstrap(pu[a], pu["metadata only"], seed=seed)})
    comps.append({"seed": seed, "variant": "imputed", "reference": "plot-less = metadata only",
                  **paired_bootstrap(pu["meta + centred plots, plot-less = median plot imputed"],
                                     pu["meta + centred plots, plot-less = metadata only"], seed=seed)})
    print(f"seed {seed}: {len(cold)} held-out movies ({has_c.mean():.0%} with plots), {len(uidx)} users", flush=True)

r = pd.DataFrame(rows)
c = pd.DataFrame(comps)
r.to_csv(RESULTS / "exp12_fresh_row.csv", index=False)
c.to_csv(RESULTS / "exp12_comparisons.csv", index=False)
pd.set_option("display.width", 220)
print(r.groupby("variant")[["ndcg@10", "plot_share_top10", "plot_share_relevant"]].mean()
      .sort_values("ndcg@10", ascending=False).round(4).to_string())
print(c.groupby(["variant", "reference"]).agg(diff=("diff", "mean"),
                                              sig_better=("ci_low", lambda x: int((x > 0).sum())),
                                              sig_worse=("ci_high", lambda x: int((x < 0).sum()))).round(4).to_string())
