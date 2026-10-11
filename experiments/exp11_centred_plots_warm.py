"""Does mean-centring the plot embeddings (adopted for item cold start, exp10) hurt or help the
existing-user settings? Same 5 seeds and settings as exp8; the only change is the plot vectors.
Compared inside the final hybrid (lambda = 0, as audited in exp8) and as content alone."""
from dataclasses import replace

import numpy as np
import pandas as pd
from exp_common import *
from exp8_robustness import COLD_NS, FINAL, SEEDS
from recsys.evaluation.evaluate import collab_scores, fit_models, popularity_prior
from recsys.evaluation.metrics import paired_bootstrap, per_user_metrics
from recsys.models.content import ContentEngine

rows, comps = [], []


def run(seed, setting, train, test, users=None):
    f = fit_models(DATA, train, replace(FINAL, seed=seed))
    rel, uidx = rel_and_users(test, users)
    relevant = [rel[u] for u in uidx]
    counts = f.item_scores["most_popular"]
    eng = {c: ContentEngine(DATA.meta_vecs, DATA.plot_vecs, DATA.items["has_plot"].to_numpy(), FINAL.w_meta,
                            FINAL.w_plot, FINAL.pos_threshold, item_counts=counts, center_plots=c) for c in (False, True)}
    h = DynamicHybridRecommender(FINAL.alpha_max, FINAL.gamma, FINAL.pop_eta)
    recs = {}
    for s in range(0, len(uidx), 1000):
        b = uidx[s:s + 1000]
        seen = [f.R[u].indices for u in b]
        collab, prior = collab_scores(f, b, FINAL), popularity_prior(f)
        for c, e in eng.items():
            content = e.score(f.R[b])
            tag = "centred" if c else "raw"
            recs.setdefault(f"hybrid, {tag} plots", []).append(
                DynamicHybridRecommender.top_k(h.blend(f.n_u[b], content, collab, prior), seen, 10))
            recs.setdefault(f"content alone, {tag} plots", []).append(DynamicHybridRecommender.top_k(content, seen, 10))
    pu = {k: per_user_metrics(np.vstack(v), relevant, 10)["ndcg@10"] for k, v in recs.items()}
    for k, v in pu.items():
        rows.append({"seed": seed, "setting": setting, "variant": k, "ndcg@10": v.mean()})
    for kind in ("hybrid", "content alone"):
        comps.append({"seed": seed, "setting": setting, "comparison": f"{kind}: centred vs raw",
                      **paired_bootstrap(pu[f"{kind}, centred plots"], pu[f"{kind}, raw plots"], seed=seed)})
    print(f"seed {seed} {setting} done", flush=True)


for seed in SEEDS:
    r_train, r_test = random_split(DATA.ratings, CFG.test_frac, seed)
    run(seed, "random", r_train, r_test)
    run(seed, "temporal", *temporal_split(DATA.ratings, CFG.test_frac))
    users, uidx = sample_users(1000, seed=seed)
    for n in COLD_NS:
        run(seed, f"cold N={n}", truncate_history(r_train, users, n), r_test, uidx)

settings = ["random", "temporal"] + [f"cold N={n}" for n in COLD_NS]
r = pd.DataFrame(rows)
c = pd.DataFrame(comps)
r.to_csv(RESULTS / "exp11_centred_plots_warm.csv", index=False)
c.to_csv(RESULTS / "exp11_comparisons.csv", index=False)
pd.set_option("display.width", 220)
print(r.groupby(["variant", "setting"])["ndcg@10"].mean().unstack()[settings].round(4).to_string())
base = r.groupby(["variant", "setting"])["ndcg@10"].mean()
summ = c.groupby(["comparison", "setting"]).agg(diff=("diff", "mean"),
                                                sig_better=("ci_low", lambda x: int((x > 0).sum())),
                                                sig_worse=("ci_high", lambda x: int((x < 0).sum()))).reset_index()
summ["rel_%"] = [100 * d / base[(cmp.split(":")[0] + ", raw plots", s)] for d, cmp, s in
                 zip(summ["diff"], summ["comparison"], summ["setting"])]
print(summ.round(4).to_string(index=False))
