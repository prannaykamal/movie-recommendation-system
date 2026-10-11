"""Robustness, plot-embedding ablation and beyond-accuracy evaluation of the FINAL system.

For 5 seeds (seed = random-split seed AND SVD/iALS initialisation; the temporal split is fixed, so
there only the model seed varies) and 6 settings (random, temporal, cold-start N = 3/5/10/20 on
1,000 sampled users), every variant below is scored on the same users. Per-user NDCG@10 is kept
so each variant can be compared with the final system by a paired bootstrap over users.

Variants answer three questions:
  decisions   do the adopted choices hold up across seeds? (gating, iALS mix, fading prior, ...)
  ablation    what do the content channels contribute inside the final hybrid?
  beyond      coverage / popularity bias / novelty / diversity of each model's top-10

usage: python experiments/exp8_robustness.py [compute|analyse]   (default: both)
"""
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
from exp_common import *
from recsys.evaluation.beyond import beyond_accuracy
from recsys.evaluation.evaluate import fit_models, genre_matrix
from recsys.evaluation.metrics import paired_bootstrap, per_user_metrics
from recsys.models.content import ContentEngine

SEEDS = (42, 43, 44, 45, 46)
COLD_NS = (3, 5, 10, 20)
FINAL = replace(Config(), n_components=32, w_meta=0.65, w_plot=0.35, content_pop_gate=True, ials_weight=0.5,
                alpha_max=0.8, gamma=0.15, pop_eta=0.2)
GENRES = genre_matrix(DATA.items)
PER_USER = RESULTS / "exp8_per_user.parquet"
BEYOND = RESULTS / "exp8_beyond.csv"


def plot_only(eng: ContentEngine, R) -> np.ndarray:
    """Content from the SBERT plot channel alone (items/users without plots score 0), gated."""
    _, plot = eng.channel_scores(R)
    mask = ~np.isnan(plot)
    filled = np.where(mask, plot, np.nanmin(np.where(mask, plot, np.inf), axis=1, keepdims=True))
    filled = np.where(np.isfinite(filled), filled, 0.0)
    s = np.where(mask, minmax_rows(filled.astype(np.float32)), 0.0).astype(np.float32)
    return minmax_rows(s * eng.pop_prior) if eng.pop_prior is not None else s


def variant_scores(f, cfg, b) -> dict[str, np.ndarray]:
    R = f.R[b]
    counts = f.item_scores["most_popular"]
    pop = minmax_rows(counts[None].astype(np.float32))[0]
    bayes = minmax_rows(f.item_scores["bayesian_average"][None].astype(np.float32))[0]
    items = DATA.items
    mk = lambda wm, wp, gate: ContentEngine(DATA.meta_vecs, DATA.plot_vecs, items["has_plot"].to_numpy(), wm, wp,
                                            cfg.pos_threshold, item_counts=counts if gate else None).score(R)
    c_final = f.content.score(R)                       # gated, w_meta .65 / w_plot .35
    c = {"final": c_final, "ungated": mk(0.65, 0.35, False), "wm08": mk(0.8, 0.2, True),
         "meta": mk(1.0, 0.0, True), "plot": plot_only(f.content, R), "orig": mk(0.5, 0.5, False)}
    svd, ials = f.collab.score(b), f.ials.score(b)
    ens = 0.5 * ials + 0.5 * svd
    n_u = f.n_u[b]

    def hyb(content, collab, alpha_max=0.8, gamma=0.15, eta=0.2, prior=pop):
        return DynamicHybridRecommender(alpha_max, gamma, eta).blend(n_u, content, collab, prior)

    return {
        # decisions
        "final": hyb(c["final"], ens),
        "original_system": hyb(c["orig"], svd, gamma=1.0, eta=None, prior=bayes),
        "no_gating": hyb(c["ungated"], ens),
        "collab_svd_only": hyb(c["final"], svd),
        "collab_ials_only": hyb(c["final"], ials),
        "w_meta_0.8": hyb(c["wm08"], ens),
        "no_fading_gamma_1.0": hyb(c["final"], ens, gamma=1.0, eta=None),
        "gamma_1.0": hyb(c["final"], ens, gamma=1.0),
        # ablation (inside the final hybrid)
        "content_meta_only": hyb(c["meta"], ens),
        "content_plot_only": hyb(c["plot"], ens),
        "no_content": hyb(ens, ens),                   # content := collab, i.e. collab + popularity prior only
        # standalone models
        "alone_most_popular": np.broadcast_to(pop, (len(b), len(pop))),
        "alone_svd": svd,
        "alone_ials": ials,
        "alone_content_both": c["final"],
        "alone_content_meta": c["meta"],
        "alone_content_plot": c["plot"],
    }


def run_setting(seed, setting, train, test, users=None):
    cfg = replace(FINAL, seed=seed)
    f = fit_models(DATA, train, cfg)
    rel, uidx = rel_and_users(test, users)
    relevant = [rel[u] for u in uidx]
    recs = {}
    for s in range(0, len(uidx), 1000):
        b = uidx[s:s + 1000]
        seen = [f.R[u].indices for u in b]
        for name, S in variant_scores(f, cfg, b).items():
            recs.setdefault(name, []).append(DynamicHybridRecommender.top_k(S, seen, 10))
    rows, beyond = [], []
    counts = f.item_scores["most_popular"]
    for name, r in recs.items():
        r = np.vstack(r)
        ndcg = per_user_metrics(r, relevant, 10)["ndcg@10"]
        rows.append(pd.DataFrame({"seed": seed, "setting": setting, "variant": name, "uidx": uidx, "ndcg": ndcg}))
        beyond.append({"seed": seed, "setting": setting, "variant": name, "ndcg@10": ndcg.mean(),
                       **beyond_accuracy(r, counts, int((f.n_u > 0).sum()), GENRES, 10, seed=seed)})
    print(f"seed {seed} {setting}: final ndcg@10 {np.mean(rows[0]['ndcg']):.4f}", flush=True)
    return rows, beyond


def compute():
    per_user, beyond = [], []
    for seed in SEEDS:
        r_train, r_test = random_split(DATA.ratings, CFG.test_frac, seed)
        t_train, t_test = temporal_split(DATA.ratings, CFG.test_frac)
        for setting, (tr, te) in {"random": (r_train, r_test), "temporal": (t_train, t_test)}.items():
            a, b = run_setting(seed, setting, tr, te)
            per_user += a; beyond += b
        users, uidx = sample_users(1000, seed=seed)
        for n in COLD_NS:
            a, b = run_setting(seed, f"cold N={n}", truncate_history(r_train, users, n), r_test, uidx)
            per_user += a; beyond += b
    pd.concat(per_user, ignore_index=True).to_parquet(PER_USER, index=False)
    pd.DataFrame(beyond).to_csv(BEYOND, index=False)


SETTINGS = ["random", "temporal"] + [f"cold N={n}" for n in COLD_NS]
COMPARISONS = {  # (variant, reference): question
    ("final", "original_system"): "overall gain vs the original system",
    ("final", "no_gating"): "S1 popularity gating",
    ("final", "collab_svd_only"): "S2 iALS in the ensemble (vs SVD only)",
    ("final", "collab_ials_only"): "S2 SVD in the ensemble (vs iALS only)",
    ("final", "w_meta_0.8"): "w_meta 0.65 vs 0.8",
    ("final", "no_fading_gamma_1.0"): "fading prior (vs previous system)",
    ("final", "gamma_1.0"): "gamma 0.15 vs 1.0 (with fading)",
    ("final", "alone_most_popular"): "final vs most-popular",
    ("final", "alone_ials"): "final vs iALS alone",
    ("final", "content_meta_only"): "ablation: plot channel (both vs meta-only content)",
    ("final", "content_plot_only"): "ablation: meta channel (both vs plot-only content)",
    ("final", "no_content"): "ablation: any content (vs collab + popularity)",
    ("alone_content_both", "alone_content_meta"): "standalone content: plots added to metadata",
}


def analyse():
    pu = pd.read_parquet(PER_USER)
    table = pu.groupby(["setting", "variant", "seed"])["ndcg"].mean().unstack("seed")
    means = pd.DataFrame({"mean": table.mean(axis=1), "std": table.std(axis=1)}).unstack("setting")
    out = []
    for (a, b), question in COMPARISONS.items():
        for setting in SETTINGS:
            seeds = []
            for seed in SEEDS:
                x = pu[(pu.seed == seed) & (pu.setting == setting)]
                va = x[x.variant == a].sort_values("uidx")["ndcg"].to_numpy()
                vb = x[x.variant == b].sort_values("uidx")["ndcg"].to_numpy()
                seeds.append(paired_bootstrap(va, vb, seed=seed))
            d = np.array([s["diff"] for s in seeds])
            ref = table.loc[(setting, b)].mean()
            out.append({"question": question, "setting": setting, "diff": d.mean(), "rel_diff": d.mean() / ref,
                        "seed_min": d.min(), "seed_max": d.max(),
                        "seeds_sig_better": sum(s["ci_low"] > 0 for s in seeds),
                        "seeds_sig_worse": sum(s["ci_high"] < 0 for s in seeds)})
    comp = pd.DataFrame(out)
    comp.to_csv(RESULTS / "exp8_comparisons.csv", index=False)
    means.to_csv(RESULTS / "exp8_means.csv")
    pd.set_option("display.max_rows", 200)
    print("\nNDCG@10 mean over 5 seeds:")
    print(table.mean(axis=1).unstack("setting")[SETTINGS].round(4).to_string())
    print("\nNDCG@10 std over seeds:")
    print(table.std(axis=1).unstack("setting")[SETTINGS].round(4).to_string())
    print("\nPaired comparisons (diff = variant - reference, mean over seeds; sig = seeds whose 95% CI excludes 0):")
    print(comp.assign(rel_diff=(comp.rel_diff * 100).round(1)).round(4).to_string(index=False))
    b = pd.read_csv(BEYOND)
    beyond = b.groupby(["setting", "variant"]).mean(numeric_only=True).drop(columns="seed")
    beyond.to_csv(RESULTS / "exp8_beyond_mean.csv")
    print("\nBeyond-accuracy @10 (random split, mean over seeds):")
    print(beyond.loc["random"].round(3).to_string())
    print("\nBeyond-accuracy @10 (temporal split):")
    print(beyond.loc["temporal"].round(3).to_string())


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("compute", "all"):
        compute()
    if stage in ("analyse", "all"):
        analyse()
