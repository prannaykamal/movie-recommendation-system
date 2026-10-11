"""Follow-up: close the N_u in [3, 5] gap where plain most-popular beats the personalised hybrid.

    S(u, i) = beta(u) * S_pop(i) + (1 - beta(u)) * S_hybrid(u, i),   beta(u) = exp(-eta * N_u)

Run against the CURRENT production system (gated content + SVD/iALS ensemble + tuned params),
not the pinned baseline. eta (and gamma, since the popularity prior now takes the early weight)
are tuned on validation histories truncated to N in {3, 5, 10, 20}; test is used once at the end.
S_pop variants: most-popular counts (as proposed) and log-counts (a smoother prior)."""
import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
from exp_common import *
from recsys.evaluation.evaluate import collab_scores, fit_models as fit
# the system as it stood before this experiment (suggestions 1, 2 and 5 accepted, tuned params)
CUR = replace(Config(), n_components=32, w_meta=0.65, w_plot=0.35, ials_weight=0.5, alpha_max=0.8, gamma=1.0,
              pop_eta=None, center_plots=False, pop_penalty=0.0)
ETAS = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, np.inf)  # inf = beta 0 for N >= 1 (current system)
GAMMAS = (0.15, 0.3, 1.0)
print("current params:", {k: getattr(CUR, k) for k in ("w_meta", "ials_weight", "alpha_max", "gamma")})


def precompute(train, uidx):
    f = fit(DATA, train, CUR)
    counts = f.item_scores["most_popular"].astype(np.float32)
    return {"R": f.R, "n_u": f.n_u[uidx], "content": f.content.score(f.R[uidx]),
            "collab": collab_scores(f, uidx, CUR),
            "pop": {"count": minmax_rows(counts[None])[0], "log": minmax_rows(np.log1p(counts)[None])[0]}}


def score_fn(P, uidx, eta, gamma, pop_kind):
    pop = P["pop"][pop_kind]
    hyb = DynamicHybridRecommender(CUR.alpha_max, gamma).blend(P["n_u"], P["content"], P["collab"], pop)
    beta = (np.exp(-eta * P["n_u"]) if np.isfinite(eta) else (P["n_u"] == 0).astype(float))[:, None]
    S = (beta * pop + (1 - beta) * hyb).astype(np.float32)
    pos = {u: r for r, u in enumerate(uidx)}
    return lambda b: S[[pos[u] for u in b]]


def sweep(train, test, users, label):
    rel, uidx = rel_and_users(test, users)
    P = precompute(train, uidx)
    rows = []
    for eta, gamma, kind in itertools.product(ETAS, GAMMAS, ("count", "log")):
        if not np.isfinite(eta) and kind == "log":
            continue
        m = evaluate(score_fn(P, uidx, eta, gamma, kind), P["R"], rel, uidx)
        rows.append({"setting": label, "eta": eta, "gamma": gamma, "pop": kind, "ndcg@10": m["ndcg@10"]})
    pop_only = evaluate(lambda b, s=P["pop"]["count"]: np.broadcast_to(s, (len(b), len(s))), P["R"], rel, uidx)
    rows.append({"setting": label, "eta": "pop-only", "gamma": None, "pop": "count", "ndcg@10": pop_only["ndcg@10"]})
    return rows


# ---- validation: tune eta, gamma, S_pop kind --------------------------------------------------
fit_tr, val = split("val")
users, uidx_s = sample_users(1500)
rows = []
for n in (3, 5, 10, 20):
    rows += sweep(truncate_history(fit_tr, users, n), val, uidx_s, f"val N={n}")
    print(f"val N={n} done", flush=True)
v = pd.DataFrame(rows)
v.to_csv(RESULTS / "exp7_val.csv", index=False)
tuned = v[v.eta != "pop-only"].groupby(["eta", "gamma", "pop"])["ndcg@10"].mean().sort_values(ascending=False)
print("validation mean NDCG@10 over N in {3,5,10,20}:\n", tuned.head(10).round(4).to_string())
cur_val = tuned.loc[(np.inf, CUR.gamma, "count")]
best_eta, best_gamma, best_kind = tuned.index[0]
print(f"current system on val: {cur_val:.4f} | best: eta={best_eta} gamma={best_gamma} pop={best_kind} "
      f"{tuned.iloc[0]:.4f} ({tuned.iloc[0] / cur_val - 1:+.1%})")

# ---- test: tuned setting vs current system vs pop-only ----------------------------------------
train, test = split("random")
t_users, t_uidx = sample_users(1000)
settings = {"current (no fading)": (np.inf, CUR.gamma, "count"),
            f"fading eta={best_eta}, gamma={best_gamma}, pop={best_kind}": (best_eta, best_gamma, best_kind),
            "fading eta=0.4 (proposal), current gamma": (0.4, CUR.gamma, "count")}
test_rows = []
for n in (0, 3, 5, 10, 20, None):
    tr = train if n is None else truncate_history(train, t_users, n)
    rel, uidx = rel_and_users(test, t_uidx)
    P = precompute(tr, uidx)
    for name, (eta, gamma, kind) in settings.items():
        m = evaluate(score_fn(P, uidx, eta, gamma, kind), P["R"], rel, uidx)
        test_rows.append({"N": "all" if n is None else n, "model": name, "ndcg@10": m["ndcg@10"]})
    m = evaluate(lambda b, s=P["pop"]["count"]: np.broadcast_to(s, (len(b), len(s))), P["R"], rel, uidx)
    test_rows.append({"N": "all" if n is None else n, "model": "most-popular only", "ndcg@10": m["ndcg@10"]})
    print(f"test N={n} done", flush=True)
for split_name in ("random", "temporal"):
    tr, te = split(split_name)
    rel, uidx = rel_and_users(te)
    P = precompute(tr, uidx)
    for name, (eta, gamma, kind) in settings.items():
        m = evaluate(score_fn(P, uidx, eta, gamma, kind), P["R"], rel, uidx)
        test_rows.append({"N": f"full {split_name}", "model": name, "ndcg@10": m["ndcg@10"]})
t = pd.DataFrame(test_rows)
t.to_csv(RESULTS / "exp7_test.csv", index=False)
print(t.pivot(index="model", columns="N", values="ndcg@10").round(4).to_string())
