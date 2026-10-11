"""Choose the iALS/SVD mix (beta) and alpha_max on BOTH validation splits (random and temporal),
scoring each setting by NDCG@10 relative to the best setting per split, averaged."""
import itertools

from exp_models import *

_, uidx_s = sample_users(1500)
res = []
for name, (fit, val) in val_splits().items():
    B = Base(fit)
    rel, uidx = rel_and_users(val, uidx_s)
    C, I, S = B.content(uidx), B.ials(uidx), B.svd(uidx)
    pos = {u: r for r, u in enumerate(uidx)}
    for beta, a in itertools.product((0, 0.25, 0.5, 0.75, 1.0), (0.6, 0.7, 0.8, 0.9)):
        fn = lambda b: (lambda ix: a * (beta * I[ix] + (1 - beta) * S[ix]) + (1 - a) * C[ix])([pos[u] for u in b])
        res.append(dict(val=name, beta_ials=beta, alpha_max=a, ndcg=evaluate(fn, B.R, rel, uidx)["ndcg@10"]))
df = pd.DataFrame(res)
p = df.pivot_table(index=["beta_ials", "alpha_max"], columns="val", values="ndcg")
p["rel_mean"] = (p / p.max()).mean(axis=1)
p.to_csv(RESULTS / "exp2_ensemble_select.csv")
print(p.sort_values("rel_mean", ascending=False).head(8).round(4).to_string())
