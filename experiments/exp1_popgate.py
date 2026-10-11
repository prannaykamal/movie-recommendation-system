"""Suggestion 1: S_content_pop = S_content * log(1 + count_i)."""
import numpy as np
import pandas as pd
from exp_common import *

rows = []


def run(label, train, test, users=None):
    f = fit_models(DATA, train, CFG)
    R = f.R
    rel, uidx = rel_and_users(test, users)
    logpop = np.log1p(item_counts(R))
    cache = {}

    def parts(b):
        key = (b[0], len(b))
        if key not in cache:
            c = f.content.score(R[b])
            cache.clear()
            cache[key] = (c, minmax_rows(c * logpop), f.collab.score(b), f.n_u[b])
        return cache[key]

    variants = {
        "content": lambda b: parts(b)[0],
        "content_pop": lambda b: parts(b)[1],
        "hybrid(γ=1.0)": lambda b: hybrid(parts(b)[3], parts(b)[0], parts(b)[2]),
        "hybrid_pop(γ=1.0)": lambda b: hybrid(parts(b)[3], parts(b)[1], parts(b)[2]),
        "hybrid(γ=0.15)": lambda b: hybrid(parts(b)[3], parts(b)[0], parts(b)[2], gamma=0.15),
        "hybrid_pop(γ=0.15)": lambda b: hybrid(parts(b)[3], parts(b)[1], parts(b)[2], gamma=0.15),
    }
    for name, fn in variants.items():
        m = evaluate(fn, R, rel, uidx)
        rows.append({"setting": label, "model": name, "ndcg@10": m["ndcg@10"], "map@10": m["map@10"]})
    print(pd.DataFrame(rows[-len(variants):]).round(4).to_string(index=False))


for s in ("random", "temporal"):
    run(s, *split(s))

train, test = split("random")
users, _ = sample_users(1000)
for n in (3, 5, 10, 20):
    run(f"cold N={n}", truncate_history(train, users, n), test, [IDX.uidx(u) for u in users])

df = pd.DataFrame(rows)
print(df.pivot(index="model", columns="setting", values="ndcg@10").round(4).to_string())
df.to_csv(RESULTS / "exp1.csv", index=False)
