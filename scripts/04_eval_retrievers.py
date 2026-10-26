"""Evaluate the candidate sources: each alone, recall curves, overlap, and the k grid.

Writes reports/<dataset>_<stage>_retrieval.md and rows in eval_results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hybridrec.cli import log, parser, timed
from hybridrec.config import REPORTS_DIR, get_config
from hybridrec.data.dataset import Dataset
from hybridrec.eval.metrics import beyond_accuracy, evaluate_topk
from hybridrec.eval.protocol import batched_topk, new_run_id, record_results, stage_data
from hybridrec.retrieval.registry import ALL_SOURCES, load_sources

CURVE = (25, 50, 100, 200, 300)
K_GRID = (25, 50, 100, 150, 200)


def top_lists(scorer, sd, users, k: int, movie_ids) -> dict[int, list[int]]:
    out = {}
    for us, idx, vals in batched_topk(scorer, sd.hist, users, k):
        for u, row, v in zip(us, idx, vals):
            out[int(u)] = movie_ids[row[np.isfinite(v)]].tolist()
    return out


def union_grid(lists: dict, truth: dict, emb: str) -> pd.DataFrame:
    rows = []
    for k in K_GRID:
        recall, sizes, hits, unique = [], [], 0, {"cb": 0, "cf": 0, "emb": 0}
        for u, rel in truth.items():
            sets = {"cb": set(lists["cb"][u][:k]), "cf": set(lists["cf"][u][:k]), "emb": set(lists[emb][u][:k])}
            union = set().union(*sets.values())
            found = rel & union
            recall.append(len(found) / len(rel))
            sizes.append(len(union))
            hits += len(found)
            for s in sets:
                others = set().union(*(sets[o] for o in sets if o != s))
                unique[s] += len(rel & (sets[s] - others))
        rows.append({"k per source": k, "union recall": np.mean(recall), "avg union size": np.mean(sizes),
                     **{f"unique hits {s}": unique[s] / max(hits, 1) for s in unique}})
    return pd.DataFrame(rows).set_index("k per source")


def main() -> None:
    args = parser(__doc__, stage=True, default_stage="s2").parse_args()
    cfg = get_config(args.dataset)
    ds = Dataset.load(cfg)
    sd = stage_data(ds, args.stage)
    users = sd.eval_users
    run_id = new_run_id()
    sources = load_sources(ds, args.stage, sd.hist)

    lists = {}
    for name, scorer in sources.items():
        with timed(f"scoring {name}"):
            lists[name] = top_lists(scorer, sd, users, max(CURVE), ds.items.movie_ids)

    lines = [f"# Candidate sources: {cfg.name}, stage {args.stage}", "",
             f"{len(users)} users with at least one liked movie in the target block; "
             f"{len(ds.items)} movies in the universe.", ""]

    rows = []
    for name in ALL_SOURCES:
        top10 = {u: l[:10] for u, l in lists[name].items()}
        m = {**evaluate_topk(top10, sd.truth),
             **beyond_accuracy(top10, ds.items.movie_ids, np.log1p(sd.hist.item_counts))}
        curve = {f"recall@{k}": float(np.mean([len(rel & set(lists[name][u][:k])) / len(rel)
                                               for u, rel in sd.truth.items()])) for k in CURVE}
        record_results(cfg, args.stage, f"source:{name}", {**m, **curve}, run_id)
        rows.append({"source": name, "P@10": m["precision@10"], "R@10": m["recall@10"],
                     "NDCG@10": m["ndcg@10"], "coverage": m["coverage"], **curve})
    lines += ["## Each source alone", "", pd.DataFrame(rows).set_index("source").to_markdown(floatfmt=".4f"), ""]

    names = [n for n in ALL_SOURCES if n != "popularity"]
    overlap = pd.DataFrame(
        [[np.mean([len(set(lists[a][u][:100]) & set(lists[b][u][:100])) / 100 for u in users]) for b in names]
         for a in names], index=names, columns=names)
    lines += ["## Overlap of top-100 lists (share of the row source's list also in the column source's)", "",
              overlap.to_markdown(floatfmt=".3f"), ""]

    for emb in ("text", "twotower"):
        grid = union_grid(lists, sd.truth, emb)
        for k, r in grid.iterrows():
            record_results(cfg, args.stage, f"union:cb+cf+{emb}",
                           {f"union_recall@k{k}": r["union recall"], f"union_size@k{k}": r["avg union size"]}, run_id)
        lines += [f"## Union of cb + cf + {emb}", "",
                  "Unique hits = share of the union's hits that only this source found.", "",
                  grid.to_markdown(floatfmt=".4f"), ""]

    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"{cfg.name}_{args.stage}_retrieval.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
