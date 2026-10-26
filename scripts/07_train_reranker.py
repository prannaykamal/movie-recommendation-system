"""Phase 8b: train the reranker on s1 (+ s0) and early-stop on s2 NDCG@10.

--ablations retrains v1 with each feature group masked out and records s2 NDCG@10.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hybridrec.cli import log, parser, timed
from hybridrec.config import REPORTS_DIR, get_config
from hybridrec.data.dataset import Dataset, genre_matrix
from hybridrec.eval.metrics import beyond_accuracy
from hybridrec.eval.protocol import new_run_id, record_results, stage_data
from hybridrec.rerank import train as rt
from hybridrec.rerank.features import FEATURE_GROUPS, FEATURES
from hybridrec.retrieval.merge import topn_by_score
from hybridrec.retrieval.registry import text_vectors


def load_lists(cfg, stage: str, norm: rt.Normalizer | None = None) -> rt.ListData:
    sdir = cfg.stage_dir(stage)
    h = np.load(sdir / "history.npz")
    return rt.ListData.from_parquet(sdir / "features.parquet", FEATURES, h["users"], h["seq"], norm)


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--model", choices=["v1", "v2", "both"], default="both")
    p.add_argument("--ablations", action="store_true")
    args = p.parse_args()
    cfg = get_config(args.dataset)
    ds = Dataset.load(cfg)
    item_text, item_genres = text_vectors(ds), genre_matrix(ds.movies)
    s2 = stage_data(ds, "s2")
    s2_log_count = np.log1p(s2.hist.item_counts)
    train_stages = [s for s in cfg.stages if s in ("s0", "s1")]

    # Load each stage once; normalization is fitted on the training rows only.
    train_sets = [load_lists(cfg, s) for s in train_stages]
    norm = rt.Normalizer.fit(*(d.X for d in train_sets))
    for d in train_sets:
        norm.apply_(d.X)
    val = load_lists(cfg, "s2", norm)
    log.info("training lists: %s rows; validation: %d rows", [len(d.X) for d in train_sets], len(val.X))

    run_id = new_run_id()
    lines = [f"# Reranker: {cfg.name}", "", f"Trained on {', '.join(train_stages)}; early stopping on s2 NDCG@10.", ""]

    def fit(kind: str, keep=None, prefix: str = ""):
        model, history = rt.train(kind, train_sets, val, s2.truth, cfg, item_text, item_genres, prefix, keep)
        recs = topn_by_score(val.frame, rt.score_lists(model, val))
        best = max(history, key=lambda h: h["ndcg@10"])
        metrics = {**{k: v for k, v in best.items() if k not in ("epoch", "loss")},
                   **beyond_accuracy(recs, ds.items.movie_ids, s2_log_count)}
        return model, history, metrics

    for kind in ["v1", "v2"] if args.model == "both" else [args.model]:
        with timed(f"training reranker {kind}"):
            model, history, metrics = fit(kind)
        rt.save(cfg.artifacts_dir / f"reranker_{kind}.pt", model, kind, FEATURES, norm, history)
        record_results(cfg, "s2", f"reranker_{kind}", metrics, run_id)
        lines += [f"## {kind}: training history (s2 metrics per epoch)", "",
                  pd.DataFrame(history).set_index("epoch").to_markdown(floatfmt=".4f"), ""]

    if args.ablations:
        rows = []
        for group, gcols in FEATURE_GROUPS.items():
            keep = np.array([c not in gcols for c in FEATURES], dtype=np.float32)
            with timed(f"ablation without {group}"):
                _, _, m = fit("v1", keep, prefix=f"[-{group}] ")
            record_results(cfg, "s2", f"ablation_v1:-{group}", m, run_id)
            rows.append({"without": group, "NDCG@10": m["ndcg@10"],
                         "NDCG@10 (users with a positive)": m["ndcg@10_with_positive"],
                         "P@10": m["precision@10"], "R@10": m["recall@10"]})
        lines += ["## Ablations (v1 retrained with one feature group masked out, s2)", "",
                  pd.DataFrame(rows).set_index("without").to_markdown(floatfmt=".4f"), ""]

    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"{cfg.name}_s2_reranker{'_ablations' if args.ablations else ''}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
