"""Phase 9: final test on s3 (fit on R+L+V, score against T), run once.

Run ``04_eval_retrievers.py --stage s3`` and ``05_candidates.py`` first: they
record each source's and RRF's s3 metrics. This script scores the rerankers on
s3, writes rerank scores into the SQL candidates table, and compiles
reports/<dataset>_final.md from eval_results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hybridrec.cli import log, parser
from hybridrec.config import REPORTS_DIR, get_config
from hybridrec.data.dataset import Dataset, genre_matrix
from hybridrec.data.load_sql import connect
from hybridrec.eval.metrics import beyond_accuracy
from hybridrec.eval.protocol import latest_results, new_run_id, record_results, stage_data
from hybridrec.rerank import train as rt
from hybridrec.rerank.rrf import rrf_scores
from hybridrec.retrieval.merge import topn_by_score
from hybridrec.retrieval.registry import text_vectors

MODELS = ["source:popularity", "source:cb", "source:cf", "source:text", "source:twotower", "rrf",
          "reranker_v1", "reranker_v2"]
COLS = {"precision@10": "P@10", "recall@10": "R@10", "recall_trunc@10": "R@10 (trunc)", "ndcg@10": "NDCG@10",
        "coverage": "coverage", "avg_log_popularity": "avg log pop"}


def main() -> None:
    args = parser(__doc__).parse_args()
    cfg = get_config(args.dataset)
    ds = Dataset.load(cfg)
    sd = stage_data(ds, "s3")
    run_id = new_run_id()
    log_count = np.log1p(sd.hist.item_counts)
    item_text, item_genres = text_vectors(ds), genre_matrix(ds.movies)

    sdir = cfg.stage_dir("s3")
    h = np.load(sdir / "history.npz")
    cands = pd.read_parquet(sdir / "candidates.parquet")
    cands = cands[cands["user_id"].isin(set(sd.truth))]
    record_results(cfg, "s3", "rrf", beyond_accuracy(topn_by_score(cands, rrf_scores(cands)),
                                                      ds.items.movie_ids, log_count), run_id)

    best = rt.best_kind(cfg)
    for kind in ("v1", "v2"):
        path = cfg.artifacts_dir / f"reranker_{kind}.pt"
        if not path.exists():
            continue
        model, cols, norm = rt.load(path, cfg, item_text, item_genres)
        data = rt.ListData.from_parquet(sdir / "features.parquet", cols, h["users"], h["seq"], norm)
        m = rt.evaluate_lists(model, data, sd.truth)
        scores = rt.score_lists(model, data)
        recs = topn_by_score(data.frame, scores)
        m.update(beyond_accuracy(recs, ds.items.movie_ids, log_count))
        record_results(cfg, "s3", f"reranker_{kind}", m, run_id)
        log.info("s3 reranker %s: P@10 %.4f R@10 %.4f NDCG@10 %.4f", kind, m["precision@10"], m["recall@10"],
                 m["ndcg@10"])
        if kind == best:
            _write_scores(cfg, data.frame.assign(rerank_score=scores))

    res = latest_results(cfg)
    lines = [f"# Final results: {cfg.name}", "",
             f"Test = block T (s3: every model refit on R+L+V). {len(sd.truth)} users with a liked test movie. "
             f"Serving uses reranker {best} (higher s2 NDCG@10).", ""]
    for stage in ("s3", "s2"):
        sub = res[(res["stage"] == stage) & res["model"].isin(MODELS) & res["metric"].isin(COLS)]
        table = sub.pivot(index="model", columns="metric", values="value").reindex(MODELS).dropna(how="all")
        table = table[[c for c in COLS if c in table]].rename(columns=COLS)
        title = "Test (s3)" if stage == "s3" else "Validation (s2), for the s2-vs-s3 gap"
        lines += [f"## {title}", "", table.to_markdown(floatfmt=".4f"), ""]
    cand = res[(res["model"] == "candidates")].pivot(index="stage", columns="metric", values="value")
    lines += ["## Candidate stage", "", cand.to_markdown(floatfmt=".4f"), ""]
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"{cfg.name}_final.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", path)
    print("\n".join(lines))


def _write_scores(cfg, frame: pd.DataFrame) -> None:
    frame = frame.sort_values(["user_id", "rerank_score"], ascending=[True, False])
    frame["final_rank"] = frame.groupby("user_id").cumcount() + 1
    con = connect(cfg.db_path)
    try:
        con.executemany(
            "UPDATE candidates SET rerank_score = ?, final_rank = ? WHERE stage = 's3' AND user_id = ? AND movie_id = ?",
            ((float(s), int(r), int(u), int(m)) for u, m, s, r in
             frame[["user_id", "movie_id", "rerank_score", "final_rank"]].itertuples(index=False)))
        con.commit()
    finally:
        con.close()


if __name__ == "__main__":
    main()
