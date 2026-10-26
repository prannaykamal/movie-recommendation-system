"""Phase 7: merge the three sources into labelled candidate lists for each stage.

Records union recall and the RRF baseline per stage. s3 candidates also go into
the SQL candidates table for the UI.
"""
from __future__ import annotations

import numpy as np

from hybridrec.cli import log, parser, timed
from hybridrec.config import get_config
from hybridrec.data.dataset import Dataset
from hybridrec.data.load_sql import connect
from hybridrec.eval.metrics import beyond_accuracy, candidate_recall, evaluate_topk
from hybridrec.eval.protocol import new_run_id, record_results, stage_data
from hybridrec.rerank.rrf import rrf_scores
from hybridrec.retrieval.merge import SOURCES, add_labels, merge_candidates, topn_by_score
from hybridrec.retrieval.registry import candidate_sources

SQL_COLS = ["user_id", "movie_id", *[f"{s}_{c}" for s in SOURCES for c in ("score", "rank")], "n_sources"]


def main() -> None:
    args = parser(__doc__, stage=True).parse_args()
    cfg = get_config(args.dataset)
    ds = Dataset.load(cfg)
    ks = {src: cfg.k_for(src) for src in SOURCES}
    run_id = new_run_id()
    for stage in cfg.stages if args.stage == "all" else (args.stage,):
        sd = stage_data(ds, stage)
        users = sd.hist.users.astype(np.int64)
        with timed(f"[{stage}] merging candidates for {len(users)} users (k={ks}, emb={cfg.emb_source})"):
            cands = merge_candidates(candidate_sources(ds, stage, sd.hist), sd.hist, users, ks)
        cands = add_labels(cands, sd.target, cfg.like_threshold)
        cands.to_parquet(cfg.stage_dir(stage) / "candidates.parquet", index=False)

        by_user = cands.groupby("user_id")["movie_id"].apply(list).to_dict()
        stats = {"union_recall": candidate_recall(by_user, sd.truth),
                 "avg_candidates": float(cands.groupby("user_id").size().mean()),
                 "positive_rate": float(cands["label"].mean())}
        evaluated = cands[cands["user_id"].isin(set(sd.truth))]
        rrf_recs = topn_by_score(evaluated, rrf_scores(evaluated))
        rrf = {**evaluate_topk(rrf_recs, sd.truth),
               **beyond_accuracy(rrf_recs, ds.items.movie_ids, np.log1p(sd.hist.item_counts))}
        record_results(cfg, stage, "candidates", stats, run_id)
        record_results(cfg, stage, "rrf", rrf, run_id)
        log.info("[%s] %s | RRF P@10 %.4f R@10 %.4f NDCG@10 %.4f", stage,
                 {k: round(v, 4) for k, v in stats.items()}, rrf["precision@10"], rrf["recall@10"], rrf["ndcg@10"])

        if stage == "s3":
            con = connect(cfg.db_path)
            try:
                con.execute("DELETE FROM candidates WHERE stage = 's3'")
                cands[SQL_COLS].assign(stage="s3").to_sql("candidates", con, if_exists="append",
                                                          index=False, chunksize=50_000)
                con.commit()
            finally:
                con.close()


if __name__ == "__main__":
    main()
