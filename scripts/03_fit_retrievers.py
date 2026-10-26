"""Phases 4 and 6: fit iALS and the two-tower model for each stage.

--tune-als runs a small grid on s2 (fit on R+L, recall@100 on V) and prints the
best setting; copy it into config.py.
"""
from __future__ import annotations

import itertools

from hybridrec.cli import log, parser, timed
from hybridrec.config import get_config
from hybridrec.data.dataset import Dataset
from hybridrec.eval.metrics import candidate_recall
from hybridrec.eval.protocol import new_run_id, recommend_all, record_results, stage_data
from hybridrec.retrieval.collaborative import ImplicitALS
from hybridrec.retrieval.registry import new_als, text_vectors
from hybridrec.retrieval.two_tower import TwoTower
from hybridrec.vectordb import qdrant_store as qs


def tune_als(ds: Dataset) -> None:
    sd = stage_data(ds, "s2")
    run_id, results = new_run_id(), []
    for factors, reg, alpha in itertools.product((16, 32, 64, 128), (0.1, 1.0), (2.0, 5.0, 20.0)):
        m = ImplicitALS(factors, reg, alpha, ds.cfg.als_iterations, ds.cfg.als_min_item_ratings, ds.cfg.seed)
        recs = recommend_all(m.fit(sd.hist), sd.hist, sd.eval_users, 100)
        r = candidate_recall(recs, sd.truth)
        results.append((r, factors, reg, alpha))
        record_results(ds.cfg, "s2", f"als_tune:f{factors}_r{reg}_a{alpha:g}", {"recall@100": r}, run_id)
        log.info("factors=%d reg=%.2f alpha=%.0f -> recall@100 %.4f", factors, reg, alpha, r)
    log.info("best: recall@100 %.4f with factors=%d regularization=%.2f alpha=%.0f", *max(results))


def main() -> None:
    p = parser(__doc__, stage=True)
    p.add_argument("--tune-als", action="store_true")
    p.add_argument("--skip", nargs="*", default=[], choices=["cf", "twotower"])
    args = p.parse_args()
    cfg = get_config(args.dataset)
    ds = Dataset.load(cfg)
    if args.tune_als:
        tune_als(ds)
        return

    fit_tt = "twotower" not in args.skip
    vectors = text_vectors(ds) if fit_tt else None
    client = qs.get_client() if fit_tt else None
    for stage in cfg.stages if args.stage == "all" else (args.stage,):
        sd = stage_data(ds, stage)
        out = cfg.stage_dir(stage)
        out.mkdir(parents=True, exist_ok=True)
        if "cf" not in args.skip:
            with timed(f"[{stage}] iALS"):
                new_als(ds).fit(sd.hist).save(out / "als.npz")
        if fit_tt:
            with timed(f"[{stage}] two-tower"):
                tt = TwoTower(cfg, ds.movies, vectors).fit(sd.hist)
                tt.save(out)
                qs.write_collection(client, qs.collection_name(cfg.name, "twotower", stage), ds.items.movie_ids,
                                    tt.item_vecs, qs.payloads(ds.movies, sd.hist.item_counts))
    if client is not None:
        client.close()


if __name__ == "__main__":
    main()
