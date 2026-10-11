"""python -m recsys.cli {extract,align,embed,train,evaluate,coldstart,tune,all}"""
import argparse
import json
import logging

import pandas as pd

from recsys.config import Config

STEPS = ["extract", "align", "embed", "train", "evaluate", "coldstart"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="recsys")
    ap.add_argument("command", choices=[*STEPS, "tune", "all"])
    ap.add_argument("--users", type=int, default=1000, help="sampled users for coldstart/tune")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    pd.set_option("display.width", 200)
    cfg = Config()
    out = cfg.paths.artifacts

    for cmd in STEPS if args.command == "all" else [args.command]:
        if cmd == "extract":
            from recsys.data.extract import extract
            extract(cfg.paths)
        elif cmd == "align":
            from recsys.data.align import align, coverage_report
            print(json.dumps(coverage_report(align(cfg)), indent=2))
        elif cmd == "embed":
            from recsys.features.plot_embed import encode_plots
            items = pd.read_parquet(cfg.paths.items)
            print("plot embeddings:", encode_plots(items, cfg.sbert_model, cfg.chunk_words, cfg.paths.plot_emb).shape)
        elif cmd == "train":
            from recsys.pipeline import train
            train(cfg)
        elif cmd == "tune":
            from recsys.evaluation.evaluate import tune
            from recsys.pipeline import save_best_params
            out.mkdir(parents=True, exist_ok=True)
            res, best = tune(cfg, n_users=args.users)
            res.to_csv(out / "tuning_results.csv", index=False)
            save_best_params(best)
            print(res.round(4).to_string(index=False))
        else:
            from recsys.evaluation import evaluate as ev
            from recsys.pipeline import load_best_params
            out.mkdir(parents=True, exist_ok=True)
            c = load_best_params(cfg)
            res = ev.evaluate(c) if cmd == "evaluate" else ev.cold_start(c, n_users=args.users)
            name = "eval_results" if cmd == "evaluate" else "coldstart_results"
            res.to_csv(out / f"{name}.csv", index=False)
            (out / f"{name}.md").write_text(res.round(4).to_markdown(index=False))
            print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
