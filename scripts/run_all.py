"""Run the whole pipeline for one dataset, in order.

Equivalent to running scripts 01-08 by hand. Tuning steps (03 --tune-als) and
the k grid (04 on s2) are not repeated here: their chosen values live in config.py.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from hybridrec.cli import log, parser

SCRIPTS = Path(__file__).resolve().parent


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--from-step", type=int, default=1, help="skip steps before this number")
    p.add_argument("--ablations", action="store_true")
    args = p.parse_args()
    d = ["--dataset", args.dataset]
    steps = [
        (1, "01_prepare_data.py", d),
        (2, "02_build_item_features.py", d),
        (3, "03_fit_retrievers.py", d),
        (4, "04_eval_retrievers.py", [*d, "--stage", "s2"]),
        (5, "05_candidates.py", d),
        (6, "06_features.py", d),
        (7, "07_train_reranker.py", [*d, *(["--ablations"] if args.ablations else [])]),
        (8, "04_eval_retrievers.py", [*d, "--stage", "s3"]),
        (9, "08_evaluate.py", d),
    ]
    for n, script, extra in steps:
        if n < args.from_step:
            continue
        start = time.perf_counter()
        log.info("step %d: %s %s", n, script, " ".join(extra))
        subprocess.run([sys.executable, "-W", "ignore", str(SCRIPTS / script), *extra], check=True)
        log.info("step %d finished in %.0fs", n, time.perf_counter() - start)


if __name__ == "__main__":
    main()
