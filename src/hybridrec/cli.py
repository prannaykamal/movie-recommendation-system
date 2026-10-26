"""Argument parsing and logging shared by the pipeline scripts."""
from __future__ import annotations

import argparse
import logging
import time
from contextlib import contextmanager

from hybridrec.config import PRESETS

log = logging.getLogger("hybridrec")


def parser(description: str, stage: bool = False, default_stage: str | None = None) -> argparse.ArgumentParser:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dataset", choices=sorted(PRESETS), default="small")
    if stage:
        p.add_argument("--stage", choices=["s0", "s1", "s2", "s3", "all"], default=default_stage or "all")
    return p


@contextmanager
def timed(label: str):
    start = time.perf_counter()
    log.info("%s ...", label)
    yield
    log.info("%s done in %.1fs", label, time.perf_counter() - start)
