"""Stage data, batch top-k scoring, and the eval_results table."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from hybridrec.config import Config
from hybridrec.data.dataset import Dataset, Histories, mask_seen, topk
from hybridrec.data.load_sql import connect
from hybridrec.data.split import stage_frames


class Scorer(Protocol):
    name: str

    def fit(self, hist: Histories) -> "Scorer": ...

    def score(self, hist: Histories, rows: np.ndarray) -> np.ndarray:
        """(len(rows), n_items) float32 scores; NaN where the item can't be scored."""
        ...


@dataclass
class StageData:
    stage: str
    fit: pd.DataFrame
    target: pd.DataFrame
    hist: Histories
    truth: dict[int, set[int]]  # user_id -> liked movie_ids in the target block

    @property
    def eval_users(self) -> np.ndarray:
        return np.array(sorted(self.truth), dtype=np.int64)


def liked_truth(target: pd.DataFrame, threshold: float) -> dict[int, set[int]]:
    liked = target[target["rating"] >= threshold]
    return {int(u): set(g.tolist()) for u, g in liked.groupby("user_id")["movie_id"]}


def stage_data(ds: Dataset, stage: str) -> StageData:
    fit, target = stage_frames(ds.ratings, stage)
    return StageData(stage, fit, target, Histories(fit, ds.items), liked_truth(target, ds.cfg.like_threshold))


def batched_topk(scorer: Scorer, hist: Histories, user_ids: np.ndarray, k: int, batch: int = 256):
    """Yield (user_ids, item_idx (B, k), scores (B, k)) with already-rated items excluded."""
    rows = hist.positions(user_ids)
    for s in range(0, len(user_ids), batch):
        r = rows[s : s + batch]
        scores = mask_seen(scorer.score(hist, r), hist, r)
        idx, vals = topk(scores, k)
        yield user_ids[s : s + batch], idx, vals


def recommend_all(scorer: Scorer, hist: Histories, user_ids: np.ndarray, k: int) -> dict[int, list[int]]:
    """Top-k movie_ids per user."""
    out = {}
    movie_ids = hist.items.movie_ids
    for users, idx, vals in batched_topk(scorer, hist, user_ids, k):
        for u, row, v in zip(users, idx, vals):
            out[int(u)] = movie_ids[row[np.isfinite(v)]].tolist()
    return out


def new_run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def record_results(cfg: Config, stage: str, model: str, metrics: dict[str, float], run_id: str) -> None:
    now = dt.datetime.now().isoformat(timespec="seconds")
    rows = [(run_id, cfg.name, stage, model, m, float(v), now) for m, v in metrics.items()]
    con = connect(cfg.db_path)
    try:
        con.execute("DELETE FROM eval_results WHERE dataset=? AND stage=? AND model=? AND run_id=?",
                    (cfg.name, stage, model, run_id))
        con.executemany("INSERT INTO eval_results VALUES (?,?,?,?,?,?,?)", rows)
        con.commit()
    finally:
        con.close()


def latest_results(cfg: Config) -> pd.DataFrame:
    """Most recent value of every (stage, model, metric)."""
    con = connect(cfg.db_path)
    try:
        df = pd.read_sql("SELECT * FROM eval_results WHERE dataset = ?", con, params=(cfg.name,))
    finally:
        con.close()
    return df.sort_values("created_at").drop_duplicates(["stage", "model", "metric"], keep="last")
