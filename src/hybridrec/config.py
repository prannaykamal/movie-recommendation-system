"""Dataset presets and paths shared by every script.

Both datasets run through the same code; a script picks its preset with
``--dataset small|full20k``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CATALOG_DIR = ROOT / "data" / "processed" / "catalog"
# Override with HYBRIDREC_QDRANT_PATH, e.g. to run the app on a copy while the pipeline writes.
QDRANT_PATH = Path(os.environ.get("HYBRIDREC_QDRANT_PATH", ROOT / "db" / "qdrant"))
REPORTS_DIR = ROOT / "reports"

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Per-user time split: cumulative ends of blocks R, L, V (T takes the rest).
SPLIT_ENDS = (0.7, 0.8, 0.9)
# Optional extra reranker-training window s0: fit on [0, 0.6), label [0.6, 0.7).
S0_END = 0.6


@dataclass(frozen=True)
class Config:
    name: str
    ratings_file: str
    sample_users: int | None = None
    seed: int = 42
    min_user_ratings: int = 20
    like_threshold: float = 4.0

    # Candidate generation: top-k per source (chosen on s2, see REPORT.md).
    k_cb: int = 100
    k_cf: int = 100
    k_emb: int = 100
    emb_source: str = "twotower"  # "twotower" or "text"
    use_s0: bool = False

    # Content-based
    min_keyword_movies: int = 2

    # iALS (confidence = 1 + als_alpha * rating / 5)
    als_factors: int = 64
    als_regularization: float = 0.05
    als_alpha: float = 10.0
    als_iterations: int = 15
    als_min_item_ratings: int = 5

    # Text-embedding v1 user profile
    emb_recent: int = 50
    emb_half_life: float = 20.0
    emb_clusters: int = 3
    emb_min_for_clusters: int = 12

    # Two-tower
    tt_dim: int = 64
    tt_history: int = 50
    tt_min_item_ratings: int = 5
    tt_epochs: int = 20
    tt_batch: int = 512
    tt_lr: float = 1e-3
    tt_temperature: float = 0.05
    tt_targets_per_user: int = 30
    tt_patience: int = 2

    # Reranker
    rr_history: int = 50
    rr_recent_sims: int = 20
    rr_neg_per_pos: int = 20
    rr_list_negatives: int = 100
    rr_epochs: int = 30
    rr_patience: int = 3
    rr_lr: float = 1e-3
    rr_dropout: float = 0.25
    rr_users_per_batch: int = 32
    rr_batch_rows: int = 1024

    @property
    def processed_dir(self) -> Path:
        return ROOT / "data" / "processed" / self.name

    @property
    def artifacts_dir(self) -> Path:
        return ROOT / "artifacts" / self.name

    def stage_dir(self, stage: str) -> Path:
        return self.artifacts_dir / stage

    @property
    def db_path(self) -> Path:
        return ROOT / "db" / f"{self.name}.db"

    @property
    def stages(self) -> tuple[str, ...]:
        return ("s0", "s1", "s2", "s3") if self.use_s0 else ("s1", "s2", "s3")

    def k_for(self, source: str) -> int:
        return {"cb": self.k_cb, "cf": self.k_cf, "emb": self.k_emb}[source]


SMALL = Config(
    name="small",
    ratings_file="ratings_small.csv",
    use_s0=True,
    als_factors=32,
    als_regularization=1.0,
    als_alpha=2.0,
    tt_dim=32,
    tt_epochs=30,
    rr_dropout=0.3,
)

FULL20K = Config(
    name="full20k",
    ratings_file="ratings.csv",
    sample_users=20_000,
    als_factors=64,
    als_regularization=1.0,
    als_alpha=5.0,
)

PRESETS = {c.name: c for c in (SMALL, FULL20K)}


def get_config(name: str, **overrides) -> Config:
    cfg = PRESETS[name]
    return replace(cfg, **overrides) if overrides else cfg
