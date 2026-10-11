"""Paths and hyperparameters."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Paths:
    root: Path = ROOT
    ml_zip: Path = ROOT / "ml-1m.zip"
    cmu_tar: Path = ROOT / "MovieSummaries.tar.gz"
    raw: Path = ROOT / "data" / "raw"
    processed: Path = ROOT / "data" / "processed"
    artifacts: Path = ROOT / "artifacts"

    @property
    def ml_dir(self) -> Path:
        return self.raw / "ml-1m"

    @property
    def cmu_dir(self) -> Path:
        return self.raw / "MovieSummaries"

    @property
    def items(self) -> Path:
        return self.processed / "items.parquet"

    @property
    def plot_emb(self) -> Path:
        return self.artifacts / "plot_emb.npy"


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    # content (defaults = validation-tuned values; the design doc used w_meta=.35, w_plot=.65)
    w_meta: float = 0.8
    w_plot: float = 0.2
    pos_threshold: float = 4.0
    content_pop_gate: bool = True  # S_content * log(1 + item interactions)
    sbert_model: str = "all-MiniLM-L6-v2"
    chunk_words: int = 150
    top_cast: int = 5
    # collaborative: S_collab = ials_weight * iALS + (1 - ials_weight) * SVD
    n_components: int = 32
    mean_shrinkage: float = 5.0
    ials_weight: float = 0.5
    ials_factors: int = 32
    ials_reg: float = 3.0
    ials_confidence: float = 3.0
    ials_iters: int = 30
    # hybrid (design doc: alpha_max=.8, gamma=.15)
    alpha_max: float = 0.80
    gamma: float = 0.15
    # fading popularity prior beta(u) = exp(-pop_eta * N_u); None = popularity only at N_u = 0
    pop_eta: float | None = 0.2
    # alignment
    fuzzy_threshold: float = 0.90
    # evaluation
    seed: int = 42
    test_frac: float = 0.2
    ks: tuple = (5, 10, 20)
    bayes_prior: int = 50
