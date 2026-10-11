"""Fit all engines on the full rating set and persist artifacts for serving."""
import json
import logging
from dataclasses import asdict, replace

import joblib
import numpy as np
from scipy import sparse

from recsys.config import Config
from recsys.evaluation.evaluate import fit_models, load_data

log = logging.getLogger(__name__)

PARAM_KEYS = ("n_components", "mean_shrinkage", "w_meta", "w_plot", "content_pop_gate", "center_plots", "ials_weight", "ials_factors",
              "ials_reg", "ials_confidence", "ials_iters", "alpha_max", "gamma", "pop_eta", "pop_penalty",
              "pos_threshold", "bayes_prior")


def _params(cfg: Config) -> dict:
    return {k: v for k, v in asdict(cfg).items() if k in PARAM_KEYS}


def load_best_params(cfg: Config) -> Config:
    """Apply tuned hyperparameters from artifacts/best_params.json if `tune` has been run."""
    path = cfg.paths.artifacts / "best_params.json"
    if path.exists():
        cfg = replace(cfg, **{k: v for k, v in json.loads(path.read_text()).items() if k in PARAM_KEYS})
    return cfg


def save_best_params(cfg: Config) -> None:
    cfg.paths.artifacts.mkdir(parents=True, exist_ok=True)
    (cfg.paths.artifacts / "best_params.json").write_text(json.dumps(_params(cfg), indent=2))


def train(cfg: Config | None = None) -> None:
    cfg = load_best_params(cfg or Config())
    data = load_data(cfg)
    f = fit_models(data, data.ratings, cfg)
    out = cfg.paths.artifacts
    out.mkdir(parents=True, exist_ok=True)
    data.items.to_parquet(out / "items.parquet", index=False)
    sparse.save_npz(out / "meta_vecs.npz", data.meta_vecs)
    sparse.save_npz(out / "ratings.npz", f.R)
    np.savez(out / "index.npz", user_ids=data.index.user_ids, movie_ids=data.index.movie_ids)
    np.save(out / "item_counts.npy", f.item_scores["most_popular"])
    joblib.dump(f.collab, out / "collab.joblib")
    if f.ials is not None:
        joblib.dump(f.ials, out / "ials.joblib")
    (out / "model_params.json").write_text(json.dumps(_params(cfg), indent=2))
    log.info("artifacts written to %s", out)
