"""Offline evaluation: baselines vs content / SVD / iALS / fixed-alpha / dynamic-alpha hybrids,
a simulated cold-start sweep over N_u, and a hyperparameter search on validation splits.

Every model is fitted on the training split only (user means, SVD, iALS, content profiles and
their popularity gate, N_u)."""
import itertools
import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.config import Config
from recsys.data import loaders
from recsys.data.interactions import Index, build_index, minmax_rows, rating_matrix
from recsys.evaluation.beyond import beyond_accuracy
from recsys.evaluation.metrics import ranking_metrics
from recsys.evaluation.split import random_split, temporal_split, truncate_history
from recsys.features.metadata import build_metadata_matrix
from recsys.features.plot_embed import encode_plots
from recsys.models import baselines
from recsys.models.collaborative import CollaborativeEngine
from recsys.models.content import ContentEngine
from recsys.models.hybrid import DynamicHybridRecommender
from recsys.models.ials import IALSEngine

log = logging.getLogger(__name__)

ALL_MODELS = ("random", "most_popular", "average_rating", "bayesian_average",
              "content", "svd", "ials", "hybrid_fixed", "hybrid_dynamic")


@dataclass
class Data:
    items: pd.DataFrame
    ratings: pd.DataFrame
    index: Index
    meta_vecs: sparse.csr_matrix
    plot_vecs: np.ndarray


def genre_matrix(items: pd.DataFrame) -> np.ndarray:
    """(n_items, n_genres) 0/1 matrix of MovieLens genres, in item-index order."""
    genres = items.sort_values("idx")["genres"].str.split("|")
    names = sorted({g for gs in genres for g in gs})
    return np.array([[g in gs for g in names] for gs in genres], dtype=np.float32)


def load_data(cfg: Config) -> Data:
    items = pd.read_parquet(cfg.paths.items)
    ratings = loaders.load_ml_ratings(cfg.paths)
    meta, _ = build_metadata_matrix(items)
    plot = encode_plots(items, cfg.sbert_model, cfg.chunk_words, cfg.paths.plot_emb)
    return Data(items, ratings, build_index(ratings, items), meta, plot)


@dataclass
class Fitted:
    R: sparse.csr_matrix
    n_u: np.ndarray
    content: ContentEngine
    collab: CollaborativeEngine
    ials: IALSEngine
    item_scores: dict  # baseline name -> per-item score vector


def fit_models(data: Data, train: pd.DataFrame, cfg: Config) -> Fitted:
    idx = data.index
    R = rating_matrix(idx.encode(train), idx.n_users, idx.n_items)
    counts = baselines.most_popular(R)
    content = ContentEngine(data.meta_vecs, data.plot_vecs, data.items["has_plot"].to_numpy(),
                            cfg.w_meta, cfg.w_plot, cfg.pos_threshold,
                            item_counts=counts if cfg.content_pop_gate else None, center_plots=cfg.center_plots)
    collab = CollaborativeEngine(cfg.n_components, cfg.seed, cfg.mean_shrinkage).fit(R)
    ials = (IALSEngine(cfg.ials_factors, cfg.ials_reg, cfg.ials_confidence, cfg.ials_iters, cfg.seed).fit(R)
            if cfg.ials_weight > 0 else None)
    item_scores = {
        "most_popular": counts,
        "average_rating": baselines.average_rating(R),
        "bayesian_average": baselines.bayesian_average(R, cfg.bayes_prior),
    }
    return Fitted(R, np.diff(R.indptr), content, collab, ials, item_scores)


def collab_scores(f: Fitted, uidx: np.ndarray, cfg: Config, svd=None, ials=None) -> np.ndarray:
    """S_collab = ials_weight * iALS + (1 - ials_weight) * SVD (both per-user min-max scaled)."""
    w = cfg.ials_weight if f.ials is not None else 0.0
    svd = f.collab.score(uidx) if svd is None and w < 1 else svd
    ials = f.ials.score(uidx) if ials is None and w > 0 else ials
    if w == 0:
        return svd
    return ials if w == 1 else w * ials + (1 - w) * svd


def popularity_prior(f: Fitted) -> np.ndarray:
    """S_popular: most-popular, min-max scaled. It is the whole score at N_u == 0 (beat Bayesian
    average by ~45% NDCG@10 there) and fades out as exp(-pop_eta * N_u)."""
    return minmax_rows(f.item_scores["most_popular"][None, :].astype(np.float32))[0]


def popularity_penalty(f: Fitted) -> np.ndarray:
    """P(i) = min-max scaled log(1 + training count); subtracted (times cfg.pop_penalty) to trade
    accuracy for catalogue exposure."""
    return minmax_rows(np.log1p(f.item_scores["most_popular"])[None, :].astype(np.float32))[0]


def score_batch(f: Fitted, uidx: np.ndarray, cfg: Config, models, rng) -> dict[str, np.ndarray]:
    """Score matrices (len(uidx) x n_items) for the requested models."""
    out = {}
    need_content = {"content", "hybrid_fixed", "hybrid_dynamic"} & set(models)
    need_hybrid = {"hybrid_fixed", "hybrid_dynamic"} & set(models)
    content = f.content.score(f.R[uidx]) if need_content else None
    svd = f.collab.score(uidx) if "svd" in models or (need_hybrid and cfg.ials_weight < 1) else None
    ials = f.ials.score(uidx) if f.ials is not None and ("ials" in models or need_hybrid) else None
    collab = collab_scores(f, uidx, cfg, svd, ials) if need_hybrid else None
    n_items = f.R.shape[1]
    for m in models:
        if m == "random":
            out[m] = rng.random((len(uidx), n_items), dtype=np.float32)
        elif m in f.item_scores:
            out[m] = np.broadcast_to(f.item_scores[m].astype(np.float32), (len(uidx), n_items))
        elif m == "content":
            out[m] = content
        elif m == "svd":
            out[m] = svd
        elif m == "ials":
            out[m] = ials
        elif m == "hybrid_fixed":
            out[m] = cfg.alpha_max * collab + (1 - cfg.alpha_max) * content
        elif m == "hybrid_dynamic":
            out[m] = DynamicHybridRecommender(cfg.alpha_max, cfg.gamma, cfg.pop_eta).blend(
                f.n_u[uidx], content, collab, popularity_prior(f))
            if cfg.pop_penalty:
                out[m] = out[m] - cfg.pop_penalty * popularity_penalty(f)
    return out


def relevant_sets(data: Data, test: pd.DataFrame, threshold: float) -> dict[int, set]:
    enc = data.index.encode(test[test["rating"] >= threshold])
    return enc.groupby("uidx")["iidx"].apply(set).to_dict()


def rank_users(f: Fitted, uidx: np.ndarray, cfg: Config, models, k_max: int, batch: int = 1000):
    rng = np.random.default_rng(cfg.seed)
    recs = {m: [] for m in models}
    for start in range(0, len(uidx), batch):
        b = uidx[start:start + batch]
        seen = [f.R[u].indices for u in b]
        for m, S in score_batch(f, b, cfg, models, rng).items():
            recs[m].append(DynamicHybridRecommender.top_k(S, seen, k_max))
    return {m: np.vstack(v) for m, v in recs.items()}


def evaluate_models(data: Data, train: pd.DataFrame, test: pd.DataFrame, cfg: Config,
                    models=ALL_MODELS, users: np.ndarray | None = None, fitted: Fitted | None = None) -> pd.DataFrame:
    f = fitted or fit_models(data, train, cfg)
    rel = relevant_sets(data, test, cfg.pos_threshold)
    uidx = np.array(sorted(rel) if users is None else [u for u in users if u in rel], dtype=np.int64)
    relevant = [rel[u] for u in uidx]
    recs = rank_users(f, uidx, cfg, models, max(cfg.ks))
    k_beyond = 10 if 10 in cfg.ks else max(cfg.ks)
    counts = f.item_scores["most_popular"]
    genres = genre_matrix(data.items)
    n_train_users = int((f.n_u > 0).sum())
    rows = []
    for m in models:
        row = {"model": m, "users": len(uidx)}
        for k in cfg.ks:
            row.update(ranking_metrics(recs[m], relevant, k))
        row.update(beyond_accuracy(recs[m], counts, n_train_users, genres, k_beyond, seed=cfg.seed))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(cfg: Config, data: Data | None = None) -> pd.DataFrame:
    """Full comparison on random and temporal splits."""
    data = data or load_data(cfg)
    frames = []
    for name, (train, test) in {"random": random_split(data.ratings, cfg.test_frac, cfg.seed),
                                "temporal": temporal_split(data.ratings, cfg.test_frac)}.items():
        log.info("evaluating split=%s", name)
        frames.append(evaluate_models(data, train, test, cfg).assign(split=name))
    return pd.concat(frames, ignore_index=True)


def cold_start(cfg: Config, data: Data | None = None, ns=(0, 3, 5, 10, 20, None), n_users: int = 1000,
               models=("most_popular", "content", "svd", "ials", "hybrid_fixed", "hybrid_dynamic")) -> pd.DataFrame:
    """Truncate a sample of users' training history to their earliest N ratings, refit, evaluate.
    N=None means the full training history."""
    data = data or load_data(cfg)
    train, test = random_split(data.ratings, cfg.test_frac, cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    users = rng.choice(data.index.user_ids, size=min(n_users, data.index.n_users), replace=False)
    uidx = np.array([data.index.uidx(u) for u in users])
    frames = []
    for n in ns:
        tr = train if n is None else truncate_history(train, users, n)
        log.info("cold-start N=%s", n)
        df = evaluate_models(data, tr, test, cfg, models, users=uidx)
        h = DynamicHybridRecommender(cfg.alpha_max, cfg.gamma, cfg.pop_eta)
        n_eff = n if n is not None else 1e9
        frames.append(df.assign(n_u="all" if n is None else n, alpha=float(h.calculate_alpha(n_eff)),
                                beta=float(h.calculate_beta(n_eff))))
    return pd.concat(frames, ignore_index=True)


def validation_splits(data: Data, cfg: Config) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Validation sets carved out of each protocol's training split; the test splits stay untouched."""
    r_train, _ = random_split(data.ratings, cfg.test_frac, cfg.seed)
    t_train, _ = temporal_split(data.ratings, cfg.test_frac)
    return {"random": random_split(r_train, cfg.test_frac, cfg.seed + 1),
            "temporal": temporal_split(t_train, cfg.test_frac)}


def tune(cfg: Config, data: Data | None = None, n_users: int = 1500, metric: str = "ndcg@10",
         grid_k=(32, 64), grid_w_meta=(0.5, 0.65, 0.8), grid_ials_weight=(0.0, 0.5, 0.75, 1.0),
         grid_alpha_max=(0.7, 0.8, 0.9), grid_gamma=(0.05, 0.15, 0.3, 0.6, 1.0),
         grid_pop_eta=(0.1, 0.2, 0.3, 0.4, 0.6, None), cold_ns=(3, 5, 10, 20)) -> tuple[pd.DataFrame, Config]:
    """Grid search on validation splits (test never touched).
    Stage 1 tunes SVD k, w_meta, the iALS/SVD mix and alpha_max on full histories, scoring each
    setting by its NDCG relative to the best setting, averaged over the random and temporal
    validation splits (the two protocols disagree, e.g. iALS alone wins one and loses the other).
    Stage 2 tunes gamma and the popularity fade pop_eta jointly on truncated histories, where
    alpha(u) and beta(u) actually vary (pop_eta=None: popularity only at N_u = 0)."""
    data = data or load_data(cfg)
    splits = validation_splits(data, cfg)
    rng = np.random.default_rng(cfg.seed)
    users = rng.choice(data.index.user_ids, size=min(n_users, data.index.n_users), replace=False)
    uidx = np.array([data.index.uidx(u) for u in users])
    k_col = int(metric.split("@")[1])
    rows = []
    for (k, w), (split_name, (fit_tr, val)) in itertools.product(itertools.product(grid_k, grid_w_meta),
                                                                 splits.items()):
        c = replace(cfg, n_components=k, w_meta=w, w_plot=1 - w, ials_weight=0.5, ks=(k_col,))
        f = fit_models(data, fit_tr, c)
        for iw, a in itertools.product(grid_ials_weight, grid_alpha_max):
            res = evaluate_models(data, fit_tr, val, replace(c, ials_weight=iw, alpha_max=a),
                                  ("hybrid_fixed",), users=uidx, fitted=f)
            rows.append({"stage": 1, "split": split_name, "n_components": k, "w_meta": w, "ials_weight": iw,
                         "alpha_max": a, "gamma": None, metric: res[metric].iat[0]})
            log.info("tune %s", rows[-1])
    stage1 = pd.DataFrame(rows)
    keys = ["n_components", "w_meta", "ials_weight", "alpha_max"]
    stage1["relative"] = stage1[metric] / stage1.groupby("split")[metric].transform("max")
    best = stage1.groupby(keys)["relative"].mean().idxmax()
    best_cfg = replace(cfg, **dict(zip(keys, best)), w_plot=1 - best[1])
    best_cfg = replace(best_cfg, n_components=int(best_cfg.n_components))

    fit_tr, val = splits["random"]
    scores = {}
    for n in cold_ns:
        tr = truncate_history(fit_tr, users, n)
        f = fit_models(data, tr, best_cfg)
        for g, eta in itertools.product(grid_gamma, grid_pop_eta):
            res = evaluate_models(data, tr, val, replace(best_cfg, gamma=g, pop_eta=eta, ks=(k_col,)),
                                  ("hybrid_dynamic",), users=uidx, fitted=f)
            scores.setdefault((g, eta), []).append(res[metric].iat[0])
    for (g, eta), v in scores.items():
        rows.append({"stage": 2, "split": "random", **{k_: getattr(best_cfg, k_) for k_ in keys},
                     "gamma": g, "pop_eta": eta, metric: float(np.mean(v))})
        log.info("tune %s", rows[-1])
    results = pd.concat([stage1, pd.DataFrame(rows[len(stage1):])], ignore_index=True)
    stage2 = results[results["stage"] == 2]
    best = stage2.loc[stage2[metric].idxmax()]
    eta = best["pop_eta"]
    best_cfg = replace(best_cfg, gamma=float(best["gamma"]), pop_eta=None if pd.isna(eta) else float(eta))
    return results, best_cfg
