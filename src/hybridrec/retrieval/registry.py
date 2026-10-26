"""Build or load each candidate source for a stage, so scripts and serving agree."""
from __future__ import annotations

from hybridrec.data.dataset import Dataset, Histories
from hybridrec.retrieval.collaborative import ImplicitALS
from hybridrec.retrieval.content import ContentTfidf
from hybridrec.retrieval.embed_v1 import TextEmbeddingV1
from hybridrec.retrieval.popularity import Popularity
from hybridrec.retrieval.text_embed import load_text_embeddings
from hybridrec.retrieval.two_tower import TwoTower

ALL_SOURCES = ("popularity", "cb", "cf", "text", "twotower")


def text_vectors(ds: Dataset):
    return load_text_embeddings(ds.items.movie_ids)


def new_als(ds: Dataset) -> ImplicitALS:
    c = ds.cfg
    return ImplicitALS(c.als_factors, c.als_regularization, c.als_alpha, c.als_iterations,
                       c.als_min_item_ratings, c.seed)


def new_text_v1(ds: Dataset, vectors=None) -> TextEmbeddingV1:
    c = ds.cfg
    return TextEmbeddingV1(text_vectors(ds) if vectors is None else vectors, c.like_threshold,
                           c.emb_recent, c.emb_half_life, c.emb_clusters, c.emb_min_for_clusters)


def load_sources(ds: Dataset, stage: str, hist: Histories, names=ALL_SOURCES) -> dict:
    """Fitted sources for a stage (popularity is refit; it takes milliseconds)."""
    stage_dir = ds.cfg.stage_dir(stage)
    out, vectors = {}, None
    for name in names:
        if name == "popularity":
            out[name] = Popularity(ds.cfg.like_threshold).fit(hist)
        elif name == "cb":
            out[name] = ContentTfidf.load(ds.cfg.artifacts_dir / "content_tfidf.npz")
        elif name == "cf":
            out[name] = ImplicitALS.load(stage_dir / "als.npz")
        elif name in ("text", "twotower"):
            vectors = text_vectors(ds) if vectors is None else vectors
            out[name] = (new_text_v1(ds, vectors) if name == "text"
                         else TwoTower.load(stage_dir, ds.cfg, ds.movies, vectors))
    return out


def candidate_sources(ds: Dataset, stage: str, hist: Histories) -> dict:
    """The three merged sources, keyed cb / cf / emb."""
    emb = ds.cfg.emb_source
    srcs = load_sources(ds, stage, hist, ("cb", "cf", emb))
    return {"cb": srcs["cb"], "cf": srcs["cf"], "emb": srcs[emb]}
