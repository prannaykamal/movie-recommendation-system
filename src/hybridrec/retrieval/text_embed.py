"""Sentence-transformer embeddings of each movie's story text.

Text = "<title> (<year>). Genres: ... <tagline> <overview>". Cast, director and
keywords stay out (they drive the content-based source), except that keywords
fill in when the overview is missing. Computed once for the whole catalog.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hybridrec.config import CATALOG_DIR, TEXT_MODEL

EMB_PATH = CATALOG_DIR / "text_emb.npy"
IDS_PATH = CATALOG_DIR / "text_emb_ids.npy"


def movie_text(row) -> str:
    title = row.title or ""
    year = f" ({int(row.year)})" if pd.notna(row.year) else ""
    parts = [f"{title}{year}."]
    if len(row.genres):
        parts.append("Genres: " + ", ".join(row.genres) + ".")
    if row.tagline:
        parts.append(str(row.tagline))
    if row.overview:
        parts.append(str(row.overview))
    elif len(row.keywords):
        parts.append("Keywords: " + ", ".join(row.keywords) + ".")
    return " ".join(parts)


def encode_catalog(movies: pd.DataFrame, batch_size: int = 128) -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(TEXT_MODEL, device="cpu")
    texts = [movie_text(r) for r in movies.itertuples(index=False)]
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    np.save(EMB_PATH, emb.astype(np.float32))
    np.save(IDS_PATH, movies["movie_id"].to_numpy(np.int64))


def load_text_embeddings(movie_ids: np.ndarray) -> np.ndarray:
    """(len(movie_ids), 384) unit vectors aligned with ``movie_ids``."""
    emb, ids = np.load(EMB_PATH), np.load(IDS_PATH)
    pos = np.searchsorted(ids, movie_ids)
    assert (ids[pos] == movie_ids).all(), "text embeddings missing for some movies"
    return emb[pos]
