"""Dense Sentence-BERT plot embeddings V_plot(i), chunked to beat the 256-token limit."""
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def plot_mean(emb: np.ndarray, has_plot: np.ndarray) -> np.ndarray:
    """Mean embedding over items that have a plot (the centre used by center_embeddings)."""
    return emb[np.asarray(has_plot, dtype=bool)].mean(axis=0).astype(np.float32)


def center_embeddings(emb: np.ndarray, has_plot: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
    """Subtract the mean plot embedding and re-normalise; rows without a plot stay zero.

    SBERT embeddings occupy a narrow cone, so raw cosines are compressed and a user's averaged
    profile looks similar to every movie. Centring restores the contrast: on movies with zero
    ratings, plot-only content went from 0.060 to 0.172 NDCG@10 and metadata + plots from -1.5%
    to +9.2% over metadata alone (experiments/exp10_item_cold_start.py).

    Pass `mean` to centre new items (e.g. movies added after training) on the catalogue's centre."""
    has = np.asarray(has_plot, dtype=bool)
    out = emb.astype(np.float32, copy=True)
    out[has] -= plot_mean(emb, has) if mean is None else mean
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out = np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)
    out[~has] = 0.0
    return out


def chunk_words(text: str, size: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)] or [""]


def _fingerprint(items: pd.DataFrame, model_name: str, size: int) -> str:
    h = hashlib.sha1(f"{model_name}|{size}".encode())
    for mid, plot in zip(items["movieId"], items["plot_summary"].fillna("")):
        h.update(f"{mid}\x00{plot}\x01".encode())
    return h.hexdigest()


def encode_plots(items: pd.DataFrame, model_name: str, chunk_size: int, cache: Path,
                 batch_size: int = 64) -> np.ndarray:
    """(n_items, d) float32, L2-normalised; rows for items without a plot are all-zero."""
    fp = _fingerprint(items, model_name, chunk_size)
    meta = cache.with_suffix(".sha1")
    if cache.exists() and meta.exists() and meta.read_text() == fp:
        return np.load(cache)

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    dim = (getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension)()

    has_plot = items["has_plot"].to_numpy()
    chunks, owner = [], []
    for i in np.flatnonzero(has_plot):
        for c in chunk_words(items["plot_summary"].iat[i], chunk_size):
            chunks.append(c)
            owner.append(i)
    log.info("encoding %d chunks for %d plots on %s", len(chunks), has_plot.sum(), device)
    chunk_emb = model.encode(chunks, batch_size=batch_size, normalize_embeddings=True,
                             show_progress_bar=True, convert_to_numpy=True)

    emb = np.zeros((len(items), dim), dtype=np.float32)
    np.add.at(emb, np.asarray(owner), chunk_emb.astype(np.float32))
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = np.divide(emb, norms, out=np.zeros_like(emb), where=norms > 0)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    meta.write_text(fp)
    return emb
