"""Qdrant (local mode) storage for movie vectors.

Local mode allows one process at a time on the storage folder, so don't run the
pipeline scripts while the Streamlit app has it open.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

from hybridrec.config import QDRANT_PATH


def get_client(path=None) -> QdrantClient:
    path = Path(path or QDRANT_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def collection_name(dataset: str, kind: str, stage: str | None = None) -> str:
    return f"{dataset}_{kind}" + (f"_{stage}" if stage else "")


def payloads(movies: pd.DataFrame, n_ratings: np.ndarray) -> list[dict]:
    return [
        {
            "movie_id": int(r.movie_id),
            "title": r.title if isinstance(r.title, str) else "",
            "year": int(r.year) if pd.notna(r.year) else None,
            "genres": list(r.genres),
            "language": r.original_language if isinstance(r.original_language, str) else None,
            "n_ratings": int(n),
        }
        for r, n in zip(movies.itertuples(index=False), n_ratings)
    ]


def _skip_disk_sync(client: QdrantClient, name: str) -> None:
    """Local mode commits its SQLite file once per point; on Windows each commit
    waits for the disk, so 45k points take hours. Turning off the sync makes the
    write take seconds. The collection is rebuilt by the pipeline, so a crash
    mid-write costs nothing. No-op if the client's internals differ."""
    try:
        con = client._client.collections[name].storage.storage
        con.execute("PRAGMA synchronous = OFF")
        con.execute("PRAGMA journal_mode = MEMORY")
    except (AttributeError, KeyError):
        pass


def write_collection(client: QdrantClient, name: str, movie_ids: np.ndarray, vectors: np.ndarray,
                     payload: list[dict], batch: int = 2048) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=models.VectorParams(size=vectors.shape[1],
                                                                      distance=models.Distance.COSINE))
    _skip_disk_sync(client, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # local mode warns above 20k points; fine at this scale
        for s in range(0, len(movie_ids), batch):
            client.upsert(name, points=models.Batch(
                ids=[int(i) for i in movie_ids[s : s + batch]],
                vectors=vectors[s : s + batch].tolist(),
                payloads=payload[s : s + batch],
            ))


def search(client: QdrantClient, name: str, vector: np.ndarray, limit: int,
           exclude_ids=()) -> list[tuple[int, float, dict]]:
    """Nearest movies by cosine, excluding ``exclude_ids``: [(movie_id, score, payload)]."""
    flt = models.Filter(must_not=[models.HasIdCondition(has_id=[int(i) for i in exclude_ids])]) if len(exclude_ids) else None
    res = client.query_points(name, query=vector.tolist(), limit=limit, query_filter=flt, with_payload=True)
    return [(int(p.id), float(p.score), p.payload) for p in res.points]


def similar_movies(client: QdrantClient, name: str, movie_id: int, limit: int = 10) -> list[tuple[int, float, dict]]:
    res = client.query_points(name, query=int(movie_id), limit=limit, with_payload=True)
    return [(int(p.id), float(p.score), p.payload) for p in res.points if int(p.id) != int(movie_id)]


def fetch_vectors(client: QdrantClient, name: str, movie_ids) -> dict[int, np.ndarray]:
    pts = client.retrieve(name, ids=[int(i) for i in movie_ids], with_vectors=True)
    return {int(p.id): np.asarray(p.vector, dtype=np.float32) for p in pts}
