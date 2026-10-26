"""Phases 3 and 5: TF-IDF item matrix, text embeddings, and the Qdrant text collection."""
from __future__ import annotations

import numpy as np

from hybridrec.cli import log, parser, timed
from hybridrec.config import get_config
from hybridrec.data.clean import load_movies
from hybridrec.data.dataset import Dataset
from hybridrec.data.split import stage_frames
from hybridrec.retrieval.content import ContentTfidf, build_item_matrix
from hybridrec.retrieval.text_embed import EMB_PATH, encode_catalog, load_text_embeddings
from hybridrec.vectordb import qdrant_store as qs


def main() -> None:
    args = parser(__doc__).parse_args()
    cfg = get_config(args.dataset)
    ds = Dataset.load(cfg)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    with timed("TF-IDF item matrix"):
        X, _ = build_item_matrix(ds.movies)
        ContentTfidf.save(cfg.artifacts_dir / "content_tfidf.npz", X)
    log.info("TF-IDF: %d movies x %d tokens", *X.shape)

    if not EMB_PATH.exists():
        with timed("encoding catalog text"):
            encode_catalog(load_movies().sort_values("movie_id"))
    E = load_text_embeddings(ds.items.movie_ids)

    # Payload rating counts come from the final fit window (R+L+V), which the UI uses.
    fit, _ = stage_frames(ds.ratings, "s3")
    counts = np.bincount(ds.items.index(fit["movie_id"].to_numpy()), minlength=len(ds.items))
    with timed("writing Qdrant text collection"):
        client = qs.get_client()
        name = qs.collection_name(cfg.name, "movie_text")
        qs.write_collection(client, name, ds.items.movie_ids, E, qs.payloads(ds.movies, counts))
        log.info("%s: %d points", name, client.count(name).count)
        client.close()


if __name__ == "__main__":
    main()
