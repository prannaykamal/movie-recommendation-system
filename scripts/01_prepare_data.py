"""Phase 1: clean the catalog, prepare ratings, split them, and load SQLite."""
from __future__ import annotations

import pandas as pd

from hybridrec.cli import log, parser, timed
from hybridrec.config import CATALOG_DIR, get_config
from hybridrec.data.clean import build_catalog, save_catalog
from hybridrec.data.load_sql import CATALOG_TABLES, build_database
from hybridrec.data.sample import load_ratings
from hybridrec.data.split import assign_splits


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--rebuild-catalog", action="store_true")
    args = p.parse_args()
    cfg = get_config(args.dataset)

    if args.rebuild_catalog or not (CATALOG_DIR / "movies.parquet").exists():
        with timed("building catalog"):
            save_catalog(build_catalog(cfg.min_keyword_movies))
    catalog = {name: pd.read_parquet(CATALOG_DIR / f"{name}.parquet") for name in ("movies", *CATALOG_TABLES)}
    movies = catalog["movies"]
    log.info("catalog: %d movies", len(movies))

    with timed(f"loading ratings from {cfg.ratings_file}"):
        ratings, universe = load_ratings(cfg, movies["movie_id"].to_numpy())
    ratings = assign_splits(ratings)
    log.info("ratings: %d rows, %d users, %d movies rated; item universe %d",
             len(ratings), ratings.user_id.nunique(), ratings.movie_id.nunique(), len(universe))
    log.info("split sizes: %s", ratings["split"].value_counts().sort_index().to_dict())

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(cfg.processed_dir / "ratings.parquet", index=False)
    pd.DataFrame({"movie_id": universe}).to_parquet(cfg.processed_dir / "items.parquet", index=False)

    with timed(f"writing {cfg.db_path.name}"):
        build_database(cfg.db_path, catalog, universe, ratings)


if __name__ == "__main__":
    main()
