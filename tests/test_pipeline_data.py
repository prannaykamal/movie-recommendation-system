"""Checks that need the prepared data or trained artifacts; skipped when missing."""
import numpy as np
import pandas as pd
import pytest

from hybridrec.config import CATALOG_DIR, RAW_DIR, SMALL
from hybridrec.data.clean import parse_literal

needs_catalog = pytest.mark.skipif(not (CATALOG_DIR / "movies.parquet").exists(), reason="run 01_prepare_data first")
needs_small = pytest.mark.skipif(not (SMALL.processed_dir / "ratings.parquet").exists(), reason="small not prepared")


def test_parse_literal_handles_single_quotes_and_garbage():
    assert parse_literal("[{'id': 16, 'name': 'Animation'}]", []) == [{"id": 16, "name": "Animation"}]
    assert parse_literal("not a literal", []) == []
    assert parse_literal(float("nan"), None) is None


@needs_catalog
def test_id_join_covers_ratings():
    movies = pd.read_parquet(CATALOG_DIR / "movies.parquet", columns=["movie_id"])
    assert movies["movie_id"].is_unique
    r = pd.read_csv(RAW_DIR / "ratings_small.csv", usecols=["movieId"])
    assert r["movieId"].isin(set(movies["movie_id"])).mean() >= 0.99


@needs_small
def test_feature_leakage_s1_ignores_later_blocks():
    """s1 features must not change when every L/V/T rating is deleted."""
    from hybridrec.data.dataset import Dataset
    from hybridrec.rerank.features import FEATURES, ItemTable, stage_features
    from hybridrec.retrieval.text_embed import EMB_PATH

    if not EMB_PATH.exists() or not (SMALL.stage_dir("s1") / "candidates.parquet").exists():
        pytest.skip("needs text embeddings and s1 candidates")
    ds = Dataset.load(SMALL)
    from hybridrec.retrieval.registry import text_vectors

    it = ItemTable.build(ds, text_vectors(ds))
    cands = pd.read_parquet(SMALL.stage_dir("s1") / "candidates.parquet")
    cands = cands[cands["user_id"].isin(cands["user_id"].unique()[:25])]
    tt = np.load(SMALL.stage_dir("s1") / "twotower_items.npy")
    full, users_a, seq_a = stage_features(ds, "s1", cands, it, tt)
    truncated = Dataset(ds.cfg, ds.ratings[ds.ratings["split"] == "R"], ds.items, ds.movies)
    cut, users_b, seq_b = stage_features(truncated, "s1", cands, it, tt)
    pd.testing.assert_frame_equal(full[FEATURES], cut[FEATURES])
    assert (users_a == users_b).all() and (seq_a == seq_b).all()


@needs_small
def test_serving_candidates_match_offline_merge():
    """Qdrant-backed serving must reproduce the offline (NumPy) s3 candidates."""
    if not (SMALL.artifacts_dir / "reranker_v1.pt").exists():
        pytest.skip("needs a trained pipeline")
    from hybridrec.serving.recommend import Recommender

    try:
        rec = Recommender(SMALL)
    except RuntimeError as e:  # Qdrant folder locked by a running app
        pytest.skip(str(e))
    offline = pd.read_parquet(SMALL.stage_dir("s3") / "candidates.parquet")
    for user in offline["user_id"].unique()[:5]:
        served = rec.candidates(rec.user_history(int(user))).set_index("movie_id").sort_index()
        off = offline[offline.user_id == user].set_index("movie_id").sort_index()
        assert set(served.index) == set(off.index)
        for col in ("cb_score", "cf_score", "emb_score"):
            np.testing.assert_allclose(served[col], off[col], rtol=1e-4, atol=1e-4)
    rec.client.close()
