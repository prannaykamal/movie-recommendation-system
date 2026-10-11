import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recsys.data.interactions import minmax_rows
from recsys.evaluation.split import random_split, temporal_split, truncate_history
from recsys.models.collaborative import CollaborativeEngine
from recsys.models.content import ContentEngine
from recsys.models.hybrid import DynamicHybridRecommender


@pytest.mark.parametrize("n, expected", [(0, 0.0), (3, 0.29), (10, 0.62), (20, 0.76)])
def test_alpha_matches_design_table(n, expected):
    assert DynamicHybridRecommender(0.8, 0.15).calculate_alpha(n) == pytest.approx(expected, abs=0.005)


def test_alpha_saturates_at_alpha_max():
    assert DynamicHybridRecommender(0.8, 0.15).calculate_alpha(1000) == pytest.approx(0.8)


def test_top_k_excludes_seen_and_orders():
    scores = np.array([[0.9, 0.8, 0.1, 0.7, 0.5]])
    top = DynamicHybridRecommender.top_k(scores, [np.array([0])], 3)[0]
    assert top.tolist() == [1, 3, 4]


def test_blend_uses_fallback_for_cold_users():
    h = DynamicHybridRecommender(0.8, 0.15)
    content = np.full((2, 3), 0.5, dtype=np.float32)
    collab = np.ones((2, 3), dtype=np.float32)
    fb = np.array([0.0, 1.0, 0.2], dtype=np.float32)
    out = h.blend(np.array([0, 20]), content, collab, fb)
    assert out[0].tolist() == fb.tolist()
    a = h.calculate_alpha(20)
    assert out[1] == pytest.approx(a * 1 + (1 - a) * 0.5)


def test_minmax_rows_constant_row_is_zero():
    out = minmax_rows(np.array([[1.0, 3.0, 2.0], [4.0, 4.0, 4.0]]))
    assert out[0].tolist() == [0.0, 1.0, 0.5]
    assert out[1].tolist() == [0.0, 0.0, 0.0]


def _toy_ratings(seed=0, n_users=60, n_items=40):
    rng = np.random.default_rng(seed)
    taste = rng.normal(size=(n_users, 3)) @ rng.normal(size=(3, n_items))
    R = np.clip(np.round(3 + taste), 1, 5)
    R[rng.random(R.shape) < 0.5] = 0
    return sparse.csr_matrix(R.astype(np.float32))


def test_svd_fold_in_matches_fitted_users():
    R = _toy_ratings()
    eng = CollaborativeEngine(n_components=5).fit(R)
    np.testing.assert_allclose(eng.fold_in(R[:5]), eng.predict_raw(np.arange(5)), atol=1e-4)


def test_svd_fold_in_uniform_ratings_still_informative():
    """A new user who rates everything 5 must not collapse to a constant score vector."""
    R = _toy_ratings()
    eng = CollaborativeEngine(n_components=5).fit(R)
    new = sparse.csr_matrix((np.full(3, 5.0, np.float32), ([0, 0, 0], [0, 1, 2])), shape=(1, R.shape[1]))
    assert np.ptp(eng.score_new(new)) > 0


def test_content_scores_prefer_similar_items():
    meta = sparse.csr_matrix(np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32))
    plot = np.array([[1, 0], [0.9, 0.1], [0, 1], [0, 0]], dtype=np.float32)
    plot /= np.maximum(np.linalg.norm(plot, axis=1, keepdims=True), 1e-9)
    eng = ContentEngine(meta, plot, has_plot=np.array([1, 1, 1, 0], bool))
    R = sparse.csr_matrix(np.array([[5, 0, 0, 0]], dtype=np.float32))
    s = eng.score(R)[0]
    assert s[1] > s[2] and s[1] > s[3]
    assert s.min() >= 0 and s.max() <= 1


def _ratings_df():
    rng = np.random.default_rng(1)
    rows = [(u, m, int(rng.integers(1, 6)), int(t)) for u in range(1, 11) for t, m in enumerate(rng.permutation(50)[:25])]
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


@pytest.mark.parametrize("split", [lambda r: random_split(r, 0.2, 0), lambda r: temporal_split(r, 0.2)])
def test_splits_are_disjoint_and_per_user(split):
    r = _ratings_df()
    train, test = split(r)
    key = ["userId", "movieId"]
    assert len(train.merge(test, on=key)) == 0
    assert len(train) + len(test) == len(r)
    assert (test.groupby("userId").size() == 5).all()


def test_temporal_split_holds_out_latest():
    train, test = temporal_split(_ratings_df(), 0.2)
    last_train = train.groupby("userId")["timestamp"].max()
    first_test = test.groupby("userId")["timestamp"].min()
    assert (last_train < first_test).all()


def test_truncate_history_keeps_earliest_n():
    r = _ratings_df()
    out = truncate_history(r, users=[1, 2], n=3)
    assert (out.groupby("userId").size().loc[[1, 2]] == 3).all()
    assert (out.groupby("userId").size().drop([1, 2]) == 25).all()
    assert out[out.userId == 1]["timestamp"].max() == 2


def test_ials_fold_in_matches_fitted_users():
    """The exact ALS user step must reproduce the trained user factors' rankings."""
    from recsys.models.ials import IALSEngine
    R = _toy_ratings()
    eng = IALSEngine(factors=4, regularization=1.0, iterations=20).fit(R)
    fitted = eng.user_factors[:5] @ eng.item_factors.T
    folded = eng.fold_in(R[:5]) @ eng.item_factors.T
    for a, b in zip(fitted, folded):
        assert np.corrcoef(a, b)[0, 1] > 0.99


def test_ials_scores_new_user_without_history_as_zero():
    from recsys.models.ials import IALSEngine
    R = _toy_ratings()
    eng = IALSEngine(factors=4, iterations=5).fit(R)
    empty = sparse.csr_matrix((1, R.shape[1]), dtype=np.float32)
    assert np.all(eng.score_new(empty) == 0)


def test_content_popularity_gate_prefers_popular_among_equals():
    meta = sparse.csr_matrix(np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32))
    plot = np.zeros((4, 2), dtype=np.float32)
    counts = np.array([10, 1, 500, 500])
    eng = ContentEngine(meta, plot, has_plot=np.zeros(4, bool), item_counts=counts)
    s = eng.score(sparse.csr_matrix(np.array([[5, 0, 0, 0]], dtype=np.float32)))[0]
    assert s[2] > s[1]          # same similarity, more popular wins
    assert s[2] > s[3]          # popular but dissimilar loses
    assert s.min() >= 0 and s.max() <= 1


@pytest.mark.parametrize("n, expected", [(0, 1.0), (3, 0.301), (8, 0.041)])
def test_beta_fades_popularity_prior(n, expected):
    assert DynamicHybridRecommender(0.8, 0.15, pop_eta=0.4).calculate_beta(n) == pytest.approx(expected, abs=0.001)


def test_blend_with_fading_popularity():
    h = DynamicHybridRecommender(0.8, 0.15, pop_eta=0.2)
    content = np.full((3, 3), 0.5, dtype=np.float32)
    collab = np.ones((3, 3), dtype=np.float32)
    pop = np.array([1.0, 0.0, 0.3], dtype=np.float32)
    out = h.blend(np.array([0, 3, 200]), content, collab, pop)
    assert out[0].tolist() == pytest.approx(pop.tolist())                 # N=0: pure popularity
    hyb3 = h.calculate_alpha(3) * 1 + (1 - h.calculate_alpha(3)) * 0.5
    b3 = np.exp(-0.6)
    assert out[1] == pytest.approx(b3 * pop + (1 - b3) * hyb3, abs=1e-6)   # N=3: mixture
    assert out[2] == pytest.approx(np.full(3, 0.8 + 0.2 * 0.5), abs=1e-6)   # large N: popularity gone


def test_explain_reports_final_scores():
    h = DynamicHybridRecommender(0.8, 0.15, pop_eta=0.2)
    content, collab = np.array([0.2, 0.9], np.float32), np.array([0.7, 0.1], np.float32)
    pop = np.array([0.5, 1.0], np.float32)
    e = h.explain(4, content, collab, np.array([0, 1]), pop)
    expected = h.blend(np.array([4]), content[None], collab[None], pop)[0]
    assert e["s_hybrid"].to_numpy() == pytest.approx(expected, abs=1e-6)


def test_center_embeddings_removes_shared_direction():
    from recsys.features.plot_embed import center_embeddings
    rng = np.random.default_rng(0)
    shared = np.ones(8, dtype=np.float32) * 3          # every raw vector points mostly the same way
    E = shared + rng.normal(0, 0.3, (50, 8)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    has = np.ones(50, bool)
    has[-1] = False
    C = center_embeddings(E, has)
    assert np.allclose(np.linalg.norm(C[:-1], axis=1), 1.0, atol=1e-5)
    assert np.all(C[-1] == 0)                          # no plot -> stays zero
    raw_cos = (E[:-1] @ E[:-1].T)[np.triu_indices(49, 1)].mean()
    cen_cos = (C[:-1] @ C[:-1].T)[np.triu_indices(49, 1)].mean()
    assert raw_cos > 0.9 and abs(cen_cos) < 0.1        # compressed cone -> spread out


def test_content_engine_center_flag_only_changes_plot_channel():
    meta = sparse.csr_matrix(np.eye(4, dtype=np.float32))
    plot = np.array([[1, 0.1], [1, 0.2], [1, -0.1], [0, 0]], dtype=np.float32)
    plot /= np.maximum(np.linalg.norm(plot, axis=1, keepdims=True), 1e-9)
    has = np.array([1, 1, 1, 0], bool)
    raw = ContentEngine(meta, plot, has, center_plots=False)
    cen = ContentEngine(meta, plot, has, center_plots=True)
    R = sparse.csr_matrix(np.array([[5, 0, 0, 0]], dtype=np.float32))
    assert np.allclose(raw.channel_scores(R)[0], cen.channel_scores(R)[0])       # metadata identical
    assert not np.allclose(np.nan_to_num(raw.channel_scores(R)[1]), np.nan_to_num(cen.channel_scores(R)[1]))


def _toy_content_engine(**kw):
    meta = sparse.csr_matrix(np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32))
    plot = np.array([[1, 0], [0.9, 0.3], [0, 1], [0, 0]], dtype=np.float32)
    plot /= np.maximum(np.linalg.norm(plot, axis=1, keepdims=True), 1e-9)
    has = np.array([1, 1, 1, 0], bool)
    return ContentEngine(meta, plot, has, 0.5, 0.5, **kw), meta, plot, has


@pytest.mark.parametrize("center", [False, True])
def test_score_items_on_own_catalogue_matches_ungated_score(center):
    eng, meta, plot, has = _toy_content_engine(center_plots=center)
    R = sparse.csr_matrix(np.array([[5, 0, 0, 0], [0, 4, 5, 0]], dtype=np.float32))
    np.testing.assert_allclose(eng.score_items(R, meta, plot, has), eng.score(R), atol=1e-6)


def test_score_items_ignores_popularity_gate_for_new_items():
    eng, meta, plot, has = _toy_content_engine(item_counts=np.array([10, 10, 10, 10]))
    R = sparse.csr_matrix(np.array([[5, 0, 0, 0]], dtype=np.float32))
    new_meta = sparse.csr_matrix(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    new_plot = np.array([[1, 0.05], [0, 1]], dtype=np.float32)
    s = eng.score_items(R, new_meta, new_plot, np.array([1, 1], bool))[0]
    assert s[0] > s[1] and s.max() == pytest.approx(1.0)   # zero-count items still ranked by taste


def test_imputation_stops_plotless_items_winning_by_default():
    eng, meta, plot, has = _toy_content_engine()
    R = sparse.csr_matrix(np.array([[5, 0, 0, 0]], dtype=np.float32))
    # candidates: same genre as the liked movie; one has a matching plot, one has no plot at all
    new_meta = sparse.csr_matrix(np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32))
    new_plot = np.array([[0.95, 0.1], [0, 0], [0, 1]], dtype=np.float32)
    new_has = np.array([1, 0, 1], bool)
    plain = eng.score_items(R, new_meta, new_plot, new_has)[0]
    imputed = eng.score_items(R, new_meta, new_plot, new_has, impute_missing_plot=True)[0]
    assert plain[1] >= plain[0]          # metadata-only fallback lets the plot-less movie tie or win
    assert imputed[0] > imputed[1]       # with a neutral plot score the plot match wins
