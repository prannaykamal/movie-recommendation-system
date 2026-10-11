import numpy as np
import pytest

from recsys.evaluation.metrics import ranking_metrics


def test_hand_computed_ranking():
    # relevant = {1, 3}; ranking hits at positions 2 and 4
    m = ranking_metrics(np.array([[0, 1, 2, 3, 4]]), [{1, 3}], k=4)
    assert m["precision@4"] == pytest.approx(0.5)
    assert m["recall@4"] == pytest.approx(1.0)
    dcg = 1 / np.log2(3) + 1 / np.log2(5)
    idcg = 1 + 1 / np.log2(3)
    assert m["ndcg@4"] == pytest.approx(dcg / idcg)
    assert m["map@4"] == pytest.approx((1 / 2 + 2 / 4) / 2)
    assert m["mrr@4"] == pytest.approx(0.5)


def test_perfect_and_empty_rankings_average():
    m = ranking_metrics(np.array([[7, 8], [1, 2]]), [{7, 8}, {9}], k=2)
    assert m["precision@2"] == pytest.approx(0.5)
    assert m["ndcg@2"] == pytest.approx(0.5)
    assert m["mrr@2"] == pytest.approx(0.5)


def test_per_user_metrics_average_to_ranking_metrics():
    from recsys.evaluation.metrics import per_user_metrics
    recs = np.array([[0, 1, 2], [3, 4, 5], [1, 0, 9]])
    rel = [{1}, {9}, {1, 9}]
    pu = per_user_metrics(recs, rel, 3)
    agg = ranking_metrics(recs, rel, 3)
    for m in agg:
        assert pu[m].mean() == pytest.approx(agg[m])
    assert pu["ndcg@3"][1] == 0.0


def test_paired_bootstrap_detects_consistent_gain_only():
    from recsys.evaluation.metrics import paired_bootstrap
    rng = np.random.default_rng(1)
    base = rng.random(2000)
    better = paired_bootstrap(base + 0.05, base)
    assert better["ci_low"] > 0 and better["diff"] == pytest.approx(0.05)
    noise = paired_bootstrap(base + rng.normal(0, 0.1, 2000), base)
    assert noise["ci_low"] < 0 < noise["ci_high"]


def test_beyond_accuracy_extremes():
    from recsys.evaluation.beyond import beyond_accuracy, gini
    counts = np.array([100, 50, 10, 5, 1, 1, 1, 1, 1, 1], dtype=float)
    genres = np.eye(10, dtype=np.float32)              # every item its own genre
    same = np.tile(np.arange(3), (20, 1))              # everyone gets the 3 most popular
    spread = np.array([[i % 10, (i + 1) % 10, (i + 2) % 10] for i in range(20)])
    a = beyond_accuracy(same, counts, 100, genres, k=3, short_head_frac=0.2)
    b = beyond_accuracy(spread, counts, 100, genres, k=3, short_head_frac=0.2)
    assert a["coverage@3"] == pytest.approx(0.3) and b["coverage@3"] == pytest.approx(1.0)
    assert a["personal@3"] == pytest.approx(0.0) and b["personal@3"] > 0.5
    assert a["gini@3"] > b["gini@3"]
    assert a["long_tail@3"] == pytest.approx(1 / 3)    # items 0,1 are the head (top 20% of 10)
    assert a["novelty@3"] < b["novelty@3"]
    assert a["ild@3"] == pytest.approx(1.0)            # distinct genres -> fully diverse lists
    assert gini(np.ones(5)) == pytest.approx(0.0)
    assert gini(np.array([0, 0, 0, 0, 1.0])) == pytest.approx(0.8)
