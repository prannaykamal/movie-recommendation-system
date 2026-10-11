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
