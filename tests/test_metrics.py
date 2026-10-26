import math

import pytest

from hybridrec.eval.metrics import candidate_recall, evaluate_topk, user_metrics


def test_user_metrics_hand_worked():
    # Hits at positions 1 and 3 out of 3 relevant items, k = 5.
    p, r, rt, ndcg = user_metrics([10, 99, 20, 98, 97], {10, 20, 30}, k=5)
    assert p == pytest.approx(2 / 5)
    assert r == pytest.approx(2 / 3)
    assert rt == pytest.approx(2 / 3)
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg == pytest.approx(dcg / idcg)


def test_truncated_recall_caps_denominator_at_k():
    relevant = set(range(30))
    p, r, rt, _ = user_metrics(list(range(10)), relevant, k=10)
    assert p == 1.0
    assert r == pytest.approx(10 / 30)
    assert rt == 1.0


def test_perfect_and_empty_rankings():
    assert user_metrics([1, 2], {1, 2}, k=10)[3] == pytest.approx(1.0)
    assert user_metrics([], {1}, k=10) == (0.0, 0.0, 0.0, 0.0)


def test_evaluate_topk_counts_missing_users_as_zero():
    truth = {1: {5}, 2: {6}}
    out = evaluate_topk({1: [5]}, truth, k=10)
    assert out["recall@10"] == pytest.approx(0.5)
    assert out["n_users"] == 2


def test_candidate_recall():
    assert candidate_recall({1: [1, 2, 3], 2: [9]}, {1: {1, 4}, 2: {9}}) == pytest.approx((0.5 + 1.0) / 2)
