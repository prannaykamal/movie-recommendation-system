import numpy as np
import pandas as pd
import pytest

from hybridrec.data.dataset import Histories, Items
from hybridrec.rerank.rrf import rrf_scores
from hybridrec.retrieval.merge import add_labels, merge_candidates, topn_by_score


class RandomScorer:
    def __init__(self, n_items, seed, nan_items=()):
        self.n_items, self.seed, self.nan_items = n_items, seed, list(nan_items)

    def score(self, hist, rows):
        S = np.random.default_rng(self.seed).random((len(rows), self.n_items)).astype(np.float32)
        S[:, self.nan_items] = np.nan
        return S


@pytest.fixture
def setup():
    items = Items(np.arange(100, 160))
    rng = np.random.default_rng(1)
    rows = [(u, int(m), float(rng.integers(1, 6)), t) for u in (1, 2, 3)
            for t, m in enumerate(rng.choice(items.movie_ids, 15, replace=False))]
    hist = Histories(pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "ts"]), items)
    scorers = {"cb": RandomScorer(60, 1), "cf": RandomScorer(60, 2, nan_items=range(10)), "emb": RandomScorer(60, 3)}
    cands = merge_candidates(scorers, hist, hist.users, {"cb": 10, "cf": 8, "emb": 12})
    return hist, cands


def test_no_duplicates_and_no_watched(setup):
    hist, cands = setup
    assert not cands.duplicated(["user_id", "movie_id"]).any()
    for p, u in enumerate(hist.users):
        watched = set(hist.items.movie_ids[hist.get(p)[0]])
        assert not watched & set(cands.loc[cands.user_id == u, "movie_id"])


def test_ranks_and_sources(setup):
    _, cands = setup
    for src, k in (("cb", 10), ("cf", 8), ("emb", 12)):
        per_user = (cands[f"{src}_rank"] > 0).groupby(cands.user_id).sum()
        assert (per_user == k).all()
        r = cands.loc[cands[f"{src}_rank"] > 0, f"{src}_rank"]
        assert r.between(1, k).all()
    assert cands["n_sources"].between(1, 3).all()
    # Every candidate has all three scores; iALS-missing items score NaN.
    assert cands["cf_score"].isna().any() and cands[["cb_score", "emb_score"]].notna().all().all()


def test_labels_and_topn(setup):
    _, cands = setup
    target = cands.groupby("user_id").head(2)[["user_id", "movie_id"]].assign(rating=[5.0, 2.0] * 3, ts=0)
    labelled = add_labels(cands, target, 4.0)
    assert labelled["label"].sum() == 3 and labelled["rated"].sum() == 6
    scores = rrf_scores(cands)
    top = topn_by_score(cands, scores, n=5)
    by_movie = pd.Series(scores, index=pd.MultiIndex.from_frame(cands[["user_id", "movie_id"]]))
    for u, mids in top.items():
        chosen = [by_movie[(u, m)] for m in mids]
        assert chosen == sorted(chosen, reverse=True)  # best first
        assert sorted(chosen) == sorted(by_movie[u].nlargest(5).tolist())  # the top 5 (ties allowed)
