import numpy as np
import pandas as pd
import pytest

from hybridrec.data.split import assign_splits, stage_frames


@pytest.fixture
def ratings():
    rng = np.random.default_rng(0)
    rows = []
    for user, n in [(1, 20), (2, 21), (3, 57)]:
        ts = rng.integers(0, 1_000, size=n)
        rows += [(user, 1000 + i, 4.0, int(t)) for i, t in enumerate(ts)]
    return assign_splits(pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "ts"]))


def test_blocks_are_in_time_order_per_user(ratings):
    order = {"R": 0, "L": 1, "V": 2, "T": 3}
    for _, g in ratings.groupby("user_id"):
        g = g.sort_values("pos")
        assert (np.diff(g["split"].map(order).to_numpy()) >= 0).all()
        assert (np.diff(g["ts"].to_numpy()) >= 0).all()


def test_block_sizes(ratings):
    sizes = ratings.groupby(["user_id", "split"]).size().unstack()
    assert sizes.loc[1].to_dict() == {"L": 2, "R": 14, "T": 2, "V": 2}
    assert (sizes.min(axis=1) >= 2).all()
    assert (sizes["R"] / sizes.sum(axis=1)).between(0.65, 0.75).all()


@pytest.mark.parametrize("stage", ["s0", "s1", "s2", "s3"])
def test_stage_fit_and_target_do_not_overlap_and_target_is_later(ratings, stage):
    fit, target = stage_frames(ratings, stage)
    pairs = lambda df: set(zip(df.user_id, df.movie_id))
    assert not pairs(fit) & pairs(target)
    last_fit = fit.groupby("user_id")["pos"].max()
    first_target = target.groupby("user_id")["pos"].min()
    assert (first_target > last_fit.loc[first_target.index]).all()


def test_stages_grow_the_fit_window(ratings):
    sizes = [len(stage_frames(ratings, s)[0]) for s in ["s0", "s1", "s2", "s3"]]
    assert sizes == sorted(sizes) and len(set(sizes)) == 4
