"""Per-user train/test splits."""
import numpy as np
import pandas as pd


def random_split(ratings: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    """Hold out `test_frac` of each user's ratings uniformly at random."""
    rng = np.random.default_rng(seed)
    r = ratings.assign(_u=rng.random(len(ratings)))
    r["_rank"] = r.groupby("userId")["_u"].rank(method="first", pct=True)
    test = r["_rank"] > 1 - test_frac
    return ratings[~test.to_numpy()].copy(), ratings[test.to_numpy()].copy()


def temporal_split(ratings: pd.DataFrame, test_frac: float = 0.2):
    """Hold out each user's most recent `test_frac` ratings."""
    rank = ratings.groupby("userId")["timestamp"].rank(method="first", pct=True)
    test = rank > 1 - test_frac
    return ratings[~test.to_numpy()].copy(), ratings[test.to_numpy()].copy()


def truncate_history(train: pd.DataFrame, users, n: int, seed: int = 42) -> pd.DataFrame:
    """Keep only each listed user's earliest `n` training ratings (simulated cold start)."""
    mask = train["userId"].isin(users)
    rank = train.groupby("userId")["timestamp"].rank(method="first")
    return train[~mask | (rank <= n)]
