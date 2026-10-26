"""Content-based source: TF-IDF over metadata tokens.

Each movie becomes a bag of typed tokens (genres, keywords, director x3, top-3
cast, collection, language, decade). A user's profile is the sum of their
movies' vectors weighted by (rating - the user's mean rating), so disliked
movies push recommendations away. The plot overview is deliberately left out;
the embedding source uses it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from hybridrec.data.clean import name_token
from hybridrec.data.dataset import Histories

DIRECTOR_REPEAT = 3
TOP_CAST = 3


def movie_tokens(row) -> str:
    toks = [f"g_{name_token(g)}" for g in row.genres]
    toks += [f"k_{name_token(k)}" for k in row.keywords]
    toks += [f"d_{name_token(d)}" for d in row.directors] * DIRECTOR_REPEAT
    toks += [f"c_{name_token(c)}" for c in list(row.cast)[:TOP_CAST]]
    if isinstance(row.collection_name, str):
        toks.append(f"col_{name_token(row.collection_name)}")
    if isinstance(row.original_language, str):
        toks.append(f"lang_{row.original_language}")
    if pd.notna(row.year):
        toks.append(f"dec_{int(row.year) // 10 * 10}")
    return " ".join(toks)


def build_item_matrix(movies: pd.DataFrame) -> tuple[sp.csr_matrix, list[str]]:
    """(N, V) TF-IDF matrix with L2-normalized rows, aligned with ``movies``."""
    docs = [movie_tokens(r) for r in movies.itertuples(index=False)]
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=2, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(docs)
    return normalize(X).tocsr().astype(np.float32), vec.get_feature_names_out().tolist()


class ContentTfidf:
    name = "cb"

    def __init__(self, item_matrix: sp.csr_matrix):
        self.X = item_matrix
        self.XT = item_matrix.T.tocsr()

    def fit(self, hist: Histories) -> "ContentTfidf":
        return self  # stateless: the item matrix doesn't depend on ratings

    def profiles(self, hist: Histories, rows: np.ndarray) -> sp.csr_matrix:
        data, cols, indptr = [], [], [0]
        for p in rows:
            items, ratings, _ = hist.get(p) if p >= 0 else (np.array([], int), np.array([]), None)
            w = ratings - ratings.mean() if len(ratings) else ratings
            if len(w) and not np.any(w):
                w = np.ones_like(ratings)  # all ratings equal: weight movies equally
            data.append(w.astype(np.float32))
            cols.append(items)
            indptr.append(indptr[-1] + len(items))
        W = sp.csr_matrix(
            (np.concatenate(data) if data else [], np.concatenate(cols) if cols else [], indptr),
            shape=(len(rows), self.X.shape[0]),
        )
        return normalize(W @ self.X)

    def score(self, hist: Histories, rows: np.ndarray) -> np.ndarray:
        P = self.profiles(hist, rows)
        return (P @ self.XT).toarray().astype(np.float32)

    @staticmethod
    def save(path, X: sp.csr_matrix) -> None:
        sp.save_npz(path, X)

    @classmethod
    def load(cls, path) -> "ContentTfidf":
        return cls(sp.load_npz(path).tocsr())
