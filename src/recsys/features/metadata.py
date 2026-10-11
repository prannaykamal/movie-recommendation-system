"""Sparse TF-IDF metadata vectors V_meta(i)."""
import re

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def _tok(prefix: str, value: str) -> str:
    """Multi-word entity -> single token, namespaced so 'drama' genre != 'drama' elsewhere."""
    return prefix + re.sub(r"\W+", "_", str(value).lower()).strip("_")


def metadata_tokens(row) -> str:
    toks = [_tok("g_", g) for g in str(row["genres"]).split("|") if g]
    toks += [_tok("cg_", g) for g in row["cmu_genres"]]
    toks += [_tok("c_", c) for c in row["countries"]]
    toks += [_tok("a_", a) for a in row["cast"]]
    if pd.notna(row["year"]):
        toks.append(f"d_{int(row['year']) // 10 * 10}s")
    return " ".join(toks)


def build_metadata_matrix(items: pd.DataFrame) -> tuple[sparse.csr_matrix, TfidfVectorizer]:
    corpus = items.apply(metadata_tokens, axis=1)
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True)
    X = normalize(vec.fit_transform(corpus).astype("float32"))
    return X.tocsr(), vec
