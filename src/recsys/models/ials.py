"""Implicit ALS (Hu, Koren & Volinsky 2008) via the `implicit` library.

Every rating counts as an interaction, with confidence c = 1 + conf * (r - 1) / 4, so a 1-star
rating is weak evidence and a 5-star rating strong evidence. Scores are u . v dot products,
min-max scaled per user to [0, 1] like the other engines."""
import numpy as np
from implicit.cpu.als import AlternatingLeastSquares
from scipy import sparse
from threadpoolctl import threadpool_limits

from recsys.data.interactions import minmax_rows


class IALSEngine:
    def __init__(self, factors: int = 32, regularization: float = 3.0, confidence: float = 3.0,
                 iterations: int = 30, random_state: int = 42):
        self.factors = factors
        self.regularization = regularization
        self.confidence = confidence
        self.iterations = iterations
        self.random_state = random_state

    def confidence_matrix(self, R: sparse.csr_matrix) -> sparse.csr_matrix:
        C = R.tocsr().astype(np.float32, copy=True)
        C.data = 1.0 + self.confidence * (C.data - 1.0) / 4.0
        return C

    def fit(self, R: sparse.csr_matrix) -> "IALSEngine":
        with threadpool_limits(1, "blas"):  # implicit parallelises itself; nested BLAS threads thrash
            model = AlternatingLeastSquares(factors=self.factors, regularization=self.regularization,
                                            iterations=self.iterations, random_state=self.random_state,
                                            calculate_training_loss=False)
            model.fit(self.confidence_matrix(R), show_progress=False)
        self.user_factors = np.asarray(model.user_factors, dtype=np.float32)
        self.item_factors = np.asarray(model.item_factors, dtype=np.float32)
        self._gram = self.item_factors.T @ self.item_factors
        return self

    def fold_in(self, R_new: sparse.csr_matrix) -> np.ndarray:
        """Exact ALS user step for users outside the training matrix:
        (Y'Y + Y_u' (C_u - I) Y_u + reg I) x_u = Y_u' c_u."""
        C = self.confidence_matrix(R_new)
        Y = self.item_factors
        X = np.zeros((C.shape[0], Y.shape[1]), dtype=np.float32)
        eye = self.regularization * np.eye(Y.shape[1], dtype=np.float32)
        for u in range(C.shape[0]):
            s, e = C.indptr[u], C.indptr[u + 1]
            if s == e:
                continue
            Yu, cu = Y[C.indices[s:e]], C.data[s:e]
            A = self._gram + (Yu.T * (cu - 1.0)) @ Yu + eye
            X[u] = np.linalg.solve(A, Yu.T @ cu)
        return X

    def score(self, uidx: np.ndarray) -> np.ndarray:
        return minmax_rows(self.user_factors[uidx] @ self.item_factors.T)

    def score_new(self, R_new: sparse.csr_matrix) -> np.ndarray:
        return minmax_rows(self.fold_in(R_new) @ self.item_factors.T)
