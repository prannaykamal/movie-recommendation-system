"""Content-based engine: user profiles over V_meta (TF-IDF) and V_plot (SBERT)."""
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from recsys.data.interactions import minmax_rows


class ContentEngine:
    def __init__(self, meta_vecs: sparse.csr_matrix, plot_vecs: np.ndarray, has_plot: np.ndarray,
                 meta_weight: float = 0.35, plot_weight: float = 0.65, pos_threshold: float = 4.0,
                 item_counts: np.ndarray | None = None):
        # Both channels are row L2-normalised, so cosine(profile, item) = item . profile_unit.
        self.meta_vecs = normalize(meta_vecs.tocsr())
        self.plot_vecs = plot_vecs.astype(np.float32)
        self.has_plot = np.asarray(has_plot, dtype=bool)
        self.meta_weight = meta_weight
        self.plot_weight = plot_weight
        self.pos_threshold = pos_threshold
        # Popularity gate: similarity alone ranks obscure near-duplicates first; multiplying by
        # log(1 + interactions) favours well-known matches (NDCG@10 0.025 -> 0.152 content-only).
        self.pop_prior = None if item_counts is None else np.log1p(np.asarray(item_counts, dtype=np.float32))

    def profile_weights(self, R: sparse.csr_matrix) -> sparse.csr_matrix:
        """Rating weights for the profile: ratings >= threshold, or all ratings if a user has none."""
        R = R.tocsr()
        pos = R.multiply(R >= self.pos_threshold).tocsr()
        pos.eliminate_zeros()
        no_pos = np.diff(pos.indptr) == 0
        if no_pos.any():
            pos = pos + sparse.diags(no_pos.astype(np.float32)) @ R
        return pos.tocsr()

    def channel_scores(self, R: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        """Raw cosine scores per channel for each row of R (users x items). Plot scores are
        NaN for items without a plot and for users whose profile has no plot items."""
        W = self.profile_weights(R)
        # Profile = rating-weighted mean of item vectors; the 1/sum(r) factor cancels in cosine.
        u_meta = normalize(W @ self.meta_vecs)
        meta = np.asarray((u_meta @ self.meta_vecs.T).todense(), dtype=np.float32)

        u_plot = W @ self.plot_vecs
        norms = np.linalg.norm(u_plot, axis=1, keepdims=True)
        u_plot = np.divide(u_plot, norms, out=np.zeros_like(u_plot), where=norms > 0)
        plot = (u_plot @ self.plot_vecs.T).astype(np.float32)
        plot[:, ~self.has_plot] = np.nan
        plot[norms.ravel() == 0, :] = np.nan
        return meta, plot

    def score(self, R: sparse.csr_matrix) -> np.ndarray:
        """S_content in [0, 1] for each row of R, popularity-gated when item_counts was given.
        Each channel is min-max scaled per user before the weighted blend so TF-IDF [0,1] and SBERT [-1,1] cosines are comparable; items with
        no plot fall back to the metadata channel alone (weights renormalised)."""
        meta, plot = self.channel_scores(R)
        meta = minmax_rows(meta)
        plot_mask = ~np.isnan(plot)
        plot_filled = np.where(plot_mask, plot, np.nanmin(np.where(plot_mask, plot, np.inf), axis=1, keepdims=True))
        plot_filled = np.where(np.isfinite(plot_filled), plot_filled, 0.0)
        plot_scaled = minmax_rows(plot_filled)
        blended = (self.meta_weight * meta + self.plot_weight * plot_scaled) / (self.meta_weight + self.plot_weight)
        out = np.where(plot_mask, blended, meta)
        empty = np.diff(R.tocsr().indptr) == 0
        out[empty] = 0.0
        out = minmax_rows(out.astype(np.float32))
        return out if self.pop_prior is None else minmax_rows(out * self.pop_prior)
