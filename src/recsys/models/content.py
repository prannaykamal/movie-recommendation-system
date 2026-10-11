"""Content-based engine: user profiles over V_meta (TF-IDF) and V_plot (SBERT)."""
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from recsys.data.interactions import minmax_rows
from recsys.features.plot_embed import center_embeddings, plot_mean


class ContentEngine:
    def __init__(self, meta_vecs: sparse.csr_matrix, plot_vecs: np.ndarray, has_plot: np.ndarray,
                 meta_weight: float = 0.35, plot_weight: float = 0.65, pos_threshold: float = 4.0,
                 item_counts: np.ndarray | None = None, center_plots: bool = False):
        # Both channels are row L2-normalised, so cosine(profile, item) = item . profile_unit.
        self.meta_vecs = normalize(meta_vecs.tocsr())
        self.has_plot = np.asarray(has_plot, dtype=bool)
        self.plot_mean = plot_mean(plot_vecs, self.has_plot) if center_plots else None
        self.plot_vecs = (center_embeddings(plot_vecs, self.has_plot, self.plot_mean) if center_plots
                          else plot_vecs.astype(np.float32))
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

    def _profiles(self, R: sparse.csr_matrix):
        """Unit profile vectors per user. The rating-weighted mean's 1/sum(r) factor cancels in cosine."""
        W = self.profile_weights(R)
        u_meta = normalize(W @ self.meta_vecs)
        u_plot = W @ self.plot_vecs
        norms = np.linalg.norm(u_plot, axis=1, keepdims=True)
        u_plot = np.divide(u_plot, norms, out=np.zeros_like(u_plot), where=norms > 0)
        return u_meta, u_plot, norms.ravel() > 0

    @staticmethod
    def _channels(profiles, meta_items, plot_items, has_plot_items) -> tuple[np.ndarray, np.ndarray]:
        u_meta, u_plot, has_profile_plot = profiles
        meta = np.asarray((u_meta @ meta_items.T).todense(), dtype=np.float32)
        plot = (u_plot @ plot_items.T).astype(np.float32)
        plot[:, ~has_plot_items] = np.nan
        plot[~has_profile_plot, :] = np.nan
        return meta, plot

    def channel_scores(self, R: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        """Raw cosine scores per channel for each row of R (users x items). Plot scores are
        NaN for items without a plot and for users whose profile has no plot items."""
        return self._channels(self._profiles(R), self.meta_vecs, self.plot_vecs, self.has_plot)

    def _blend(self, meta: np.ndarray, plot: np.ndarray, R: sparse.csr_matrix,
               impute_missing_plot: bool = False) -> np.ndarray:
        """Each channel is min-max scaled per user before the weighted blend so TF-IDF [0,1] and
        SBERT [-1,1] cosines are comparable. Items with no plot fall back to the metadata channel
        alone (weights renormalised) or, with `impute_missing_plot`, are blended with the user's
        median plot score so they do not outrank plot-bearing items just by skipping a channel.
        Users with no ratings score 0 everywhere."""
        meta = minmax_rows(meta)
        plot_mask = ~np.isnan(plot)
        plot_filled = np.where(plot_mask, plot, np.nanmin(np.where(plot_mask, plot, np.inf), axis=1, keepdims=True))
        plot_filled = np.where(np.isfinite(plot_filled), plot_filled, 0.0)
        plot_scaled = minmax_rows(plot_filled)
        blended = (self.meta_weight * meta + self.plot_weight * plot_scaled) / (self.meta_weight + self.plot_weight)
        if impute_missing_plot:
            masked = np.where(plot_mask, plot_scaled, np.nan)
            med = np.nanmedian(np.where(plot_mask.any(axis=1, keepdims=True), masked, 0.0), axis=1, keepdims=True)
            imputed = (self.meta_weight * meta + self.plot_weight * med) / (self.meta_weight + self.plot_weight)
            has_profile_plot = plot_mask.any(axis=1, keepdims=True)
            out = np.where(plot_mask, blended, np.where(has_profile_plot, imputed, meta))
        else:
            out = np.where(plot_mask, blended, meta)
        out[np.diff(R.tocsr().indptr) == 0] = 0.0
        return minmax_rows(out.astype(np.float32))

    def score(self, R: sparse.csr_matrix) -> np.ndarray:
        """S_content in [0, 1] for each row of R over the catalogue, popularity-gated when
        item_counts was given."""
        out = self._blend(*self.channel_scores(R), R)
        return out if self.pop_prior is None else minmax_rows(out * self.pop_prior)

    def score_items(self, R: sparse.csr_matrix, meta_vecs: sparse.csr_matrix, plot_vecs: np.ndarray,
                    has_plot: np.ndarray, impute_missing_plot: bool = False) -> np.ndarray:
        """Content scores in [0, 1] of external items (e.g. movies with no ratings at all) against
        each user's profile, scaled over those items only and never popularity-gated: gating
        multiplies a zero-interaction item by log(1 + 0) = 0. Plot vectors are centred on the
        catalogue's mean, like the catalogue itself."""
        has = np.asarray(has_plot, dtype=bool)
        plots = center_embeddings(plot_vecs, has, self.plot_mean) if self.plot_mean is not None \
            else plot_vecs.astype(np.float32)
        meta, plot = self._channels(self._profiles(R), normalize(meta_vecs.tocsr()), plots, has)
        return self._blend(meta, plot, R, impute_missing_plot)
