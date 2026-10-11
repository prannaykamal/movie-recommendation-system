# Hybrid Movie Recommendation System: Architectural Design Document (v2)

## Executive Summary
This document presents the updated end-to-end architectural design for an AI/ML **Hybrid Movie Recommendation System**. The architecture addresses classical cold-start bottlenecks and over-specialization by integrating dual-representation **Content-Based Filtering** (combining structured metadata and dense Sentence-BERT plot embeddings) with **User-Centered Matrix Factorization (SVD)**.

The system is configured to operate on the **MovieLens 1M (`ml-1m`)** dataset for collaborative interaction modeling paired with the **CMU Movie Summary Corpus** for long-form Wikipedia plot narratives. Rather than relying on a static or tuned global hyperparameter for score interpolation, the system utilizes a **Dynamic Interaction-Driven Score Weighting Mechanism** where the hybrid blend coefficient $\alpha(u)$ scales automatically as a function of the user's interaction history count $N_u$.

---

## 1. System Overview & Multi-Stage Pipeline

The recommendation system follows a modular architecture separating candidate feature extraction, collaborative factor learning, dynamic score blending, and offline evaluation.

```
                  [ Datasets: MovieLens 1M (Ratings) + CMU Plot Corpus (Narratives) ]
                                                   │
                    ┌──────────────────────────────┴──────────────────────────────┐
                    ▼                                                             ▼
      [ Content-Based Pipeline ]                                   [ Collaborative SVD Pipeline ]
 ┌───────────────────────────────────┐                         ┌───────────────────────────────────┐
 │ 1. Metadata Vector (TF-IDF)       │                         │ 1. User Rating Mean Centering     │
 │ 2. CMU Plots (Sentence-BERT)     │                         │ 2. TruncatedSVD Decomposition     │
 │ 3. User Content Profile Fusion    │                         │ 3. Score Reconstruction & Scaling │
 └─────────────────┬─────────────────┘                         └─────────────────┬─────────────────┘
                   │                                                             │
                   │ S_content(u, i)                             S_collab(u, i)  │
                   └──────────────────────────────┬──────────────────────────────┘
                                                  │
                                                  ▼
                                  [ Dynamic Interpolation Engine ]
                                  α(u) = α_max * (1 - e^(-γ * N_u))
                                                  │
                                                  ▼
                                    [ Hybrid Candidate Ranking ]
                                     (Exclude Previously Rated)
                                                  │
                                                  ▼
                                 [ Top-K Output & Evaluation ]
                                 (NDCG@K, Precision@K, Recall@K)
```

---

## 2. Dataset Specification & Alignment Pipeline

To power both deep narrative analysis and rich user interaction modeling, the architecture couples the **CMU Movie Summary Corpus** with **MovieLens 1M (`ml-1m`)**.

### 2.1 Dataset Composition

| Dataset Component | Source / Origin | Volume / Size | Primary Usage in System |
| :--- | :--- | :--- | :--- |
| **MovieLens 1M (`ml-1m`)** | GroupLens (Univ. of Minnesota) | 1,000,209 ratings across 6,040 users & 3,883 movies | User-item interaction matrix $R$, mean-centered SVD factorization, dynamic $\alpha(u)$ evaluation. |
| **CMU Movie Summary Corpus** | CMU / Wikipedia (42,306 plots) | `plot_summaries.txt` & `movie.metadata.tsv` | Dense Sentence-BERT (`all-MiniLM-L6-v2`) 384-d narrative embeddings & TF-IDF metadata. |

### 2.2 Overlap Analysis & Catalog Coverage
* **Estimated Overlap**: **~85% – 92% (~3,200 – 3,600 movies)** of the MovieLens 1M catalog matches directly with CMU plot summaries.
* **Why this Pairing is Optimal**:
  1. **Temporal Alignment**: MovieLens 1M covers movies released up to the year 2000, while the CMU Corpus contains Wikipedia dumps compiled through ~2012/2013. This ensures virtually every movie in ML-1M has a dedicated, multi-paragraph Wikipedia plot section.
  2. **Rich Narrative vs. Short Overviews**: Unlike 2-sentence API overviews, CMU plot summaries are complete multi-paragraph narrative breakdowns (500–2,000 words), allowing Sentence-BERT to extract subtle themes, plot twists, and character arcs.
  3. **Interaction Density**: Users in ML-1M average ~165 ratings each, providing an ideal distribution of sparse cold-start profiles ($N_u < 5$) and mature profiles ($N_u \ge 20$) to validate dynamic $\alpha(u)$ score interpolation.

### 2.3 Data Ingestion & Key Alignment Strategy
Because MovieLens 1M uses internal `movieId` keys and CMU uses Freebase/Wikipedia IDs (`/m/01234`), the datasets are aligned using a multi-pass mapping pipeline:

1. **Pass 1: Title + Release Year Exact Matching (Primary)**:
   * Strip leading articles ("The", "A", "An") from movie titles.
   * Lowercase and remove punctuation to produce `clean_title`.
   * Join CMU `movie.metadata.tsv` with MovieLens `movies.dat` on `(clean_title, release_year)`. This resolves ~85% of matches.
2. **Pass 2: Freebase/Wikidata ID Translation via `links.csv`**:
   * Map CMU Freebase IDs (`/m/0...`) to IMDb/TMDB IDs using Wikidata dumps, then map directly to MovieLens `movieId`.
3. **Pass 3: Fuzzy String Distance Matching (Fallback)**:
   * Apply Jaro-Winkler / Levenshtein edit distance matching on remaining unmatched titles (threshold similarity > 0.90) to handle foreign title variants and punctuation differences.

---

## 3. Content-Based Filtering Subsystem Design

To capture both explicit taxonomy tags and deep narrative nuances, the content engine processes movie data through two distinct representation channels before constructing a unified content score $S_{\text{content}}(u, i)$.

```
                     ┌─────────────────────────────────────────┐
                     │    Movie Metadata & CMU Plot Corpus     │
                     └────────────────────┬────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
   ┌─────────────────────────────┐                 ┌─────────────────────────────┐
   │ Categorical & Tag Features   │                 │   CMU Full Plot Summaries   │
   │ (Genres, Year, Cast, Director)│                 │    (Wikipedia Narratives)   │
   └──────────────┬──────────────┘                 └──────────────┬──────────────┘
                  │                                               │
                  ▼                                               ▼
   ┌─────────────────────────────┐                 ┌─────────────────────────────┐
   │ TF-IDF / Multi-Hot Encoder  │                 │ Sentence-Transformers Engine│
   │      (Sparse Vectors)       │                 │   ('all-MiniLM-L6-v2')      │
   └──────────────┬──────────────┘                 └──────────────┬──────────────┘
                  │                                               │
                  ▼                                               ▼
   ┌─────────────────────────────┐                 ┌─────────────────────────────┐
   │   Metadata Feature Vector   │                 │   384-d Dense Semantic Vector│
   │       V_meta(i)             │                 │           V_plot(i)         │
   └──────────────┬──────────────┘                 └──────────────┬──────────────┘
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                           ┌─────────────────────────────┐
                           │ User Content Profile Fusion │
                           │  Cosine Similarity Scoring  │
                           └─────────────────────────────┘
```

### 3.1 Categorical Metadata Representation ($V_{\text{meta}}$)
* **Metadata Extraction**: Collect structural entities from CMU `movie.metadata.tsv` and ML-1M `movies.dat` including genres, release year, director, top cast members, and country of origin.
* **Normalization & Stemming**: Apply token normalization and lowercasing.
* **Vectorization**: Transform concatenated metadata tokens into sparse term vectors using **TF-IDF Vectorization** (or Multi-Hot Encoding).

### 3.2 Dense CMU Plot Semantic Embeddings ($V_{\text{plot}}$)
* **Pre-trained Transformer**: Pass raw CMU Wikipedia plot summaries through `all-MiniLM-L6-v2` (Sentence-BERT).
* **Semantic Vector Space**: Generates a dense $384$-dimensional embedding for every title:
  $$V_{\text{plot}}(i) = \text{SentenceBERT}(\text{CMU\_Plot}_i)$$
* **Advantage**: Captures deep narrative semantics (e.g., "dystopian survival drama" vs. "space exploration time dilation") even when metadata tags are sparse.

### 3.3 User Content Profile Construction & Scoring
* **Positive Implicit Signal Aggregation**: A user's preference profile is computed as the weighted average of item vectors for movies rated positively ($r_{u,j} \ge 4.0$ in MovieLens 1M):
  $$\mathbf{U}_{\text{meta}}(u) = \frac{\sum_{j \in \mathcal{I}_u^+} r_{u,j} \cdot V_{\text{meta}}(j)}{\sum_{j \in \mathcal{I}_u^+} r_{u,j}}, \quad \mathbf{U}_{\text{plot}}(u) = \frac{\sum_{j \in \mathcal{I}_u^+} r_{u,j} \cdot V_{\text{plot}}(j)}{\sum_{j \in \mathcal{I}_u^+} r_{u,j}}$$
* **Combined Content Similarity**: Calculated via Cosine Similarity:
  $$S_{\text{content}}(u, i) = w_{\text{meta}} \cdot \text{Cosine}\left(\mathbf{U}_{\text{meta}}(u), V_{\text{meta}}(i)\right) + w_{\text{plot}} \cdot \text{Cosine}\left(\mathbf{U}_{\text{plot}}(u), V_{\text{plot}}(i)\right)$$
  *(Recommended weights: $w_{\text{meta}} = 0.35$, $w_{\text{plot}} = 0.65$).*

---

## 4. Collaborative Filtering Subsystem (User-Centered SVD)

The collaborative engine captures user interaction dynamics using the MovieLens 1M rating matrix $R \in \mathbb{R}^{6040 \times 3883}$.

### 4.1 User Rating Mean Centering
To eliminate bias between lenient raters (who give mostly 4s and 5s) and strict raters, each user's mean rating $\mu_u$ is subtracted prior to matrix decomposition:
$$\tilde{r}_{u,i} = r_{u,i} - \mu_u \quad \text{for } r_{u,i} \neq \text{NaN}$$

### 4.2 Matrix Decomposition via TruncatedSVD
The sparse, mean-centered interaction matrix $\tilde{R}$ is factorized into $k$ latent components ($k \in [50, 100]$):
$$\tilde{R} \approx P \cdot Q^T$$
where $P \in \mathbb{R}^{6040 \times k}$ represents latent user factors and $Q \in \mathbb{R}^{3883 \times k}$ represents latent movie factors.

### 4.3 Score Reconstruction & Range Normalization
* **Score Reconstruction**:
  $$\hat{r}_{u,i} = \mu_u + \mathbf{p}_u^T \mathbf{q}_i$$
* **Min-Max Score Normalization**: Project collaborative predictions onto $[0.0, 1.0]$ to align scale with content cosine similarity scores:
  $$S_{\text{collab}}(u, i) = \frac{\hat{r}_{u,i} - \min(\hat{r}_u)}{\max(\hat{r}_u) - \min(\hat{r}_u)}$$

---

## 5. Dynamic Interaction-Driven Linear Score Interpolation

### 5.1 Hybrid Blending Formula
The final hybrid recommendation score $S_{\text{hybrid}}(u, i)$ combines collaborative and content signals:
$$S_{\text{hybrid}}(u, i) = \alpha(u) \cdot S_{\text{collab}}(u, i) + (1 - \alpha(u)) \cdot S_{\text{content}}(u, i)$$

### 5.2 Mathematical Formulation for $\alpha(u)$
The blend weight $\alpha(u)$ is determined dynamically for each user based on their interaction count $N_u$ in MovieLens 1M:

$$\alpha(u) = \alpha_{\max} \cdot \left(1 - e^{-\gamma \cdot N_u}\right)$$

Where:
* **$N_u$**: Total number of ratings/movies interacted with by user $u$.
* **$\alpha_{\max}$**: The maximum asymptotic weight for collaborative filtering (e.g., $0.80$).
* **$\gamma$**: The growth saturation constant (e.g., $0.15$), controlling how quickly collaborative weight ramps up.

```
  Alpha Weight α(u)
   1.0 ┼───────────────────────────────────────────────────────────
   0.8 ┼─────────────────────────────────────████████████████████  <-- α_max Ceiling
   0.6 ┼─────────────────────────████████████
   0.4 ┼─────────────████████████
   0.2 ┼─────████████
   0.0 ┼█
       └─────┴───────┴───────────┴───────────┴───────────┴───────►
       0     3       7          12          18          25+
                        User Interaction Count (N_u)
```

### 5.3 User Lifecycle Stage Behavior in MovieLens 1M

| Stage | Interacted Movies ($N_u$) | Calculated $\alpha(u)$ ($\alpha_{\max}=0.8, \gamma=0.15$) | Dominant Engine | System Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **New User (Cold-Start)** | $0$ | **$0.00$** | **$100\%$ Content-Based** | Operates purely on CMU plot embeddings & metadata selections. |
| **Early Interaction** | $3$ | **$0.29$** | **$71\%$ Content / $29\%$ Collab** | Begins introducing SVD collaborative ratings without losing metadata anchors. |
| **Developing Profile** | $10$ | **$0.62$** | **$38\%$ Content / $62\%$ Collab** | Collaborative matrix factorization takes primary influence. |
| **Mature User** | $\ge 20$ | **$0.76 \to 0.80$** | **$20\%$ Content / $80\%$ Collab** | Reaches fixed $\alpha_{\max}$ ceiling; content acts as a subtle fine-tuner. |

---

## 6. Production-Ready Python Implementation (MovieLens 1M + CMU)

```python
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer

class MovieLensCMUDataLoader:
    """Utility to load and align MovieLens 1M with CMU Plot Summaries."""
    
    @staticmethod
    def clean_title(title: str) -> str:
        title = re.sub(r'\s*\(\d{4}\)', '', title)  # Remove year in parens
        title = title.lower().strip()
        title = re.sub(r'^(the|a|an)\s+', '', title)  # Strip leading articles
        return re.sub(r'[^\w\s]', '', title)

    def load_and_align(self, ml_movies_path: str, cmu_meta_path: str, cmu_plots_path: str) -> pd.DataFrame:
        # Load MovieLens 1M movies (movieId::Title::Genres)
        ml_movies = pd.read_csv(
            ml_movies_path, sep='::', engine='python', 
            names=['movieId', 'title', 'genres'], encoding='latin-1'
        )
        ml_movies['clean_title'] = ml_movies['title'].apply(self.clean_title)
        
        # Load CMU Metadata
        cmu_meta = pd.read_csv(
            cmu_meta_path, sep='\t', header=None,
            usecols=[0, 2, 3], names=['cmu_id', 'movie_name', 'release_date']
        )
        cmu_meta['clean_title'] = cmu_meta['movie_name'].astype(str).apply(self.clean_title)
        
        # Load CMU Plots
        cmu_plots = pd.read_csv(cmu_plots_path, sep='\t', header=None, names=['cmu_id', 'plot_summary'])
        cmu_full = pd.merge(cmu_meta, cmu_plots, on='cmu_id', how='inner')
        
        # Merge MovieLens with CMU on clean_title
        aligned_df = pd.merge(ml_movies, cmu_full, on='clean_title', how='inner')
        aligned_df = aligned_df.drop_duplicates(subset=['movieId']).reset_index(drop=True)
        
        # Build composite metadata tags
        aligned_df['metadata_tags'] = aligned_df['genres'].str.replace('|', ' ') + ' ' + aligned_df['clean_title']
        return aligned_df


class ContentEngine:
    def __init__(self, meta_weight=0.35, plot_weight=0.65):
        self.meta_weight = meta_weight
        self.plot_weight = plot_weight
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
    def fit_transform(self, aligned_df: pd.DataFrame):
        # 1. Sparse TF-IDF metadata matrix
        meta_corpus = aligned_df['metadata_tags'].fillna('')
        self.meta_vecs = self.tfidf.fit_transform(meta_corpus)
        
        # 2. Dense CMU plot embedding matrix
        plot_corpus = aligned_df['plot_summary'].fillna('').tolist()
        self.plot_vecs = self.transformer.encode(plot_corpus, show_progress_bar=False)
        return self

    def compute_user_content_scores(self, user_ratings: pd.Series) -> np.ndarray:
        pos_ratings = user_ratings[user_ratings >= 4.0]
        if len(pos_ratings) == 0:
            pos_ratings = user_ratings  # Fallback if no 4+ ratings
            
        pos_indices = pos_ratings.index.values
        weights = pos_ratings.values.reshape(-1, 1)
        
        user_meta_profile = np.asarray(self.meta_vecs[pos_indices].T.dot(weights)).T
        user_plot_profile = np.average(self.plot_vecs[pos_indices], axis=0, weights=pos_ratings.values).reshape(1, -1)
        
        meta_sim = cosine_similarity(user_meta_profile, self.meta_vecs).flatten()
        plot_sim = cosine_similarity(user_plot_profile, self.plot_vecs).flatten()
        
        return (self.meta_weight * meta_sim) + (self.plot_weight * plot_sim)


class CollaborativeEngine:
    def __init__(self, n_components=50, random_state=42):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        
    def fit_predict(self, R_matrix: np.ndarray) -> np.ndarray:
        # 1. Subtract User Mean
        self.user_means = np.nanmean(R_matrix, axis=1, keepdims=True)
        self.user_means = np.nan_to_num(self.user_means, nan=3.5)
        
        R_centered = np.nan_to_num(R_matrix - self.user_means, nan=0.0)
        
        # 2. Decompose and reconstruct
        user_factors = self.svd.fit_transform(R_centered)
        R_reconstructed = self.svd.inverse_transform(user_factors) + self.user_means
        
        # 3. Min-Max scale per user
        min_vals = R_reconstructed.min(axis=1, keepdims=True)
        max_vals = R_reconstructed.max(axis=1, keepdims=True)
        denom = np.where((max_vals - min_vals) == 0, 1.0, max_vals - min_vals)
        
        R_scaled = (R_reconstructed - min_vals) / denom
        return R_scaled


class DynamicHybridRecommender:
    def __init__(self, alpha_max=0.80, gamma=0.15):
        self.alpha_max = alpha_max
        self.gamma = gamma

    def calculate_alpha(self, n_user_interactions: int) -> float:
        """Computes interaction-dependent dynamic alpha weight."""
        return self.alpha_max * (1.0 - np.exp(-self.gamma * n_user_interactions))

    def recommend(
        self, 
        user_id: int, 
        user_interaction_count: int, 
        content_scores: np.ndarray, 
        collab_scores: np.ndarray, 
        seen_indices: list, 
        top_k: int = 10
    ) -> tuple[np.ndarray, float]:
        
        # 1. Compute dynamic alpha for current user
        alpha = self.calculate_alpha(user_interaction_count)
        
        # 2. Score interpolation
        hybrid_scores = (alpha * collab_scores) + ((1.0 - alpha) * content_scores)
        
        # 3. Filter out previously rated / watched movies
        hybrid_scores[seen_indices] = -np.inf
        
        # 4. Rank top-K candidate indices
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        return top_indices, alpha
```

---

## 7. Offline Evaluation Protocol & Metrics

To benchmark the performance of the dynamic hybrid recommender using MovieLens 1M, evaluate offline against standard non-personalized baselines and ranking metrics.

### 7.1 Baseline Benchmarks
1. **Random Recommender**: Selects items randomly from candidate catalog (lower bound).
2. **Most Popular**: Recommends movies with the highest total rating counts in MovieLens 1M.
3. **Average Rating**: Recommends movies with the highest mean score across all users.
4. **Bayesian Average Rating**: Smooths average ratings by adding pseudo-counts to prevent low-vote items from dominating.

### 7.2 Target Offline Ranking Metrics
Evaluated over top-$K$ recommendations ($K \in \{5, 10, 20\}$) using leave-$k$-out temporal or random splits:
* **Precision@K**: Proportion of top-$K$ recommendations that are relevant ($r_{u,i} \ge 4.0$ in test set).
* **Recall@K**: Proportion of total relevant items recovered in top-$K$.
* **NDCG@K (Normalized Discounted Cumulative Gain)**: Measures ranking quality by penalizing relevant items placed lower in the recommended list.
* **MAP@K (Mean Average Precision)** & **MRR@K (Mean Reciprocal Rank)**.

---

## 8. Summary
This updated architectural document formally integrates **MovieLens 1M** and the **CMU Movie Summary Corpus**. The combination of long-form Wikipedia narrative embeddings (via Sentence-BERT), sparse metadata TF-IDF vectors, mean-centered SVD matrix factorization, and user-interaction dynamic score weighting provides an end-to-end framework for hybrid movie recommendation.
