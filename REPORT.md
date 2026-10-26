# Hybrid Movie Recommender: Results

**Summary.** The two-stage pipeline beats every single model on both datasets:
- Stage 1 merges three candidate sources: content-based, iALS and the two-tower model.
- Stage 2 is a deep reranker that picks the top 10.

On the 20,000-user sample of the full ratings, the test results are:

| Model | NDCG@10 |
|---|---|
| Deep reranker (v2) | **0.146** |
| Rank fusion of the three sources, no learning (RRF) | 0.128 |
| Best single source (two-tower) | 0.096 |
| iALS alone | 0.082 |
| Popularity | 0.050 |

That's +14% over RRF and +52% over the best single source. Precision@10 is 0.080 and recall@10 is 0.166.

The detailed tables are generated into [reports/](reports/) by the pipeline scripts.

## Setup

**Data:**
- `small` = ratings_small.csv: 670 users and 99.8k ratings, recommending from 9,025 movies.
- `full20k` = a fixed random sample (seed 42) of 20,000 users with 20+ ratings from ratings.csv: 2.92M ratings, recommending from 44,732 movies.
- Recommendable movies: every catalog movie rated at least once in the source file.

**Labels and splits:**
- A movie is relevant when rated 4.0 or higher.
- Each user's ratings, oldest first, are split 70/10/10/10 into blocks R/L/V/T.
- Every model is fitted only on the blocks before the one it is scored on:

| Stage | Fit on | Scored on | Used for |
|---|---|---|---|
| s1 | R | L | Reranker training labels |
| s2 | R+L | V | All tuning and early stopping |
| s3 | R+L+V | T | Final test, run once |
| s0 (small only) | First 60% | 60-70% | Extra reranker training data |

**Metrics:**
- P@10, R@10 and NDCG@10, with 0/1 relevance.
- Averaged over users with at least one liked movie in the scored block.
- Movies the user already rated are excluded.
- Users whose candidates contain no liked movie still count, scoring 0.
- Also reported: R@10 divided by min(10, number of liked movies), catalogue coverage, and the mean log rating count of recommended movies ("avg log pop", higher means more popular).

## Test results (s3)

### full20k (18,674 test users)

| model | P@10 | R@10 | R@10 (trunc) | NDCG@10 | coverage | avg log pop |
|---|---|---|---|---|---|---|
| popularity | 0.0310 | 0.0563 | 0.0631 | 0.0504 | 0.0047 | 8.84 |
| content-based (cb) | 0.0080 | 0.0161 | 0.0174 | 0.0144 | 0.2248 | 3.36 |
| iALS (cf) | 0.0480 | 0.1012 | 0.1094 | 0.0820 | 0.0450 | 7.93 |
| text embeddings v1 | 0.0020 | 0.0055 | 0.0058 | 0.0044 | 0.1416 | 1.85 |
| two-tower | 0.0494 | 0.1180 | 0.1252 | 0.0956 | 0.0855 | 5.76 |
| RRF of cb + cf + two-tower | 0.0690 | 0.1537 | 0.1645 | 0.1279 | 0.0740 | 7.53 |
| reranker v1 (MLP) | **0.0800** | 0.1653 | **0.1804** | 0.1445 | 0.0568 | 7.67 |
| **reranker v2 (attention + DCN-v2)** | 0.0797 | **0.1655** | **0.1804** | **0.1457** | 0.0655 | 7.64 |

**Candidate stage on s3:**
- About 261 candidates per user.
- Union recall of 0.558: 56% of each user's liked test movies reach the reranker.

**Validation vs test (NDCG@10):**

| Model | s2 | s3 |
|---|---|---|
| v2 | 0.1461 | 0.1457 |
| v1 | 0.1433 | 0.1445 |
| RRF | 0.1284 | 0.1279 |

The tuned models lose nothing from s2 to s3, which is the check the plan set for leakage.

**What the reranker does to the list.** It recommends slightly more popular movies than RRF (avg log pop 7.6 vs 7.5) and covers less of the catalogue:

| Model | coverage |
|---|---|
| RRF | 0.074 |
| v2 | 0.066 |
| v1 | 0.057 |

The two-tower source alone reaches the widest share of the catalogue among the strong sources (0.086).

### small (629 test users)

| model | P@10 | R@10 | NDCG@10 | coverage |
|---|---|---|---|---|
| popularity | 0.0254 | 0.0515 | 0.0430 | 0.0119 |
| content-based (cb) | 0.0110 | 0.0188 | 0.0170 | 0.2133 |
| iALS (cf) | 0.0437 | 0.0988 | 0.0794 | 0.0829 |
| text embeddings v1 | 0.0041 | 0.0104 | 0.0079 | 0.1033 |
| two-tower | 0.0367 | 0.0810 | 0.0694 | 0.0517 |
| RRF of cb + cf + two-tower | 0.0510 | 0.1077 | 0.0947 | 0.0747 |
| **reranker v1 (MLP)** | **0.0574** | **0.1207** | **0.1050** | 0.0801 |
| reranker v2 (attention + DCN-v2) | 0.0536 | 0.1145 | 0.1000 | 0.0884 |

On small, the validation (s2) NDCG@10 values are:

| Model | s2 NDCG@10 |
|---|---|
| RRF | 0.0984 |
| v1 | 0.1014 |
| v2 | 0.1001 |

That is a tie within noise. The standard error is about 0.008 on 615 users, and CPU training varies by about 0.004 from run to run. On s3, v1 leads RRF by 0.010.

## Candidate generation

### Each source alone (s2, recall@100)

| source | small | full20k |
|---|---|---|
| popularity | 0.236 | 0.260 |
| content-based | 0.092 | 0.058 |
| iALS | **0.434** | **0.488** |
| text embeddings v1 | 0.043 | 0.022 |
| two-tower | 0.302 | 0.415 |

**iALS:**
- The best recall@100.
- Tuned on s2 with a 24-point grid. Best settings:
  - small: 32 factors, regularization 1.0, α 2 (recall@100 rose from 0.378 with the defaults to 0.434)
  - full20k: 64 factors, regularization 1.0, α 5

**Two-tower:**
- Second in recall, but the best single source at the top of the list on full20k (s2 NDCG@10 0.099, against 0.087 for iALS).
- Only about 35% of its top-100 overlaps with iALS.
- In the union, about 19% of hits (full20k) come only from the two-tower model.

**Text embeddings v1** (MiniLM over plot, tagline and genres):
- Far below even popularity.
- Story similarity alone is a poor predictor of which movie a MovieLens user rates next.
- The plan's rule was to keep whichever embedding option won, so the two-tower model is the embedding source.
- The text vectors still feed the two-tower model, the reranker's similarity features and the UI's "similar movies".

**Content-based:**
- Weak alone.
- Still contributes hits no other source finds: about 9% of union hits on small, 3% on full20k.
- The mean-centred user profile from the design beat three alternatives:
  - liked movies only
  - liked movies with recency weighting
  - mean-centred with recency weighting

### Choosing k

Union recall of cb + cf + two-tower on s2:

| k per source | small recall | small avg candidates | full20k recall | full20k avg candidates |
|---|---|---|---|---|
| 25 | 0.268 | 67 | 0.326 | 68 |
| 50 | 0.390 | 132 | 0.452 | 133 |
| 100 | **0.524** | **259** | **0.588** | **260** |
| 150 | 0.606 | 386 | 0.665 | 388 |
| 200 | 0.659 | 511 | 0.717 | 515 |

The plan's rule was to stop when +50 per source adds less than 1 point of recall. It never triggers here: each step still adds 5-13 points. So k was set by an end-to-end check on small instead:
- Rebuild the candidates, features and v1 reranker at each k.
- Measure s2 NDCG@10 with two seeds.

| k per source | union recall | RRF NDCG@10 | reranker v1 NDCG@10 (2 seeds) |
|---|---|---|---|
| 50 | 0.390 | 0.0930 | 0.1036 / 0.1013 |
| 100 | 0.524 | 0.0984 | 0.1013 / 0.1035 |
| 200 | 0.659 | 0.0962 | 0.1002 / 0.1015 |

Final quality is flat in k. The extra candidates sit deep in every source's list and rarely reach the top 10. **k = 100 per source** was chosen for both datasets:
- It keeps a high candidate recall.
- It costs nothing in final quality.
- full20k's feature tables (about 5M rows per stage) still fit in 8 GB of RAM. k = 200 would not.

## Reranker

**Inputs:** 79 features in four groups:
- candidate stage: each source's score, rank, in-top-k flag and percentile; RRF score
- user: activity, rating habits, genre taste, era, how mainstream they are
- movie: genres, year, runtime, language, collection, rating count and average in the fit window
- user × movie: genre match; shared director, cast and collection with liked movies; year and popularity gaps; text and two-tower similarity to the last 20 liked movies

**Models:**
- **v1:** an MLP (256-128-64) with BCE loss. Each epoch uses every positive, every movie rated below 4, and 20 sampled never-rated candidates per positive.
- **v2:** adds a learned movie vector (from text and genres) and attention over the user's last 50 liked movies, weighted by similarity to the candidate. These feed 2 DCN-v2 cross layers, trained with a listwise softmax over each user's list.

**Training:** on s1 (plus s0 on small), early stopping on s2 NDCG@10 over the full candidate lists. Serving uses whichever version has the higher s2 NDCG@10:
- small: v1
- full20k: v2 (s2 NDCG@10 0.1461, against 0.1433 for v1)

### Ablations (v1 retrained with one feature group masked out, s2 NDCG@10)

| without | small | full20k |
|---|---|---|
| nothing (full v1) | 0.1014 | 0.1433 |
| candidate-stage features | 0.0881 | 0.1129 |
| user features | 0.0967 | 0.1414 |
| item features | 0.0987 | 0.1416 |
| cross features | 0.1053 | 0.1397 |

What the ablations show:
- **Candidate-stage features carry most of the signal** on both datasets. Without them, full20k NDCG@10 falls from 0.143 to 0.113.
- **Cross features matter on full20k:** removing them costs 0.0036, about 2 standard errors with 18.7k users. On small, removing them doesn't hurt.
- **User and item features each add a little** (about 1 standard error on full20k).
- **v2 vs v1:** v2's attention and cross layers add about 0.001–0.003 over v1 on full20k (s2 and s3). That's a small but consistent gain, and it only shows up once there is enough training data.

## Changes from the plan, and why

- **Package name.** The package is `hybridrec`, not `recsys`. Another project on this machine installs a `recsys` package into the system Python, which the venv reuses.
- **Qdrant local mode write speed.** Local mode commits its SQLite file once per point, which made writing 9k vectors take more than 10 minutes. `qdrant_store` turns off the disk sync while writing; the collections are rebuilt by the pipeline anyway.
- **Serving uses Qdrant.** The embedding source in serving searches Qdrant (exact search in local mode) and fetches candidate vectors from it. A test checks that serving reproduces the offline NumPy candidates.
- **Reranker v2's movie vectors use text and genres only.** Each stage's two-tower model has its own vector space, so its vectors can't be model inputs across stages. Two-tower information reaches the reranker through the similarity and score features.
- **full20k memory.** Feature tables are streamed from Parquet into float16 matrices. The plain pandas path used about 8 GB and caused swapping.
- **Ablations by masking.** Ablations mask a feature group to zero instead of rebuilding tables. For these models a constant-zero input carries no information.
- **UI.** Expanders (which draw a chevron icon) were replaced with sidebar checkboxes, and Streamlit's toolbar is set to minimal.
- **Skipped optional step.** The plan's optional step of retraining the reranker on s2 labels before testing was not done.

## Caveats

- **Noise.** On small, reranker differences under about 0.01 NDCG@10 are within noise.
- **Run-to-run variation.** CPU training is not bit-for-bit reproducible.
- **Split leakage.** The per-user time split lets one user's late ratings inform another user's earlier test period. The design discussion accepted this; a global date cut would leave too few test users with history.
- **Two-tower early stopping is noisy.** It uses a 2,000-user holdout with patience 2. On full20k, the s2 model stopped at epoch 6 (best was epoch 4).
- **Two-tower "similar movies".** In the UI, these group movies by co-viewing, often by release era. That's useful as a contrast to text similarity, but it is not a content similarity.
