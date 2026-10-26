# Hybrid Movie Recommender

A two-stage recommender on Kaggle's "The Movies Dataset":

1. **Candidate generation (high recall).** Three sources each return their top k movies, which are merged and de-duplicated:
   - **content-based:** TF-IDF over genres, keywords, director, cast and collection
   - **collaborative:** implicit ALS
   - **embedding:** a two-tower model over MiniLM text vectors, served from Qdrant
2. **Reranking (high precision).** A PyTorch model scores the candidates and returns the top 10.
   - v1: MLP
   - v2: attention over the user's history plus DCN-v2

Metrics: P@10, R@10, NDCG@10. Storage: SQLite, plus Qdrant for vectors. UI: Streamlit.

Results and the decisions behind them are in [REPORT.md](REPORT.md).

## Setup

```
python -m venv --system-site-packages .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e . --no-deps --no-build-isolation
```

Put the Kaggle CSVs in `data/raw/`.

The Python package is called `hybridrec`, not `recsys`, because another project on this machine already installs a `recsys` package.

## Running the pipeline

```
.venv\Scripts\python scripts\run_all.py --dataset small
.venv\Scripts\python scripts\run_all.py --dataset full20k
```

**Datasets:**
- `small`: ratings_small.csv (670 users).
- `full20k`: a fixed random sample of 20,000 users (each with 20+ ratings) from ratings.csv.

**Steps** (each script takes `--dataset`, and `--stage` where relevant):

| Script | Does |
|---|---|
| `01_prepare_data.py` | Clean the catalog, sample and split ratings, build `db/<dataset>.db` |
| `02_build_item_features.py` | TF-IDF matrix, text embeddings, Qdrant `movie_text` collection |
| `03_fit_retrievers.py` | iALS and two-tower per stage (`--tune-als` for the grid) |
| `04_eval_retrievers.py` | Each source alone, recall curves, overlap, k grid -> `reports/` |
| `05_candidates.py` | Merge sources into labelled candidate lists; RRF baseline |
| `06_features.py` | Reranker features per stage |
| `07_train_reranker.py` | Train v1/v2 on s1 (+s0), early stop on s2 (`--ablations`) |
| `08_evaluate.py` | Final test on s3 -> `reports/<dataset>_final.md` |

**Split and stages:**
- Each user's ratings, oldest first, are cut 70/10/10/10 into blocks R/L/V/T.
- A rating of 4.0 or more counts as liked.
- s1: models fit on R, labels from L. Used for reranker training.
- s2: fit on R+L, evaluated on V. Used for all tuning.
- s3: fit on R+L+V, evaluated on T. The final test, run once.
- s0 (small only): fit on the first 60%, labels from 60-70%. Extra reranker training data.

## UI

```
.venv\Scripts\python -m streamlit run app\streamlit_app.py
```

**What's in the app:**
- **Sidebar:**
  - Mode: an existing test user, or a new user who picks 3+ liked movies
  - Checkboxes for watch history and pipeline details
- **Tabs:**
  - Recommendations: top 10 with the sources that found each movie, plus a held-out check for test users
  - Similar movies: from Qdrant, by story text or two-tower vectors
  - Evaluation: the results tables

The UI is text and tables only. `.streamlit/config.toml` hides Streamlit's developer toolbar.

**Qdrant and the pipeline:**
- Qdrant runs in local mode, which allows one process at a time on `db/qdrant`. Close the app before re-running the pipeline scripts.
- To use the app while the pipeline is writing, copy `db/qdrant` elsewhere and point `HYBRIDREC_QDRANT_PATH` at the copy.

## Tests

```
.venv\Scripts\python -m pytest
```

The tests cover:
- metrics, checked against hand-worked examples
- the split rules
- merge invariants
- ID-join coverage
- a feature-leakage check: s1 features are unchanged when every later rating is deleted
- serving vs offline: the Qdrant-backed serving path reproduces the offline candidates
