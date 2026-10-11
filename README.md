# Hybrid Movie Recommender (MovieLens 1M + CMU Plot Summaries)

This project implements the system described in
[hybrid-movie-recommender-architecture.md](hybrid-movie-recommender-architecture.md), plus the improvements from `suggestions.md` that held up in testing. It has four parts:
- **Content engine:** TF-IDF metadata vectors plus Sentence-BERT (`all-MiniLM-L6-v2`) embeddings of the Wikipedia plots. Scores are popularity-gated.
- **Collaborative engine:** an equal-weight ensemble of TruncatedSVD on mean-centred ratings and implicit ALS (iALS).
- **Blend:** the two scores are combined with a dynamic weight α(u) = α_max · (1 − e^(−γ·N_u)). A popularity prior carries new users and fades out as β(u) = e^(−η·N_u); it can be boosted by the genres a new user picks.
- **Evaluation:** offline comparison against baselines.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

`ml-1m.zip` and `MovieSummaries.tar.gz` must be in the repository root.

## Running

```bash
python -m recsys.cli all            # extract → align → embed → train → evaluate → coldstart
python -m recsys.cli tune           # optional: grid search, writes artifacts/best_params.json
python -m recsys.cli train          # retrain using the tuned parameters
streamlit run app/streamlit_app.py  # demo
pytest
```

| Step | What it does | Output |
| --- | --- | --- |
| `extract` | Unpacks both archives | `data/raw/` |
| `align` | Matches MovieLens movies to CMU plots (title + year, then a guarded fuzzy pass) | `data/processed/items.parquet` |
| `embed` | Encodes plots with SBERT in chunks and caches the result. Takes about 11 minutes on CPU | `artifacts/plot_emb.npy` |
| `train` | Fits every engine on all ratings | `artifacts/*` |
| `evaluate` | Runs baselines and models on random and temporal 80/20 splits | `artifacts/eval_results.{csv,md}` |
| `coldstart` | Cuts 1,000 users' history down to N ratings, refits and evaluates | `artifacts/coldstart_results.{csv,md}` |
| `tune` | Grid search on validation splits carved from train: blend weights first, then γ and η on truncated histories | `artifacts/tuning_results.csv`, `best_params.json` |

## Layout

```
src/recsys/
  data/        extract, loaders, align (title matching), interactions (index + CSR)
  features/    metadata (TF-IDF), plot_embed (chunked SBERT)
  models/      content (popularity-gated), collaborative (SVD + fold-in), ials (iALS + fold-in),
               hybrid (α(u) blend + fading popularity β(u)), baselines
  evaluation/  split, metrics, evaluate (comparison, cold-start sweep, tuning)
  pipeline.py  train + persist artifacts
  service.py   RecommenderService used by the app
app/streamlit_app.py
experiments/   one script per evaluated suggestion, with results (see experiments/README.md)
notebooks/01_eda.ipynb
```

## Where this differs from the design document

1. **Title alignment.** MovieLens writes titles as `Matrix, The (1999)` and adds alternate titles in parentheses, such as `Postino, Il (The Postman)`. Each movie therefore gets several match keys: the trailing article is moved to the front, and every alternate title becomes its own key. Matching uses `(clean_title, year ± 1)`.
2. **Pass 2 (Freebase→IMDb via Wikidata) is skipped.** ML-1M has no `links.csv`. `align.pass2_external(mapping_csv)` is left as a hook for a mapping file.
3. **Fuzzy matching has guards.** Jaro-Winkler on its own accepted pairs such as *Sleepover → Sleepers* and *Godzilla 2000 → Godzilla*. A match now also needs an indel ratio of at least 0.88 and the same numbers in both titles, with Roman numerals and `½` normalised first.
4. **Plot coverage is about 81%, not the 85–92% the document estimates.** About 92% of movies match a CMU record. Many CMU records have metadata but no plot summary, such as *Last Action Hero* and *Girlfight*.
5. **The content engine covers every rated movie.** Movies without a plot still get a metadata vector, and their content score falls back to the metadata channel alone.
6. **Content features.** CMU genres and countries are stored as Freebase JSON and get parsed. Cast is taken from `character.metadata.tsv`. **Neither dataset has a director field.**
7. **Score scales.** SBERT cosines lie in [−1, 1] and TF-IDF cosines in [0, 1]. Each channel is min-max scaled per user before blending, so the content score lands on the same [0, 1] scale as the SVD score.
8. **Long plots are chunked.** MiniLM truncates input at 256 tokens. Plots are split into 150-word chunks, encoded, mean-pooled and L2-normalised.
9. **Damped user mean in the SVD.** Each user's mean is shrunk toward the global mean with 5 pseudo-ratings. Otherwise a new user who rates everything 5★ would have an all-zero centred row, and fold-in would return nothing.
10. **New users.** The final score is β(u)·S_popular + (1 − β(u))·S_hybrid, with β(u) = e^(−η·N_u). So a user with no ratings gets pure most-popular, boosted by any genres picked in the app as pop · (1 + genre match). Popularity keeps a large share for the first few ratings. New users get collaborative scores by fold-in: p_u = r̃_u·Q for SVD, and the exact ALS user step for iALS.
11. **No leakage.** User means, the SVD, iALS, content profiles, the popularity gate and N_u are all fitted on the training split only.
12. **Collaborative engine and content weights.** These follow the evaluated improvements below: an equal-weight SVD + iALS ensemble, popularity-gated content, and w_meta / w_plot = 0.65 / 0.35 instead of the document's 0.35 / 0.65.

## Results

The tuned parameters come from `python -m recsys.cli tune` on validation splits carved out of the training data. Each setting is scored by its NDCG@10 relative to the best setting, averaged over random and time-ordered validation. The test splits were never used for tuning.

| Parameter | Value |
| --- | --- |
| k (SVD components) | 32 |
| iALS weight in S_collab | 0.5 |
| iALS | 32 factors, reg 3, confidence 3, 30 iterations |
| w_meta / w_plot | 0.65 / 0.35 |
| α_max | 0.8 |
| γ | 0.15 |
| η (popularity fade) | 0.2 |

Relevance means a test rating of 4 or more. Seen items are excluded from recommendations, and every candidate is ranked against the full catalogue of 3,706 movies.

**NDCG@10, all users**

| Model | Random 80/20 | Temporal 80/20 |
| --- | ---: | ---: |
| Random | 0.0060 | 0.0050 |
| Average rating | 0.0001 | 0.0002 |
| Bayesian average | 0.0920 | 0.0525 |
| Content only (popularity-gated) | 0.1517 | 0.0686 |
| Most popular | 0.1746 | 0.0918 |
| SVD only | 0.2375 | 0.1185 |
| iALS only | 0.3516 | 0.1107 |
| **Hybrid (dynamic α + fading popularity)** | **0.3668** | **0.1335** |
| *Previous hybrid (SVD, ungated content)* | *0.2520* | *0.1192* |

The hybrid leads every other model on every metric, on both splits:

| Metric @10 | Hybrid (random) | Previous hybrid (random) | Hybrid (temporal) | Previous hybrid (temporal) |
| --- | ---: | ---: | ---: | ---: |
| Precision | 0.301 | 0.207 | 0.110 | 0.101 |
| Recall | 0.212 | 0.127 | 0.093 | 0.071 |
| MAP | 0.234 | 0.148 | 0.066 | 0.059 |
| MRR | 0.599 | 0.472 | 0.254 | 0.237 |

Full tables for K = 5, 10 and 20 are in `artifacts/eval_results.md`.

**Cold-start simulation (NDCG@10).** For 1,000 users, training history was cut down to their earliest N ratings, and every model was refitted:

| Model | N=0 | 3 | 5 | 10 | 20 | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Most popular | 0.085 | 0.088 | 0.091 | 0.097 | 0.103 | 0.165 |
| Content only (popularity-gated) | 0.006 | 0.041 | 0.052 | 0.077 | 0.094 | 0.141 |
| SVD only | 0.006 | 0.062 | 0.071 | 0.093 | 0.114 | 0.223 |
| iALS only | 0.006 | 0.063 | 0.080 | **0.131** | **0.192** | 0.346 |
| **Hybrid (dynamic α + fading popularity)** | **0.085** | **0.089** | **0.093** | 0.128 | 0.186 | **0.359** |
| *Hybrid without the fading prior* | *0.085* | *0.062* | *0.079* | *0.127* | *0.186* | *0.358* |
| *Original hybrid (SVD, ungated content)* | *0.059* | *0.061* | *0.071* | *0.095* | *0.120* | *0.237* |

### Findings

- **The accepted changes lift NDCG@10 by 46% on the random split and 12% on the temporal split.** They also help at every cold-start depth, by 31–55%: from 0.059 to 0.085 at N=0, 0.061 to 0.089 at N=3, and 0.120 to 0.186 at N=20.
- **The SVD + iALS ensemble is what makes the hybrid robust across protocols.** iALS alone wins clearly on the random split but loses to SVD on the temporal one, and the ensemble beats both on both.
- **Gating content by popularity makes it competitive on its own,** taking content-only NDCG@10 from 0.025 to 0.152.
- **The fading popularity prior closed the 3–5-rating gap.** Before it, plain most-popular (0.088–0.091) beat the hybrid (0.062–0.079) at those depths. With β(u) = e^(−0.2·N_u), the hybrid scores 0.089 at N=3 and 0.093 at N=5, so it now matches or beats most-popular at every depth. Full-history results are unchanged, because every full-history user has at least 16 training ratings, so β ≤ 0.04. At N=3 the gain is mostly from leaning on popularity, not from better personalisation (0.0887 vs 0.0882).
- **With popularity carrying new users, the design document's γ = 0.15 is optimal again.** Without the prior, tuning wanted a fast collaborative ramp (γ = 1.0). With it, the slow ramp wins (0.111 vs 0.103 mean validation NDCG@10 over N = 3–20).
- **Remaining gaps.** iALS alone is still slightly ahead of the hybrid at N = 10–20 (0.131 vs 0.128; 0.192 vs 0.186). In the demo, a single rating now barely changes the list (β(1) = 0.82). That is the accuracy-optimal behaviour offline, but it feels less responsive; genre picks or a higher η in the sidebar change it.

## Evaluated improvements (`suggestions.md`)

Each suggestion was implemented as an experiment and kept only if it improved held-out NDCG@10 without hurting another protocol. Hyperparameters were always chosen on validation splits. Numbers are test NDCG@10 unless marked *(val)*.

| # | Suggestion | Evidence | Decision |
| --- | --- | --- | --- |
| 1 | **Popularity-gated content**: S_content × log(1 + count) | Content-only: 0.025 → 0.152 (random), 0.014 → 0.070 (temporal). Hybrid: +2% (random), +4% (temporal). Better at every cold-start N ≥ 3. It cannot help at N = 0, where there is no content profile. | ✅ **Accepted** (`content_pop_gate`) |
| 2a | **iALS** instead of SVD | iALS 0.215 vs SVD 0.138 *(val)*. A hybrid built on iALS alone gains 41% on the random test (0.356 vs 0.252) but trails the gated SVD hybrid on temporal (0.116 vs 0.124); retuning on temporal validation only reaches parity (0.124). The 50/50 ensemble with SVD wins both: 0.367 random, 0.134 temporal. | ✅ **Accepted** as an SVD + iALS ensemble (`ials_weight`) |
| 2b | **BPR** | Best of 16 configurations 0.124 *(val)*, below SVD (0.138) and far below iALS (0.215). | ❌ Rejected |
| 3 | **Recency decay** w(t) = e^(−λΔt) | Day-based decay changes nothing (temporal 0.1238 → 0.1239). The median user rated everything within one day, and 66% of training ratings fall within a day of the user's last rating. Order-based decay (half-life in ratings) gains 1.8% on temporal but loses 3.2% on random. Decaying the SVD input hurts both splits, by 5–30%. | ❌ Rejected |
| 4 | **Two-stage retrieval + LightGBM LambdaRank** (top-100 candidates, 18 features) | Trained on random-style validation: +1.1% on random, −3.5% on temporal (0.134 → 0.129), mixed on cold start (−6% at N = 5). Trained on temporal validation: +8.6% on temporal (0.134 → 0.146), but the same ranker loses 33% on random (0.367 → 0.247) and 10–19% at every cold-start depth. The gains are tied to the protocol it was trained on, so they don't carry over. | ❌ Rejected |
| 5 | **Onboarding seeds** for N_u = 0 | In a simulation where each user's answers come from their hidden history: previous fallback (Bayesian average) 0.059 → most-popular 0.085 → most-popular × (1 + genre match) 0.096. A TF-IDF genre pseudo-profile (U_seed) scores 0.024. Three seed favourites through the hybrid's fold-in score 0.098; the app already supports this by rating movies. | ✅ **Accepted**: most-popular fallback and multiplicative genre boost. The TF-IDF U_seed variant was rejected. |
| F | **Follow-up: fading popularity prior** β(u) = e^(−η·N_u) (`experiments/exp7_fading_popularity.py`) | Tuned on validation: η = 0.2, γ = 0.15, raw counts beat log-counts. Test: N = 3 0.062 → 0.089 (+43%), N = 5 0.079 → 0.093 (+18%), N = 10 +1%, full histories unchanged. The proposed η = 0.4 with the then-current γ = 1.0 helps less (N = 3: 0.081, still below most-popular). | ✅ **Accepted** with the tuned η and γ (`pop_eta`) |
| 6 | **LightGCN** | A CPU implementation took 99 s per epoch on an idle machine (191 s under load). After 10 epochs it reached 0.166 *(val)*, against iALS at 0.215, which fits in about 2 s. Convergence would take hundreds of epochs, which is hours on this machine. It also has no way to embed a new user without retraining. It was not run to convergence. | ❌ Rejected (cost, plus no cold-start path) |
