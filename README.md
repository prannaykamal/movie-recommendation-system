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
12. **Collaborative engine and content weights.** These follow the evaluated improvements below: an equal-weight SVD + iALS ensemble, popularity-gated content, and w_meta / w_plot = 0.65 / 0.35 instead of the document's 0.35 / 0.65. Inside the hybrid, metadata matters more than plots (see *What the content channels contribute*).

## Results

The tuned parameters come from `python -m recsys.cli tune` on validation splits carved out of the training data. Each setting is scored by its NDCG@10 relative to the best setting, averaged over random and time-ordered validation. The test splits were never used for tuning.

| Parameter | Value |
| --- | --- |
| k (SVD components) | 32 |
| iALS weight in S_collab | 0.5 |
| iALS | 32 factors, reg 3, confidence 3, 30 iterations |
| w_meta / w_plot | 0.65 / 0.35 (indistinguishable from 0.8 / 0.2 across seeds) |
| α_max | 0.8 |
| γ | 0.15 (indistinguishable from 1.0 across seeds) |
| η (popularity fade) | 0.2 |
| λ (popularity penalty, optional) | 0, off; see *Beyond accuracy* |

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

### Robustness across seeds

The single-split results above were re-checked with `experiments/exp8_robustness.py`:
- **Five seeds.** Each seed sets both the random split and the SVD/iALS initialisation. The temporal split is fixed, so only the model seed varies there.
- **Paired bootstrap.** Each comparison uses a 95% bootstrap confidence interval over users, computed per seed.
- **Noise level.** The seed-to-seed spread of NDCG@10 is about ±0.0005 on full histories and about ±0.004 in cold start, so differences below about 0.3% (full histories) or about 3–4% (cold start) are noise.

Each cell is the relative NDCG@10 difference, averaged over seeds, followed by the number of seeds whose confidence interval excludes zero:

| Comparison | Random | Temporal | N=3 | N=5 | N=10 | N=20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Final vs original system | +45.8% (5/5) | +12.4% (5/5) | +44.9% (5/5) | +33.2% (5/5) | +36.4% (5/5) | +52.1% (5/5) |
| iALS in the ensemble (vs SVD-only collab) | +40.4% (5/5) | +8.2% (5/5) | +0.3% (0/5) | +3.5% (3/5) | +21.2% (5/5) | +43.6% (5/5) |
| SVD in the ensemble (vs iALS-only collab) | +2.7% (5/5) | +16.8% (5/5) | −0.2% (0/5) | +2.2% (1/5) | −2.6% (2/5 worse) | −2.4% (1/5 worse) |
| Fading popularity prior (vs none, γ = 1) | −0.1% (1/5 worse) | 0.0% (0/5) | +31.1% (5/5) | +15.3% (5/5) | −0.1% (0/5) | −0.3% (0/5) |
| Popularity-gated content (vs ungated) | +0.1% (0/5) | +1.7% (5/5) | −0.1% (0/5) | −0.1% (0/5) | +0.3% (0/5) | −0.3% (0/5) |
| γ = 0.15 vs 1.0 | 0.0% (0/5) | +0.1% (1/5) | +0.5% (0/5) | +1.8% (0/5) | −0.5% (0/5) | −0.4% (0/5) |
| w_meta = 0.65 vs 0.8 | 0.0% (0/5) | +0.1% (0/5) | −0.5% (0/5) | −0.4% (0/5) | +0.4% (1/5) | +0.2% (1/5) |
| Final vs most-popular | +112% (5/5) | +45.5% (5/5) | +2.0% (1/5) | +4.4% (2/5) | +29.7% (5/5) | +79.2% (5/5) |
| Final vs iALS alone | +4.7% (5/5) | +21.2% (5/5) | +35.7% (5/5) | +18.8% (5/5) | −1.5% (0/5) | −1.4% (0/5) |

### What the content channels contribute

The question is how much the Sentence-BERT plot embeddings add. Plot embeddings are the design document's centrepiece and the slowest pipeline step. The table shows NDCG@10, averaged over five seeds:

| Content used inside the final hybrid | Random | Temporal | N=3 | N=5 | N=10 | N=20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Metadata + plots (final) | 0.3675 | 0.1335 | 0.0934 | 0.0987 | 0.1310 | 0.1947 |
| Metadata only | 0.3663 | 0.1326 | 0.0940 | 0.0987 | 0.1293 | 0.1934 |
| Plots only | 0.3540 | 0.1297 | 0.0917 | 0.0954 | 0.1255 | 0.1885 |
| No content (collaborative + popularity prior) | 0.3629 | 0.1291 | 0.0925 | 0.0955 | 0.1301 | 0.1939 |
| *Content alone: metadata + plots* | *0.1518* | *0.0686* | *0.0454* | *0.0534* | *0.0768* | *0.0989* |
| *Content alone: metadata only* | *0.1401* | *0.0623* | *0.0367* | *0.0457* | *0.0652* | *0.0862* |
| *Content alone: plots only* | *0.0806* | *0.0520* | *0.0469* | *0.0503* | *0.0588* | *0.0626* |

### Beyond accuracy

These metrics come from `recsys.evaluation.beyond` and are computed over the top-10 lists on the random split, averaged over five seeds. The rows with λ come from seed 42 only.
- **Coverage:** the share of the catalogue that is recommended to anyone.
- **Long tail:** the share of recommendation slots that go to items outside the 20% most-rated.
- **Pop. percentile:** the mean popularity percentile of recommended items.
- **Novelty:** −log2 of each item's popularity, in bits.
- **ILD:** intra-list genre diversity.
- **Personal.:** 1 − the overlap between two users' lists.

| Model | NDCG@10 | Coverage | Long tail | Pop. percentile | Novelty | ILD | Personal. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Most popular | 0.173 | 1.9% | 0.0% | 0.998 | 1.53 | 0.65 | 0.63 |
| Content alone (metadata) | 0.140 | 22.3% | 19.1% | 0.881 | 3.24 | 0.47 | 0.94 |
| SVD alone | 0.237 | 13.5% | 0.2% | 0.979 | 2.16 | 0.72 | 0.95 |
| iALS alone | 0.351 | 25.2% | 1.3% | 0.960 | 2.44 | 0.61 | 0.97 |
| Original system | 0.252 | 22.4% | 4.1% | 0.948 | 2.47 | 0.68 | 0.96 |
| **Final system** | **0.367** | 20.2% | 0.6% | 0.969 | 2.30 | 0.62 | 0.96 |
| Final, ungated content | 0.367 | 23.1% | 1.3% | 0.964 | 2.38 | 0.61 | 0.97 |
| Final, λ = 0.1 | 0.367 | 21.8% | 1.0% | 0.966 | 2.35 | 0.61 | 0.97 |
| Final, λ = 0.2 | 0.366 | 24.2% | 1.6% | 0.961 | 2.41 | 0.60 | 0.97 |
| Final, λ = 0.3 | 0.364 | 27.2% | 2.6% | 0.954 | 2.50 | 0.60 | 0.97 |

λ is an optional popularity penalty: the final score minus λ · minmax(log(1 + count)), set with `pop_penalty` in the config or the *Discovery* slider in the app. On the temporal split the same λ values cost 0.8%, 2.2% and 4.0% NDCG@10, for +10%, +19% and +38% coverage. At λ ≈ 0.5 accuracy collapses (−39% to −57%). The full curve is in `experiments/results/exp9_popularity_tradeoff.csv`.

### Findings

- **The headline gains hold up.** Across five seeds the final system beats the original by 46% (random), 12% (temporal) and 33–52% in cold start, significant for every seed in every setting.
- **Every component that was adopted helps somewhere, but less than the single-split numbers suggested:**
  - **The SVD + iALS ensemble** is the main driver, and both halves matter on full histories. At N = 10–20 the SVD half costs about 2.5%, which is significant in only 1–2 seeds.
  - **The fading popularity prior** gives +31% at N = 3 and +15% at N = 5, and does nothing elsewhere.
  - **Popularity gating**, inside the *final* system, only helps the temporal split (+1.7%). It is neutral elsewhere and costs exposure: coverage falls from 23% to 20% and the long-tail share halves. Its earlier +2–4% was measured on the SVD-only system.
- **Some earlier claims were noise:**
  - w_meta 0.65 vs 0.8, and γ = 0.15 vs 1.0, are indistinguishable. The README previously said γ = 0.15 was "optimal again"; that is not supported.
  - At N = 3 the hybrid is **tied** with most-popular (+2%, 1/5 seeds), not ahead. Its edge only becomes real from about N = 10.
  - At N = 10–20 the hybrid and iALS alone are tied (−1.5%, 0/5 seeds). Earlier I said iALS was slightly ahead.
- **The plot embeddings barely matter inside the hybrid.** They add 0.3% (random) and 0.7% (temporal) over metadata-only content, significant in only 2–3 of 5 seeds, and nothing measurable in user cold start. On their own they are a real signal: adding plots to metadata improves content-only recommendation by 8–24% in every seed. But the collaborative engines already capture what plots add. Metadata is the stronger content channel, and content as a whole adds 1.3% (random) and 3.4% (temporal). One case this protocol never tests is **new items with no ratings**, where content is the only signal. That is where the plots would have to prove their 11-minute cost.
- **The system is strongly popularity-biased.** It covers 20% of the catalogue, and only 0.6% of slots go to long-tail movies, against 4.1% for the original system. Gating and iALS are what shifted it. The optional penalty λ buys back coverage almost for free on the random split, but costs 1–4% on the temporal one, and it cannot reach real long-tail exposure (above about 4%) before accuracy collapses. Offline accuracy on MovieLens itself rewards popularity, because people mostly rate popular films, so these costs are probably overstated. Choosing λ is a product decision; the default stays at 0.
- **The demo still reacts slowly to the first rating.** With one rating, β(1) = 0.82, so a single rating barely changes the list. That is accuracy-optimal offline but feels unresponsive; genre picks, a higher η or the Discovery slider change it.

## Evaluated improvements (`suggestions.md`)

Each suggestion was implemented as an experiment and kept only if it improved held-out NDCG@10 without hurting another protocol. Hyperparameters were always chosen on validation splits. Numbers are test NDCG@10 unless marked *(val)*.

| # | Suggestion | Evidence | Decision |
| --- | --- | --- | --- |
| 1 | **Popularity-gated content**: S_content × log(1 + count) | Content-only: 0.025 → 0.152 (random), 0.014 → 0.070 (temporal). Hybrid: +2% (random), +4% (temporal). Better at every cold-start N ≥ 3. It cannot help at N = 0, where there is no content profile. **Re-check inside the final system over 5 seeds:** +1.7% temporal (5/5 seeds), no significant effect elsewhere, and coverage drops from 23% to 20%. | ✅ **Accepted**, with a smaller effect than first measured (`content_pop_gate`) |
| 2a | **iALS** instead of SVD | iALS 0.215 vs SVD 0.138 *(val)*. A hybrid built on iALS alone gains 41% on the random test (0.356 vs 0.252) but trails the gated SVD hybrid on temporal (0.116 vs 0.124); retuning on temporal validation only reaches parity (0.124). The 50/50 ensemble with SVD wins both: 0.367 random, 0.134 temporal. | ✅ **Accepted** as an SVD + iALS ensemble (`ials_weight`) |
| 2b | **BPR** | Best of 16 configurations 0.124 *(val)*, below SVD (0.138) and far below iALS (0.215). | ❌ Rejected |
| 3 | **Recency decay** w(t) = e^(−λΔt) | Day-based decay changes nothing (temporal 0.1238 → 0.1239). The median user rated everything within one day, and 66% of training ratings fall within a day of the user's last rating. Order-based decay (half-life in ratings) gains 1.8% on temporal but loses 3.2% on random. Decaying the SVD input hurts both splits, by 5–30%. | ❌ Rejected |
| 4 | **Two-stage retrieval + LightGBM LambdaRank** (top-100 candidates, 18 features) | Trained on random-style validation: +1.1% on random, −3.5% on temporal (0.134 → 0.129), mixed on cold start (−6% at N = 5). Trained on temporal validation: +8.6% on temporal (0.134 → 0.146), but the same ranker loses 33% on random (0.367 → 0.247) and 10–19% at every cold-start depth. The gains are tied to the protocol it was trained on, so they don't carry over. | ❌ Rejected |
| 5 | **Onboarding seeds** for N_u = 0 | In a simulation where each user's answers come from their hidden history: previous fallback (Bayesian average) 0.059 → most-popular 0.085 → most-popular × (1 + genre match) 0.096. A TF-IDF genre pseudo-profile (U_seed) scores 0.024. Three seed favourites through the hybrid's fold-in score 0.098; the app already supports this by rating movies. | ✅ **Accepted**: most-popular fallback and multiplicative genre boost. The TF-IDF U_seed variant was rejected. |
| F | **Follow-up: fading popularity prior** β(u) = e^(−η·N_u) (`experiments/exp7_fading_popularity.py`) | Tuned on validation: η = 0.2, γ = 0.15, raw counts beat log-counts. Test: N = 3 0.062 → 0.089 (+43%), N = 5 0.079 → 0.093 (+18%), N = 10 +1%, full histories unchanged. The proposed η = 0.4 with the then-current γ = 1.0 helps less (N = 3: 0.081, still below most-popular). **Over 5 seeds:** +31% at N = 3 and +15% at N = 5 (5/5 seeds each). The γ change is indistinguishable from γ = 1.0, so η is what matters. | ✅ **Accepted** (`pop_eta`) |
| 6 | **LightGCN** | A CPU implementation took 99 s per epoch on an idle machine (191 s under load). After 10 epochs it reached 0.166 *(val)*, against iALS at 0.215, which fits in about 2 s. Convergence would take hundreds of epochs, which is hours on this machine. It also has no way to embed a new user without retraining. It was not run to convergence. | ❌ Rejected (cost, plus no cold-start path) |
