# Hybrid Movie Recommender (MovieLens 1M + CMU Plot Summaries)

This project implements the system described in
[hybrid-movie-recommender-architecture.md](hybrid-movie-recommender-architecture.md), plus the improvements from `suggestions.md` that held up in testing. It has four parts:
- **Content engine:** TF-IDF metadata vectors plus mean-centred Sentence-BERT (`all-MiniLM-L6-v2`) embeddings of the Wikipedia plots. Scores are popularity-gated.
- **Collaborative engine:** an equal-weight ensemble of TruncatedSVD on mean-centred ratings and implicit ALS (iALS).
- **Blend:** the two scores are combined with a dynamic weight α(u) = α_max · (1 − e^(−γ·N_u)). A popularity prior carries new users and fades out as β(u) = e^(−η·N_u); it can be boosted by the genres a new user picks. A small popularity penalty (λ = 0.2) widens catalogue coverage.
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
8. **Long plots are chunked, and embeddings are centred.** MiniLM truncates input at 256 tokens, so plots are split into 150-word chunks, encoded, mean-pooled and L2-normalised. The embeddings are then mean-centred. Raw SBERT vectors sit in a narrow cone where every movie looks similar to an averaged profile; centring is what makes plots useful for new movies (see *Item cold start*).
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
| w_meta / w_plot | 0.5 / 0.5 (centred plots) |
| α_max | 0.7 |
| γ | 0.6 |
| η (popularity fade) | 0.2 |
| λ (popularity penalty) | 0.2, a product choice rather than a tuned value; see *Beyond accuracy* |

The five-seed audit below found that differences in w_meta and γ of this size are within noise, so the re-tune's shifts in them are not meaningful.

Relevance means a test rating of 4 or more. Seen items are excluded from recommendations, and every candidate is ranked against the full catalogue of 3,706 movies.

**NDCG@10, all users**

| Model | Random 80/20 | Temporal 80/20 |
| --- | ---: | ---: |
| Random | 0.0060 | 0.0050 |
| Average rating | 0.0001 | 0.0002 |
| Bayesian average | 0.0920 | 0.0525 |
| Content only (popularity-gated, centred plots) | 0.1653 | 0.0754 |
| Most popular | 0.1746 | 0.0918 |
| SVD only | 0.2375 | 0.1185 |
| iALS only | 0.3516 | 0.1107 |
| **Hybrid, deployed default (λ = 0.2)** | **0.3581** | **0.1311** |
| Hybrid, λ = 0 (accuracy-optimal) | 0.3604 | 0.1353 |
| *Hybrid before centring and re-tune (λ = 0)* | *0.3668* | *0.1335* |
| *Original hybrid (SVD, ungated content)* | *0.2520* | *0.1192* |

Centring plus the re-tune moved accuracy from the random split to the temporal one: −1.7% random and +1.3% temporal at λ = 0. Validation chose that trade-off, and it was not overridden using test results. The λ = 0.2 default then costs another 0.6% (random) and 3.1% (temporal) in exchange for +18–20% coverage.

The deployed hybrid leads every other model on precision, NDCG, MAP and MRR on both splits. iALS alone has slightly higher recall@10 on the random split (0.211 vs 0.206).

| Metric @10 | Deployed hybrid (random) | Original hybrid (random) | Deployed hybrid (temporal) | Original hybrid (temporal) |
| --- | ---: | ---: | ---: | ---: |
| Precision | 0.293 | 0.207 | 0.107 | 0.101 |
| Recall | 0.206 | 0.127 | 0.093 | 0.071 |
| MAP | 0.226 | 0.148 | 0.064 | 0.059 |
| MRR | 0.593 | 0.472 | 0.251 | 0.237 |

Full tables for K = 5, 10 and 20 are in `artifacts/eval_results.md`.

**Cold-start simulation (NDCG@10).** For 1,000 users, training history was cut down to their earliest N ratings, and every model was refitted:

| Model | N=0 | 3 | 5 | 10 | 20 | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Most popular | 0.085 | 0.088 | 0.091 | 0.097 | 0.103 | 0.165 |
| Content only (popularity-gated, centred plots) | 0.006 | 0.042 | 0.052 | 0.070 | 0.089 | 0.154 |
| SVD only | 0.006 | 0.062 | 0.071 | 0.093 | 0.114 | 0.223 |
| iALS only | 0.006 | 0.063 | 0.080 | **0.131** | **0.192** | 0.346 |
| **Hybrid, deployed default (λ = 0.2)** | **0.085** | 0.087 | 0.091 | 0.126 | 0.185 | **0.349** |
| Hybrid, λ = 0 | 0.085 | 0.088 | 0.091 | 0.125 | 0.185 | 0.353 |
| *Hybrid before centring and re-tune* | *0.085* | *0.089* | *0.093* | *0.128* | *0.186* | *0.359* |
| *Hybrid without the fading prior* | *0.085* | *0.062* | *0.079* | *0.127* | *0.186* | *0.358* |
| *Original hybrid (SVD, ungated content)* | *0.059* | *0.061* | *0.071* | *0.095* | *0.120* | *0.237* |

### Robustness across seeds

The system as it stood before plot centring, the re-tune and λ (raw plots, w_meta 0.65, α_max 0.8, γ 0.15, λ = 0) was re-checked with `experiments/exp8_robustness.py`:
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

These rows use raw plot embeddings. For existing users, centring them changes the hybrid by between −0.4% and +0.9% (`exp11_centred_plots_warm.py`).

#### Item cold start: new movies with zero ratings

The setup is `experiments/exp10_item_cold_start.py`, run over 5 seeds:
- **Held-out movies:** in each seed, 10% of the movies with at least 20 ratings (304 movies) lose 100% of their ratings from training.
- **Task:** each user ranks only those movies, using their history on the rest of the catalogue.
- **Collaborative filtering is blind here,** since the held-out movies have no interactions.

| Ranking zero-rating movies | NDCG@10 | vs metadata only |
| --- | ---: | ---: |
| Random | 0.040 | |
| Plots only, raw embeddings | 0.060 | −69% (5/5 worse) |
| Plots only, centred | 0.172 | |
| Metadata 0.35 + raw plots 0.65 (design doc) | 0.155 | −19% (5/5 worse) |
| Metadata 0.65 + raw plots 0.35 | 0.188 | −1.5% (3/5 worse) |
| Metadata only | 0.191 | |
| **Metadata 0.65 + centred plots 0.35** | **0.209** | **+9.2% (5/5 better)** |
| Deployed system before the fix (gated content + priors) | 0.043 | |

**Raw embeddings were the problem.** Raw SBERT plots are almost useless for new movies, and adding them to metadata hurts. Once the embeddings are mean-centred, plots plus metadata beat metadata alone by 9% in every seed, so the Sentence-BERT step earns its cost as a cold-start safeguard. Centring is now on (`center_plots`). The mix was fixed in advance at 0.65 / 0.35, not tuned on these held-out movies. The re-tuned production mix (0.5 / 0.5) has not been run through this test.

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
| **Deployed (centred, re-tuned), λ = 0** | 0.360 | 20.4% | 0.9% | 0.968 | 2.31 | 0.59 | 0.96 |
| **Deployed (centred, re-tuned), λ = 0.2 (default)** | 0.358 | 24.4% | 2.1% | 0.958 | 2.45 | 0.58 | 0.97 |

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
- **Plot embeddings matter for new movies, and only once they are centred.** For existing users they barely affect the hybrid (0.3–0.7% over metadata alone), because the collaborative engines already capture what they add. For movies with zero ratings, raw plot embeddings made things worse. Mean-centred, metadata plus plots beat metadata alone by **9.2% in every seed**, so the 11-minute Sentence-BERT step is justified as a cold-start safeguard.
- **Known defect: the deployed system never shows a new movie.** Popularity gating multiplies a zero-rating movie's content score by log(1 + 0) = 0, and the popularity prior and SVD/iALS also give it nothing. In the cold-start test, held-out movies got **0% of top-10 slots**, with or without λ. So the safeguard can rank new movies well (0.209 NDCG@10 among themselves), but they never reach the main list. Fixing this needs a product decision about how new movies should compete with established ones, for example a reserved slot or a separate *new for you* row ranked by ungated content. It is not fixed yet. The current MovieLens catalogue has no zero-rating movies, so the demo cannot show the problem.
- **The system is strongly popularity-biased.** At λ = 0 it covers 20% of the catalogue, and only 0.6–0.9% of slots go to long-tail movies, against 4.1% for the original system. Gating and iALS are what shifted it. The optional penalty λ buys back coverage almost for free on the random split, but costs 1–4% on the temporal one, and it cannot reach real long-tail exposure (above about 4%) before accuracy collapses. Offline accuracy on MovieLens itself rewards popularity, because people mostly rate popular films, so these costs are probably overstated. λ = 0.2 is now the default. It buys +18–20% coverage and roughly twice the long-tail share, for −0.6% NDCG@10 on the random split and −3.1% on the temporal one. The *Discovery* slider in the app ranges from 0 to 0.5.
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
| C | **Follow-up: item cold-start test** of the plot embeddings (`exp10`, `exp11`) | On zero-rating movies, raw plots hurt (−1.5% to −69% vs metadata), while mean-centred plots plus metadata gain +9.2% (5/5 seeds). For existing users, centring is neutral to +0.9% in the hybrid. Also found: the deployed system gives new movies 0% of top-10 slots. | ✅ **Accepted**: plots kept, now mean-centred (`center_plots`). New-movie exposure is an open defect |
| λ | **Follow-up: default popularity penalty λ = 0.2** (product choice) | +20% coverage on random and +18% on temporal; long-tail share 0.9% → 2.1% (random); NDCG@10 −0.6% (random) and −3.1% (temporal) vs λ = 0. | ✅ **Adopted** as the default (`pop_penalty`); slider range 0–0.5 |
| 6 | **LightGCN** | A CPU implementation took 99 s per epoch on an idle machine (191 s under load). After 10 epochs it reached 0.166 *(val)*, against iALS at 0.215, which fits in about 2 s. Convergence would take hundreds of epochs, which is hours on this machine. It also has no way to embed a new user without retraining. It was not run to convergence. | ❌ Rejected (cost, plus no cold-start path) |
