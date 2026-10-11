# Experiments

These scripts test each proposal in `suggestions.md`, plus a follow-up fix for the cold-start gap. Each change was adopted only if it improved held-out NDCG@10 without hurting another evaluation protocol. The decisions and headline numbers are summarised in the main [README](../README.md#evaluated-improvements-suggestionsmd).

## Running

Run everything from the repository root after the data steps:

```bash
python -m recsys.cli extract && python -m recsys.cli align && python -m recsys.cli embed
python experiments/exp1_popgate.py
```

Each script writes its tables to `experiments/results/`. `experiments/logs/` holds the stdout from the last full re-run.

## Protocol

The protocol matches `recsys.evaluation`:
- **Splits:** random and temporal 80/20 splits per user.
- **Relevance:** a test rating of 4 or more.
- **Ranking:** previously seen items are excluded, and every candidate is ranked against all 3,706 movies.
- **Hyperparameter selection:** always on validation data carved out of the training split, never on test.
  - *Random-style validation* is a random 80/20 split of the random-split train.
  - *Temporal-style validation* takes the last 20% of each user's ratings in the temporal-split train.
- **Cold start:** simulated by cutting 1,000 sampled users' training history down to their earliest N ratings and refitting every model.

`exp_common.CFG` is pinned to the **baseline the experiments started from**: SVD only (k = 32), ungated content with w_meta = 0.5, α_max = 0.8 and γ = 1.0. So the scripts still reproduce their original numbers after the project defaults changed. `exp7` pins its own reference instead: the system after suggestions 1, 2 and 5 were adopted.

## Scripts

| Script | Tests | Main output | Outcome |
| --- | --- | --- | --- |
| `exp1_popgate.py` | **S1:** popularity-gated content, S_content × log(1 + count). Run on both splits and at cold-start N = 3–20, for γ ∈ {0.15, 1.0} | `exp1.csv` | ✅ Content-only 0.025 → 0.152; hybrid +2% (random), +4% (temporal) |
| `exp2_ranking_mf.py tune` | **S2:** iALS and BPR grids (`implicit`) against SVD, on random-style validation | `exp2_tune.csv` | iALS 0.203 vs SVD 0.138; BPR at most 0.124 ❌ |
| `exp2_ranking_mf.py refine` | Second, finer iALS grid (factors, reg, confidence, iterations) | `exp2_refine.csv` | Plateau around 0.215 with 24–32 factors |
| `exp2_hybrid_tune.py` | Blend weights (w_meta, α_max) and γ for SVD- and iALS-based hybrids, on validation | `exp2_hybrid_s{1,2}.csv` | iALS hybrid 0.221 vs SVD hybrid 0.162 *(val)* |
| `exp2_temporal.py` | Whether iALS's temporal-test loss is a tuning artefact: temporal-validation tuning, plus an iALS/SVD ensemble | stdout / log | iALS alone only reaches parity; the 50/50 ensemble scores 0.134 on temporal test |
| `exp2_ensemble_select.py` | iALS/SVD mix and α_max, chosen on **both** validation splits | `exp2_ensemble_select.csv` | ✅ β = 0.5, α_max = 0.8 |
| `exp3_recency.py` | **S3:** recency decay on the content profile and on the SVD input, by days or by rating order | `exp3.csv` | ❌ Day-based has no effect; order-based helps temporal (+1.8%) but hurts random (−3.2%) |
| `exp4_ltr.py` | **S4:** top-100 candidate retrieval + LightGBM LambdaRank (18 features). Each ranker is also transferred to the other split and to cold start | `exp4.csv` | ❌ Each ranker gains only on its own protocol: the random-trained one +1.1% random and −3.5% temporal; the temporal-trained one +8.6% temporal and −33% random |
| `exp5_onboarding.py` | **S5:** N = 0 onboarding, with answers simulated from hidden history: popularity fallbacks, genre blends, a TF-IDF genre pseudo-profile, three seed favourites | `exp5.csv` | ✅ Most-popular × (1 + genre match) scores 0.096, vs 0.059 for the old fallback |
| `exp6_lightgcn.py` | **S6:** LightGCN with BPR loss on CPU. Arguments: `<val\|random\|temporal> <all\|pos> [max_epochs] [eval_every]` | `lightgcn_*.npz` | ❌ About 100 s per epoch on an idle CPU; 0.166 after 10 epochs (`logs/exp6_lightgcn_val_all_10epochs_original.log`) vs iALS 0.215 *(val)*. Not run to convergence |
| `exp_final_test.py` | Test-set comparison: baseline, then + gating, iALS alone, then + iALS + gating | `final_test.csv` | Basis for adopting S1 and S2 |
| `exp7_fading_popularity.py` | **Follow-up:** fading popularity prior β(u) = e^(−η·N_u), with η, γ and S_pop tuned on truncated validation histories | `exp7_{val,test}.csv` | ✅ N = 3: 0.062 → 0.089 (+43%); N = 5: 0.079 → 0.093; full histories unchanged |

Helper modules:
- `exp_common.py`: data, splits, evaluation harness and the pinned baseline config.
- `exp_models.py`: base engines (SVD, iALS, gated content, popularity) shared by `exp2_ensemble_select` and `exp4`.

## Notes

- **LightGCN is slow.** At about 100 s per epoch on this machine's CPU (190 s when sharing it with other jobs), a converged LightGCN run would take many hours, so the reported result is an early checkpoint.
- **Runtime.** A full re-run of the other scripts takes about 40 minutes on CPU.
