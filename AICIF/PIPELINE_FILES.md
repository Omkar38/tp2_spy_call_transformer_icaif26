# Pipeline Files (Start to End)

Every file below sits on the actual data/analysis lineage behind the numbers, tables,
and figures in `tp2_spy_call_acm_draft.tex`. Diagnostic, ablation, and superseded
scripts that are *not* upstream of anything in the paper are intentionally left out.
Paths are relative to the repo root.

## Stage 0 — Raw Data
- `Archive-2/` — professor-provided raw SPY option data (untouched, not in git).

## Stage 1 — Ingestion and Cleaning
- `src/tp2rr2/archive2_ingest.py` — ingestion/cleaning logic.
- `scripts/run_ingest_archive2.py` — runs Stage 1.
  → `data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet`

## Stage 2 — Quote Availability (No Spread Deletion)
- `scripts/build_archive2_quote_availability.py`
  → `data/processed/archive2_spy_call_quote_availability_nospread_2013_2026.parquet`

## Stage 3 — Forward-Adjusted TP2 Rectangle Detection
- `src/tp2rr2/detect_forward_tp2.py`, `src/tp2rr2/detect_violations.py`, `src/tp2rr2/features.py`
- `scripts/run_detect_archive2_lastquote_by_year.py` — runs Stage 3.
  → `data/features/archive2_lastquote_forward/tp2_call_forward_features_lastquote_2013_2026.parquet`

## Stage 4 — Conservative Bid/Ask TP2 Subset
- `src/tp2rr2/conservative_bidask.py`
- `scripts/build_conservative_bidask_subset.py`
  → `data/labels/archive2_episode_start_labels_conservative_bidask_2013_2026.parquet`

## Stage 5 — Episode Tracking (Section 3.3 of the paper)
- `src/tp2rr2/episodes.py` — groups consecutive daily rectangle rows into pair episodes.
- `src/tp2rr2/episode_end_states.py` — classifies each episode's end state (true reversion, expiry, liquidity failure, ...).
- `scripts/run_build_episodes.py`, `scripts/run_classify_episode_end_states.py`
  → `data/features/archive2_episodes/tp2_call_pair_episodes_2013_2026.parquet`
  → `data/features/archive2_episodes/tp2_call_pair_episode_end_states_2013_2026.parquet`
- `reports/episode_tracking_explanation_report.tex` — episode-count/duration write-up the Section 3.3 prose statistics are drawn from (2,430,220 daily rows → 2,098,213 pair episodes, 88.66% single-session, 50.6% true reversion).

## Stage 6 — Episode Lifecycle Panel (Section 3.3 figure)
- `scripts/build_violation_lifecycle_panel.py` — tracks each conservative TP2 episode across a fixed event-time window (20-session pre-event lookback through expiration).
  → `data/features/archive2_lifecycle/tp2_conservative_episode_lifecycle_2013_2026.parquet`
- `scripts/generate_lifecycle_tracking_assets.py` — external analysis that produced the lifecycle-path figure used directly in the paper (Figure 1).
  → `reports/figures/lifecycle_normalized_pressure_band.png`
- `scripts/generate_lifecycle_latex_report.py` — write-up this analysis is drawn from.
  → `reports/lifecycle_tracking_report.tex` / `.pdf`

## Stage 7 — Episode Start Labels and Mini-Backtest
- `src/tp2rr2/episode_start_labels.py`, `src/tp2rr2/strategies.py`
- `src/tp2rr2/episode_mini_backtest.py`, `src/tp2rr2/episode_mini_backtest_unit.py`
- `scripts/run_build_episode_start_labels.py`, `scripts/run_episode_mini_backtest.py`, `scripts/run_episode_mini_backtest_unit_sized.py`
  → feeds Stage 6's lifecycle panel (unit-sized conservative mini-backtest).

## Stage 8 — Cost-Aware Labels (Section 3.4 of the paper)
- `scripts/build_liquid_exit_labels.py` — builds the expiration-payoff labels used throughout the paper, with entry-side bid/ask spread + $0.65/contract commission baked in. (This script also builds an alternative first-liquid exit label as a byproduct; that label policy and its comparison report, `scripts/generate_exit_policy_comparison_report.py` / `reports/tp2_exit_policy_comparison_report.pdf`, are not part of the paper's current direction and are excluded from the results below.)
  → `data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet`

## Stage 9 — ML Episode Dataset
- `scripts/build_ml_dataset.py`
  → `data/ml/archive2_ml_episode_dataset_2013_2026.parquet`

## Stage 10 — Feature Dictionary and Transformer Token Dataset (Section 4.2)
- `TP2MVTransformer/01_build_feature_dictionary.py` — LightGBM-driven feature discovery, 149 → 48 features ($F^{\ast}$); builds the shared T1/K2 token dataset used by the MLP and transformer.
  → `TP2MVTransformer/data/tp2_t1k2_token_dataset.parquet`
  → `TP2MVTransformer/tables/feature_dictionary.csv`

## Stage 11 — Final Models (Section 4.3–4.5, Section 5)
- `TP2MVTransformer/27_train_lightgbm_final.py` — LightGBM, all 4 strategies + T1/K2 subset.
- `TP2MVTransformer/28_train_mlp_final.py` — MLP, single seed.
- `TP2MVTransformer/34_mlp_5seed_ensemble.py` — MLP, 5-seed ensemble (Section 5.3).
- `TP2MVTransformer/23_train_mv_alibi_transformer_v3_final.py` — Transformer, single seed.
- `TP2MVTransformer/24_train_mv_alibi_transformer_v4_final.py` — Transformer, 5-seed ensemble (shipped model).
- `TP2MVTransformer/35_transformer_10seed_final.py` — Transformer, 10-seed ensemble (Section 5.3).

## Stage 12 — Supporting Analysis Cited in the Paper
- `TP2MVTransformer/20_training_window_sensitivity.py`, `TP2MVTransformer/32_expanding_window_remaining_folds.py` — training-window-length sensitivity (Section 5.5).
- `TP2MVTransformer/33_v4_mvalibi_diagnostics.py` — learned MV-ALiBi attention-slope diagnostics (Section 4.5).
- `TP2MVTransformer/29_make_model_comparison_report.py` — assembles the main model-comparison table and the T1/K2 structural-prior check, all 6 years (Section 5.2).
- `TP2MVTransformer/30_make_professor_update_2026_07_14.py` — paired-bootstrap/sign-test significance testing (Section 5.3) and the SPX benchmark comparison (Section 5.6).

## Stage 13 — Final Report-Generation Artifacts (Reproducibility Trace)
Every number, table, and figure in the paper traces to one of these generated report
artifacts (all produced by the Stage 11/12 scripts above; listed here so each is
independently locatable rather than only implied by the script that made it):
- `TP2MVTransformer/tables/lgbm_final_summary.json`, `TP2MVTransformer/tables/lgbm_final_yearly_diagnostics.csv` — LightGBM final results (all 4 strategies + T1/K2 subset), Table 1, Section 5.2.
- `TP2MVTransformer/tables/mlp_final_results.csv` — MLP single-seed yearly results, Tables 1 and 4.
- `TP2MVTransformer/tables/mlp_5seed_ensemble_results.csv`, `TP2MVTransformer/tables/mlp_5seed_ensemble_summary.json` — MLP 5-seed ensemble, Table 3 (Section 5.3).
- `TP2MVTransformer/tables/mv_alibi_transformer_v3final_results.csv`, `TP2MVTransformer/tables/mv_alibi_transformer_v3final_summary.json` — transformer single-seed final results, Tables 1, 3, 4.
- `TP2MVTransformer/tables/mv_alibi_transformer_v4final_results.csv`, `TP2MVTransformer/tables/mv_alibi_transformer_v4final_summary.json` — transformer 5-seed ensemble final results (shipped model), Tables 1, 3, 4.
- `TP2MVTransformer/tables/mv_alibi_transformer_10seedfinal_summary.json` — transformer 10-seed ensemble, Table 3 (Section 5.3).
- `TP2MVTransformer/tables/mv_alibi_v4_seed_attention_parameters.csv` — learned attention-slope numbers cited in Section 4.5.
- `TP2MVTransformer/model_comparison_report.pdf` — rendered output of Stage 12's `29_make_model_comparison_report.py`; source of the Table 1 and Section 5.2 numbers.
- `TP2MVTransformer/professor_update_2026-07-14.pdf` — rendered output of Stage 12's `30_make_professor_update_2026_07_14.py`; source of the Section 5.3 significance-test p-values and the Section 5.6 SPX benchmark table.

## Stage 14 — Paper
- `AICIF/tp2_spy_call_acm_draft.tex`
- `AICIF/figures/lifecycle_normalized_pressure_band.png` — copied from Stage 6's output for direct inclusion in the paper.
