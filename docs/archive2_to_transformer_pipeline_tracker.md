# Archive-2 to Transformer Pipeline Tracker

Last updated: 2026-07-12

This file tracks the current research pipeline from the professor-provided
Archive-2 raw SPY option data through TP2 detection, episode tracking, labels,
ML datasets, LightGBM/MLP baselines, and the TP2-MV-ALiBi transformer.

The goal is reproducibility: every stage should have a clear input, script,
output, and research purpose.

## Current Research Scope

- Underlying: `SPY`
- Option type used for the main paper path: American-style call options
- Data period: 2013-2026
- Current paper-style label: expiration-payoff labels
- Current transformer strategy universe: `T1` and `K2`
- Current transformer input style: entry-time TP2 episode features plus
  historical TP2 episode tokens
- Current risk-control candidate: post-model fixed probability-floor family
  `0.10`, `0.20`, `0.30`

## High-Level Flow

```text
Archive-2 raw data
  -> cleaned last-quote SPY call dataset
  -> forward-adjusted TP2 rectangle construction
  -> conservative bid/ask TP2 episode starts
  -> episode tracking and lifecycle panel
  -> strategy labels / expiration payoff labels
  -> ML episode dataset
  -> rolling LightGBM feature discovery
  -> frozen transformer feature dictionary F*
  -> MLP flat-feature baseline
  -> TP2-MV-ALiBi transformer
  -> threshold/floor risk-control evaluation
  -> diagnostics, drift/calibration audit, ablations
```

## Stage 0: Raw Data

### Input

```text
data/raw/
Archive-2/
```

The raw option data itself may be stored outside git or in the untracked
Archive-2 folder. Do not manipulate or overwrite professor-provided raw files.

### Purpose

Keep the original professor-provided data untouched. All project data should be
derived into `data/processed`, `data/features`, `data/labels`, or `data/ml`.

### Main EDA Outputs

```text
reports/raw_archive2_eda/tables/raw_*.csv
reports/archive2_dataset_ingestion_report.pdf
```

Useful audit tables include:

- `raw_yearly_summary.csv`
- `raw_missingness_summary.csv`
- `raw_duplicate_summary.csv`
- `raw_quote_time_summary.csv`
- `raw_top_invalid_bidask_dates.csv`
- `raw_skipped_dates_presence_check.csv`

## Stage 1: Raw Ingestion and Clean Last-Quote Dataset

### Script

```bash
python scripts/run_ingest_archive2.py
```

### Main Outputs

```text
data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet
data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026_summary.json
```

### Purpose

Build the cleaned SPY call dataset using the last available quote convention.
This is the active cleaned call-option universe for downstream TP2 work.

### Important Conceptual Choice

We no longer discard quotes just because quote time is earlier than 15:55. The
working assumption is that Archive-2 provides last available quote data for that
contract/session. Early quote times are tracked as quote-quality information,
not automatically deleted at ingestion.

## Stage 2: Quote Availability Dataset Without Spread Deletion

### Script

```bash
python scripts/build_archive2_quote_availability.py
```

### Main Outputs

```text
data/processed/archive2_spy_call_quote_availability_nospread_2013_2026.parquet
data/processed/archive2_spy_call_quote_availability_nospread_2013_2026_summary.json
```

### Purpose

Preserve quote availability for lifecycle tracking. Wide-spread quotes should
not disappear before we can study whether a TP2 episode persists, reverts, or
becomes tradable later.

## Stage 3: Last-Quote Forward TP2 Feature Construction

### Script

```bash
python scripts/run_detect_archive2_lastquote_by_year.py
```

### Main Outputs

```text
data/features/archive2_lastquote_forward/tp2_call_forward_features_lastquote_2013_2026.parquet
data/features/archive2_lastquote_forward/tp2_call_forward_summary_lastquote_2013_2026.json
```

### Purpose

Construct TP2 option-price rectangles using forward-adjusted strikes/maturities.
This is where the Glasserman-style TP2 inequality is evaluated on SPY calls.

### TP2 Condition Reminder

For a rectangle with two strikes and two expirations, we compare:

```text
A * B < C * D
```

where:

- `A = C(K1, T1)`
- `B = C(K2, T2)`
- `C = C(K1 adjusted to T2, T2)`
- `D = C(K2 adjusted to T1, T1)`

A violation means the right-hand product is overpriced relative to the
left-hand product under the TP2 ordering.

## Stage 4: Conservative Bid/Ask TP2 Subset

### Script

```bash
python scripts/build_conservative_bidask_subset.py
```

### Main Outputs

```text
data/labels/archive2_episode_start_labels_conservative_bidask_2013_2026.parquet
data/labels/archive2_episode_start_labels_conservative_bidask_summary_2013_2026.json
reports/raw_archive2_eda/tables/conservative_bidask_subset/
```

### Purpose

Move from mid-price violations to more conservative tradable violations using
bid/ask logic. This avoids treating every mid-price inequality as executable.

### Current Paper Path

The transformer pipeline uses conservative TP2 episode starts and then focuses
on `T1` and `K2` strategy labels.

## Stage 5: Episode Tracking

### Scripts

```bash
python scripts/run_build_episodes.py
python scripts/run_classify_episode_end_states.py
```

### Main Outputs

```text
data/features/archive2_episodes/tp2_call_pair_episodes_2013_2026.parquet
data/features/archive2_episodes/tp2_call_episode_members_2013_2026.parquet
data/features/archive2_episodes/tp2_call_pair_episode_end_states_2013_2026.parquet
data/features/archive2_episodes/tp2_call_pair_episode_end_state_summary_2013_2026.json
```

### Purpose

Track the same economic TP2 rectangle across dates instead of counting the same
persistent violation as unrelated daily events. This is required to study
reversion, persistence, expiration, and quote availability.

### Key Intuition

If the same TP2 rectangle is violated on consecutive dates, it is one episode,
not multiple independent discoveries.

## Stage 6: Lifecycle Panel

### Script

```bash
python scripts/build_violation_lifecycle_panel.py
```

### Main Outputs

```text
data/features/archive2_lifecycle/tp2_conservative_episode_lifecycle_2013_2026.parquet
data/features/archive2_lifecycle/tp2_conservative_episode_lifecycle_summary_2013_2026.json
reports/lifecycle_tracking_report.pdf
reports/raw_archive2_eda/tables/lifecycle_tracking/
```

### Purpose

Track each conservative TP2 episode from a pre-event lookback window through
`T1` expiration. This lets us study:

- pressure before violation;
- whether the violation reverts;
- when it becomes tradable;
- whether quote/contract availability disappears;
- whether lifecycle information can become an ML feature without leakage.

### Leakage Rule

Only lifecycle information available at or before the signal date can enter ML
features. Future reversion information is label/outcome information, not input.

## Stage 7: Episode Start Labels and Strategy Backtests

### Scripts

```bash
python scripts/run_build_episode_start_labels.py
python scripts/run_episode_mini_backtest.py
python scripts/run_episode_mini_backtest_unit_sized.py
```

### Main Outputs

```text
data/labels/archive2_episode_start_labels_2013_2026.parquet
data/labels/archive2_episode_strategy_mini_backtest_2013_2026.parquet
data/labels/archive2_episode_strategy_mini_backtest_conservative_bidask_2013_2026.parquet
data/labels/archive2_episode_strategy_mini_backtest_unit_conservative_bidask_2013_2026.parquet
```

### Purpose

Attach strategy-level outcomes to TP2 episode starts. Earlier versions included
multiple strategy/backtest variants; the current transformer path uses
expiration-payoff labels with approximately normalized exposure.

## Stage 8: Liquid-Exit Labels and Exit-Policy Comparison

### Script

```bash
python scripts/build_liquid_exit_labels.py
python scripts/generate_exit_policy_comparison_report.py
```

### Main Outputs

```text
data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet
data/labels/archive2_liquid_exit_strategy_labels_summary_2013_2026.json
reports/tp2_exit_policy_comparison_report.pdf
reports/raw_archive2_eda/tables/exit_policy_comparison/
```

### Purpose

Compare first-liquid exit and expiration-payoff accounting. For the current
paper-style transformer path, expiration payoff is the main label because it is
closest to the Glasserman-style strategy evaluation and is more stable in this
dataset.

### Important Distinction

- First-liquid exit: execution-realistic, but much weaker in this dataset.
- Expiration payoff: paper-style label, not a live intraday strategy claim.

## Stage 9: ML Episode Dataset

### Script

```bash
python scripts/build_ml_dataset.py
```

### Main Outputs

```text
data/ml/archive2_ml_episode_dataset_2013_2026.parquet
data/ml/archive2_ml_episode_dataset_summary_2013_2026.json
```

### Purpose

Create the leakage-safe per-episode ML table. Features include entry-time
option-surface information, liquidity, moneyness, DTE, severity, and
pre-signal lifecycle summaries.

### Feature Leakage Rule

No future reversion date, exit date, realized P&L, post-signal lifecycle row, or
future label column may be used as an input feature.

## Stage 10: Rolling LightGBM Baseline

### Script

```bash
python scripts/run_ml_rolling_lightgbm.py
python scripts/generate_ml_lightgbm_report.py
```

### Main Outputs

```text
data/ml/archive2_ml_lightgbm_rolling_predictions*.parquet
data/ml/archive2_ml_lightgbm_rolling_summary*.csv
reports/ml_lightgbm_rolling_report.pdf
reports/raw_archive2_eda/tables/ml_lightgbm/
```

### Purpose

Provide a leakage-safe rolling baseline and feature-discovery layer. LightGBM
is not the final model for the transformer paper path; it helps identify stable
features and benchmark flat-feature performance.

### Split Logic

Rolling walk-forward style:

```text
train years -> validation year -> test year
```

Test-year P&L is evaluated once and is not used for threshold or feature
selection.

## Stage 11: Transformer Feature Dictionary

### Script

```bash
python TP2MVTransformer/01_build_feature_dictionary.py
```

### Main Outputs

```text
TP2MVTransformer/data/tp2_t1k2_token_dataset.parquet
TP2MVTransformer/tables/feature_dictionary.csv
TP2MVTransformer/tables/fold_feature_importance.csv
TP2MVTransformer/tables/build_summary.json
```

### Purpose

Freeze the transformer feature dictionary `F*`. LightGBM is used only for
feature discovery/stability scoring. Booster predictions, leaves, SHAP vectors,
and scores do not enter the transformer.

### Current Feature Count

The current transformer feature dictionary selects 48 features.

## Stage 12: MLP Flat-Feature Baseline

### Script

```bash
python TP2MVTransformer/02_train_mlp_baseline.py
```

### Main Outputs

```text
TP2MVTransformer/tables/mlp_baseline_results.csv
TP2MVTransformer/tables/mlp_baseline_summary.json
TP2MVTransformer/data/mlp_baseline_test_predictions.parquet
```

### Purpose

Test whether the selected `F*` features contain signal without historical
attention.

### Historical Information Clarification

MLP and LightGBM use some historical summaries as flat columns, such as
violation age, one-day severity change, and pre-signal lifecycle aggregates.
They do not use transformer-style cross-episode historical attention.

## Stage 13: TP2-MV-ALiBi Transformer

### Main Script

```bash
python TP2MVTransformer/03_train_mv_alibi_transformer.py
```

### Legacy No-History Reproduction

```bash
python TP2MVTransformer/03_train_mv_alibi_transformer_legacy_no_history.py
```

Use the legacy script only for ablation/sanity comparison. It reproduces the
old date-conversion behavior that effectively removed historical attention.

### Main Outputs

```text
TP2MVTransformer/tables/mv_alibi_transformer_v2_results.csv
TP2MVTransformer/tables/mv_alibi_v2_yearly_diagnostics.csv
TP2MVTransformer/tables/mv_alibi_v2_training_history.csv
TP2MVTransformer/tables/mv_alibi_v2_distance_diagnostics.csv
TP2MVTransformer/tables/mv_alibi_v2_attention_parameters.csv
TP2MVTransformer/data/mv_alibi_transformer_v2_test_predictions.parquet
TP2MVTransformer/data/mv_alibi_transformer_v2_val_predictions.parquet
TP2MVTransformer/data/mv_alibi_transformer_v2_selected_trades.parquet
```

### Purpose

Score each current TP2 episode/strategy using:

- current episode features;
- up to 64 historical TP2 episode tokens;
- MV-ALiBi distances: time, surface, short-D-leg, liquidity, lifecycle;
- validation-only threshold tuning.

### Current Known Issue

The transformer learns training-year patterns, but validation loss can worsen
in later folds. This points to probability calibration/generalization instability,
not necessarily zero learning.

## Stage 14: Transformer Diagnostics

### Scripts

```bash
python TP2MVTransformer/05_make_transformer_diagnostics_report.py
python TP2MVTransformer/08_make_professor_update_report.py
```

### Main Outputs

```text
TP2MVTransformer/transformer_diagnostics_report.pdf
TP2MVTransformer/professor_transformer_update_report.pdf
```

### Purpose

Document:

- training loss and validation loss;
- gradient norms;
- validation/test AUC;
- Sortino and drawdown;
- MV-ALiBi distance scale diagnostics;
- learned attention slopes;
- raw transformer over-selection issue.

## Stage 15: Post-Model Risk-Control Floors

### Script

```bash
python TP2MVTransformer/07_conservative_transformer_floor_selection.py
```

### Main Outputs

```text
TP2MVTransformer/tables/conservative_transformer_floor_summary_by_floor.csv
TP2MVTransformer/tables/conservative_transformer_floor_yearly.csv
TP2MVTransformer/conservative_transformer_floor_report.pdf
```

### Purpose

Evaluate fixed post-model probability floors:

```text
0.10, 0.20, 0.30
```

The floor is not inside the transformer. It is an execution/risk-control rule
applied after the transformer outputs probabilities.

### Threshold Logic

For floor `0.20`, validation searches:

```text
K = 0.20, 0.225, 0.25, ..., 0.95
```

and then applies the chosen `K` once to the test year. Test P&L is not used to
choose `K`.

## Stage 16: Transformer Training Audit and Ablations

### Scripts

```bash
python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py
python TP2MVTransformer/10_transformer_drift_calibration_audit.py
python TP2MVTransformer/11_summarize_transformer_ablation_runs.py
```

### Main Outputs

```text
TP2MVTransformer/ablation_runs/<tag>/
TP2MVTransformer/ablation_runs/_audit/tables/
TP2MVTransformer/ablation_runs/ablation_run_comparison.csv
```

### Purpose

Implement professor's training checklist without overwriting the main model:

- best validation-loss checkpointing;
- learning-rate sweep;
- warm-up / scheduler tests;
- gradient clipping;
- hidden size / attention head / history length ablations;
- dropout and weight decay ablations;
- distance-bias and strategy-bonus ablations;
- seed stability checks;
- feature drift and target drift audit;
- Brier score / calibration audit.

### Colab Command Sheet

```text
TP2MVTransformer/transformer_ablation_colab_commands.md
```

Use this for parallel GPU runs.

## Reports to Show Professor

Main technical reports:

```text
reports/archive2_dataset_ingestion_report.pdf
reports/lifecycle_tracking_report.pdf
reports/tp2_exit_policy_comparison_report.pdf
reports/ml_lightgbm_rolling_report.pdf
TP2MVTransformer/transformer_diagnostics_report.pdf
TP2MVTransformer/conservative_transformer_floor_report.pdf
TP2MVTransformer/professor_transformer_update_report.pdf
ProfessorMeetingScript/transformer_update_meeting_script.pdf
```

## Current Open Issues

1. The raw transformer often chooses very low probability thresholds such as
   `K = 0.05`, causing over-selection.
2. Validation loss worsens in several later folds even when training loss
   improves, indicating overfitting or regime instability.
3. Label-rate drift is large across years, especially around 2020-2022 and
   2024-2025.
4. The probability floor improves robustness, but it must be presented as
   post-model risk control, not as the transformer itself solving the problem.
5. The transformer ablation suite must be run and summarized before claiming
   architecture-level improvement.

## Minimal Rebuild Order

For a full rebuild from cleaned Archive-2 outputs:

```bash
python scripts/run_ingest_archive2.py
python scripts/build_archive2_quote_availability.py
python scripts/run_detect_archive2_lastquote_by_year.py
python scripts/build_conservative_bidask_subset.py
python scripts/run_build_episodes.py
python scripts/run_classify_episode_end_states.py
python scripts/build_violation_lifecycle_panel.py
python scripts/run_build_episode_start_labels.py
python scripts/run_episode_mini_backtest.py
python scripts/run_episode_mini_backtest_unit_sized.py
python scripts/build_liquid_exit_labels.py
python scripts/build_ml_dataset.py
python scripts/run_ml_rolling_lightgbm.py
python TP2MVTransformer/01_build_feature_dictionary.py
python TP2MVTransformer/02_train_mlp_baseline.py
python TP2MVTransformer/03_train_mv_alibi_transformer.py
python TP2MVTransformer/05_make_transformer_diagnostics_report.py
python TP2MVTransformer/07_conservative_transformer_floor_selection.py
python TP2MVTransformer/10_transformer_drift_calibration_audit.py
```

For current transformer ablations, use Colab/GPU:

```bash
python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag best_loss_lr3e4 --checkpoint-metric val_loss --scheduler warmup_cosine --lr 3e-4 --clip-norm 1.0
```

Then summarize:

```bash
python TP2MVTransformer/11_summarize_transformer_ablation_runs.py
```

## Do Not Overwrite

- Do not edit raw professor data.
- Do not replace the main transformer outputs with ablation outputs.
- Do not use test-year P&L to choose a threshold, architecture, or floor.
- Do not present the probability floor as part of the neural architecture.

