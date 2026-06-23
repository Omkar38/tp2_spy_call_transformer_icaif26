# Report Index

This file lists the reports that should be shown first during professor handoff.
PDFs are the readable deliverables; `.tex` files are kept for reproducibility
when a report was generated from LaTeX.

## Read First

1. `reports/lifecycle_tracking_report.pdf`
   - Explains episode/lifecycle tracking and how repeated daily violations are
     converted into event-level observations.
2. `reports/liquid_exit_rebuild_report.pdf`
   - Explains the first-liquid tradable exit rebuild and why liquidity is
     enforced at entry/exit rather than deleting wide-spread observations before
     lifecycle tracking.
3. `reports/ml_lightgbm_rolling_report.pdf`
   - Main non-Transformer rolling LightGBM result using `skip/T1/T2/K1/K2`.
4. `reports/ml_quality_gate_comparison_report.pdf`
   - Compares base, volume+delta, and strict market-quality ML variants.
5. `reports/profitability_stress_report.pdf`
   - Shows commission robustness and tail-concentration risk.

## Supporting Reports

- `reports/archive2_dataset_ingestion_report.pdf`
  - Dataset ingestion, clean sample, and historical pipeline notes.
- `reports/raw_archive2_eda/archive2_raw_data_eda_report.pdf`
  - Raw Archive-2 EDA.
- `reports/strategy_exit_robustness_report.pdf`
  - Exit-policy and strategy robustness.
- `reports/market_quality_robustness_report.pdf`
  - Market-quality gate diagnostics before ML.
- `reports/conservative_bidask_subset_report.pdf`
  - Conservative bid/ask TP2 subset accounting.
- `reports/episode_end_state_classification_report.pdf`
  - Episode end-state definitions and counts.
- `reports/episode_start_label_dataset_report.pdf`
  - Episode-start label construction.
- `reports/episode_mini_backtest_report.pdf`
  - Earlier mini-backtest diagnostics.
- `reports/episode_tracking_explanation_report.pdf`
  - Plain-language episode tracking explanation.
- `reports/pre_ml_gating_and_rolling_plan.pdf`
  - Earlier pre-ML gating plan and diagnostics.

## Local-Only Data

Raw and generated parquet files are intentionally not tracked:

```text
Archive-2/spy_options_daily/year=*/data.parquet
data/**/*.parquet
reports/**/*.parquet
```

Small summary JSON/CSV tables and figures are tracked so the reports can be
audited without committing the raw dataset.
