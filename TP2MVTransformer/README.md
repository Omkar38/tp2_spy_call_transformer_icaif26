# TP2-MV-ALiBi Transformer Build

This folder implements the professor-proposed transformer layer on top of the
corrected TP2 expiry pipeline.

Current scope:

1. Build a T1/K2-only episode-strategy token dataset from the corrected expiry
   labels.
2. Use rolling LightGBM only for feature discovery and stability scoring.
3. Save a frozen feature dictionary `F_star` for the transformer.
4. Train an MLP ablation on `F_star`.
5. Train the first TP2-MV-ALiBi cross-attention transformer with historical
   TP2 token context.
6. Generate a LaTeX/PDF implementation report with comparison tables and
   figures.

## Inputs

- `data/ml/archive2_ml_episode_dataset_2013_2026.parquet`
- `data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet`

## Outputs

- `TP2MVTransformer/data/tp2_t1k2_token_dataset.parquet`
- `TP2MVTransformer/tables/feature_dictionary.csv`
- `TP2MVTransformer/tables/fold_feature_importance.csv`
- `TP2MVTransformer/tables/build_summary.json`
- `TP2MVTransformer/tables/mlp_baseline_results.csv`
- `TP2MVTransformer/tables/mv_alibi_transformer_results.csv`
- `TP2MVTransformer/data/mv_alibi_transformer_test_predictions.parquet`
- `TP2MVTransformer/tables/mv_alibi_transformer_v2_results.csv`
- `TP2MVTransformer/data/mv_alibi_transformer_v2_test_predictions.parquet`
- `TP2MVTransformer/data/mv_alibi_transformer_v2_val_predictions.parquet`
- `TP2MVTransformer/data/mv_alibi_transformer_v2_selected_trades.parquet`
- `TP2MVTransformer/tables/mv_alibi_transformer_v2_selected_trades.csv`
- `TP2MVTransformer/tables/mv_alibi_transformer_v2_selected_daily_pnl.csv`
- `TP2MVTransformer/transformer_implementation_report.pdf`

## Run Order

```bash
python TP2MVTransformer/01_build_feature_dictionary.py
python TP2MVTransformer/02_train_mlp_baseline.py
python TP2MVTransformer/03_train_mv_alibi_transformer.py
python TP2MVTransformer/04_make_transformer_report.py
python TP2MVTransformer/05_make_transformer_diagnostics_report.py
python TP2MVTransformer/06_risk_gated_transformer_selection.py
python TP2MVTransformer/07_conservative_transformer_floor_selection.py
python TP2MVTransformer/10_transformer_drift_calibration_audit.py
cd TP2MVTransformer
pdflatex -interaction=nonstopmode -halt-on-error transformer_implementation_report.tex
pdflatex -interaction=nonstopmode -halt-on-error transformer_implementation_report.tex
pdflatex -interaction=nonstopmode -halt-on-error transformer_diagnostics_report.tex
pdflatex -interaction=nonstopmode -halt-on-error transformer_diagnostics_report.tex
pdflatex -interaction=nonstopmode -halt-on-error conservative_transformer_floor_report.tex
pdflatex -interaction=nonstopmode -halt-on-error conservative_transformer_floor_report.tex
```

The v2 transformer script uses CUDA, then Apple MPS, then CPU in priority order.
On CPU-only machines it can take several hours. The v2 run writes separate
`*_v2_*` outputs so the earlier transformer result files are preserved.

## Professor Training-Audit Ablations

Professor's transformer-training feedback is implemented as a separate
experiment layer so the main `03_train_mv_alibi_transformer.py` result remains
reproducible.

Use this configurable runner for GPU ablations:

```bash
python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py \
  --tag best_loss_lr3e4 \
  --checkpoint-metric val_loss \
  --scheduler warmup_cosine \
  --lr 3e-4 \
  --clip-norm 1.0
```

Each run writes to:

```text
TP2MVTransformer/ablation_runs/<tag>/
```

The runner supports:

- best validation-loss versus best validation-AUC checkpointing;
- learning-rate sweep;
- AdamW weight-decay sweep;
- warm-up plus cosine schedule or plateau schedule;
- gradient clipping;
- hidden dimension / attention-head / history-length ablations;
- dropout and random-seed ablations;
- distance-bias and strategy-bonus ablations;
- Brier score and expected calibration error.

Use this audit script for target drift, feature drift, and calibration tables:

```bash
python TP2MVTransformer/10_transformer_drift_calibration_audit.py
```

After several ablations finish, summarize them with:

```bash
python TP2MVTransformer/11_summarize_transformer_ablation_runs.py
```

For Colab copy/paste commands, use:

```text
TP2MVTransformer/transformer_ablation_colab_commands.md
```

## Corrected vs Legacy Transformer

Use this as the main transformer:

```bash
python TP2MVTransformer/03_train_mv_alibi_transformer.py
```

This corrected runner converts dates with calendar-day units and actually fills
the historical-attention memory. The console should show nonzero history
coverage.

For comparison only, the previous behavior is preserved here:

```bash
python TP2MVTransformer/03_train_mv_alibi_transformer_legacy_no_history.py
```

The legacy runner intentionally reproduces the earlier datetime conversion
behavior that led to empty historical attention. Keep it only for ablation or
sanity comparison; it should not be treated as the main publishable model.

## Current Risk-Control Candidate

The raw corrected transformer is diagnostic. The practical candidate is:

```bash
python TP2MVTransformer/07_conservative_transformer_floor_selection.py
```

It uses a fixed probability floor of `0.20` and then tunes the exact threshold
on the validation year only. This is intended to control low-confidence
over-selection in adverse years. Treat the floor as a pre-specified risk-control
rule for future runs, not a test-year optimized arbitrage condition.

For Sortino reporting, bring back at least
`TP2MVTransformer/data/mv_alibi_transformer_v2_selected_trades.parquet` from
Colab/GPU. The comparison report can then recompute V2 daily Sortino directly.
For professor-facing transformer diagnostics, also bring back:

- `TP2MVTransformer/tables/mv_alibi_v2_training_history.csv`
- `TP2MVTransformer/tables/mv_alibi_v2_distance_diagnostics.csv`
- `TP2MVTransformer/tables/mv_alibi_v2_attention_parameters.csv`
- `TP2MVTransformer/tables/mv_alibi_v2_yearly_diagnostics.csv`
- `TP2MVTransformer/tables/mv_alibi_transformer_v2_summary.json`

## Design Rules

- Strategy universe is fixed to `T1` and `K2`.
- Labels are expiry-payoff labels only.
- No first-liquid labels enter the transformer target.
- No `label_*`, future P&L, future exit, or `signal_year` features enter the
  model.
- Boosting outputs are used only to select features; boosting predictions,
  leaves, SHAP vectors, and scores are not transformer inputs.
- Historical attention uses prior TP2 tokens only. Validation thresholds are
  tuned on validation years, and test years are evaluated once.
