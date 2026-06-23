# Professor Handoff Summary

Date: 2026-06-23

Project: ML-Based Trading of TP2-Inspired Violations in SPY American Call Options

## What Is Ready

The current repo contains the complete non-Transformer baseline pipeline:

1. Archive-2 SPY call ingestion and TP2 detection.
2. Episode tracking from repeated daily violations.
3. Lifecycle tracking before and after violation start.
4. True reversion and first-liquid tradable exit labeling.
5. Four TP2 strategy labels: `T1`, `T2`, `K1`, `K2`.
6. Leakage-safe rolling LightGBM five-action model:
   `skip`, `T1`, `T2`, `K1`, `K2`.
7. Market-quality robustness.
8. Profitability stress tests.
9. Moneyness and pricing-quality diagnostics.

## Main Empirical Results

Base rolling LightGBM, test years 2021-2026:

```text
Net P&L: +$1.99M
Selected: 14,874
Executed: 8,672
Hit rate: 50.3%
Positive years: 4 of 6
```

Quality-gated rolling ML:

```text
Base:              +$1.99M, hit 50.3%
Volume + Delta:   +$5.7K, hit 55.2%
Strict Quality: +$195.0K, hit 61.3%
```

Profitability stress:

```text
Base at $2.00 commission: +$1.51M
Strict quality at $2.00 commission: +$127.0K
Base remove top 1% trades: +$186.6K
Base remove top 5% trades: -$2.41M
```

## Important Interpretation

This should not be framed as a risk-free arbitrage result.

The stronger and more accurate claim is:

```text
TP2-inspired SPY call violations form short-lived relative-value episodes.
Profitability depends on liquidity, moneyness, exit timing, and trade selection.
ML can help select the tradeable subset, but the base result is tail-concentrated.
```

## Main Caveats

- SPY options are American-style; the motivating paper is cleaner for SPX
  European options.
- Latest-available EOD quotes are not live intraday executable quotes.
- Base P&L is concentrated in a small number of large winning episodes.
- Some base-model P&L comes from ITM/intrinsic-value-dominated calls.
- OTM options have their own quote-quality issue: wide relative spreads and
  low/zero volume.

## Reports To Read First

```text
reports/lifecycle_tracking_report.pdf
reports/liquid_exit_rebuild_report.pdf
reports/ml_lightgbm_rolling_report.pdf
reports/strategy_exit_robustness_report.pdf
reports/market_quality_robustness_report.pdf
reports/ml_quality_gate_comparison_report.pdf
reports/profitability_stress_report.pdf
```

Each recent report also has a `.tex` source in `reports/`.

## Rebuild Path

The repository is intended to be readable without pushing the raw data. To
rebuild from raw Archive-2 locally, use the sequence in `README.md`:

```text
raw Archive-2 parquet
-> clean SPY call contract-day panel
-> quote-availability panel
-> forward-adjusted TP2 detection
-> pair episodes and lifecycle panel
-> end-state and liquid-exit labels
-> episode-level ML dataset
-> rolling LightGBM reports
```

Large parquet files in `Archive-2/` and `data/` are intentionally ignored; the
repo tracks the code, reports, figures, and small summary tables.

## Transformer Handoff

Recommended Transformer input is the lifecycle sequence before violation start:

```text
event_day_index = -20 ... 0
```

Allowed features:

- normalized TP2 pressure;
- observable/missing/liquidity status;
- spread and volume/open-interest features;
- prior violation persistence;
- point-in-time A/B/C/D geometry.

Forbidden features:

- future reversion date;
- future exit date or exit availability;
- realized P&L;
- any `label_*` or post-event outcome columns.

Target:

```text
skip / T1 / T2 / K1 / K2
```

Professor's meeting note that most violations revert within about 10 days is
consistent with using a 20-session pre-signal context window. The model should
learn the pressure path before the violation, not the future reversion path.

## Suggested Next Work

1. Deep-ITM/pricing-quality filter robustness.
2. Concentration-control model:
   capped objective, ranker, top-k/day, or exposure cap.
3. Transformer lifecycle model.
4. SHAP/feature attribution for tabular models.
5. Final paper writing and positioning around TP2-inspired episode selection.
