# ML-Based Trading of TP2-Inspired Violations in SPY Call Options

This repository contains the current Archive-2 research pipeline for detecting,
tracking, labeling, and modeling TP2-inspired relative-value violations in SPY
ETF call options.

The project uses the Glasserman-Li-Pirjol TP2 call inequality as theoretical
motivation, but it is not a direct replication of their SPX European-option
study. This is an empirical SPY American-option project with liquidity,
dividend, early-exercise, and EOD quote caveats.

## Current Status

- Underlying: SPY ETF options.
- Option side: calls only.
- Sample: professor-provided Archive-2 data, 2013-04-01 to 2026-06-04.
- Quote convention: latest available quote per contract-day.
- Detector: forward-adjusted call TP2 inequality with rounded adjusted strikes.
- Unit of analysis: episode-level TP2 violation starts, not repeated daily rows.
- Execution labels: bid/ask execution, first-liquid exit after true TP2
  reversion, and multiple exit-policy robustness checks.
- ML baseline: leakage-safe rolling LightGBM five-action policy:
  `skip`, `T1`, `T2`, `K1`, `K2`.

## Data Policy

Large raw and derived data files are intentionally not pushed to GitHub.

Local-only:

```text
Archive-2/spy_options_daily/year=*/data.parquet
data/**/*.parquet
reports/**/*.parquet
```

Tracked:

- source code;
- docs;
- report PDFs and report `.tex` sources;
- summary JSON/CSV tables that are small enough for Git;
- figures and final report assets.

## Main Results So Far

### Episode And Lifecycle Pipeline

The pipeline now tracks TP2 violations as episodes. This avoids treating the
same economic event on repeated days as independent observations.

Key files:

```text
scripts/run_build_episodes.py
scripts/build_violation_lifecycle_panel.py
scripts/run_build_episode_start_labels.py
scripts/build_liquid_exit_labels.py
```

Important reports:

```text
reports/lifecycle_tracking_report.pdf
reports/liquid_exit_rebuild_report.pdf
```

### Strategy And Exit Robustness

All four Glasserman-style denominations are tracked:

```text
T1, T2, K1, K2
```

Exit-policy robustness shows unconditional trading is fragile:

```text
positive policy-strategy cells: 2 of 24
```

First-liquid premium-resale aggregate P&L:

```text
T1: -$49.09M
T2:  -$3.34M
K1: -$19.40M
K2: -$24.42M
```

Expiration payoff aggregate P&L:

```text
T1:  -$32.16M
T2: +$194.35M
K1: -$118.56M
K2: +$173.84M
```

Report:

```text
reports/strategy_exit_robustness_report.pdf
```

### Rolling LightGBM Baseline

The current base rolling ML protocol uses:

```text
train 3 years -> validate 1 year -> test 1 year
```

Primary test years are 2021-2026.

Base LightGBM result:

```text
Net P&L: +$1.99M
Selected: 14,874
Executed: 8,672
Hit rate: 50.3%
Positive years: 4 of 6
```

Report:

```text
reports/ml_lightgbm_rolling_report.pdf
```

### Market-Quality Robustness

Market-quality filters improve label quality and hit rate but reduce coverage:

```text
Base positive-label share: 26.3%
Volume + Delta positive-label share: 75.1%
Strict Quality positive-label share: 90.0%
```

Rolling ML quality-gate comparison:

```text
Base:              +$1.99M, hit 50.3%
Volume + Delta:   +$5.7K, hit 55.2%
Strict Quality: +$195.0K, hit 61.3%
```

Reports:

```text
reports/market_quality_robustness_report.pdf
reports/ml_quality_gate_comparison_report.pdf
```

### Profitability Stress Tests

Commission stress:

```text
Base remains positive at $2.00 commission: +$1.51M
Strict quality remains positive at $2.00 commission: +$127.0K
```

Tail stress:

```text
Base original: +$1.99M
Base remove top 1% executed trades: +$186.6K
Base remove top 5% executed trades: -$2.41M
```

Interpretation: transaction costs are not the primary weakness. Tail
concentration is the main caveat.

Report:

```text
reports/profitability_stress_report.pdf
```

### Moneyness And Pricing-Quality Caveat

The base model selects many OTM pairs, but much of the base P&L comes from ITM
or mixed-moneyness pairs. Deep ITM calls are intrinsic-value dominated, so these
results should not be used as the cleanest TP2-arbitrage headline before adding
deep-ITM/pricing-quality filters.

Diagnostic tables:

```text
reports/raw_archive2_eda/tables/ml_moneyness/
reports/raw_archive2_eda/tables/pricing_quality/
```

## Current Caveats

- This is not a risk-free arbitrage claim.
- SPY options are American style and dividend/friction affected.
- EOD/latest-available quotes are not live executable intraday quotes.
- Base P&L is tail-concentrated.
- ITM/intrinsic-value-dominated rows need further pricing-quality filtering.
- Very strict quality filters improve hit rate but reduce recent-year sample
  size, especially 2026.

## Transformer Handoff

The next modeling step is temporal structure / Transformer modeling.

Recommended input:

```text
event_day_index = -20 ... 0
```

Allowed sequence features:

- TP2 pressure / normalized pressure;
- observable/missing status;
- liquidity flags;
- spread stats;
- violation persistence before day 0;
- point-in-time A/B/C/D geometry and liquidity.

Forbidden features:

- reversion date;
- future lifecycle rows after day 0;
- future exit availability;
- realized P&L;
- label/end-state columns.

Target:

```text
skip / T1 / T2 / K1 / K2
```

The professor noted that most violations revert within about 10 days. A
20-session pre-signal lookback is still reasonable because it lets the model see
whether pressure was building before the violation.

## Main Docs

```text
docs/pipeline.md
docs/rigorous_execution_plan.md
docs/ml_phase_plan.md
docs/violation_lifecycle_tracking_plan.md
docs/liquid_exit_rebuild_plan.md
```

## Report Standard

Project reports should be generated from `.tex` sources and compiled to PDF.
Recent final reports follow:

```text
reports/lifecycle_tracking_report.tex
reports/liquid_exit_rebuild_report.tex
reports/ml_lightgbm_rolling_report.tex
reports/strategy_exit_robustness_report.tex
reports/market_quality_robustness_report.tex
reports/ml_quality_gate_comparison_report.tex
reports/profitability_stress_report.tex
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run scripts with:

```bash
PYTHONPATH=src python3 <script>
```

## Important Rebuild Commands

These commands require the local Archive-2 raw data.

```bash
PYTHONPATH=src python3 scripts/run_ingest_archive2.py
PYTHONPATH=src python3 scripts/build_archive2_quote_availability.py
PYTHONPATH=src python3 scripts/run_detect_archive2_lastquote_by_year.py --progress
PYTHONPATH=src python3 scripts/run_build_episodes.py
PYTHONPATH=src python3 scripts/run_classify_episode_end_states.py
PYTHONPATH=src python3 scripts/run_build_episode_start_labels.py
PYTHONPATH=src python3 scripts/build_conservative_bidask_subset.py
PYTHONPATH=src python3 scripts/build_liquid_exit_labels.py
PYTHONPATH=src python3 scripts/run_episode_mini_backtest.py \
  --labels-input data/labels/archive2_episode_start_labels_conservative_bidask_2013_2026.parquet \
  --output data/labels/archive2_episode_strategy_mini_backtest_conservative_bidask_2013_2026.parquet \
  --summary-output data/labels/archive2_episode_strategy_mini_backtest_conservative_bidask_summary_2013_2026.json \
  --diagnostics-dir reports/raw_archive2_eda/tables/episode_mini_backtest_conservative_bidask
PYTHONPATH=src python3 scripts/run_episode_mini_backtest_unit_sized.py
PYTHONPATH=src python3 scripts/build_violation_lifecycle_panel.py
PYTHONPATH=src python3 scripts/build_ml_dataset.py
PYTHONPATH=src python3 scripts/run_ml_rolling_lightgbm.py --target net_pnl --quality-gate base
PYTHONPATH=src python3 scripts/generate_ml_lightgbm_report.py
PYTHONPATH=src python3 scripts/generate_ml_quality_gate_report.py
PYTHONPATH=src python3 scripts/generate_profitability_stress_report.py
```

The current reports and small summary tables are already committed. The raw
Archive-2 parquet files and regenerated large parquet outputs are local-only by
design.

## Near-Term Next Steps

Before final paper submission:

1. Add deep-ITM and pricing-quality filters.
2. Add concentration-control modeling:
   capped objective, ranking objective, top-k/day, and exposure limits.
3. Add Transformer temporal model using lifecycle sequences.
4. Add SHAP/feature attribution for model interpretation.
5. Compare final models under the same rolling splits and robustness tables.
