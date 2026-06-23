# Rigorous Execution Plan

Project: ML-Based Trading of TP2-Inspired Violations in SPY American Call Options

Date: 2026-06-22

## Goal

Turn the TP2-inspired SPY call project into a publishable empirical ML/quant
finance paper by making the signal definition, labeling, execution assumptions,
and out-of-sample testing defensible.

## Reporting Standard

All project PDFs generated from this point forward should have a corresponding
`.tex` source file in `reports/`, and the PDF should be compiled from that
source. Direct PDF-only generation should be avoided except for quick temporary
diagnostics that are not part of the final project record.

Recent robustness reports have been converted to this standard:

```text
reports/strategy_exit_robustness_report.tex
reports/strategy_exit_robustness_report.pdf
reports/market_quality_robustness_report.tex
reports/market_quality_robustness_report.pdf
```

## Current Idea-Level Assessment

The idea is good to very good, not poor. Its strength is that it starts from a
known option-surface inequality rather than generic price prediction, then asks
whether TP2-inspired violation episodes in American SPY calls contain learnable
relative-value structure.

The project becomes publishable only if the empirical claim survives realistic
execution, liquidity, robustness, and leakage-safe ML evaluation.

## Execution Steps

1. Label robustness.
   - Keep the current dollar net-P&L label.
   - Add return-normalized labels.
   - Add capped/winsorized-P&L labels.
   - Compare whether selected strategies remain stable.
   - This directly tests whether T2 is genuinely useful or just larger in
     dollar exposure.

2. Strategy robustness.
   - Evaluate T1, T2, K1, and K2 under multiple exit definitions:
     reversion-day, first-liquid-after-reversion, reversion +1, reversion +3,
     reversion +5, and expiration payoff.
   - Report where results agree and where they break.

3. Market-quality robustness.
   - Positive-volume sensitivity.
   - Delta screen sensitivity, where delta is available.
   - Open-interest sensitivity.
   - Rounded-strike-distance sensitivity.
   - Static vertical/calendar no-arbitrage sanity filters.

4. ML robustness.
   - Use the same rolling folds for every target.
   - Run simple baselines first.
   - Then run LightGBM/CatBoost.
   - Add Transformer only after the tabular pipeline is stable.

5. Interpretability.
   - Add SHAP or equivalent feature attribution.
   - Confirm the model uses economically meaningful variables such as severity,
     spread, moneyness, DTE, liquidity, and prior pressure path.

6. Profitability stress tests.
   - Higher transaction costs.
   - Remove or cap top tail trades.
   - Report median return and hit rate, not only total P&L.
   - Report invalid selected trades explicitly.

## Current First Step

Implement label robustness:

- `net_pnl`: choose the strategy with the largest positive dollar net P&L.
- `return`: choose the strategy with the largest positive realized return.
- `capped_pnl`: choose the strategy with the largest positive net P&L after a
  fixed symmetric cap.

The cap must be fixed ex ante, not estimated from the full future sample, to
avoid a subtle form of distribution leakage.

Status: completed.

Result:

```text
net-P&L vs return label mismatches: 0 of 226,577
net-P&L vs capped-P&L label mismatches: 12 of 226,577
```

This means the current strategy labels are not materially driven by dollar
notional alone.

## Current Second Step

Implement strategy robustness across exit policies:

- reversion-day exit;
- first-liquid-on-or-after-reversion exit;
- reversion +1 session;
- reversion +3 sessions;
- reversion +5 sessions;
- selected-leg expiration payoff.

Status: completed.

Outputs:

```text
reports/strategy_exit_robustness_report.pdf
reports/raw_archive2_eda/tables/strategy_exit_robustness/strategy_exit_policy_summary.csv
reports/raw_archive2_eda/tables/strategy_exit_robustness/strategy_exit_policy_year_summary.csv
reports/raw_archive2_eda/tables/strategy_exit_robustness/strategy_exit_policy_rankings.csv
reports/raw_archive2_eda/tables/strategy_exit_robustness/strategy_exit_robustness_summary.json
```

Main result:

```text
positive policy-strategy cells: 2 of 24
first-liquid net P&L:
  T1 -$49.09M
  T2  -$3.34M
  K1 -$19.40M
  K2 -$24.42M
expiration net P&L:
  T1 -$32.16M
  T2 +$194.35M
  K1 -$118.56M
  K2 +$173.84M
```

Interpretation:

- Unconditional premium-resale trading is fragile.
- T2 is least negative under first-liquid premium resale, which explains why
  the ML target often selects T2 episodes.
- Expiration payoff is structurally different and should be reported as a
  separate benchmark, not mixed with premium-resale exits.
- The defensible research claim is ML-based selection of tradeable TP2 episodes,
  not a universal arbitrage from every violation.

## Current Third Step

Implement market-quality robustness diagnostics:

- all four legs have positive daily volume;
- all four legs have positive open interest;
- all four deltas satisfy `0.01 <= abs(delta) <= 0.99`;
- stricter max spread gates at 25% and 10%;
- rounded-strike distance gates at 1, 2.5, and 5 dollars;
- combined volume/delta and strict quality gates.

Status: completed.

Outputs:

```text
reports/market_quality_robustness_report.pdf
reports/raw_archive2_eda/tables/market_quality_robustness/market_quality_gate_summary.csv
reports/raw_archive2_eda/tables/market_quality_robustness/market_quality_strategy_summary.csv
reports/raw_archive2_eda/tables/market_quality_robustness/market_quality_gate_year_summary.csv
reports/raw_archive2_eda/tables/market_quality_robustness/market_quality_robustness_summary.json
```

Main result:

```text
base conservative episodes: 226,577
all legs positive volume: 30,468 episodes, 45.2% positive-label share
paper delta 0.01-0.99: 60,532 episodes, 61.9% positive-label share
volume + delta: 15,772 episodes, 75.1% positive-label share
strict quality combo: 8,953 episodes, 90.0% positive-label share
```

Interpretation:

- The quality gates materially improve the economic label universe.
- The strict combo is promising but keeps only 4.0% of episodes, so it should be
  treated as a robustness universe before becoming the main sample.
- The next model run should compare the base universe against at least
  `volume_and_delta` and `strict_quality_combo`.

## Current Fourth Step

Rerun rolling LightGBM on quality-filtered universes:

- base conservative universe;
- `volume_and_delta`;
- `strict_quality_combo`.

Status: completed.

Outputs:

```text
reports/ml_quality_gate_comparison_report.tex
reports/ml_quality_gate_comparison_report.pdf
reports/raw_archive2_eda/tables/ml_quality_gate/ml_quality_gate_aggregate_comparison.csv
reports/raw_archive2_eda/tables/ml_quality_gate/ml_quality_gate_year_comparison.csv
reports/raw_archive2_eda/tables/ml_quality_gate/ml_quality_gate_action_mix.csv
```

Main result:

```text
Base:              +$1.99M, selected 14,874, executed 8,672, hit 50.3%
Volume + Delta:   +$5.7K, selected 1,865, executed 1,720, hit 55.2%
Strict Quality: +$195.0K, selected 1,650, executed 1,489, hit 61.3%
```

Interpretation:

- The base universe produces the largest total P&L because it keeps the full
  opportunity set.
- The filtered universes greatly improve execution rate and hit rate, but they
  remove most candidates and therefore much of the P&L opportunity.
- `strict_quality_combo` is a useful robustness result, not a replacement for
  the main model, because recent-year sample sizes are very small.

## Current Fifth Step

Run profitability stress tests on selected rolling ML trades:

- commission sensitivity at `$0.65`, `$1.00`, `$1.50`, and `$2.00` per contract
  per side;
- top-tail removal at top 1% and 5% of executed trades;
- winsorization at 1/99% and 5/95%;
- max drawdown and concentration diagnostics.

Status: completed.

Outputs:

```text
reports/profitability_stress_report.tex
reports/profitability_stress_report.pdf
reports/raw_archive2_eda/tables/profitability_stress/profitability_stress_aggregate.csv
reports/raw_archive2_eda/tables/profitability_stress/profitability_stress_commission.csv
reports/raw_archive2_eda/tables/profitability_stress/profitability_stress_tail.csv
reports/raw_archive2_eda/tables/profitability_stress/profitability_stress_yearly.csv
```

Main result:

```text
Base P&L:
  $0.65 commission: +$1.99M
  $1.00 commission: +$1.87M
  $1.50 commission: +$1.69M
  $2.00 commission: +$1.51M

Strict quality P&L:
  $0.65 commission: +$195.0K
  $1.00 commission: +$177.4K
  $1.50 commission: +$152.2K
  $2.00 commission: +$127.0K
```

Tail sensitivity:

```text
Base original: +$1.99M
Base remove top 1%: +$186.6K
Base remove top 5%: -$2.41M

Strict quality original: +$195.0K
Strict quality remove top 1%: +$90.0K
Strict quality remove top 5%: -$49.4K
```

Interpretation:

- Commission sensitivity is acceptable for base and strict-quality universes.
- Tail concentration is the main economic weakness.
- The paper should explicitly report that aggregate P&L is concentrated in a
  small number of high-payoff episodes.
- The next model improvement should target concentration control or ranking
  rather than only higher aggregate P&L.
