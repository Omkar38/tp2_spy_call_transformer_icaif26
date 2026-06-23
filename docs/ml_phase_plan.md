# ML Phase Plan

Project: ML-Based Trading of TP2-Inspired Violations in SPY American Call Options

Date: 2026-06-22

## Objective

Build a leakage-safe rolling ML pipeline that selects which conservative TP2 violation episodes to trade, and optionally which strategy/exit policy to use. The model should improve over unconditional trading and be strong enough for research reporting.

## Current Inputs

Use the regenerated post-cleaning pipeline outputs:

- `data/labels/archive2_episode_start_labels_conservative_bidask_2013_2026.parquet`
- `data/features/archive2_lifecycle/tp2_conservative_episode_lifecycle_2013_2026.parquet`
- `data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet`
- `data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet`
- `data/processed/archive2_spy_call_quote_availability_nospread_2013_2026.parquet`

The main ML table should be episode-level. One row should represent one possible decision at the violation start date.

## First Target Choice

Use the full strategy-choice setup:

1. Predict whether a conservative episode should be traded under the first-liquid exit policy.
2. Compare all four Glasserman-style TP2 strategies.
3. Build a five-action decision layer:
   - `skip`
   - `trade_T1`
   - `trade_T2`
   - `trade_K1`
   - `trade_K2`

Initial label candidates:

- Binary classification per strategy: `net_pnl > 0`
- Regression per strategy: `net_pnl`
- Risk-adjusted regression target: capped or winsorized `net_pnl`
- Final decision target: choose the strategy with highest positive net P&L across T1/T2/K1/K2; otherwise skip

Recommended first implementation:

- Build one episode-level row with four realized strategy labels under `exit_first_liquid_on_or_after_reversion`.
- Train either four separate strategy profitability/expected-P&L models or one multiclass five-action model.
- Select trades only when the predicted expected value or predicted probability exceeds a validation-tuned threshold.
- Choose the highest-scored action among T1/T2/K1/K2; choose `skip` when no action passes the threshold.

## Implemented Label Robustness Step

The ML dataset now stores three five-action target definitions for the same
episode rows:

- `net_pnl`: choose the strategy with the largest positive dollar net P&L;
  otherwise skip.
- `return`: choose the strategy with the largest positive realized return;
  otherwise skip.
- `capped_pnl`: choose the strategy with the largest positive net P&L after a
  fixed symmetric `$10,000` cap; otherwise skip.

The cap is fixed ex ante rather than estimated from the full sample, so the
target definition does not depend on future distribution quantiles.

Current label comparison:

```text
rows: 226,577
net-P&L vs return mismatches: 0
net-P&L vs capped-P&L mismatches: 12
return vs capped-P&L mismatches: 12
```

This is a strong diagnostic: the current T2 label frequency is not simply a
dollar-notional artifact, because the return-normalized target chooses the same
actions as dollar net P&L. The capped-P&L target is also nearly identical at the
label level, but its rolling model selects fewer trades and reduces some tail
exposure.

Robustness outputs:

```text
data/ml/archive2_ml_lightgbm_rolling_summary_net_pnl.csv
data/ml/archive2_ml_lightgbm_rolling_summary_return.csv
data/ml/archive2_ml_lightgbm_rolling_summary_capped_pnl.csv
reports/raw_archive2_eda/tables/ml_lightgbm/lightgbm_label_robustness_comparison.csv
reports/raw_archive2_eda/tables/ml_lightgbm/lightgbm_label_robustness_labels.json
```

## Implemented Exit-Policy Robustness Step

The strategy label layer has been audited across six exit policies:

- reversion-day exit;
- first-liquid-on-or-after-reversion exit;
- reversion +1 session;
- reversion +3 sessions;
- reversion +5 sessions;
- selected-leg expiration payoff.

Output report:

```text
reports/strategy_exit_robustness_report.pdf
```

Output tables:

```text
reports/raw_archive2_eda/tables/strategy_exit_robustness/strategy_exit_policy_summary.csv
reports/raw_archive2_eda/tables/strategy_exit_robustness/strategy_exit_policy_year_summary.csv
reports/raw_archive2_eda/tables/strategy_exit_robustness/strategy_exit_policy_rankings.csv
reports/raw_archive2_eda/tables/strategy_exit_robustness/strategy_exit_robustness_summary.json
```

Main result:

```text
positive policy-strategy cells: 2 of 24
first-liquid aggregate P&L: all four strategies negative
expiration aggregate P&L: T2 and K2 positive, T1 and K1 negative
```

This confirms that the final paper should separate premium-resale exits from
expiration-payoff benchmarks. It also confirms that unconditional trading is not
the claim; ML selection is the core empirical contribution.

## Implemented Market-Quality Robustness Step

The current episode-level ML dataset has been audited under stricter
market-quality gates:

- all four entry legs have positive volume;
- all four entry legs have positive open interest;
- all four deltas satisfy `0.01 <= abs(delta) <= 0.99`;
- max entry spread no greater than 25% or 10%;
- max rounded-strike distance no greater than 1, 2.5, or 5 dollars;
- combined volume/delta gate;
- strict quality gate combining volume, open interest, delta, 25% spread, and
  2.5-dollar rounded-strike distance.

Output report:

```text
reports/market_quality_robustness_report.pdf
```

Output tables:

```text
reports/raw_archive2_eda/tables/market_quality_robustness/market_quality_gate_summary.csv
reports/raw_archive2_eda/tables/market_quality_robustness/market_quality_strategy_summary.csv
reports/raw_archive2_eda/tables/market_quality_robustness/market_quality_gate_year_summary.csv
reports/raw_archive2_eda/tables/market_quality_robustness/market_quality_robustness_summary.json
```

Main result:

```text
base conservative episodes: 226,577, positive-label share 26.3%
all legs positive volume: 30,468, positive-label share 45.2%
paper delta 0.01-0.99: 60,532, positive-label share 61.9%
volume + delta: 15,772, positive-label share 75.1%
strict quality combo: 8,953, positive-label share 90.0%
```

This is strong evidence that market-quality conditions matter. The next rolling
ML step should rerun the same leakage-safe folds on at least two filtered
universes: `volume_and_delta` and `strict_quality_combo`.

## Implemented ML Quality-Gate Comparison

Rolling LightGBM has now been rerun on three candidate universes using the same
net-P&L five-action target:

- base conservative bid/ask TP2 universe;
- `volume_and_delta`;
- `strict_quality_combo`.

Output report:

```text
reports/ml_quality_gate_comparison_report.tex
reports/ml_quality_gate_comparison_report.pdf
```

Output tables:

```text
reports/raw_archive2_eda/tables/ml_quality_gate/ml_quality_gate_aggregate_comparison.csv
reports/raw_archive2_eda/tables/ml_quality_gate/ml_quality_gate_year_comparison.csv
reports/raw_archive2_eda/tables/ml_quality_gate/ml_quality_gate_action_mix.csv
```

Aggregate test results:

```text
Base:              +$1.99M, selected 14,874, executed 8,672, hit 50.3%
Volume + Delta:   +$5.7K, selected 1,865, executed 1,720, hit 55.2%
Strict Quality: +$195.0K, selected 1,650, executed 1,489, hit 61.3%
```

This supports a layered paper result: use the base model as the main broad
sample and use filtered universes as robustness checks showing that cleaner
market-quality data improves execution and hit rate but reduces opportunity
coverage.

## Implemented Profitability Stress Tests

The selected rolling ML trades have been stress-tested without retraining:

- commission sensitivity at `$0.65`, `$1.00`, `$1.50`, and `$2.00` per contract
  per side;
- top 1% and 5% executed-trade removal;
- 1/99% and 5/95% winsorization;
- max drawdown and top-tail concentration.

Output report:

```text
reports/profitability_stress_report.tex
reports/profitability_stress_report.pdf
```

Output tables:

```text
reports/raw_archive2_eda/tables/profitability_stress/profitability_stress_aggregate.csv
reports/raw_archive2_eda/tables/profitability_stress/profitability_stress_commission.csv
reports/raw_archive2_eda/tables/profitability_stress/profitability_stress_tail.csv
reports/raw_archive2_eda/tables/profitability_stress/profitability_stress_yearly.csv
```

Main result:

```text
Base remains positive through $2.00 commission: +$1.51M
Strict quality remains positive through $2.00 commission: +$127.0K
Volume + Delta turns negative above $0.65 commission
```

Tail result:

```text
Base remove top 1% executed trades: +$186.6K
Base remove top 5% executed trades: -$2.41M
Strict quality remove top 1%: +$90.0K
Strict quality remove top 5%: -$49.4K
```

This means commission robustness is acceptable, but concentration is a major
research caveat. The next modeling step should control tail dependence with
ranking, capped objectives, position sizing, or maximum exposure limits.

## Strict Leakage Rules

Allowed features:

- Signal-date option surface features from A/B/C/D.
- Entry-time bid, ask, mid, spread, normalized severity, moneyness, DTE, expiration gap.
- Liquidity features available on signal date.
- Calendar features available on signal date.
- Prior lifecycle features from `event_day_index <= 0` only.
- Historical rolling aggregates computed using past dates only.

Forbidden features:

- `exit_date`
- `exit_reason`
- realized P&L columns
- reversion day
- future lifecycle rows where `event_day_index > 0`
- future quote availability
- future SPY returns
- labels or end-state columns

The model must never see information that becomes known after the signal date.

## Feature Groups

### Signal-Date Rectangle Features

- Mid-price TP2 pressure.
- Conservative bid/ask pressure.
- Normalized pressure.
- A/B/C/D bid, ask, mid.
- A/B/C/D spread percentage.
- All-leg liquidity flag.
- Individual-leg liquidity flags.
- Product ratio: `(C_mid * D_mid) / (A_mid * B_mid)`.
- Bid/ask executable ratio: `(C_bid * D_bid) / (A_ask * B_ask)`.

### Contract Geometry Features

- A strike, B strike, C strike, D strike.
- Moneyness for each leg.
- DTE for T1 and T2.
- Term gap: `T2_DTE - T1_DTE`.
- Strike gap: `K2 - K1`.
- Forward-adjustment gap.
- Whether adjusted right-side strikes are far from original strikes.

### Liquidity Features

- Bid/ask spread percentage per leg.
- Max spread percentage across legs.
- Mean spread percentage across legs.
- Volume per leg, if available.
- Open interest per leg, if available.
- Zero-volume flags.
- Low-open-interest flags.

### Historical Lifecycle Features

Use only rows with `event_day_index <= 0`:

- Prior pressure slope.
- Prior median pressure.
- Prior max pressure.
- Prior count of near-violation days.
- Prior count of actual violation days.
- Pressure volatility.
- Whether pair was close to zero before day 0.
- Number of observable prior days.
- Number of missing prior days.
- Prior liquidity failure share.

### Market Context Features

Only if available without future leakage:

- SPY close return over past 1, 5, 10 sessions.
- Realized SPY volatility over past 5, 10, 20 sessions.
- Calendar year, month, day-of-week.
- VIX-like external feature only if data is added cleanly and timestamped.

## Rolling Evaluation

Do not use random train/test split.

Primary rolling setup:

- Train: 3 years
- Validation: next 1 year
- Test: next 1 year
- Roll forward by 1 year
- Primary evaluation starts from the stable 2017-forward sample.

Example:

- Train 2017-2019, validate 2020, test 2021
- Train 2018-2020, validate 2021, test 2022
- Train 2019-2021, validate 2022, test 2023
- Train 2020-2022, validate 2023, test 2024
- Train 2021-2023, validate 2024, test 2025
- Train 2022-2024, validate 2025, test 2026

The early 2013--2016 sample should be used as a robustness appendix, not the primary ML claim. The reason is that 2013 is partial and 2014--2016 contain fewer conservative executable episodes than later years. This does not invalidate the data, but it makes early-window ML validation noisier.

Threshold tuning must happen only on validation data. The test year must be untouched until final evaluation.

## Baseline Models

Start with strong, explainable baselines:

1. Logistic regression
2. Random forest
3. XGBoost or LightGBM, if installed
4. CatBoost, if installed

Then add deep models:

1. MLP on tabular features
2. Transformer over prior lifecycle sequences

The Transformer should use event-time sequences from `event_day_index <= 0`, not future lifecycle paths.

## Transformer Design

Input:

- Sequence length up to 21 days: event days `-20` through `0`.
- Per-day features: pressure, liquidity status, observability, spread stats, near-violation flags.
- Static features: signal-date geometry and liquidity features.

Architecture:

- Transformer encoder for lifecycle sequence.
- Static tabular encoder for signal-date features.
- Concatenate sequence embedding and static embedding.
- Output either:
  - T1 profitability probability
  - K1 profitability probability
- expected net P&L for each action: T1, T2, K1, K2, skip

Keep the first Transformer small. The conference value is not only model size; it is the clean arbitrage/lifecycle formulation plus leakage-safe evaluation.

## Metrics

Classification metrics:

- AUC
- precision
- recall
- precision at top k
- calibration

Trading metrics:

- selected trades
- executed trades
- hit rate
- gross P&L
- net P&L after fees
- mean and median return
- Sharpe-like return statistic
- max drawdown by signal date
- yearly P&L
- top 1%, 5%, 10 trade contribution
- P&L after winsorization/capping

Always compare against:

- unconditional trade-all baseline
- random selection with same trade count
- severity threshold baseline
- liquidity threshold baseline

## Transaction Costs

Keep bid/ask execution as the main cost assumption. Also report contract fee sensitivity:

- `$0.00`
- `$0.65` per contract
- higher stress case if needed

Because strategy quantities can be product-weighted, report both:

- product-weighted P&L
- capped or unit-normalized P&L

## Robustness Checks

Required before paper-style claims:

- year-by-year performance
- excluding 2013 partial data
- excluding 2026 partial data
- excluding top 1% P&L episodes
- capped position size
- wide-spread sensitivity: 25%, 50%, 100%
- separate T1 and K1 results
- recent period only: 2024-2026
- rolling-window stability

## Expected Deliverables

Code:

- `scripts/build_ml_dataset.py`
- `scripts/run_ml_rolling_baselines.py`
- `scripts/run_ml_rolling_transformer.py`
- `scripts/generate_ml_report.py`

Data:

- `data/ml/archive2_ml_episode_dataset_2013_2026.parquet`
- `data/ml/archive2_ml_rolling_predictions.parquet`
- `data/ml/archive2_ml_rolling_trade_results.parquet`

Reports:

- `reports/ml_dataset_report.pdf`
- `reports/ml_rolling_baseline_report.pdf`
- `reports/ml_transformer_report.pdf`

## Go/No-Go Checks Before Modeling

Proceed to ML only after:

- bad 2017 raw quote artifact is removed from labels
- lifecycle report is regenerated
- liquid-exit labels are regenerated
- target columns are clearly selected
- feature leakage list is enforced in code
- rolling split logic is fixed and tested

Current status: ready to start ML dataset construction.
