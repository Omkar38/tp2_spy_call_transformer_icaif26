# Next Phase Guidelines: Episode End-State Labels And Mini-Backtest

This note records the agreed rules before starting the next implementation
phase. Follow this file when continuing from episode tracking to end-state
classification, labels, mini-backtest, and ML.

## Current State

Completed:

- Clean all-call lastquote SPY dataset, 2013-2026.
- Forward-adjusted TP2 detector on daily four-leg rectangles.
- Pair-level episode tracking using the left-side A/B pair.
- Episode end-state classification.
- Episode-start reversion label dataset.
- Conservative episode-level mini-backtest baseline.
- Conservative bid/ask TP2 subset and four-strategy mini-backtest rerun.
- Pre-ML gating and year-stability diagnostics for the conservative subset.
- Episode validation:
  - daily detector rows: 2,430,220
  - pair episodes: 2,098,213
  - no lost `violation_id`
  - no duplicate `violation_id`
  - no missing `episode_id`

Current episode output:

```text
data/features/archive2_episodes/tp2_call_pair_episodes_2013_2026.parquet
data/features/archive2_episodes/tp2_call_episode_members_2013_2026.parquet
data/features/archive2_episodes/tp2_call_episode_summary_2013_2026.json
```

Current label output:

```text
data/features/archive2_episodes/tp2_call_pair_episode_end_states_2013_2026.parquet
data/labels/archive2_episode_start_labels_2013_2026.parquet
data/labels/archive2_episode_start_label_summary_2013_2026.json
data/labels/archive2_episode_strategy_mini_backtest_2013_2026.parquet
data/labels/archive2_episode_strategy_mini_backtest_summary_2013_2026.json
data/labels/archive2_episode_start_labels_conservative_bidask_2013_2026.parquet
data/labels/archive2_episode_strategy_mini_backtest_conservative_bidask_2013_2026.parquet
reports/pre_ml_gating_and_rolling_plan.pdf
reports/raw_archive2_eda/tables/pre_ml_gating/
```

## Important Conceptual Separation

Keep these three stages separate:

1. **Detection**
   - Finds TP2-inspired violations.
   - Uses mid-price inequality as the main research signal.
   - Should not be changed yet.

2. **End-state classification**
   - Explains why an episode stopped appearing.
   - Must not treat disappearance as automatic reversion.
   - Uses clean-data availability and liquidity status to classify the next
     state.

3. **Mini-backtest / backtest**
   - Tests whether the violation is actually tradable.
   - Uses bid/ask execution, transaction costs, exit quote availability, and
     invalid-selected accounting.

## Bid/Ask Filter Decision

Do **not** change the main detector to require:

```text
A_ask * B_ask < C_bid * D_bid
```

at this stage.

Reason:

- The detector's job is to find a TP2-inspired mid-price relative-value signal.
- Tradability should be tested in the mini-backtest and final backtest using
  actual bid/ask execution.
- If bid/ask is enforced inside detection too early, we may throw away useful
  research signal before ML can learn which violations are economically useful.

For the paper, conservative bid/ask TP2 detection can still be added later as a
robustness table. It is not the next coding priority.

## What Liquidity Filter Failure Means

Liquidity filter failure belongs mainly in end-state classification and
backtesting.

An episode can stop appearing because all required contracts still technically
exist, but one or more legs fail tradability or quality rules, such as:

- invalid bid/ask
- bid <= 0
- mid <= 0
- spread percentage above threshold
- missing quote
- optionally later: zero volume, zero open interest, stale quote, or too-small
  bid/ask size

This should be classified as:

```text
liquidity_filter_failure
```

not:

```text
true_reversion
```

## Correct End-State Classification

For each episode, inspect the next available trading session after
`episode_end_date`.

Classification priority:

```text
1. sample_ended
2. left_leg_expired_or_missing
3. right_leg_missing_or_expired
4. liquidity_filter_failure
5. true_reversion
6. still_violation_or_reappeared
7. unresolved_other
```

Interpretation:

`sample_ended`

The episode ends on the final available clean-data session. We do not know what
happened next.

`left_leg_expired_or_missing`

The A or B contract from the pair key cannot be observed on the next trading
session, or has expired. This is not reversion.

`right_leg_missing_or_expired`

A and B are observable, but the required adjusted C/D legs cannot be formed or
are expired/missing. This is not reversion.

`liquidity_filter_failure`

All required legs can be identified, but at least one fails the clean/tradable
quote rules. This is not reversion.

`true_reversion`

All four legs are observable and pass required quote-quality rules, and the
TP2 inequality no longer violates:

```text
lhs >= rhs
```

`still_violation_or_reappeared`

All four legs are observable and pass required quote-quality rules, and:

```text
lhs < rhs
```

This should be rare immediately after a consecutive-session episode end. If it
appears often, the episode grouping logic or rounded-leg reconstruction needs
review.

## What Counts As Reversion

Only this should count as true reversion:

```text
same A/B pair can be checked
required C/D legs can be formed
all four legs pass quote-quality rules
lhs >= rhs
```

Do not count these as reversion:

- A/B expired
- C/D expired
- one leg missing
- one leg fails liquidity/quote-quality filters
- dataset ends

## Mini-Backtest Rule

The mini-backtest should handle tradability directly.

Entry/exit should use bid/ask-side execution rather than assuming mid-price
execution. Therefore, bid/ask costs are encountered during the mini-backtest
and final backtest.

Minimum backtest accounting:

- selected episodes
- executable selected episodes
- invalid selected episodes
- invalid reason
- gross P&L
- transaction costs
- net P&L
- holding days
- exit reason

Do not silently remove selected trades that cannot be executed. Count them as
invalid selected.

## Pre-ML Rule

Do not train ML directly on daily violation rows. The ML observation unit is:

```text
episode_id + strategy
```

Use the conservative bid/ask episode subset as the first ML candidate universe.
The active strategies are:

```text
T1, T2, K1, K2
```

`full4` is no longer part of the active ML or backtest pipeline.

The first primary strategy families are `K1` and `T1`, because the conservative
mini-backtest has positive aggregate P&L for those strategies. `K2` and `T2`
remain controls and possible secondary selections.

Because episode arrivals and P&L are uneven by year, do not use random splits.
Use an expanding walk-forward design:

- train from 2013 forward;
- first test year should be 2017 or 2018, depending on minimum row checks;
- validation should be the last 20% of training dates or the prior calendar
  year when enough observations exist;
- one-year test windows are the first paper-quality baseline;
- split by `signal_date` and keep each episode intact;
- report invalid selected rows in the final selected-trade accounting.

Eligible model inputs are point-in-time `feature_*` columns and entry-time
conservative bid/ask diagnostics. Never use future end-state fields, episode
end dates, realized P&L, realized returns, or future exit availability as model
features.

Start with logistic regression and gradient boosting. Transformer models should
come only after the tabular walk-forward baseline is stable.

## Completed Implementation Step

The end-state classification layer and episode-start reversion-label layer have
now been implemented.

Implemented files:

```text
src/tp2rr2/episode_end_states.py
scripts/run_classify_episode_end_states.py
src/tp2rr2/episode_start_labels.py
scripts/run_build_episode_start_labels.py
```

Implemented outputs:

```text
data/features/archive2_episodes/tp2_call_pair_episode_end_states_2013_2026.parquet
data/features/archive2_episodes/tp2_call_pair_episode_end_state_summary_2013_2026.json
reports/raw_archive2_eda/tables/episode_end_states/
data/labels/archive2_episode_start_labels_2013_2026.parquet
data/labels/archive2_episode_start_label_summary_2013_2026.json
reports/raw_archive2_eda/tables/episode_start_labels/
```

Episode-start label validation:

```text
rows: 2,098,213
unique episode IDs: 2,098,213
duplicate episode IDs: 0
missing end-state labels: 0
missing start detector features: 0
signal-date/start-date mismatches: 0
observable next-state rows: 1,093,735
true reversion rows: 1,065,111
```

Modeling rule:

- `feature_*` columns are point-in-time detector features.
- `label_*` columns are future labels or diagnostics and must not be model
  inputs.
- `ml_binary_reversion_target` is currently a reversion target, not a realized
  profitability target.

## Completed Mini-Backtest Step

The first episode-level mini-backtest layer has been implemented as a new
derived layer, not a rewrite of the detector or episode tracker.

Implemented files:

```text
src/tp2rr2/episode_mini_backtest.py
scripts/run_episode_mini_backtest.py
scripts/generate_episode_mini_backtest_report_assets.py
```

Implemented outputs:

```text
data/labels/archive2_episode_strategy_mini_backtest_2013_2026.parquet
data/labels/archive2_episode_strategy_mini_backtest_summary_2013_2026.json
reports/raw_archive2_eda/tables/episode_mini_backtest/
reports/episode_mini_backtest_report.pdf
```

Implemented rules:

- one candidate trade per episode start and strategy family;
- entry prices use bid/ask, not mid;
- transaction cost is \$0.65 per option contract side unless changed;
- invalid selected trades are counted, not silently dropped;
- model features must come only from `signal_date` and `feature_*` columns.

Broad mid-detected baseline result:

```text
strategy rows: 10,491,065
valid conservative mini-backtest rows: 5,239,276
valid row share: 49.94%
```

Important interpretation:

The first premium-resale baseline is strongly negative after bid/ask execution.
Do not train the final ML model on this target as if it were final. It is a
diagnostic baseline showing that clean TP2 reversion and realized resale profit
are different objects.

## Active Conservative Bid/Ask Subset

The active strategy set is now:

```text
T1, T2, K1, K2
```

`full4` has been removed from the active pipeline.

The conservative bid/ask TP2 filter is:

```text
feature_A_ask * feature_B_ask < feature_C_bid * feature_D_bid
```

Current active outputs:

```text
data/labels/archive2_episode_start_labels_conservative_bidask_2013_2026.parquet
data/labels/archive2_episode_start_labels_conservative_bidask_summary_2013_2026.json
data/labels/archive2_episode_strategy_mini_backtest_conservative_bidask_2013_2026.parquet
data/labels/archive2_episode_strategy_mini_backtest_conservative_bidask_summary_2013_2026.json
reports/conservative_bidask_subset_report.pdf
```

Current result:

```text
conservative bid/ask episode starts: 228,241
kept share: 10.88%
strategy rows: 912,964
valid mini-backtest rows: 427,924
valid net P&L: +$1.17B
median valid return: -1.05%
```

Interpretation:

The conservative subset fixes the main spread problem in aggregate, but median
valid returns remain negative. This means the ML target should not blindly use
all conservative rows. The next ML-ready target should test additional gates or
an expiration-payoff label.

## Next Implementation Step

Add robustness mini-backtest variants before model training:

- expiration-payoff labels, closer to the paper's profitability framing;
- shorter premium-resale holding rules;
- severity, spread, volume, open-interest, DTE, and moneyness entry gates;
- a final model-ready label table that explicitly chooses the paper's main
  trading target.

Do not overwrite:

```text
data/features/archive2_lastquote_forward/tp2_call_forward_features_lastquote_2013_2026.parquet
data/features/archive2_episodes/tp2_call_pair_episodes_2013_2026.parquet
data/features/archive2_episodes/tp2_call_episode_members_2013_2026.parquet
```

The end-state output should be a new derived layer.

## Validation Required

Before accepting the end-state classifier, check:

- every episode receives exactly one end-state label
- episode count in equals episode count out
- no duplicate `episode_id`
- `sample_ended` only occurs at the sample boundary
- `true_reversion` requires all four legs present and clean
- expired/missing legs are never counted as `true_reversion`
- classification counts by year are written to report tables

## After End-State Classification

Next steps after this layer:

1. Build episode-start training rows.
2. Attach reversion labels and mini-backtest labels to episode starts.
3. Run simple baselines:
   - trade all episode starts
   - severity threshold
   - liquidity threshold
4. Run ML on episode starts only.
5. Compare models:
   - logistic regression
   - random forest / XGBoost / LightGBM / CatBoost
   - Transformer or temporal model
6. Add explainability:
   - SHAP for tree models
   - attention or feature attribution for Transformer

## Paper Framing

Use careful language:

```text
TP2-inspired relative-value violations in American SPY calls
```

Do not call every detected mid-price violation a guaranteed arbitrage.

The strong claim should only be made after:

- bid/ask execution,
- transaction costs,
- invalid-selected accounting,
- liquidity controls,
- and leakage-safe episode-start ML.
