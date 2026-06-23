# Violation Lifecycle Tracking Plan

## Purpose

This note records the next research direction before implementation. The goal is
to study how a TP2 candidate pair behaves before the first detected violation,
on the violation day, after the violation, and until the earlier expiration
`T1`.

This is different from the current episode pipeline:

- Current episode tracking groups repeated daily violations so the same
  economic event is not counted many times.
- Lifecycle tracking will create a time series around each episode start, even
  on days where the pair is not violating.

The purpose is to understand whether the path of the TP2 gap helps decide when
to trade and which violations are likely to become profitable.

## Pre-Violation Pair Universe Count

The clean all-call lastquote dataset has:

```text
clean call rows: 4,912,891
trading sessions: 3,256
date-expiration pairs: 385,222
```

There are several valid meanings of "pair" in this project.

### All Clean Strikes

Using every clean strike in each expiration chain:

```text
raw cross-expiration A/B pairs: 4,140,337,398
forward-ordered A/B pairs:      2,119,251,213
valid four-leg rectangles:      1,695,489,562
```

`raw cross-expiration A/B pairs` means every possible `(A = K1,T1)` and
`(B = K2,T2)` with `T1 < T2`.

`forward-ordered A/B pairs` applies the TP2 ordering:

```text
K1 / F_T1 < K2 / F_T2
```

`valid four-leg rectangles` means the adjusted right-side contracts `C` and `D`
can be formed after forward adjustment and round-up strike selection.

### Detector-Equivalent Universe

The active detector limits each expiration chain to at most 80 near-underlying
strikes. In that detector-equivalent universe:

```text
raw cross-expiration A/B pairs: 1,908,491,013
forward-ordered A/B pairs:      1,062,894,485
valid four-leg rectangles:        848,906,267
```

The current detector found:

```text
daily TP2 violation rows: 2,430,220
pair episodes:            2,098,213
```

So the violation set is a small subset of the valid pre-violation rectangle
universe.

Generated count tables:

```text
reports/raw_archive2_eda/tables/pre_violation_pair_universe/
```

## Gap Definition

Use one signed TP2 gap consistently.

For calls:

```text
lhs = A_mid * B_mid
rhs = C_mid * D_mid
```

The no-violation condition is:

```text
lhs >= rhs
```

To match the intuitive statement "violation day is +200", define:

```text
violation_pressure = rhs - lhs
```

Interpretation:

- `violation_pressure > 0`: TP2 violation.
- `violation_pressure = 0`: boundary.
- `violation_pressure < 0`: no violation; inequality has safe margin.

Also store normalized versions:

```text
normalized_pressure = (rhs - lhs) / (rhs + lhs)
log_pressure = log(rhs / lhs)
```

The normalized metrics are better for charts because raw dollar-product gaps are
not comparable across years, price levels, or deep ITM/OTM contracts.

## Lifecycle Unit

The primary lifecycle unit should be the episode-start pair:

```text
episode_id
pair_key = A_expiration | A_strike | B_expiration | B_strike
```

For each episode start, track the same A/B pair backward and forward.

Recommended first universe:

```text
conservative bid/ask episode starts
```

Reason: the full 2.1M episode universe is large, and the conservative subset is
the one intended for trading research. The full universe can be added later as a
robustness aggregation.

## Time Window

Track:

```text
lookback: 20 trading sessions before first violation
event day: 0 = episode_start_date
forward: until min(T1 expiration, sample end)
```

Do not track beyond `T1` expiration in the main lifecycle panel. After `T1`, the
left-side short-maturity leg no longer exists, so the original TP2 rectangle is
no longer economically the same object.

## Daily Reconstruction Rule

For each lifecycle day:

1. Check whether A and B contracts exist and have valid quotes.
2. Compute current forwards for `T1` and `T2`.
3. Recompute the adjusted right-side strikes:

```text
C target = K1 * F_T2 / F_T1
D target = K2 * F_T1 / F_T2
```

4. Round up to the nearest available strike in the relevant expiration chain.
5. If C and D exist with valid quotes, compute:

```text
lhs, rhs, violation_pressure, normalized_pressure, log_pressure
```

6. Store a status if the pair cannot be evaluated:

```text
not_yet_listed
missing_A_or_B
missing_C_or_D
liquidity_failure
not_forward_ordered
expired_T1
sample_end
observable
```

Important: the C/D legs can change across dates because forwards and available
strikes change. That should be stored, not hidden.

## Spread Filter Caveat

The active clean dataset currently removes option contract-days where:

```text
spread_pct > 50%
```

after applying these earlier conditions:

```text
calls only
DTE 1-90
valid bid/ask
ask > bid
mid > 0
bid > 0
latest available quote
```

Exact impact of the 50% spread rule:

```text
eligible call contract-days before spread filter: 5,128,025
removed by spread > 50%:                         215,134
kept after spread filter:                       4,912,891
removed share of eligible:                          4.20%
```

Yearly removal ranges from about 3.1% to 5.2% of otherwise eligible call
contract-days. So this filter is not deleting most of the option universe, but
it can still remove important legs for some TP2 rectangles.

This matters for lifecycle tracking and end-state classification:

- If a leg exists in raw/vendor data but has spread above 50%, the active clean
  dataset treats it as unavailable.
- That can contribute to `left_leg_expired_or_missing`,
  `right_leg_missing_or_expired`, or `liquidity_filter_failure` labels.
- For behavior analysis, this may hide the path of the TP2 gap.
- For tradable backtesting, this is still a useful execution-quality filter.

Recommended fix:

1. Build a separate **quote-availability lifecycle panel** with no 50% spread
   exclusion after valid bid/ask and positive bid checks.
2. Store the spread and a `tradable_under_50pct_spread` flag each day.
3. Use the wider panel to study gap behavior.
4. Use the spread/tradability flag for trade execution and ML robustness.

In other words, do not permanently discard wide-spread observations before
behavior tracking. Keep them as observable-but-illiquid rows.

## What To Store

Output one row per:

```text
episode_id + lifecycle_date
```

Core columns:

```text
episode_id
pair_key
lifecycle_date
event_day_index
episode_start_date
T1_expiration
T2_expiration
days_to_T1
status
A/B/C/D contract identifiers
A/B/C/D strikes
A/B/C/D bid/ask/mid/spread
lhs
rhs
violation_pressure
normalized_pressure
log_pressure
is_violation
is_conservative_bidask_violation
```

Join later:

```text
strategy P&L labels
product-weighted P&L labels from the Glasserman-style strategy sizing
unit-sized P&L labels only as a secondary diagnostic
end-state label
```

## Charts

Do not make one chart per violation. There are too many episodes.

Use aggregate event-time charts:

1. Median and percentile band:
   - x-axis: event day index, e.g. `-20` to `T1`.
   - y-axis: `normalized_pressure`.
   - lines: median, 25/75 percentile, 10/90 percentile.

2. Grouped lifecycle chart:
   - split by eventual outcome:
     - profitable T1/K1
     - unprofitable T1/K1
     - true reversion
     - missing/expired/liquidity failure

3. Heatmap:
   - x-axis: event day index.
   - y-axis: normalized pressure bin.
   - color: number/share of episodes.

4. Small sample traces:
   - only for the top 20 high-profit and bottom 20 low-profit episodes.
   - useful for intuition, not as headline evidence.

5. Hazard-style chart:
   - x-axis: days since first violation.
   - y-axis: probability of first reversion or probability of profitable exit.

## Conceptual Critique

The intuition is mostly correct:

- Tracking only the violation day is incomplete.
- The pre-violation path can reveal whether a violation is sudden or slowly
  building.
- The post-violation path can show whether the gap typically closes quickly,
  persists, worsens, or disappears due to missing/expired legs.
- This can create useful ML features such as slope, acceleration, persistence,
  and prior near-violation counts.

But there are important caveats:

1. Do not use future path values as trade-entry features.
   - The future lifecycle is for label analysis and model evaluation.
   - Only pre-event and event-day values can be used for live trade selection.

2. Raw gap size is not comparable across years.
   - Use normalized pressure or log pressure for charts and ML features.

3. Reversion of the inequality is not the same as profit.
   - A gap can revert but still lose money after bid/ask and commissions.
   - Lifecycle labels must be joined to actual strategy P&L.

4. Tracking until expiration is useful, but the main object should stop at `T1`.
   - After `T1`, the original short-maturity leg disappears.
   - For expiration-payoff research, create a separate payoff label.

5. The full pair universe is huge.
   - Start with conservative episode starts.
   - Store full lifecycle rows for that universe first.
   - Use streaming aggregation for the full 2.1M episode universe later if
     needed.

## Implementation Plan

1. Build lifecycle panel for conservative bid/ask episode starts.
2. Validate row counts and statuses.
3. Create event-time aggregate tables.
4. Create lifecycle figures.
5. Join lifecycle features to strategy P&L labels.
6. Decide which past-only lifecycle features are eligible for rolling ML.

Suggested first output paths:

```text
data/features/archive2_lifecycle/tp2_conservative_episode_lifecycle_2013_2026.parquet
reports/raw_archive2_eda/tables/pre_violation_pair_universe/
reports/raw_archive2_eda/tables/lifecycle_tracking/
reports/figures/lifecycle_*.png
```

## Implemented Outputs

Implemented on the conservative bid/ask episode universe:

```text
data/processed/archive2_spy_call_quote_availability_nospread_2013_2026.parquet
data/features/archive2_lifecycle/tp2_conservative_episode_lifecycle_2013_2026.parquet
data/features/archive2_lifecycle/tp2_conservative_episode_lifecycle_summary_2013_2026.json
reports/raw_archive2_eda/tables/lifecycle_tracking/
reports/figures/lifecycle_normalized_pressure_band.png
reports/figures/lifecycle_violation_tradability_share.png
reports/figures/lifecycle_status_mix_by_event_day.png
reports/figures/lifecycle_pressure_by_T1_outcome.png
reports/figures/lifecycle_pressure_heatmap.png
reports/lifecycle_tracking_report.pdf
```

The lifecycle panel has:

```text
episodes: 228,241
lifecycle rows: 6,409,304
event day range: -20 to +53
```

Day 0 is anchored to the original detected violation's C/D contracts, so the
event-day row matches the conservative bid/ask detector by construction:

```text
day 0 mid-price violation share: 100%
day 0 conservative bid/ask violation share: 100%
day 0 median normalized pressure: 0.1927
```

Other dates dynamically reconstruct the C/D legs using detector-equivalent
forward-adjusted round-up strike selection on the no-spread quote availability
panel.
