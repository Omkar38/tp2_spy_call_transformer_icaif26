# Current Archive-2 TP2 Pipeline

This document describes the current project pipeline after the Archive-2
dataset reset. It supersedes the older 2019-2024 JSON, put/call, five-strategy
pipeline.

## 1. Research Scope

- Underlying: SPY ETF options.
- Option type: calls only.
- Style caveat: SPY options are American-style and dividend/friction affected.
- Theory base: forward-adjusted call TP2 inequality inspired by
  Glasserman-Li-Pirjol.
- Sample: professor-provided Archive-2 data, 2013-04-01 to 2026-06-04.
- Current stage: full non-Transformer baseline is built. The repo contains
  EDA, full detector output, pair-episode tracking, lifecycle tracking,
  end-state classification, episode-start labels, liquid-exit strategy labels,
  leakage-safe rolling LightGBM, market-quality robustness, profitability
  stress tests, and moneyness/pricing-quality diagnostics.
- Next stage: handoff to temporal modeling / Transformer work, after adding a
  final deep-ITM/pricing-quality robustness pass.

This is SPY EOD relative-value research, not a pure SPX European arbitrage
replication.

## 1.1 Paper Alignment And Current Deviations

The motivating paper uses SPX European options from OptionMetrics and observes
quotes at market close, more precisely around 3:59pm. Our project uses SPY ETF
options and the latest available quote per contract-day. These differences are
acceptable for an empirical SPY study, but they must remain explicit in the
paper and reports.

Concepts we intentionally keep aligned with the paper:

- Forward-adjusted TP2 call inequality.
- `T1 < T2` and `K1/F_T1 < K2/F_T2` ordering.
- Rounded-up adjusted strikes for the right-hand-side options.
- Baseline use of mid prices for detecting violations.
- Reporting both violation counts and rates/densities, because the number of
  candidate pairs changes over time.

Differences and robustness items to disclose or add before final ML/backtest:

- Our quote time is latest available quote, not guaranteed exact 3:59pm.
- Our active clean data does not currently require positive daily volume.
- Our active clean data does not currently impose the paper's delta screen
  `0.01 <= abs(delta) <= 0.99`.
- Our detector checks adjusted strikes are available in bounds, but does not yet
  impose the paper's maximum distance rule for rounded strikes.
- The paper adds no-arbitrage filters before dynamic strategy testing; we should
  add analogous vertical/calendar sanity filters before final trading results.
- The paper's primary trading profitability is expiration-payoff based; any
  premium-resale or flexible-exit backtest should be reported as our extension,
  not as the paper's exact strategy.

## 2. Raw Data

Raw input lives locally under:

```text
Archive-2/spy_options_daily/year=*/data.parquet
```

The raw files are not committed to GitHub because several yearly parquet files
exceed GitHub's normal 100 MB file limit. The repository tracks metadata,
summaries, scripts, EDA tables, and reports.

One raw row is an option quote observation. The raw files can contain multiple
observations for the same `quote_date, contract_id`, so ingestion must collapse
raw observations to one quote per contract-day.

## 3. Active Clean And Quote-Availability Datasets

Active detector clean file:

```text
data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet
```

Cleaning rules:

- SPY calls only.
- All moneyness: ITM, ATM, and OTM calls are included.
- DTE 1-90.
- Latest available quote per contract-day.
- Valid bid/ask.
- Positive bid and positive mid.
- Spread percentage no greater than 50%.
- One row per `date, contractID`.

The active clean dataset has 4,912,891 SPY call contract-day rows across 3,256
sessions.

The lifecycle and liquid-exit rebuild also uses a wider quote-availability
file:

```text
data/processed/archive2_spy_call_quote_availability_nospread_2013_2026.parquet
```

That file keeps valid positive-bid call quotes without deleting observations
only because the spread is above 50%. It stores `spread_pct`,
`tradable_under_50pct_spread`, and `wide_spread_over_50pct`. This lets the
pipeline track whether a TP2 rectangle remains observable through time while
still enforcing the 50% spread rule only when a trade entry or exit is counted
as executable.

Rebuild command:

```bash
PYTHONPATH=src python3 scripts/run_ingest_archive2.py
```

The script defaults now point to the active full-sample all-call lastquote
dataset. Use `--otm-only` only for a narrower diagnostic universe.

## 4. Forward-Adjusted TP2 Detection

For calls with `T1 < T2` and `K1/F_T1 < K2/F_T2`, the no-violation condition is:

```text
C(K1,T1) C(K2,T2)
>=
C(K1 F_T2/F_T1, T2) C(K2 F_T1/F_T2, T1)
```

The detector defines:

```text
lhs = A_mid * B_mid
rhs = C_mid * D_mid
```

where:

```text
A = C(K1, T1)
B = C(K2, T2)
C = C(K1 adjusted to T2, T2)
D = C(K2 adjusted to T1, T1)
```

The call violation condition is:

```text
lhs < rhs
```

Adjusted strikes are rounded up to the nearest available strike in the target
expiration chain. The default detector threshold is:

```text
normalized_severity >= 0.0001
```

Full detector command:

```bash
PYTHONPATH=src python3 scripts/run_detect_archive2_lastquote_by_year.py --progress
```

Output:

```text
data/features/archive2_lastquote_forward/tp2_call_forward_features_lastquote_YYYY.parquet
data/features/archive2_lastquote_forward/tp2_call_forward_features_lastquote_2013_2026.parquet
data/features/archive2_lastquote_forward/tp2_call_forward_summary_lastquote_2013_2026.json
```

The combined detector output has 2,430,220 daily TP2 violation rows.

## 5. What One Violation Row Means

One clean row is one individual option quote for one contract on one date.

One violation row is one four-contract TP2 rectangle on one date. It is not one
option contract, not one selected ML trade, and not one executed trade.

Example interpretation:

```text
date = 2024-03-15
A = one call at K1,T1
B = one call at K2,T2
C = adjusted K1 call at T2
D = adjusted K2 call at T1
```

If `A_mid * B_mid < C_mid * D_mid`, the detector writes one violation row.

For 2013:

```text
99,129 clean option rows
17,847,362 raw K1/K2 strike pairs across expiration pairs
8,470,629 after forward-order condition
6,886,206 valid four-leg TP2 rectangles
73,619 TP2 violation rows
```

## 6. EDA Reports

Current report outputs:

```text
reports/raw_archive2_eda/archive2_raw_data_eda_report.pdf
reports/archive2_dataset_ingestion_report.pdf
```

Main EDA tables:

```text
reports/raw_archive2_eda/tables/lastquote_tp2_violation_counts_by_year.csv
reports/raw_archive2_eda/tables/lastquote_tp2_violation_counts_by_month.csv
reports/raw_archive2_eda/tables/lastquote_forward_coverage_by_year.csv
reports/raw_archive2_eda/tables/clean_dataset_version_comparison.csv
```

Regenerate TP2 violation EDA:

```bash
PYTHONPATH=src python3 scripts/generate_lastquote_tp2_violation_eda.py
```

Regenerate raw EDA report source:

```bash
PYTHONPATH=src python3 scripts/generate_archive2_raw_eda_report.py
```

Compile PDF:

```bash
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory reports/raw_archive2_eda \
  reports/raw_archive2_eda/archive2_raw_data_eda_report.tex
```

## 7. Why Episode Tracking Is Next

The current detector output is daily rectangle-level. The same economic
rectangle can violate on adjacent days. Counting each day independently can
inflate the sample size and can leak repeated observations across rolling
train/test splits.

The next pipeline stage should:

1. Build a primary paper-style pair key from the left-side options:
   `A = (K1,T1)` and `B = (K2,T2)`.
2. Also build an exact rectangle key from all four legs `A/B/C/D` as a
   diagnostic. This matters because the rounded adjusted `C` and `D` legs may
   shift slightly as forwards move.
3. Group repeated appearances of the same primary pair key into an episode.
4. Use the active clean dataset's trading-date calendar, not calendar-day
   differences, so weekends and holidays do not break episodes.
5. Define episode start date, last observed violation date, duration, row count,
   max severity, start severity, end severity, and whether exact rectangles
   changed inside the pair episode.
6. Label episode starts for ML.
7. Build episode-level mini-backtest profitability labels.
8. Use rolling-window ML and backtesting on episode-level observations.

This makes the final research design more defensible because the model learns
from economic events rather than repeated daily rows of the same event.

## 8. Episode Tracking Convention

The paper identifies a TP2 violation by the pair of left-side parameters
`(K1,T1)` and `(K2,T2)`, while noting that the inequality actually involves four
contracts. Therefore, the primary episode unit should be:

```text
pair_key = A_expiration | A_strike | B_expiration | B_strike
```

The exact four-leg rectangle should still be tracked:

```text
rectangle_key =
A_expiration | A_strike |
B_expiration | B_strike |
C_expiration | C_strike |
D_expiration | D_strike
```

Recommended outputs:

```text
data/features/archive2_episodes/tp2_call_pair_episodes_2013_2026.parquet
data/features/archive2_episodes/tp2_call_episode_members_2013_2026.parquet
data/features/archive2_episodes/tp2_call_episode_summary_2013_2026.json
```

Current full-sample episode build:

```text
Daily violation rows: 2,430,220
Pair episodes: 2,098,213
Unique pair keys: 1,792,462
Single-day episodes: 1,860,180 (88.66%)
Episodes with exact rectangle changes: 54,996 (2.62%)
Median duration: 1 session
Mean duration: 1.16 sessions
Max duration: 14 sessions
```

Minimum validation checks:

- Daily violation rows in equals episode-member rows out.
- Every member row receives exactly one `episode_id`.
- `episode_start_date <= episode_end_date`.
- `duration_sessions >= 1`.
- Episode count is no larger than daily violation-row count.
- No duplicate `violation_id` values are created.
- Episode split diagnostics identify any episode crossing train/validation/test
  boundaries.

Important diagnostic tables:

```text
episode_starts_by_year
episode_starts_by_month
daily_rows_vs_pair_episodes_by_year
duration_distribution
single_day_episode_share
exact_rectangle_changes_within_pair_episode
top_longest_episodes
```

The paper's correction analysis allows a pair to disappear for intermediate
days and later violate again, then defines the last violation day as the final
observed violation day. For our ML pipeline, the safer first implementation is
to build consecutive-session episodes and report gap diagnostics. If many pair
episodes reappear after short gaps, add a second "paper-style survival" view
that tracks the first and final observed violation day for each pair regardless
of gaps.

## 9. End-State And Episode-Start Label Layers

Episode tracking produces one row per consecutive-session pair event. End-state
classification then inspects the next available trading session after
`episode_end_date` and labels why the event stopped appearing.

Current end-state output:

```text
data/features/archive2_episodes/tp2_call_pair_episode_end_states_2013_2026.parquet
data/features/archive2_episodes/tp2_call_pair_episode_end_state_summary_2013_2026.json
reports/raw_archive2_eda/tables/episode_end_states/
```

Full-sample end-state counts:

```text
true_reversion: 1,065,111
right_leg_missing_or_expired: 536,599
liquidity_filter_failure: 313,781
left_leg_expired_or_missing: 152,779
still_violation_or_reappeared: 28,624
sample_ended: 1,319
```

True reversion means all four next-session legs are observable, pass clean
quote-quality rules, and `lhs >= rhs`. Missing, expired, liquidity-filtered, or
sample-ended episodes are not treated as reversion.

The newest derived dataset creates one ML-ready row per episode start:

```text
data/labels/archive2_episode_start_labels_2013_2026.parquet
data/labels/archive2_episode_start_label_summary_2013_2026.json
reports/raw_archive2_eda/tables/episode_start_labels/
```

This dataset joins the representative start violation row back to the full
detector feature table by `violation_id`, then joins the future end-state labels
by `episode_id`.

Validation:

```text
episode-start label rows: 2,098,213
unique episode IDs: 2,098,213
duplicate episode IDs: 0
missing end-state labels: 0
missing start detector features: 0
signal-date/start-date mismatches: 0
observable next-state rows: 1,093,735
observable next-state share: 52.13%
true reversion rows: 1,065,111
true reversion share among observable rows: 97.38%
```

Column convention:

- `signal_date` is the episode start date.
- `feature_*` columns are point-in-time detector features and are eligible ML
  inputs.
- `label_*` columns are future labels or future diagnostics and must not be
  used as model inputs.
- Episode duration and end-date metadata are retained for diagnostics, but are
  future-known and should not be model inputs.
- `ml_binary_reversion_target` is `1` for clean true reversion, `0` for clean
  still-violation next state, and missing for unobservable/invalid/censored
  rows.

The first episode-level mini-backtest has also been built:

```text
data/labels/archive2_episode_strategy_mini_backtest_2013_2026.parquet
data/labels/archive2_episode_strategy_mini_backtest_summary_2013_2026.json
reports/raw_archive2_eda/tables/episode_mini_backtest/
reports/episode_mini_backtest_report.pdf
```

Initial broad mid-detected policy:

- one candidate row per episode start and strategy family;
- legacy broad output included `full4`, `T1`, `T2`, `K1`, `K2`;
- enter the original episode-start contracts at bid/ask;
- exit only when the episode end-state is clean `true_reversion`;
- close the same opened contracts at exit-day bid/ask;
- charge `$0.65` per option contract side;
- retain invalid/unobservable rows instead of silently dropping them.

Results:

```text
strategy rows: 10,491,065
valid conservative mini-backtest rows: 5,239,276
valid row share: 49.94%
```

The premium-resale baseline is strongly negative after bid/ask execution. This
does not reject the TP2 research idea, but it means clean TP2 reversion should
not be treated as automatically profitable. The next step is to compare
expiration-payoff, shorter premium-resale, liquidity-gated, and conservative
bid/ask-detection variants before final ML training.

The active strategy set has now been narrowed to the four Glasserman-style
denominations:

```text
T1, T2, K1, K2
```

`full4` is no longer part of the active pipeline.

A conservative bid/ask TP2 subset has been added:

```text
data/labels/archive2_episode_start_labels_conservative_bidask_2013_2026.parquet
data/labels/archive2_episode_start_labels_conservative_bidask_summary_2013_2026.json
data/labels/archive2_episode_strategy_mini_backtest_conservative_bidask_2013_2026.parquet
data/labels/archive2_episode_strategy_mini_backtest_conservative_bidask_summary_2013_2026.json
reports/conservative_bidask_subset_report.pdf
```

Filter rule:

```text
feature_A_ask * feature_B_ask < feature_C_bid * feature_D_bid
```

Results:

```text
full episode-start rows: 2,098,213
conservative bid/ask TP2 rows: 228,241
kept share: 10.88%
strategy rows after removing full4: 912,964
valid mini-backtest rows: 427,924
valid net P&L: +$1.17B
median valid return: -1.05%
```

Interpretation: the conservative bid/ask filter fixes the major spread problem
and produces positive aggregate P&L, especially in `T1` and `K1`. But median
returns are still negative, so the result is tail-driven and should be refined
before ML training.

## 10. Pre-ML Gating Diagnostics

The pre-ML gating analysis has been added to avoid training directly on a
noisy, uneven episode universe.

Implemented file:

```text
scripts/run_pre_ml_gating_analysis.py
```

Outputs:

```text
reports/pre_ml_gating_and_rolling_plan.pdf
reports/raw_archive2_eda/tables/pre_ml_gating/
reports/figures/pre_ml_strategy_year_stability.png
reports/figures/pre_ml_top_gates_by_net_pnl_per_candidate.png
```

Base conservative strategy result:

```text
K1: 228,241 candidates, 107,326 valid, +$639.36M net P&L, -0.68% median return
T1: 228,241 candidates, 106,636 valid, +$604.68M net P&L, -1.00% median return
K2: 228,241 candidates, 106,636 valid, -$53.56M net P&L, -1.44% median return
T2: 228,241 candidates, 107,326 valid, -$18.94M net P&L, -1.12% median return
```

Interpretation:

- `K1` and `T1` are the first primary ML strategy families.
- `K2` and `T2` remain useful controls and may still matter in a strategy
  selector, but they are not the first headline models.
- Aggregate P&L is positive for `K1` and `T1`, but all four strategy medians
  are negative. This means the current opportunity is tail-driven and must be
  evaluated with calendar-respecting out-of-sample splits.
- 2017 contributes unusually large positive P&L, while 2013 is negative across
  all active strategies. Random train/test splits would hide this instability.

Use the gate-search tables as diagnostics and model features. Do not choose a
single best in-sample gate as the final research result.

## 11. Rolling ML Design And Current LightGBM Run

Because episode arrivals are uneven across time, the ML design does not use
random splits or fixed three-month windows as the main paper result.

Current protocol:

- Observation unit: one `episode_id` row.
- Candidate universe: conservative bid/ask TP2 subset only.
- Active actions: `skip`, `T1`, `T2`, `K1`, `K2`.
- Label: choose the strategy with the largest positive first-liquid net P&L;
  otherwise choose `skip`.
- Split: 3-year train, next 1-year validation, next 1-year test, rolled forward
  by one year.
- Current primary test years: 2021-2026.
- Purge: split by `signal_date`; keep the episode unit intact across train,
  validation, and test.

Current folds:

```text
train 2017-2019, validate 2020, test 2021
train 2018-2020, validate 2021, test 2022
train 2019-2021, validate 2022, test 2023
train 2020-2022, validate 2023, test 2024
train 2021-2023, validate 2024, test 2025
train 2022-2024, validate 2025, test 2026
```

Current LightGBM outputs:

```text
data/ml/archive2_ml_episode_dataset_2013_2026.parquet
data/ml/archive2_ml_lightgbm_rolling_predictions.parquet
data/ml/archive2_ml_lightgbm_rolling_summary.csv
reports/ml_lightgbm_rolling_report.pdf
```

The test-year selected rows are all counted. If the model selects a strategy
but no valid liquid exit is available, the row is counted as
`invalid_selected`, not silently removed.

Eligible inputs are point-in-time `feature_*` columns plus entry-time
conservative bid/ask diagnostics. Do not train on `label_*`, episode end dates,
future reversion states, realized P&L, or future exit availability.

The current LightGBM label is dollar-P&L based. Before final paper claims, add
return-normalized and capped-P&L variants so strategy selection is not driven
only by larger dollar exposure.

The transformer should come after this tabular walk-forward baseline is stable.

## 12. Current Robustness Status And Remaining Work

The pipeline has moved past the original pre-ML stage. The following robustness
layers are now implemented:

- episode-level observation units rather than repeated daily violation rows;
- conservative bid/ask entry filter for TP2 violations;
- first-liquid exit labels with invalid selections counted explicitly;
- four active strategy actions only: `T1`, `T2`, `K1`, `K2`;
- leakage-safe rolling LightGBM using 3-year train, 1-year validation, and
  1-year test folds;
- market-quality gates using volume, delta, spread, and open-interest
  diagnostics;
- commission and tail-concentration stress tests;
- moneyness and intrinsic-value pricing diagnostics.

Before professor Transformer handoff / final paper claims, the remaining
highest-priority additions are:

1. Positive volume sensitivity:
   keep the existing volume-gate results and decide whether the final headline
   model should require positive volume.
2. Delta sensitivity:
   keep the existing delta-gate results and decide whether the final headline
   model should require `0.01 <= abs(delta) <= 0.99`.
3. Rounded-strike distance:
   track the distance between forward-adjusted target strikes and selected
   rounded strikes; optionally discard very distant rounded matches.
4. Static no-arbitrage sanity filters:
   add vertical and calendar checks before claiming trade profitability.
5. Deep-ITM/pricing-quality filter:
   reduce intrinsic-value dominated ITM rows and audit whether OTM/ATM pricing
   quality remains acceptable.
6. Concentration control:
   compare capped-P&L, return-normalized, top-k/day, and exposure-limited
   objectives against the current dollar-P&L LightGBM action label.

These additions are not required to detect TP2 violations, but they make the
research pipeline cleaner and more defensible for publication.
