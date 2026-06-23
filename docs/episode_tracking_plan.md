# Episode Tracking Plan

This note records the intended episode-tracking design before implementation.
It is meant to keep the research logic clear while we move from daily TP2
violation rows to episode-level observations.

## What Episode Tracking Means

Right now, the detector gives daily violation rows.

Example:

```text
2021-09-01: rectangle A-B-C-D violates TP2
2021-09-02: same rectangle A-B-C-D still violates TP2
2021-09-03: same rectangle A-B-C-D still violates TP2
```

Currently, this is counted as three violation rows.

Economically, it is one continuing violation event. Episode tracking means:

```text
Group repeated same/similar TP2 violation across consecutive trading days
into one violation episode.
```

Instead of asking:

```text
How many daily violation rows exist?
```

we ask:

```text
How many unique violation events started?
How long did they last?
Did they revert?
Was there a profitable trade opportunity?
```

This is stronger for research and ML.

## Implementation Design

### Step 1: Create A Stable Pair Key

For each violation row, create the primary episode key from the paper-style
left-side pair:

```text
A option = C(K1,T1)
B option = C(K2,T2)
```

The implemented primary key is:

```text
pair_key =
A_expiration|A_strike|B_expiration|B_strike
```

The exact four-leg rectangle is still tracked as a diagnostic:

```text
rectangle_key =
A_expiration|A_strike|B_expiration|B_strike|C_expiration|C_strike|D_expiration|D_strike
```

### Step 2: Sort Rows

Sort by:

```text
pair_key, date
```

### Step 3: Detect Consecutive Appearances

If the same `pair_key` appears on adjacent available trading days, it
belongs to the same episode.

Example:

```text
2021-09-01 same key
2021-09-02 same key
2021-09-03 same key
```

This is one episode.

If there is a gap:

```text
2021-09-01 same key
2021-09-02 same key
2021-09-08 same key
```

Then this is likely two episodes, unless we deliberately allow small gaps.

### Step 4: Assign Episode Fields

Each daily violation row should receive:

```text
episode_id
episode_start_date
episode_end_date
episode_duration_days
episode_row_count
max_normalized_severity
start_normalized_severity
end_normalized_severity
```

### Step 5: Define Episode Status

Potential status values:

```text
active_until_last_seen
reverted
expired_or_untradeable
gap_ended
sample_ended
```

First implemented version ends an episode mechanically:

```text
episode ends when the same pair no longer appears on the next trading day
```

Later, improve this by checking whether the TP2 inequality actually reverted,
not just disappeared due to missing quote/liquidity.

## Potential Problems And Handling

### 1. Slightly Changing Adjusted Strikes

The same economic violation may change slightly because adjusted strikes round
differently.

Example:

```text
Day 1: C rounded strike = 451
Day 2: C rounded strike = 452
```

Economically similar, but an exact key says it is a different episode.

Handling:

- Use the left-side pair key as the primary episode unit.
- Track exact four-leg rectangle changes as diagnostics:

```text
How many episodes last only 1 day?
How often do near-identical keys appear next day?
```

If pair-level matching is still too strict or too loose, later test an alternate
survival view using:

```text
K1, K2, T1, T2
```

instead of all four rounded legs.

Implementation decision: use pair-level episodes as the primary unit and exact
four-leg rectangle changes as a diagnostic. This matches the paper's convention
of identifying violations by the left-side pair while still preserving the fact
that a TP2 inequality uses four option contracts.

### 2. Non-Trading Days And Missing Sessions

Friday to Monday should still be consecutive in trading time.

Do not use calendar-day difference like:

```text
date_diff == 1
```

Instead, build a trading calendar from the active clean dataset:

```text
sorted unique clean dates
```

Consecutive means:

```text
next date in our available trading-date list
```

This avoids breaking episodes over weekends and holidays.

### 3. Missing Quote Can Falsely End An Episode

If one leg is missing or filtered out, the detector cannot form the rectangle,
so the episode disappears. That does not always mean the arbitrage reverted.

Handling:

First version:

```text
not_detected_next_session
```

Later classification:

```text
reverted_price
missing_leg_quote
expired
liquidity_filtered
```

### 4. Leakage Across ML Splits

This is one of the main reasons episode tracking matters.

Bad situation:

```text
Episode starts in train period
Same episode continues into validation/test
```

If we train on early days and test on later days of the same episode, that leaks
information.

Handling:

For ML, split by:

```text
episode_start_date
```

Optionally purge/embargo episodes that overlap split boundaries.

### 5. Episode Count May Become Much Smaller

This is expected and good.

Full current sample:

```text
Daily rows: 2,430,220
Pair episodes: 2,098,213
```

Episode count may be much lower. That is not a problem; it is more honest.

### 6. Huge Data Size

The combined TP2 parquet is about 514 MB locally. Episode tracking should avoid
unnecessary memory blowups.

Use only necessary columns:

```text
date
violation_id
K1
K2
T1_expiration
T2_expiration
K1_forward_rounded
K2_forward_rounded
A_option_symbol / A_expiration / A_strike
B_option_symbol / B_expiration / B_strike
C_option_symbol / C_expiration / C_strike
D_option_symbol / D_expiration / D_strike
normalized_severity
lhs
rhs
```

## Files To Add

Suggested implementation files:

```text
src/tp2rr2/episodes.py
scripts/run_build_episodes.py
```

Suggested outputs:

```text
data/features/archive2_episodes/tp2_call_episodes_2013_2026.parquet
data/features/archive2_episodes/tp2_call_episode_members_2013_2026.parquet
data/features/archive2_episodes/tp2_call_episode_summary_2013_2026.json
```

Meaning:

`episode_members`

Daily violation rows with assigned `episode_id`.

`episodes`

One row per episode.

`summary.json`

Counts, duration stats, year-wise episode starts, etc.

## How To Avoid Breaking The Pipeline

Do not modify the detector output:

```text
data/features/archive2_lastquote_forward/tp2_call_forward_features_lastquote_2013_2026.parquet
```

Episode tracking is a new layer:

```text
TP2 detector rows -> episode builder -> episode table
```

If episode logic is wrong, detector and EDA remain safe.

First run a small test:

```text
2013 only
```

Validate:

```text
daily rows in = daily rows out with episode_id
no duplicate violation_id lost
episode_start <= episode_end
duration >= 1
episode count <= daily row count
```

Then run full 2013-2026.

## Recommended First Version

Start with exact rectangle tracking:

```text
rectangle_key = A/B/C/D contract identifiers
consecutive = next available trading session
```

Generate:

```text
episode_id
episode_start_date
episode_end_date
duration_sessions
episode_row_count
start_severity
max_severity
mean_severity
end_severity
start_year
end_year
```

Then inspect:

```text
episode duration distribution
single-day episode percentage
episode starts by year
daily rows vs episode count by year
top long episodes
```

After that, decide whether exact matching is too strict and whether fuzzy
matching is needed.
