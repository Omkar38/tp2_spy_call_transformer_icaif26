# Liquid-Exit Rebuild Plan

## Purpose

This plan records the next pipeline rebuild before implementation. The main
change is to stop using the 50% bid/ask spread rule as an early deletion rule
for violation behavior. Instead:

- keep wide-spread quotes in the research/behavior panel;
- tag liquidity every day;
- require liquidity only when a trade is actually entered or exited.

This avoids losing useful TP2 lifecycle information while keeping the trading
backtest realistic.

## Core Research Question

For a TP2 violation that is tradable at entry, does the best defensible exit
occur:

- exactly on the first true reversion day;
- on the first liquid date after true reversion;
- a fixed number of sessions after reversion;
- or at T1 expiration?

This matters because many violations revert quickly, but the exact reversion
day may not be the best or most executable exit date. A paper-quality pipeline
should measure these exit policies explicitly rather than forcing every trade to
close on the first reversion day.

## Key Conceptual Rules

### 1. Do Not Delete Wide-Spread Quotes Before Tracking

The quote availability universe should keep valid call quotes with:

```text
bid > 0
ask > bid
mid > 0
DTE in active range
latest available quote for that contract-day
```

Do not apply:

```text
spread_pct <= 50%
```

as an early row deletion rule for lifecycle tracking. Instead store:

```text
spread_pct
tradable_under_50pct_spread
wide_spread_over_50pct
```

### 2. Detection And Execution Are Different

Detection asks whether the TP2 inequality is violated.

Execution asks whether the selected option legs can actually be bought/sold at
reasonable bid/ask prices.

The pipeline should not confuse these.

### 3. Entry Must Be Executable

For the main trading backtest, do not enter a trade unless entry is executable.

Primary entry condition:

```text
conservative bid/ask TP2 violation:
A_ask * B_ask < C_bid * D_bid
```

plus liquidity:

```text
all four A/B/C/D legs pass the entry liquidity threshold
```

This keeps the paper's main trading subset defensible. A later robustness table
can test strategy-leg-only liquidity, but the main version should keep all four
legs clean at signal time.

### 4. Reversion Does Not Require Immediate Exit Liquidity

True reversion should be measured from the TP2 inequality becoming non-violating
again:

```text
A_mid * B_mid >= C_mid * D_mid
```

using the no-spread quote availability universe.

If the reversion day is not executable, the trade should not be thrown away
immediately. Instead, the liquid-exit policy waits for the first later date
where the selected strategy legs can be closed.

### 5. Exit Must Be Executable

For each selected strategy, exit requires executable quotes for the selected
legs only.

For example, T1 uses selected legs:

```text
A and D
```

So after reversion, T1 exit is valid on the first date where A and D have:

```text
bid > 0
ask > bid
mid > 0
spread_pct <= threshold
date < selected leg expiration
```

## Proposed Rebuilt Pipeline

### Step 1: Quote Availability Dataset

Input:

```text
raw Archive-2 professor dataset, 2013-2026
```

Output:

```text
data/processed/archive2_spy_call_quote_availability_nospread_2013_2026.parquet
```

Rules:

- calls only;
- valid bid/ask and positive mid;
- positive bid;
- DTE range retained from current pipeline;
- latest available quote per contract-day;
- no 50% spread deletion;
- store spread and liquidity flags.

This step mostly already exists, but the rebuild should verify it is the only
quote source used by lifecycle and liquid-exit logic.

### Step 2: TP2 Candidate And Violation Detection

Use the no-spread quote availability dataset.

For each valid four-leg rectangle:

```text
lhs_mid = A_mid * B_mid
rhs_mid = C_mid * D_mid
mid_violation = lhs_mid < rhs_mid

lhs_conservative = A_ask * B_ask
rhs_conservative = C_bid * D_bid
conservative_bidask_violation = lhs_conservative < rhs_conservative
```

Store both. Do not make the 50% spread filter decide whether a violation exists.

### Step 3: Episode Tracking

Group repeated violations by the same left-side economic pair:

```text
pair_key = A_expiration | A_strike | B_expiration | B_strike
```

Episode start is the first day of a consecutive violation run.

Store at episode start:

- mid violation severity;
- conservative bid/ask severity;
- all four leg spreads;
- all four leg volume/open interest;
- whether all four legs are tradable under the chosen liquidity threshold.

Main trading universe:

```text
conservative_bidask_violation == True
entry_all_four_legs_liquid == True
```

Behavior/EDA universe:

```text
mid_violation == True or conservative_bidask_violation == True
```

with liquidity flags retained.

### Step 4: Lifecycle Tracking

For each episode, track from:

```text
20 sessions before episode start
through T1 expiration
```

On every lifecycle date:

- reconstruct A/B from original contracts;
- recompute current forwards;
- reconstruct C/D by forward-adjusted round-up strike rule;
- compute mid pressure;
- compute conservative bid/ask pressure;
- store all spreads and liquidity flags;
- classify row status without hiding wide-spread observations.

Important status distinction:

```text
observable_but_wide_spread
```

should be different from:

```text
missing_leg
missing_forward
not_forward_ordered
expired_T1
```

### Step 5: Reversion Event Label

For each episode, find:

```text
first_reversion_date
```

where:

```text
A/B/C/D are reconstructable in the no-spread quote availability universe
A_mid * B_mid >= C_mid * D_mid
date <= T1 expiration
```

If there is no such date before T1:

```text
end_state = no_reversion_before_T1
```

If legs disappear before reversion:

```text
end_state = unobservable_before_reversion
```

If sample ends first:

```text
end_state = sample_end
```

### Step 6: Multi-Exit Strategy Labels

For each episode and each strategy:

```text
T1, T2, K1, K2
```

compute product-weighted P&L using Glasserman-style quantities:

```text
T1: long B_mid units of A, short C_mid units of D
T2: long A_mid units of B, short D_mid units of C
K1: long B_mid units of A, short D_mid units of C
K2: long A_mid units of B, short C_mid units of D
```

Entry execution:

```text
long leg: buy at ask
short leg: sell at bid
```

Exit execution:

```text
close long leg: sell at bid
close short leg: buy at ask
```

Create separate exit-policy labels:

```text
exit_at_reversion_day
exit_first_liquid_on_or_after_reversion
exit_reversion_plus_1_session
exit_reversion_plus_3_sessions
exit_reversion_plus_5_sessions
exit_T1_expiration_payoff
best_exit_before_T1_upper_bound
```

The paper's live-tradable primary candidate should be:

```text
exit_first_liquid_on_or_after_reversion
```

The `best_exit_before_T1_upper_bound` label is diagnostic only and must not be
reported as an implementable strategy.

### Step 7: ML Dataset

ML observation unit:

```text
episode_id + strategy + exit_policy
```

Features must be available at entry time only:

- entry severity;
- conservative bid/ask severity;
- moneyness;
- DTE and term gap;
- spreads;
- volume/open interest;
- delta if available;
- prior lifecycle pressure slope from days <= 0;
- prior near-violation count from days <= 0;
- year/month/calendar controls.

Never use:

- first reversion date;
- future pressure path;
- future liquidity;
- realized P&L;
- exit availability;
- end-state label

as model inputs.

Targets:

```text
trade_profitable
net_pnl
realized_return
best_strategy_exit_policy among finite live-tradable policies
```

### Step 8: Required Reports

Produce a new PDF report after rebuild:

```text
reports/liquid_exit_rebuild_report.pdf
```

Required tables/figures:

1. Number of violations by year under:
   - mid condition;
   - conservative bid/ask condition;
   - conservative + entry-liquid condition.

2. Episode counts by year:
   - total episodes;
   - entry-liquid episodes;
   - wide-spread-at-entry episodes.

3. Reversion timing:
   - first reversion day distribution;
   - no reversion before T1;
   - unobservable before reversion.

4. Exit availability:
   - reversion day liquid;
   - first liquid after reversion;
   - days waited after reversion.

5. Strategy P&L by year and exit policy:
   - T1/T2/K1/K2;
   - gross P&L;
   - net P&L;
   - hit rate;
   - invalid selected count.

6. ML-ready label summary:
   - class balance;
   - P&L distribution;
   - available observations by year.

## Expected Benefits

This rebuild should make the project stronger because it:

- preserves TP2 behavior even when a quote is temporarily illiquid;
- avoids counting missing liquidity as economic reversion;
- gives a realistic exit rule for trades whose reversion day is not executable;
- creates richer ML labels for selecting both strategy and exit policy;
- gives a clear robustness bridge between Glasserman-style expiration payoff and
  premium-resale trading.

## Implementation Guardrails

- Do not overwrite raw data.
- Keep the current reports available until the new report is validated.
- Use new filenames for rebuilt labels and reports.
- Keep product-weighted strategies as the primary P&L calculation.
- Keep unit-sized strategy code only as a diagnostic, not as the main research
  result.
- Do not push changes until explicitly requested.

## Implemented Outputs

Implemented as a parallel rebuild so the prior pipeline remains intact.

```text
scripts/build_liquid_exit_labels.py
scripts/generate_liquid_exit_report.py
data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet
data/labels/archive2_liquid_exit_strategy_labels_summary_2013_2026.json
reports/liquid_exit_rebuild_report.pdf
reports/raw_archive2_eda/tables/liquid_exit_rebuild/
reports/figures/liquid_exit_*.png
```

The label dataset contains one row per:

```text
episode_id + strategy + exit_policy
```

Current implemented exit policies:

```text
exit_at_reversion_day
exit_first_liquid_on_or_after_reversion
exit_reversion_plus_1_session
exit_reversion_plus_3_sessions
exit_reversion_plus_5_sessions
exit_selected_leg_expiration_payoff
```

The main live-style policy is:

```text
exit_first_liquid_on_or_after_reversion
```

The expiration-payoff policy is included as a Glasserman-style benchmark, not as
the same execution question as premium resale.
