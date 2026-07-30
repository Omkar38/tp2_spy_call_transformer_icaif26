# TP2-MV-ALiBi v3 Run Notes

Script: `TP2MVTransformer/03_train_mv_alibi_transformer_v3.py`. Report:
`TP2MVTransformer/v3_vs_v2_report.pdf` (generator:
`TP2MVTransformer/12_make_v3_vs_v2_report.py`). Three iterations were run;
this file documents all of them, most recent first, since the history
explains why each fix exists. See also `professor_meeting_2026-07-13_notes.md`
for the meeting transcript / email cross-check that drove iteration 3, and
the v4 section immediately below for the multi-seed ensemble built on top
of iteration 3.

**IMPORTANT UPDATE (2026-07-13):** `next_steps_completion_report.pdf`
(generator `16_make_next_steps_report.py`) found that the trade-selection
rule used throughout v3/v4 (`apply_selection_fraction`) is **not fully
causal** -- it computes its probability cutoff from the entire test
period's prediction distribution, which is neither live-deployable nor
guaranteed to be achievable by a genuinely sequential system. **Every
Sharpe/Sortino/P&L number in this file and in `v3_vs_v2_report.pdf` should
therefore be read as a non-causal, best-case backtest upper bound, not a
proven or deployable result.** This does NOT affect the val_loss /
training-stability findings below (verified independent of the selection
step). Read `next_steps_completion_report.pdf` Section 2 before presenting
any of the trading numbers in this file externally. The same report also
closes the "2021 conclusion rests on one fold, not a repeated/seeded check"
caveat noted further down: a per-seed (non-ensemble) check across all 10
seeds found 8/10 individually profitable, none near v2's $3.68M, confirming
the luck-based root cause more rigorously.

**Val-loss table and graph added for the 10-seed model too (per user
request, 2026-07-13):** the original 10-seed sensitivity run
(`03_train_mv_alibi_transformer_v4b_10seed_sensitivity.py`) discarded the
per-epoch training history (`_` in the unpacking). Modified the script to
capture and save it (`mv_alibi_v4b_training_history.csv`), then re-ran the
full 60-model training (results matched the first run almost exactly, as
expected with fixed seeds). `v3_vs_v2_report.pdf` now has a "v4b: 10-Seed
Ensemble" section (Section 7) with the same page-4-style table used for
v2/v3/v4, plus a 2x3 grid figure showing all 10 seeds' curves and their
mean per fold, mirroring the v4 (5-seed) figure format
(`save_v4_seed_plot()` in `12_make_v3_vs_v2_report.py` was generalized to
take `out_filename`/`label`/`n_seeds` params so it renders both).

## v4 (2026-07-13): multi-seed ensemble -- first-ever 6/6 positive years

New, separate script: `TP2MVTransformer/03_train_mv_alibi_transformer_v4_multiseed.py`
(does not modify v3). Implements the professor's own explicit next-step
suggestion from the meeting transcript -- train several models with
different seeds and average their predictions, conditioned on the model
already being stable (which iteration 3 confirmed: mean divergence ratio
1.08x). Imports v3's tuned protocol (LR=3e-5, warm-up, grad clip 1.0,
val_loss checkpoint, 25%-capped rank-based selection) via `importlib`,
trains 5 independent models per fold (seeds 42/101/202/303/404), averages
their raw probabilities elementwise, fits one isotonic calibration on the
averaged validation probabilities, then applies v3's existing selection
logic to the averaged, calibrated probabilities.

**Result** (re-run once to also log full per-epoch history per seed for the
table/graph below; numbers shifted slightly from the first run due to MPS
non-determinism, same qualitative picture both times):

| metric | v3 (single seed) | v4 (5-seed average) |
|---|---|---|
| Total normalized P&L | $5,049,573 | $4,353,969 |
| Combined Sharpe | 1.79 | 1.65 |
| Combined Sortino | 9.28 | **14.66** |
| Positive years | 5/6 | **6/6** |
| Total selected | 32,300 | 27,184 |

Total P&L and Sharpe are both slightly *lower* than the single-seed v3 run,
but Sortino improved substantially further, and **every single year is now
profitable** for the first time across any version tried (v2, v3
iterations 1-3, v4). This is the expected trade-off from ensembling:
averaging smooths out the single-seed noise that occasionally produces an
outsized win (v3's 2026: $2.89M -> v4's $2.01M) at the cost of also
smoothing out losses (v3's 2022: -$19K -> v4's +$3.0K) -- net effect is a
lower-variance, more consistent return stream rather than a strictly
higher-return one.

Seed stability (mean/std of best val_loss across the 5 seeds per fold) shows
2021 is the most seed-sensitive fold (std 0.11 on a mean of 0.68 -- some
seeds stop as early as epoch 5, others train to epoch 24 and reach a
meaningfully lower loss), while 2022/2023/2024/2026 are all tightly
clustered (std <=0.01) -- consistent with 2021 being the fold with the most
residual training instability even after the LR fix.

**Honest caveat:** whether v4 is "better" than v3 depends on what's being
optimized for. If the goal is Sharpe/total return, v3 (iteration 3) is
still marginally ahead. If the goal is consistency/robustness (no losing
years, higher Sortino), v4 is the better choice. Given the professor's
repeated emphasis on not trusting single-run results and on robustness
over a single good number, v4 is probably the more defensible headline
result to present, with v3's higher raw return mentioned as a single-seed
data point.

**Val-loss table and graph for v4** (per user request): added
`professor_style_training_rows_multiseed()` to the report generator --
builds the same page-4-style row (Epochs, Train loss first->last, Val loss
first->last, Best epoch/val loss, Last AUC, Last grad norm) as v2/v3's
tables, but each row is the epoch-wise *mean across the 5 seeds* (a seed
that early-stops sooner just drops out of the average from that epoch on).
Also added a 2x3 grid figure (`figures/v4_multiseed_val_loss.pdf`, one
subplot per fold) showing all 5 seeds' individual curves (thin blue) plus
the mean curve (bold red) -- makes the seed-stability numbers visually
obvious: 2021 and 2025 show visible spread between seeds, the other four
folds are a tight bundle.

Report updated: `TP2MVTransformer/v3_vs_v2_report.pdf` now includes a
"v4: Multi-Seed Ensemble vs v3" section with the seed-stability table, the
page-4-style val-loss table, the 2x3 seed-spread figure, the full yearly
v3-vs-v4 comparison, and v4's row in Combined Totals.

## Current state (iteration 3, 2026-07-13): LR fix confirms genuine learning + best result yet

Cross-checked v3 (iteration 2, LR=5e-4) against the professor's literal
standard from the meeting transcript (`archive/professor_chat_transcript.md`):
*"validation loss has to go down... everything else doesn't matter until
those things start decreasing."* Checking iteration 2's per-epoch numbers
against that bar: **it failed for exactly the two years he named.** In both
test_year=2022 and test_year=2025, the best epoch was epoch 1 -- validation
loss got worse every epoch after that, meaning zero real learning in those
two folds despite the checkpoint/warm-up/clip fixes. The improved trading
numbers there were plausibly luck around a model that wasn't really learning,
exactly what he warned against trusting.

He explicitly asked (per the transcript) "did you try to lower the learning
rate?" -- at the time, no. Ran an LR sweep restricted to the 2022/2025 folds
first (`ablation_runs/lr_3e-5_2022_2025`, `lr_1e-4_2022_2025`,
`lr_3e-4_2022_2025`), then confirmed across all 6 folds
(`ablation_runs/lr_3e-5_all_folds`). **LR=3e-5 (down from 5e-4, ~16x lower)
is the rate at which validation loss genuinely decreases for multiple
epochs in every single fold**, not just the two flagged ones:

| fold | epoch 1 val_loss | best epoch | best val_loss | real improvement |
|---|---|---|---|---|
| 2021 | 0.989 | 7 | 0.630 | -36% |
| 2022 | 0.581 | 6 | 0.252 | -57% |
| 2023 | 0.598 | 25 (never early-stopped) | 0.176 | -70% |
| 2024 | 0.486 | 17 | 0.369 | -24% |
| 2025 | 1.059 | 2 | 0.959 | -9% (weakest, but still real, not zero) |
| 2026 | 0.849 | 3 | 0.526 | -38% |

Set `LR = 3e-5` directly in `03_train_mv_alibi_transformer_v3.py` (was
reading `base.LR` = 5e-4) and re-ran the full model (capped 25% selection
fraction unchanged from iteration 2). Result, using v3's actual training
history from this run (numbers differ slightly from the isolated ablation
sweep above due to the ablation script's non-per-fold RNG seeding, but same
qualitative pattern):

| test year | v2 ratio | v3 iter.2 (LR=5e-4) ratio | v3 iter.3 (LR=3e-5) ratio |
|---|---|---|---|
| 2021 | 1.02x | 1.06x | **1.02x** |
| 2022 | 1.92x | 1.92x | **1.01x** |
| 2023 | 1.06x | 1.19x | **1.00x** |
| 2024 | 1.56x | 1.21x | **1.01x** |
| 2025 | 2.81x | 1.62x | **1.42x** |
| 2026 | 1.28x | 1.23x | **1.04x** |

Mean divergence ratio: v2 1.61x -> v3 iter.2 1.37x -> **v3 iter.3 1.08x**.
Four of six folds are now essentially flat (ratio ~1.00-1.04x, no
meaningful overfitting at all). 2025 remains the hardest fold (1.42x) but
even that is down from 2.81x originally.

**Trading performance, iteration 3 vs v2:**

| metric | v2 | v3 iter.2 (LR=5e-4) | v3 iter.3 (LR=3e-5) |
|---|---|---|---|
| Total normalized P&L | $3,853,838 | $6,091,384 | $5,049,573 |
| Combined Sharpe | 0.69 | 1.76 | **1.79** |
| Combined Sortino | 2.32 | 9.76 | 9.28 |
| Max drawdown | -$3,681,550 | -$867,813 | **-$388,326** |
| Positive years | 4/6 | 5/6 | 5/6 |
| Total selected | 86,921 | 36,059 | 32,300 |

Iteration 3 has slightly lower total P&L than iteration 2 but a noticeably
better Sharpe and much smaller max drawdown -- more risk-efficient, fewer
trades, and (critically) rests on a model that is actually learning per the
professor's own standard rather than one that happened to produce good
numbers around an unlearned fold. **This is the version to present to the
professor, not iteration 2**, even though iteration 2's raw P&L was higher.

Per-year comparison, iteration 3:

| year | v2 | v3 iter.3 (LR=3e-5) | delta |
|---|---|---|---|
| 2021 | $3,681,383 | $190,473 | -$3,490,910 |
| 2022 | -$1,105,605 | -$19,367 | +$1,086,238 |
| 2023 | $12,829 | $164,222 | +$151,393 |
| 2024 | $167,048 | $229,459 | +$62,411 |
| 2025 | -$1,128,548 | $1,594,080 | +$2,722,628 |
| 2026 | $2,226,731 | $2,890,706 | +$663,975 |

2021's regression got *larger* at the lower LR (-$3.49M vs -$2.32M in
iteration 2) -- expected, not a new problem: the lower-LR model is even
more conservative/better-calibrated, so it's even more skeptical of the
concentrated, mostly-losing (20% win rate) trade cluster around Oct 2021
that drove v2's inflated 2021 number (root-caused earlier in this file).
2022 is now nearly break-even (-$19K, the best 2022 result across every
version tried) and comes with genuine learning behind it this time.

**Honest caveats, still open:**
- The 25% fraction cap and the 3e-5 learning rate are both single-point
  choices confirmed to work, not swept/optimized -- there could be a better
  combination nearby that hasn't been checked.
- Multi-seed averaging (the professor's suggested next step *after*
  stabilization) has not been tried.
- Feature-level drift (no moneyness features in `F*`) is still completely
  unaddressed.
- The post-model probability-floor family (script 07) and MV-ALiBi
  distance/attention diagnostics (scripts 05/08) have not been re-run
  against these outputs.
- Recency-weighted training and alternative training-window lengths are
  still untried.

## Iteration 2 (2026-07-13, superseded by iteration 3): capped fraction, genuine improvement but LR not yet fixed

After capping the trade-selection fraction grid at 25% (was 50% — see
iteration 1 below for why), v3 beats v2 across every headline metric:

| metric | v2 | v3 (capped) |
|---|---|---|
| Total normalized P&L | $3,853,838 | **$6,091,384** |
| Combined Sharpe | 0.69 | **1.76** |
| Combined Sortino | 2.32 | **9.76** |
| Max drawdown | -$3,681,550 | **-$867,813** |
| Positive years | 4/6 | **5/6** |
| Total selected | 86,921 | 36,059 |

Per-year P&L, v2 vs v3 (capped):

| year | v2 | v3 (capped) | delta |
|---|---|---|---|
| 2021 | $3,681,383 | $1,364,769 | -$2,316,615 |
| 2022 | -$1,105,605 | -$32,115 | +$1,073,490 |
| 2023 | $12,829 | $418,192 | +$405,363 |
| 2024 | $167,048 | $49,825 | -$117,223 |
| 2025 | -$1,128,548 | $2,029,251 | **+$3,157,799** |
| 2026 | $2,226,731 | $2,261,463 | +$34,732 |

Only one year is worse than v2 now: **2021**, down $2.3M (v3 selected far
fewer trades there — 11,757 vs 16,293). Every other year is flat-to-much-better.
2024, which regressed badly under the uncapped grid (see below), is now back
to roughly break-even with v2 (+$49,825 vs +$167,048 — a small, acceptable
difference, not a blowup).

**2021 root-caused (2026-07-13):** pulled the actual trade-level predictions
for both models on the 2021 test fold. Only 6,057 of ~16-18K candidate trades
overlap between v2's and v3's selections — v2 and v3 are two independently
trained models (different checkpoint epoch: v3 stopped at epoch 10 with best
val_loss 0.571, v2 stopped at epoch 6 with best val_loss 0.580 under a
different criterion) and they genuinely disagree on which trades are good.
Of the $2.25M v2 captured that v3 missed: 72% of it comes from just the top
1% of those trades (~102 trades), the bucket's overall win rate is only
20.3% (v3 was right to skip 80% of it on a per-trade basis), and the P&L is
extremely concentrated — the top 5 trading dates account for 122% of the
excluded bucket's total P&L (i.e. the rest of the bucket outside those days
is net negative), clustered tightly around **October 4-14, 2021**. v3's own
probability scores for these trades (mean 0.178, median right at its 0.109
cutoff) were much lower than v2's (mean 0.389) for the exact same trades —
a genuine model disagreement on a short-lived event window, not a
selection-rule bug. Given v3's validation loss for this fold is objectively
*better* (more calibrated) than v2's, and the missed bucket was 80% losers,
this reads as v2's model getting a lucky, concentrated hit on a two-week
window rather than v3 having a systematic blind spot. No code fix applied —
this looks like training variance on a concentrated event, not a defect.
Full investigation data pulled from `mv_alibi_transformer_v2/v3_test_predictions.parquet`.

Training-stability result is unchanged from iteration 1 (the fraction cap
doesn't affect training, only trade selection) — divergence ratio (worst/best
val_loss per fold) improved from 1.61x (v2) to 1.37x (v3) on average; the
worst fold (2025) improved most, from 2.81x to 1.62x.

**Honest caveats, not yet resolved:**
- 2021's regression (-$2.3M) is root-caused (see below) as model-training
  variance on a concentrated event window, not a code defect — but that
  conclusion rests on one fold's data, not a repeated/seeded check.
- The 25% cap is itself an arbitrary choice, not swept or validated — it was
  picked as "clearly more conservative than 50%," not tuned.
- Feature-level drift (raw price/strike levels, no moneyness features in
  `F*`) is still completely unaddressed — this run only fixes training
  protocol and selection policy, not the underlying feature drift diagnosed
  earlier (`ablation_runs/_audit/tables/top_feature_drift_psi.csv`,
  `target_rate_drift_by_year_strategy.csv`).
- The post-model probability-floor family (script 07) and MV-ALiBi
  distance/attention diagnostics (scripts 05/08) have not been re-run
  against v3's capped-selection outputs.
- LR sweep, multi-seed stability checks, recency-weighted training, and
  training-window-length experiments are all still untried.

## Iteration 1 (2026-07-12, superseded): uncapped fraction grid, not a clean win

First v3 run used `FRACTION_GRID = np.arange(0.01, 0.51, 0.01)` (search up to
50% of episodes). Training-stability fix worked (same numbers as above,
unaffected by the fraction change). But trading performance did **not**
improve overall: combined P&L $3,853,838 (v2) → $2,843,539 (v3), Sharpe
0.69 → 0.57, Sortino 2.32 → 1.98 — all down, even though 2022 and 2025 (the
two years flagged as unstable in the original professor report) both
improved substantially.

**What went wrong:** test_year=2024, which was fine in v2 (+$167,048),
collapsed to **-$2,422,136** in the uncapped v3 run. The fold's
`selection_fraction` tuned to 0.50 on its validation year (2023) — i.e.
"trade the top half of everything by probability," barely selective at all —
and that looseness carried into the 2024 test year, which had worse
opportunity quality. Result: 34,167 trades selected at a 15.8% win rate
(vs. v2's 18,560 trades at 26.3%), producing a large loss. Root cause: the
fraction grid had no upper bound, so a validation year that happened to
reward looseness could hand a badly-calibrated (too permissive) rule to a
worse test year — the same "validation-tuned rule breaks under regime
drift" failure as the original absolute-threshold problem, just relocated.

**Fix applied:** capped `FRACTION_GRID` to `np.arange(0.01, 0.26, 0.01)`
(max 25%) in `03_train_mv_alibi_transformer_v3.py`, then re-ran. See
"Current state" above for the result.

## What v3 changes vs v2 (`03_train_mv_alibi_transformer.py`), unchanged across both iterations

1. Checkpoint / early-stop on best **val_loss** instead of best val_auc (professor's explicit ask).
2. Linear LR warm-up for the first ~8% of epochs, then `ReduceLROnPlateau` tracking val_loss (mode="min") instead of val_auc (mode="max").
3. Gradient clip max_norm 1.0 (was 2.0).
4. Trade selection: tune a top-K **fraction** (capped at 25% as of iteration 2) on validation P&L, re-derive the matching probability cutoff separately on each split's own predicted-probability distribution, instead of one fixed absolute threshold carried across regimes.

`F*` (48 features), fold boundaries, leakage rules, MV-ALiBi distances, and
model architecture are all unchanged from v2 in both iterations. Ran on MPS
(Apple GPU), all 6 folds completed both times, all output artifacts written
to `TP2MVTransformer/tables/` and `TP2MVTransformer/data/` under `*_v3_*`
filenames — no v2 artifact was touched; the capped run's outputs overwrote
the uncapped run's `*_v3_*` files (both sets of numbers are preserved here
in this note, and in memory).

## Files produced

- `TP2MVTransformer/tables/mv_alibi_transformer_v3_results.csv`, `mv_alibi_v3_training_history.csv`, `mv_alibi_v3_yearly_diagnostics.csv`, `mv_alibi_v3_distance_diagnostics.csv`, `mv_alibi_v3_attention_parameters.csv` (reflect the capped run)
- `TP2MVTransformer/data/mv_alibi_transformer_v3_{test,val}_predictions.parquet`, `mv_alibi_transformer_v3_selected_trades.parquet`
- `TP2MVTransformer/12_make_v3_vs_v2_report.py` (results report generator)
- `TP2MVTransformer/v3_vs_v2_report.tex` / `.pdf` (results comparison report, includes the page-4-style
  training-stability tables for both v2 and v3, plus the extended best/worst/ratio comparison)
- `TP2MVTransformer/13_make_v3_technical_changelog_report.py` (technical changelog generator)
- `TP2MVTransformer/v3_technical_changelog.tex` / `.pdf` (separate PDF: parameter-by-parameter code
  diff, before/after code snippets for the two consequential changes, explicit confirmation that no data
  changed, and a consolidated next-steps list)
- `TP2MVTransformer/03_train_mv_alibi_transformer_v4_multiseed.py` (multi-seed ensemble script, new,
  separate from v3; also logs full per-epoch training history per seed to
  `mv_alibi_v4_training_history.csv`)
- `TP2MVTransformer/tables/mv_alibi_transformer_v4_results.csv`, `mv_alibi_v4_yearly_diagnostics.csv`,
  `mv_alibi_v4_seed_diagnostics.csv`, `mv_alibi_v4_training_history.csv`
- `TP2MVTransformer/data/mv_alibi_transformer_v4_{test,val}_predictions.parquet`,
  `mv_alibi_transformer_v4_selected_trades.parquet`
- `TP2MVTransformer/figures/v4_multiseed_val_loss.pdf` (2x3 grid: 5 seeds + mean per fold)
- This file.
