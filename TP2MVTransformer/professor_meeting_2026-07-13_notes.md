# Professor Meeting Notes — Validation-Loss Gate and LR Sweep

Sources: `archive/professor_chat_transcript.md` (meeting recording),
`archive/Stony Brook University Mail - Feedback for training transformer.pdf`
(checklist email). Cross-checked against `TP2MVTransformer/tables/mv_alibi_v3_training_history.csv`.

## The professor's actual standard (from the meeting, not just the email)

He is explicit and strict: **do not discuss trading performance, thresholds,
or P&L until validation loss is actually decreasing during training** — not
"less bad than before," not "peaks lower before early stopping." His words:
*"Everything else doesn't matter until those things start decreasing. You
don't look at anything else."* And on results from years where validation
loss doesn't decrease: *"it would just be a lucky result."*

He named **2022 and 2025** specifically as the two years where this fails,
matching the years flagged as unstable in our own earlier diagnostics. He
rejected a market-regime-similarity theory floated in the meeting ("no,
guys") — he wants mechanical fixes (learning rate, data size, checkpointing),
not narrative explanations, until the loss curves behave.

He explicitly asked (student answered "no" both times, at the time of the
meeting):
- "Did you try to lower the learning rate?"
- "Did you try to change the size of the data on which you are training?"

He also said, explicitly, that seed-averaging/ensembling (training 5-10
models with different seeds and averaging predictions — common in asset-
pricing transformer papers) should come **last**, only "once things
stabilize and we have a decent model."

## Checked v3 against this exact standard (2026-07-13): gate NOT cleared

| test year | epoch 1 val_loss | best epoch | trajectory after |
|---|---|---|---|
| 2022 | 0.255 | **1** | 0.281 → 0.348 → 0.380 → 0.446 → 0.489 (worse every epoch) |
| 2025 | 1.272 | **1** | 1.415 → 1.527 → 1.831 → 2.040 → 2.064 (worse every epoch) |

In both folds, training loss falls every epoch while validation loss rises
every epoch after the first — the textbook overfitting pattern the professor
described. v3's fixes (checkpoint-on-val_loss, warm-up, grad clip 1.0,
rank-based selection) made the *system* respond correctly — it now detects
this immediately and stops at epoch 1 instead of training on for 25 more
epochs into a worse state, and the selection rule is more robust to a
badly-calibrated model. But the underlying model for 2022 and 2025
specifically is **not learning anything past epoch 1**. The improved trading
numbers we measured for those years likely come from smarter selection/
calibration around a model that isn't really learning there — exactly the
"lucky result" he warned against presenting.

**Conclusion: real progress on the training-protocol bugs, but the
professor's actual bar is not yet met for 2022/2025. Do not present trading
performance for those two years as evidence of anything until this is fixed.**

## Technical details for presenting to the professor

**Loss function:** `BCEWithLogitsLoss` (binary cross-entropy) — the model
predicts the probability a trade will be profitable, and the loss measures
how far that probability was from the true 0/1 outcome. It is class-balanced
(`pos_weight`, recomputed per fold from that fold's training years) since
profitable trades are a minority class. Plus a small auxiliary term (weight
0.02, Huber loss) that also asks the model to predict return size — used
only for diagnostics, not for trading decisions.

**Validation type:** chronological, walk-forward (rolling-window) —
**not** random splitting. Each fold: 3 years train → next 1 year validation
→ year after that for test. Rolls forward one year at a time across 6 folds
(train 2017-19/val 2020/test 2021 ... train 2022-24/val 2025/test 2026).
Validation year picks the checkpoint and tunes trade selection; test year is
touched exactly once, never used for any decision.

**Optimizer / architecture:** AdamW, small custom Transformer (64-dim
hidden, 4 attention heads) with a custom distance-based attention bias
("MV-ALiBi") over up to 64 historical similar trades. Early stopping:
5 epochs without a new best val_loss. Calibration: isotonic regression
fit on the validation year only, applied to both validation and test
probabilities before any trading decision.

## LR sweep result (2026-07-13): lowering LR fixes exactly what he flagged

Ran `09_train_mv_alibi_transformer_ablation.py --fold-years 2022 2025` at
three learning rates below the original 5e-4, all other settings matching
v3 (checkpoint on val_loss, warm-up, grad clip 1.0, isotonic calibration).

Best epoch and validation-loss trajectory:

| fold | LR=3e-5 | LR=1e-4 | LR=3e-4 (~= old 5e-4) |
|---|---|---|---|
| 2022 | 0.503→0.310→0.264→**0.254 (epoch 6)** | 0.300→**0.256 (epoch 2)**→rises | **0.258 (epoch 1)**→rises immediately |
| 2025 | 1.024→0.858→**0.804 (epoch 3)** | **0.867 (epoch 1)**→rises immediately | **0.884 (epoch 1)**→rises immediately |

**LR=3e-5 is the clear winner and directly answers the professor's question.**
It is the only rate tested where validation loss genuinely decreases for
multiple consecutive epochs in *both* flagged folds — 2022 drops ~49% over
6 epochs before overfitting resumes; 2025 drops ~21% over 3 epochs. At
1e-4, only 2022 gets one extra real epoch of improvement; 2025 is already
back to "best = epoch 1" (zero real learning). At 3e-4 (close to the
original 5e-4), both folds are back to the original broken pattern. This
directly confirms his hypothesis that the learning rate was too high for
this problem.

Trading P&L at LR=3e-5, reported here only for completeness (per his
explicit instruction, not to be weighted until loss behaves everywhere):
2022 test_pnl -$1,239,089 (sharpe -4.90, still negative even with genuine
learning — consistent with 2022 being a fundamentally hard regime-transition
year, matching LightGBM's independent struggles there); 2025 test_pnl
+$959,212 (sharpe 3.10).

Ablation run tags: `ablation_runs/lr_3e-5_2022_2025/`,
`ablation_runs/lr_1e-4_2022_2025/`, `ablation_runs/lr_3e-4_2022_2025/`.

## Confirmed across all 6 folds (2026-07-13): LR=3e-5 fixes the pattern everywhere

Ran `--tag lr_3e-5_all_folds --lr 3e-5` (all other settings same as above,
no `--fold-years` restriction). Every fold now shows genuine multi-epoch
validation-loss decrease before overfitting resumes -- this satisfies the
professor's literal bar broadly, not just for the two folds he named:

| fold | epoch 1 val_loss | best epoch | best val_loss | real improvement |
|---|---|---|---|---|
| 2021 | 0.989 | 7 | 0.630 | -36% |
| 2022 | 0.581 | 6 | 0.252 | -57% |
| 2023 | 0.598 | 25 (never early-stopped) | 0.176 | -70% |
| 2024 | 0.486 | 17 | 0.369 | -24% |
| 2025 | 1.059 | 2 | 0.959 | -9% (weakest of the six, but still real, not zero) |
| 2026 | 0.849 | 3 | 0.526 | -38% |

Ablation run tag: `ablation_runs/lr_3e-5_all_folds/`.

**Important caveat:** this run used the *old* absolute-probability-threshold
trade selection (`base.tune_threshold`), not v3's rank-based capped-fraction
selection, so as to isolate the learning-rate effect cleanly. Its P&L
numbers (2021 +$119K, 2022 -$1.16M, 2023 -$25K, 2024 +$481K, 2025 +$1.52M,
2026 +$1.63M; combined ~+$2.56M) are **not directly comparable** to v3's
$6.09M, since v3 also changed the selection rule. 2022 stays sharply
negative even with genuine learning now happening -- consistent with 2022
being a fundamentally hard regime-transition year (also seen independently
in LightGBM), not something LR alone fixes.

**Minor methodological note for rigor:** the ablation script sets
`torch.manual_seed(args.seed)` once at the start of `main()`, not per fold,
so a given fold's random weight initialization depends on how many other
folds were trained before it in the same run. This is why the 2025 fold's
exact trajectory differs slightly between the 2-fold-only sweep (best epoch
3, val_loss 0.804) and the full 6-fold run (best epoch 2, val_loss 0.959) --
same fold, same nominal seed, different RNG state going in. Doesn't change
the qualitative conclusion (both show genuine multi-epoch learning, zero in
neither), but worth knowing before claiming exact reproducibility.

## Combined and confirmed (2026-07-13): LR=3e-5 + capped selection in one model

Set `LR = 3e-5` directly in `03_train_mv_alibi_transformer_v3.py` (previously
read `base.LR` = 5e-4) alongside the existing capped-fraction selection, and
re-ran the full 6-fold model. Full write-up in `v3_run_notes.md` ("iteration
3"); headline numbers:

- Training stability now satisfies the professor's literal bar broadly, not
  just for the two folds he named: mean divergence ratio 1.61x (v2) -> 1.08x
  (v3 iter.3). Four of six folds are essentially flat (ratio ~1.00-1.04x --
  no meaningful overfitting). 2025 remains the hardest fold at 1.42x, down
  from 2.81x originally, but still real, non-zero learning.
- Trading performance: combined Sharpe 0.69 (v2) -> **1.79**, Sortino 2.32 ->
  9.28, max drawdown -$3.68M -> **-$388K**, positive years 4/6 -> 5/6, total
  P&L $3.85M -> $5.05M (slightly lower than the LR=5e-4 version's $6.09M,
  but that version was resting on unlearned folds -- this is the version
  that should actually be presented).
- 2022 (one of the two years he flagged) is now nearly break-even (-$19K,
  the best result across every version tried), and this time it comes with
  genuine validation-loss learning behind it, not luck.
- 2021's regression grew larger (-$3.49M vs v2, up from -$2.32M at the
  higher LR) -- expected, not new: the more conservative, better-calibrated
  model at the lower LR is even more skeptical of the concentrated,
  mostly-losing Oct-2021 trade cluster that inflated v2's number (root
  cause investigated earlier in this file).

Updated report: `TP2MVTransformer/v3_vs_v2_report.pdf`
(`12_make_v3_vs_v2_report.py`). This is the version to bring to the
professor -- it satisfies his explicit "loss must decrease" bar, not just
"loss diverges less than before."
