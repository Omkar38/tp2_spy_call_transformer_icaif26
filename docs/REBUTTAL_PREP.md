# Rebuttal Preparation — Adversarial Review Findings (2026-08-09)

An adversarial AI-reviewer simulation was run against the submission PDF. 24 flags
were raised. The text-addressable ones were fixed in the submission (see commit log).
The items below need **coauthor numbers or answers** — compute them now so the
rebuttal is ready when reviews arrive. Ordered by priority.

## P0 — could decide the review outcome

1. **Commission convention vs hold-to-expiry (reviewer's #1, reject-risk).**
   The pipeline charges $1.30 per contract per leg (two $0.65 sides) while trades
   are held to expiration (no closing execution; IBKR charges no exercise/assignment
   fee). Questions for the code: (a) is $1.30 what the label code actually charges
   per contract-leg? (b) If it is a doubled/conservative convention, compute the
   **entry-only sensitivity**: unconditional P&L and each model's P&L at $0.65 per
   contract per leg. The reviewer computes unconditional flips to ≈ +$24M entry-only —
   if true, the framing "signal survives the spread but not commissions" needs the
   sensitivity stated. Also quantify omitted exchange/regulatory/OCC pass-through fees,
   which cut the other way. THIS IS THE SINGLE MOST IMPORTANT NUMBER TO HAVE READY.

2. **Joint quote-quality screen subsample (reviewer's #2).** For the 5–7% of candidates
   surviving all screens jointly: N, P&L, and a date-block CI. If the CI is wide and
   includes zero, the honest statement is "too few trades to measure," which defuses
   the stale-quote reading. Also send the exact screen thresholds.

3. **12.9% vs 13.8% (reviewer's #12).** Unconditional win rate (12.9%, 250,004 trades)
   vs pooled positive-label rate (13.8%) should be the same quantity. Explain the
   difference (weighting? population? rounding vintage?) — one sentence.

4. **Category episode counts (existing VERIFY).** 85,668 + 771 + 33,036 = 119,475 vs
   125,002 test-span episodes. Explain the 5,527 gap; counts return to Table 2 at
   camera-ready with the correct qualifier.

## P1 — likely reviewer questions

5. **Matched-random (L3) for MLP, ALiBi, logistic** (only LightGBM 67% and plain 8%
   exist). Same artifact, three more rows.
6. **Calendar-day Sharpe** for all five models alongside active-day (recompute_sharpe
   artifacts exist). Also report active-day counts per model.
7. **Label overlap at fold boundaries** (validation labels resolving inside the test
   year feed checkpointing/calibration). Response: quantify holding periods (median /
   p95 days signal→expiry for validation-year episodes) to show the overlap is days.
8. **Logistic row for the theory-category table** (Table 2 asymmetry).
9. **Integer-sizing 31.5% for logistic** in Table 3 — verify or explain (rounding
   instability on small selections?).
10. **Hyperparameter provenance** — one sentence per model on how/when chosen
    (defaults? tuned on which folds?).
11. **Per-seed dispersion** of the five-seed ensembles (min/max Sharpe per model).
12. **Deflated-Sharpe / multiple-testing response** — DSR artifacts exist internally
    (PI decision 2026-08-05 excluded them from the paper); have the numbers ready
    for the rebuttal only.

## P2 — have an answer, no computation needed

13. Active-day convention inflation critique → cite explicit definition + offer
    calendar-day numbers (item 6).
14. LightGBM-on-149 asymmetry → circularity rationale + the disclosed +0.46; offer
    the 48-feature LightGBM row in rebuttal.
15. 2013 exclusion evidence → weekly-expiry concentration + 30x rate (in paper);
    vendor coverage note if available.
16. Figure legibility at reduced width → regenerate with larger fonts, same aspect
    ratio (also good for camera-ready).
17. Reviewer noted Reg-T growth (1–2%) is below T-bill yields for much of the sample —
    acknowledge; the experiment is an accounting normalization and is labeled as such.

## Fixed in the submission text (no action needed)

- "for a round trip" wording reverted to the artifact's "in total, as two $0.65 sides".
- Division-of-labor claims made exactly true to Table 2 (gains "split along the same
  boundary"; "most of the loss"; MLP-specific concentration).
- Per-model 2026 contributions added ($32K/$179K/$182K/$608K).
- Exit-policy conclusion restated precisely (convention not concealing a better exit).
- Dividend look-ahead statement reconciled with the theory side (fields finalized
  after signal date; classification interprets, does not trade).
- Quote screens and the independent forward defined in text.
- Cross-run $686.2K/$734.9K figures removed from the training-window run-in.
- Terminology unified (date-block); antecedent and section-self-reference fixed.
