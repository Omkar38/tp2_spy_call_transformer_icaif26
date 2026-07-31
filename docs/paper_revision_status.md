# TP2 Paper (ICAIF 2026) — Revision Status & Handoff

**Updated:** 2026-07-31 · **Deadline:** Aug 2, 2026 · **Authors:** Pawel Polak, Jyotir Manishkumar Patel, Omkar Lashkare (Stony Brook University)

Revision pass against Prof. Polak's action list (`docs/professor_action_list_tracker.md`).

## Files
- `AICIF/tp2_spy_call_acm_draft_v2.tex` — working draft, **anonymous** (`review,anonymous`).
- `AICIF/tp2_spy_call_acm_draft_v2_1.tex` / `.pdf` — **same body + the three named authors** (`review`, de-anonymized). Bodies are in sync; edit both (or keep v2_1 as the live named copy).
- `docs/professor_action_list_tracker.md` — granular item-by-item tracker.

## How to build (this machine)
LaTeX is via **tectonic** (installed with `brew install tectonic`; auto-fetches `acmart`). No system TeX / `pdflatex`.
```bash
tectonic -X compile AICIF/tp2_spy_call_acm_draft_v2_1.tex --outdir <dir>
pdfinfo <dir>/..._v2_1.pdf | grep Pages     # page check (poppler)
```
Non-transformer results harness: `scratchpad/harness.py` (reads the committed token dataset; reproduces MLP/LightGBM/etc.). Reproduced **MLP(1) $1.41M/2.60 and MLP(5) $3.43M/1.66 exactly**.

## Done this session (in v2 and v2_1)
| # | Item | Summary |
|--|--|--|
| 6 | Significance framing | Two-sided sign-test p (0.219/0.688/1.000) + power ≈0.26; "underpowered null, not equivalence" (§4.6, §5.2, §6.2, Concl.) |
| 7 | Per-year table | New table §3.1 (episodes, pairs, base rate π_t 2014–2026) |
| 8 | Leakage disclosures | (a) F* selected once globally over folds' val years 2020–2025 → mild feature-selection leakage disclosed; (b) lifecycle features truncated at event day 0 (`≤0`) |
| 9 | History-token selection | The ≤64 tokens = 64 most-recently-dated episodes strictly before the current day, by date alone (never label/outcome) — causal & reproducible |
| 10 | 2026 partial year | ≈107 trading days noted in Table (per-year) caption |
| 15 | Baselines | Logistic-L2 **$2.02M/2.27** + Random@N **−$0.77M** rows in Table 2 (linear ≈ neural nets) |
| 5 | Seed dispersion | MLP single-seed Sharpe **1.47–2.60**, median 2.07; the "2.60" is seed 42 |
| 16 | Hyperparameters + seed | Inline "Implementation (all models)" spec; all single-seed rows use fixed seed 42 (a priori, not best-of-k) |
| 21/22 | Table 5 caption | ROC base-rate-invariant; PR-AUC rationale; base rate 13.8% identical across models |
| 23 | Sharpe formula | Defined + trades/day non-comparability caveat |
| 24 | Abstract | 450 → 243 words |
| 26 | VIX | "did not merge external VIX (CBOE/FRED)" |
| 27 | Weight-decay claim | Removed — false cumulatively over T steps (the professor's point) |
| 28 | Pair mapping | 177,487 episodes × 2 = 354,974; 250,004 in 2021–2026 |

Space reclaimed by tightening intro, conclusion, §4.6, §5.6, calibration, and limitations — no substantive claim dropped except the (incorrect) weight-decay aside.

### Validated harness numbers (2021–2026, $1,000 convention; source of #2/#5/#15)
| Model | N | P&L | Sharpe | Sortino | Win% | Mean ret | Skew |
|--|--|--|--|--|--|--|--|
| Unconditional | 234,885 | −$12.52M | −1.02 | −2.15 | 13.8 | −5.3% | 16.1 |
| MLP (1, seed 42) | 14,258 | $1.41M | 2.60 | 6.89 | 54.0 | 9.9% | 9.6 |
| MLP (5) | 24,977 | $3.43M | 1.66 | 6.12 | 33.0 | 13.7% | 8.9 |
| Logistic (L2) | 13,841 | $2.02M | 2.27 | 4.20 | 53.5 | 14.6% | 5.5 |
| Random @N | 14,258 | −$0.77M | −1.14 | — | — | — | — |
MLP single-seed Sharpe by seed [42,101,202,303,404] = [2.60, 1.47, 2.13, 2.02, 2.07] (median 2.07).

## Left to do
### Blocked on teammate's transformer Colab outputs (per-trade + per-seed)
Needed files (written by the transformer final scripts, not committed): `TP2MVTransformer/data/*_test_predictions.parquet`, `*_val_predictions.parquet`, `*_selected_trades.parquet`; `TP2MVTransformer/tables/*_seed_diagnostics.csv` — **especially the plain-attention primary** (Table 2's $686K/$695K) and **no-history** outputs. Note: committed `v3final`/`v4final` are the **MV-ALiBi** variant, not the plain primary.
- **#1** selectivity curves (transformer arm) · **#2** transformer & LightGBM per-trade rows (Sortino/mean/skew) · **#3** no-history at 5 seeds · **#4** Table 4 / Fig 3 at matched seeds · **#5** transformer seed dispersion · **#13** permuted-history control · **#18** calibration robustness.
- Fig 3 in the current build is a **regenerated stand-in** (uses ALiBi-5 for the transformer series) — replace with the real figure once plain-attention per-year P&L is available.

### Authorial / decisions needed
- **Page limit:** currently **9 pages; page 9 is references only.** Confirm whether ICAIF's 8-page limit **excludes references** (common for ACM). If it **includes** them, cut one element — recommended: **Table 4** (T1/K2 breakdown; #4 flags it for rework anyway).
- **#19 (25–30 refs) + #25 (un-`\footnotesize` bib):** physically exceed 8pp — depend on the decision above / a table cut.
- **#11** promote capital-scaled to primary · **#20** reposition contribution (1) + hedge title.
- **Author order** currently Polak, Patel, Lashkare — confirm or reorder (e.g., student-first).
- **Double-blind:** v2_1 shows author names; if the ICAIF submission is double-blind, submit the anonymous `v2.tex` and keep v2_1 for internal use.

### Not yet done, mechanical (need a table cut for space)
- **#14** Deflated Sharpe + degrees-of-freedom appendix · **#12** trade-level market-neutrality (needs underlying return over each holding window — data not in repo) · **#17** exit-policy comparison.
