# Professor Action List — Tracker (Polak email, 2026-07-30)

Working tracker for the ICAIF v2 draft (`AICIF/tp2_spy_call_acm_draft_v2.tex`).
Deadline **Aug 2, 2026**. Hard **8-page** limit (paper currently 8/8).

Legend — **Feasibility**: ✅ doable now, no training; 🧮 compute from committed token
dataset; 🔁 needs a training rerun or owner-supplied per-trade/per-seed artifacts;
📄 text/tex only; ⚠️ need a fact from code first; 🖨️ changes length → must page-check.
**Status**: TODO / WIP / DONE / BLOCKED.

## Environment constraints (established this session)
- `pdflatex` **not installed** → I can edit `.tex` but cannot compile / verify page
  count locally. Need an agreed compile path.
- Raw pipeline data (`data/`, `Archive-2/`) **absent** (local-only by policy).
- **Present & usable**: `TP2MVTransformer/data/tp2_t1k2_token_dataset.parquet` (97 MB,
  354,974 rows = 177,487 episodes × {T1,K2}, 199 cols incl. all labels/returns),
  baseline MLP + baseline transformer prediction parquets, and all summary CSV/JSON
  in `TP2MVTransformer/tables/`.
- **Missing for granular items**: final-model (lgbm_final / mlp_final / mlp_5seed /
  v3final / v4final) *per-trade selected-trade* sets and *per-seed* logs. → #1/#2(model
  rows)/#5/#18 need these, via owner Colab artifacts or a rerun.
- Token dataset reproduces paper headline exactly: 2021–2026 unconditional
  = −$12.52M, Sharpe −1.02, Sortino −2.15, base rate 13.78%, 250,004 pairs / 234,885 valid.

## Open flag
- Commit `f28cf6d` message says "final MLP-5-seed removal", but on-disk `v2.tex` **and**
  committed `v2.pdf` both still contain MLP-5-seed. Professor email treats it as present.
  → Proceeding with v2.tex as canonical (MLP-5-seed IN). Confirm if that's wrong.

## Tier 1 — blocking
| # | tag | item | feas | status |
|--|--|--|--|--|
| 1 | new | Selectivity curves: sweep calibrated threshold per model×seed, record (N,S,Sortino,P&L,r̄); compare MLP vs transformer at matched N | 🔁 | TODO |
| 2 | table | Add Sortino, win rate, mean return/trade, skew to Table 2 | 🧮/🔁 | WIP (uncond. row computed) |
| 3 | rerun | No-history ablation at 5 seeds (currently single-seed 42) | 🔁 | BLOCKED (transformer train) |
| 4 | rerun | Rebuild Table 4 & Figure 3 at matched seed counts | 🔁 | TODO |
| 5 | table | Seed-level dispersion: min/IQR/max single-seed Sharpe, stratified-bootstrap CIs, IQM (Agarwal 2021) | 🔁 | TODO |
| 6 | text | Fix significance framing: disclose one-sided; give two-sided (0.219/0.688/1.000); reframe "not sig ⇒ neither superior" as power limitation (n=6, power≈0.26) | ✅📄 | TODO |
| 7 | table | Per-year 2014–2026: episodes, candidate pairs, base rate π_t | 🧮 | **DONE (numbers computed)** |
| 8 | text | Close leakage: (a) F* selection year range + stability-penalty fold span; (b) which lifecycle features truncated at violation date | ⚠️📄 | TODO |
| 9 | text | Specify how ≤64 history tokens are selected from all prior episodes | ⚠️📄 | TODO |
| 10 | text | Disclose test-year 2026 ≈107 trading days (ends 2026-06-04) in Table 1 caption; add sensitivity excluding it | ✅📄 | TODO |

## Tier 2 — reviewers will ask
| # | tag | item | feas | status |
|--|--|--|--|--|
| 11 | text | Promote capital-scaled backtest to primary; demote flat-notional to selection-quality diagnostic; state reversal absent under capital-scaled sizing | 📄 (authorial) | TODO |
| 12 | new | Market-neutrality on trade-level returns vs underlying over each trade's holding window + realized-vol change | 🔁 (needs underlying series) | TODO |
| 13 | new | Permuted-history control at identical parameter count | 🔁 (transformer train) | TODO |
| 14 | new | Deflated Sharpe Ratio + degrees-of-freedom appendix (≥20 configs) | 🧮📄🖨️ | TODO |
| 15 | new | Two baselines: L2 logistic regression on F*, random selection at matched N | 🧮 (train logistic; random trivial) | TODO |
| 16 | table | Full hyperparameter table; state which seed backs single-seed rows | ⚠️📄 | TODO |
| 17 | new | Exit-policy comparison (reversion vs hold-to-expiry) — runs said to exist | 🔁 | TODO |
| 18 | new | Calibration robustness: Platt vs isotonic, 1 vs 2 val years, bootstrap CI on N_m | 🔁 | TODO |

## Tier 3 — polish / compliance
| # | tag | item | feas | status |
|--|--|--|--|--|
| 19 | cite | 10 → ~25–30 refs in 4 blocks (benchmarking-variance, tabular-DL-vs-GBDT, relative-position-bias, backtest-multiple-testing) + Bali RFS 2023 | ✅📄🖨️ | TODO |
| 20 | text | Reposition contribution (1) around ensembling × decision-rule interaction; hedge title | 📄 (authorial) | TODO |
| 21 | text | Fix Table 5 caption: ROC-AUC is base-rate invariant; PR-AUC argued on positive-class informativeness; reconcile with §5.7 ROC-AUC | ✅📄 | TODO |
| 22 | text | Clarify Table 5 "13.0–13.8%": per-fold or per-model? (per-year valid base rates are 11.05–15.96%, pooled 13.78%) | ⚠️📄 | TODO |
| 23 | text | Define Sharpe formula; note it scales with trades/day → Table 2 values not directly comparable across models differing in N | ✅📄 | TODO |
| 24 | text | Abstract to ≤250 words (currently ~450) | ✅📄🖨️ (frees space) | TODO |
| 25 | tex | \usepackage{float} for [H]; fix 0.54\textwidth TikZ in figure*; drop \addtolength{\textheight}{18pt} and \footnotesize bib before ACM checker | ✅📄🖨️ | TODO |
| 26 | text | §5.6: VIX is on FRED/CBOE — merge it or say we didn't merge external data | ✅📄 | TODO |
| 27 | text | §6.2: give T and (1−ηλ)^T (cumulative, not per-step) | 🧮📄 | TODO |
| 28 | text | Use "episode-strategy pairs" consistently; state 177,487 → 250,004 mapping | ✅📄 | TODO (mapping now known) |

## Professor's "if you only do five": #1, #2, #5, #7, #19
- #7 ✅ (done, this session) · #19 ✅ (no runs) · #2 partial (uncond. done; model rows need artifacts)
- #1, #5 need per-trade / per-seed final-model outputs → owner Colab artifacts or rerun.

## Suggested iteration order
1. **Free space + zero-risk corrections (no compute, ~length-neutral/negative):** #24, #25, #6, #21, #22, #23, #26, #28.
2. **Text needing a code fact:** #8, #9, #16, #27 (read the generating scripts, state true).
3. **Add costed content (page-check each):** #7 table, #19 citations, #14 appendix.
4. **Quantitative experiments (pending compute decision):** #15, #1, #2, #5, #18, then transformer-gated #3, #4, #13, #12, #17.
5. **Authorial calls (owner sign-off):** #11, #20.

## Progress — session 1 (2026-07-31)
**Paper compiles at 8/8 pages** (tectonic; poppler `pdfinfo`/`pdftoppm` for checks). Done & verified in the rebuilt PDF:
- **#6** two-sided sign-test p (0.219 / 0.688 / 1.000) + power ($\approx$0.26) framing in §4.6, §5.2, §6.2, Conclusion.
- **#7** per-year table (§3.1) from the token dataset; **#10** 2026 $\approx$107-day partial year noted in its caption (sensitivity-excluding-2026 not yet added).
- **#15** Logistic-L2 ($2.02M/2.27) + Random@N ($-$0.77M) rows in Table 2 + framing (strong: linear model ≈ neural nets).
- **#5** MLP seed dispersion (Sharpe 1.47–2.60, median 2.07; the 2.60 is seed 42, the max) in §5.2. Transformer dispersion pending Colab per-seed.
- **#21/#22** Table 5 caption (ROC base-rate-invariant; PR-AUC rationale; base rate $\approx$13.8% identical across models).
- **#23** Sharpe formula + trades/day non-comparability caveat (§5). **#24** abstract 450→243 words. **#26** VIX wording. **#27** removed the (cumulatively false) weight-decay claim. **#28** 177,487→250,004 mapping.
- Space reclaimed by trimming verbose prose (intro, conclusion, §4.6, §5.6, calibration, limitations) — no substantive claim dropped except the incorrect weight-decay aside.
- NOTE: Fig 3 in this build is a **regenerated stand-in** (uses ALiBi-5 as the transformer series pending the plain-attention per-year data); the real figure differs.

**Blocked on Colab transformer data** (per-trade + per-seed; plain-attention primary + no-history): #1 (transformer curve), #2 (transformer/LGBM per-trade skew/mean), #3, #4-transformer, #5-transformer, #13, #18.
**Authorial / infeasible in 8pp:** #11, #20; **#19+#25** (25–30 refs + un-footnoted bib physically exceeds 8pp → must cut a table).
**Doable next, costs space (need prose cuts):** #8, #9, #16, #14.
