# Revision Brief Status

**Brief:** 30-item revision brief of 2026-08-02 (Parts I–IV). Applied 2026-08-03. **Updated 2026-08-05** (second pass; see §0).

---

## 0. 2026-08-05 update (reconciliation + data/feature definitions + regime shares)

- **Repo reconciliation.** The brief-applied draft was merged with the repo line (`fee0d96` P0 artifacts, `3afcd53` selectivity fold-in). Ported from the repo: the four-findings selectivity paragraph — five-seed MLP Sharpe **1.48** at N=14,338 vs transformer 2.18 (`mlp5_sweep_verdict.json`, `reproduction_ok: true`; the guide's "ranking survives" branch, no abstract change) — and the five-series figure caption. The matched-selectivity `TODO-RERUN[67]` slot is resolved; one `TODO-RERUN[67]` marker remains (171-trade exit-policy check, guide Task 3.3).
- **Data provenance added** (Source/Scope): Databento `OPRA.PILLAR` closing NBBO (last 1-minute quote ≤ 16:00 ET), volume/OI from the same feed, full listed chain (~55–60M contract-session rows); SPDR distribution series; computed fields (mid, BS IV from mid, delta/theta/vega, parity-implied forward); dividends/rates enter regime labels only.
- **Feature definitions by generative rule** (Feature Construction): A/B/C/D leg convention; per-leg field list; rectangle-level formulas (violation size, executable ratio, IV slopes/terms, forwards, gaps, month, pre-signal pressure max); named released dictionary `F_star_48_features_with_definitions.csv`; no-dividend-feature causality statement.
- **Regime shares in text** (§2, brief item 15 partial closure): 59.3% certified-European (52.2% no ex-date before T₂ + 7.1% small-dividend), 4.7% dividend-robust, 36.0% unresolved over all 177,487 episodes; footnote: small-dividend share zero through 2022, 10–19% in 2023–2025 tracking the short rate, no-dividend share calendar-driven. Artifact `tables/action_list/regime_partition_full_by_year.csv` (verified: pooled shares recomputed from the by-year file). Remaining for brief-15: per-model selected-trade shares/P&L by regime as a committed artifact.
- **Coauthors' review fixes applied:** training-from-2017 clause; base-rate range corrected to 27–73%; logistic comparison rephrased as total P&L (its Sharpe 2.27 < MLP-1's 2.60); abstract "We refer to this as the operating-point confound." The "Sharpe 1.79→1.66" item is moot — that sentence no longer exists in the merged draft.
- **Offsetting cuts** (both variants at exactly 8pp): deduplicated 88.66% and the no-arb-filter clause; compressed the halves clause, MV-ALiBi mechanism sentence, lifecycle truncation sentence, checkpointing parenthetical, partial-year run-in, ROC/PR sentence, denominations premium sentence.
- **Self-containment (same day, second edit; PI instruction).** All companion-note references removed from the paper — the submission carries its own derivations. Inline now: proofs of the three classical facts (no-dividend reduction, between-dividends reduction, and the dividend sandwich π ≤ D̄(T) via the European lower bound + optional sampling), a proof of Prop. 2.1 (per-ex-date continuation-vs-exercise comparison), and Prop. 2.2's chain proof (already present). The concentration/perturbation/Topkis inventory sentence was dropped; Topkis survives only as a cited open-problem direction; the orphaned Vershynin bib entry was removed. Abstract says "We prove…"; the conclusion's open problem is "an American TP2 theorem without the European reduction." The theory note remains in the repo as internal documentation only — not part of the submission, never cited by the paper.
**Sources revised (deliverables 1–2):** `AICIF/tp2_spy_call_acm_draft_v4.tex` (+ `_v4_1` twin), `AICIF/tp2_american_theory_note.tex`.
**Verification:** paper compiles to 8 pages, 0 errors, 0 undefined references (both variants); note compiles clean with the new appendix. No empirical number was invented; every correction requiring a rerun is a `RERUN-REQUIRED[brief-n]` comment in the paper source and an entry in §2 below.

Locate every paper edit by grepping the quoted anchor, never by subsection number (the paper has run-in headers only). Note locations are given by label or grep anchor.

---

## 1. Change log (deliverable 3)

Status legend: **Done** = fully applied in text. **Partial** = language applied, table/number awaits the rerun in §2. **Rerun** = no text change possible without new computation; placeholder comment anchored in the source.

### Part I — theory note (`tp2_american_theory_note.tex`)

| # | Status | Location (grep anchor / label) | Change |
|---|--------|-------------------------------|--------|
| 1 | Done | `0.8185`, `break-even` (line ~78); paper §2 rate sentence | Corrected 60-day example: RHS ≈ 0.8185%·K at r=5%, 0.1642%·K at r=1% (old 0.41%/0.08% removed); added break-even display T−t_i ≥ −log(1−δ_i/K)/r with ≈55 cal. days at r=2% and ≈22 at r=5% for δ/K=0.003. "Generically true in 2023–2026" removed from both documents; replaced with the contract-specific sentence from the brief. |
| 2 | Done | `m_S` (line ~125); paper post-certificate ¶ | M identified as carrying currency² units; normalizations m_S = M/S₀², m_R = M/[C(K̃₁,T₂)C(K̃₂,T₁)] defined; magnitude statement now attaches to each D̄_i (≈0.3–0.6% of underlying), not to M. Paper's "M is about 0.3–0.6% of the underlying value" deleted; replaced by the normalization sentence. Empirical normalized summary awaits guide Task 2.2. |
| 3 | Done | `calendar years` (line ~192) | Chernoff examples recomputed in calendar-year time: q ≈ 0.015 (12%-OTM, 14 cal. days, σ̄=20%), q ≈ 0.696 (5%-OTM, 30 cal. days). Old 0.056/0.78 (mixed 252-day convention) removed. Time convention stated directly below the display. |
| 4 | Done | Prop. "Model-dependent Chernoff bound"; Corollary `Two sharpened margins` (line ~196) | Sub-Gaussian variance proxy v_t is now an explicit hypothesis; v_t = σ̄²t labeled an additional modeling approximation, not a consequence of quotes. Sharpened margin split into M^quote (Doob only; inherits certificate rigor) and M^model (adds Chernoff; labeled model-assisted/diagnostic, never "certified"). |
| 5 | Done | Optimal-stopping intro | Coordinate fixed to k = log K: TP2 of C^am ⇔ supermodularity of (k,T) ↦ log C^am(e^k,T). All increasing-differences statements in (k,T); the (−log K,T)-supermodular statement is gone (verified by grep). |
| 6 | Done | `\begin{conjecture}` / `Proposed route to American TP2` (`thm:topkis`, line ~267) | Former "Theorem (Program)" relabeled a Conjecture with an explicit 7-gap list (lattice structure; increasing differences; closure under σ∧τ, σ∨τ; measurability/existence; stopped-tail preservation; positive-rate normalization; ITM mechanism). Optional-sampling sentence replaced (supplies expectation identities under integrability, not conditional tail order). "Finite family of policy-stopped laws" → finite-dimensional threshold-policy family, with the value sets noted as generally uncountable. |
| 7 | Done | `sign-determined` (line ~248); paper §2 | Premium-monotonicity proposition removed (preferred correction). Replaced by the Observation: expanding (C_i^eu+π_i) products puts premium terms on both sides; the sandwich 0 ≤ π ≤ D̄ does not sign the net perturbation. Paper no longer relies on π nonincreasing in strike anywhere. |
| 8 | Done | `C^{\mathrm{am},*}` (line ~132) | Latent-value vs quote distinction: C^bid ≤ C^{am,*} ≤ C^ask stated as an assumption on quote quality, not a theorem. Executable certificate rewritten directionally: asks on the undervalued pair, bids on the overvalued pair, C^ask·C^ask < C^bid·C^bid − M^bid. "Holds for any price inside the no-arbitrage band" removed. Lemmas/Props restated for latent values (`lem:merton`, `lem:smalldiv`, `prop:window`, `prop:certificate`). |
| 9 | Done | Doob prop.; appendix `Positive-Rate Doob` (line ~304) | Bound now uses C^ask; λ restricted to (Σ_{t_j≤τ} δ_j, K_i) so the shifted strike is positive; unlisted shifted strike → round the strike **down** (conservative). r>0 case derived in full in the appendix via M_u = e^{−ru}S_u + Σ_{t_j≤u} e^{−rt_j}δ_j and the rescaled strike; r=0 kept as the clean main proposition. |
| 10 | Done | `first-order` (line ~236) | Spread modulus is the conservative ε = −log(1−s/C), not log(1+s/C); strike rounding bounded by ∫|∂_u log C| du (or |ΔK|·sup_I |∂C|/C) over the bracketing interval; the local ΔK|∂_K C|/C form retained only as a labeled first-order approximation. |
| 11 | Done | `Common-driver` (line ~210); `operational assignment` (line ~218) | Portfolio claim weakened: at a fixed ex-date the any-exercise event is *bounded through* the minimum boundary (inequality; threshold-region caveat stated). Equivalence limited to cash flows **through** that date, future optionality may remain. "Irrational assignment helps the short" removed; replaced by the remark that assignment can differ from rational exercise and alters exposure, financing, dividends, and costs — the bounds address rational exercise only. |

### Part II — main paper (`tp2_spy_call_acm_draft_v4.tex`)

| # | Status | Location (grep anchor) | Change |
|---|--------|------------------------|--------|
| 12 | Done | §2 throughout; `A full American-TP2 theorem remains open` | Theory summary conformed to the corrected note: latent-value phrasing in Props 2.1–2.2; blanket r≳2% rule → contract-specific break-even horizons; no dimensional M claim; quote-certified vs model-assisted margins distinguished; no premium-monotonicity reliance; Topkis described as an open program; the brief's companion-note summary sentence inserted essentially verbatim. |
| 13 | Partial | `Feature Construction.` | Resolved as "LightGBM uses 149": final LightGBM trains on the full 149-candidate set; MLP/transformer/logistic share the identical 48-feature F*; architecture comparisons drawn within the 48-feature group and LightGBM comparisons qualified as also reflecting its feature set. Same-input LightGBM-on-48 row = rerun brief-13. |
| 14 | Rerun | `RERUN-REQUIRED[brief-14]` (after the four-denominations ¶) | Forward-estimator robustness. No text change yet; the parity-implied forward is described as such, with dividends/rates entering through it. |
| 15 | Mostly done (2026-08-05) | `three regimes`; `RERUN-REQUIRED[brief-15, remaining scope]` | Pooled and by-year shares now in the text with artifact `tables/action_list/regime_partition_full_by_year.csv` (59.3/4.7/36.0; small-dividend condition tracks the rate, otherwise calendar-driven). No "only R3 is where selection operates" claim. Remaining: per-model selected-trade shares and P&L by regime as a committed artifact (guide Task 2.4). |
| 16 | Partial | Abstract; intro ¶3; `Execution and early assignment.`; conclusion; `RERUN-REQUIRED[brief-16]` | All "survives early-assignment accounting/check" language removed. Current text: exposure identified and analyzed separately (16–22% of selections; LightGBM's exposed subset −$33K; transformer's higher win rate reported without a causal account); conclusion says "identified and analyzed separately rather than assignment-simulated." Primary-sample exclusion rerun = brief-16. |
| 17 | Done | `Labels, Costs, and the T1/K2 Restriction.` | Terminal-valuation disclaimer: terminal intrinsic value as a frictionless equivalent of expiration settlement; pin risk, exercise thresholds, post-expiration stock execution not modeled; labels described as "cost-aware at option entry with terminal-value settlement, not fully execution-realistic." |
| 18 | Rerun | `RERUN-REQUIRED[brief-18]` (cash-budgeted ¶) | Integer-contract sizing rerun. The $1,000 normalization and its raw-exposure range ($7–$814K) are disclosed in Labels. |
| 19 | Done (language) / Rerun (margin) | `Cash-Budgeted Sizing Experiment.`; Table 3 caption; `RERUN-REQUIRED[brief-19]` | Section, caption, abstract, intro, and conclusion renamed to "cash-budgeted sizing experiment"; "no leverage," "deployment lens," "realistic sizing," and deployment-CAGR interpretation removed; explicit sentence that the experiment caps premium outlay and does not model broker margin or collateral; margin-aware accounting deferred and queued. |
| 20 | Done (claim removal) / Rerun (alternative rule) | `Calibration and Trade Selection.`; `RERUN-REQUIRED[brief-20]` | No "natural boundary" claim (verified by grep); the rule is stated as a common p≥0.5 operating rule across models and folds, for comparability. Value-sensitive E[R|X]>0 rule = rerun brief-20. |
| 21 | Rerun | `RERUN-REQUIRED[brief-21]` | Isotonic output described accurately (fit on validation-year predictions, applied unchanged to test). Per-year calibration table (base rate, mean selected p, selected win rate, Brier, ECE) = rerun brief-21. |
| 22 | Rerun | `RERUN-REQUIRED[brief-22]` (Results opening) | Existing block-bootstrap CIs retained where already computed; date/expiry/episode-clustered inference for headline comparisons and beta t-statistics = rerun brief-22. |
| 23 | Partial | `matched draws`; `RERUN-REQUIRED[brief-23]` | Random baseline restated as what the artifact contains: mean over 30 matched draws (−$0.77M, Sharpe −1.14), with the 1,000-draw matched distribution deferred to the released implementation. Distribution row = rerun brief-23. |
| 24 | Done | selection-protocol sentences; guide Task 3.1 | One action per episode is the stated convention (highest-probability strategy at p≥0.5, ties toward T1), applied to Table 2 and the selectivity figures; the LightGBM selected-trades file deviation (19,636 vs sweep 18,665) is flagged and its reconciliation queued (guide 3.1). |
| 25 | Partial | intro contribution 1; Results baseline ¶; `RERUN-REQUIRED[brief-25]` | Causal attribution removed. The brief's "Under the SPY implementation studied here—…—" sentence is in place; comparison described as documenting the boundary under a changed implementation, with underlying, period, quote convention, and costs changed together. Decomposition table = rerun brief-25. |
| 26 | Rerun | `RERUN-REQUIRED[brief-26]` (Source/Scope ¶) | Yearly denominator diagnostics. The 2013 exclusion is currently justified by the raw count anomaly only and is stated as such. |
| 27 | Rerun | `RERUN-REQUIRED[brief-27]` (Episode Detection ¶) | The paper claims only executable-side quote construction (asks on undervalued, bids on overvalued); no "simultaneously executable" claim exists (verified by grep). Quote-quality filter ladder = rerun brief-27. |
| 28 | Done | intro ¶4; related work | Operating-point confound framed as "a trading-specific manifestation of the established dependence of classifier comparisons on decision threshold and selection coverage," per the brief's sentence; related work acknowledges threshold-dependent evaluation, coverage/selectivity, and selective prediction; contribution restated as applying and quantifying the issue in model-selected trading portfolios. |
| 29 | Done | abstract | Companion-analysis sentence and by-regime/by-exposure reporting sentence per the brief's templates. None of the seven banned abstract claims present (checked individually). |
| 30 | Done | `\section{Conclusion}` | Conformed to the brief's form: unconditionally-unprofitable / conditional-information sentence; certified-European / dividend-robust / unresolved three-subset sentence; "strongest deployment conclusions" sentence naming independent forward construction, exposure exclusion or modeling, integer sizing, and short-leg capital; no completed-American-TP2 claim. |

---

## 2. Pending empirical reruns (deliverable 4)

Authoritative queue. Every item has an anchored `RERUN-REQUIRED[brief-n]` comment in `tp2_spy_call_acm_draft_v4.tex`; none blocks submission (the current text is truthful without them). Rules that bind every item: reproduce the task's gate first; artifact committed under `tables/action_list/` before any tex edit; artifact and tex edit in one commit; match the paper's plain style.

| Brief | What to compute | Output artifact | Paper slot |
|-------|-----------------|-----------------|------------|
| 13 | Retrain final LightGBM on the 48-feature F* (same folds/calibration/rule) for a same-input architecture row | `lgbm_f48_comparison.json` + Table 2 row | after `Feature Construction.` |
| 14 | Re-detect under an independent forward (spot − PV(divs) compounded at r; futures where available); overlap + P&L vs parity-implied | `forward_robustness.json` | after the four-denominations ¶ |
| 15 (remaining) | Per-episode regime labels artifact + per-model selected-trade shares/P&L by regime (guide Tasks 2.2–2.4; sharpened margins 2.3b use M^quote as primary). Pooled/by-year shares are done and committed. | `episode_regime_labels.parquet`, per-model regime JSON | replaces the footnote's deferral clause in §2 |
| 16 | Primary tables excluding trades whose short leg spans an ex-date; full sample as robustness | `exdiv_exclusion_rerun.json` | `Execution and early assignment.` run-in |
| 18 | Integer-contract sizing (q = round(n·w), ≥1, ratio tolerance 1/2.5/5%); principal tables | `integer_sizing.json` | cash-budgeted ¶ |
| 19 | Margin-aware capital (Reg-T or OCC-style spread margin) with reserved-margin tracking | `margin_aware_capital.json` | cash-budgeted ¶ / Table 3 |
| 20 | Value-sensitive rule: E[R\|X] regression or two-part model, traded when positive, beside p≥0.5 | `value_rule_comparison.json` | `Calibration and Trade Selection.` |
| 21 | Per-year calibration: base rate, mean selected p, selected win rate, Brier, ECE | `calibration_by_year.json` | same run-in |
| 22 | Date-, expiry-, and episode-clustered (or block-bootstrap) intervals for headline comparisons and beta t-stats | `clustered_inference.json` | Results opening; `Market exposure.` |
| 23 | 1,000-draw random selection matched by year, denomination, and per-day count; median, 5–95%, P(random ≥ model) | `random_selection_distribution.json` | Random @N row of Table 2 |
| 25 | SPY decomposition ladder: mid → bid/ask → +commissions → full model → no exposed shorts → R1/R2 only | `spy_decomposition.json` | Results baseline ¶ |
| 26 | Yearly denominators: eligible rectangles, violation rate per million, median spread, zero-bid share, weekly-expiry share; 2013 diagnosis | `denominator_diagnostics.json` | Source/Scope ¶ |
| 27 | Quote-quality ladder: no zero bids, max relative spread, min OI/volume, synchronized subset | `quote_filter_ladder.json` | Episode Detection ¶ |

Separate, submission-relevant: the `TODO-RERUN[67]` matched-selectivity slot (guide Task 0.1) — Omkar's P0 run of `67_mlp5_selectivity_and_dsr_fix.py`.

---

## 3. Claim-support consistency table (deliverable 5)

Support classes: **T** theorem (complete proof, latent values) · **C** rigorous certificate (theorem + quote-bracket assumption) · **M** model-dependent assumption · **E** empirical result (committed artifact) · **J** conjecture / open · **P** pending rerun.

| Headline claim | Support |
|----------------|---------|
| No ex-date before expiry ⇒ latent American call = European (Merton reduction) | T |
| Multi-dividend sufficient Europeanization condition; break-even horizons 55d@2% / 22d@5% (δ/K=0.003) | T (numbers are illustrations of the condition, contract-specific) |
| Dividend sandwich 0 ≤ π ≤ D̄(T) | T |
| Executable-side violation > M cannot be explained by early exercise (dividend-robust certificate) | C (requires C^bid ≤ C^{am,*} ≤ C^ask) |
| Sharpened margin M^quote via Doob maximal inequality (ask side) | C |
| Sharpened margin M^model via Chernoff with v_t = σ̄²t | M (explicit sub-Gaussian assumption; diagnostic only) |
| Perturbation lemma: premium + spread + rounding as one additive log-defect | T (with the conservative moduli; local form is labeled first-order) |
| Common-driver portfolio bound at an ex-date | T as inequality (threshold-region caveat; rational exercise only) |
| American TP2 via supermodular optimal stopping | J (Conjecture; 7 stated gaps) |
| Sign of the American TP2 perturbation from the sandwich alone | Not determined (Observation) |
| Unconditional T1/K2 trading loses $12.52M, 12.9% win rate, Sharpe −1.01 | E (`tables/action_list/`) |
| Each of four selector families yields positive OOS P&L at p≥0.5 (Table 2) | E |
| Logistic on F* earns $2.02M (Sharpe 2.27): signal largely linear in the features | E |
| Ensembling moves trade counts +75%/−13% and flips Sharpe rankings (operating-point confound) | E |
| No architecture wins under matched-selectivity + seed-dispersion controls | E (MLP-5 matched-N point resolved 2026-08-05: 1.48 at N=14,338, `mlp5_sweep_verdict.json`) |
| Cash-budgeted sizing: LightGBM 14.6%, MLP 17.2%, transformers ≈0 | E (normalized sizing experiment; no margin — stated) |
| Random baseline −$0.77M | E (mean of 30 matched draws; full distribution P, brief-23) |
| Exposed-short-leg trades are 16–22% of selections; mixed subset results | E (descriptive; exclusion rerun P, brief-16) |
| SPX→SPY performance reversal attributable to a single friction | Not claimed (documented boundary; decomposition P, brief-25) |
| Regime shares R1/R2/R3 pooled and by year | E (`regime_partition_full_by_year.csv`; per-model split still P, brief-15 remaining scope) |
| Positive realized beta vs net-short entry delta | E, flagged open (guide 3.4) |

---

## 4. Obsolete-phrase audit (deliverable 7; run 2026-08-03)

Greps over both tex sources. **All banned phrases absent.**

| Phrase | Status |
|--------|--------|
| "M is 0.3–0.6% of the underlying" | Absent. "0.3–0.6%" survives only as the per-dividend D̄_i magnitude inside the approved replacement sentence (note abstract + §2 remark). |
| "generically true in 2023–2026" / "generically" | Absent. |
| "quoted contracts satisfy C^am = C^eu" | Absent; latent-value phrasing throughout. |
| "implied volatility provides a sub-Gaussian upper bound" | Absent. "Sub-Gaussian" appears 3× in the note, all inside the explicit-assumption framing (abstract, Prop. hypothesis, M^model qualification). |
| "supermodular in (−log K, T)" | Absent; convention is (k,T), k = log K. |
| "finite family of policy-stopped laws" | Absent. |
| "irrational assignment helps the short" | Absent. |
| "early-assignment accounting" | Absent. "Early-assignment" appears twice in the paper, both descriptive: "early-assignment risk the expiry labels do not price" and the run-in header "Execution and early assignment." No "survives … accounting/check" claim remains (intro reworded 2026-08-03). |
| "no leverage" | Absent; the sizing section is a "cash-budgeted sizing experiment" with the margin caveat. |
| "natural boundary" (brief 20) | Absent. |
| "simultaneously executable" (brief 27) | Absent; "simultaneously" occurs once, in the confound definition ("the threshold simultaneously determines the selected portfolio"). |

---

## 5. Updated tables (deliverable 6)

None added: all eight requested tables (forward estimators, regimes, ex-dividend exclusion, integer sizing, margin-aware capital, yearly calibration, random-selection distribution, quote filters) require reruns and are queued in §2 with anchored placeholders. Per the brief, no empirical results were invented.
