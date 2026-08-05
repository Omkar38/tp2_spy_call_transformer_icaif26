# Agent Implementation Guide — TP2 ICAIF '26 (paper v4)

**Audience:** coding agents (Claude Code / Codex / etc.) working with Jyotir and Omkar.
**Repo:** `tp2_spy_call_transformer_icaif26`, branch `fresh-files-2026-05-12`.
**Paper:** `AICIF/tp2_spy_call_acm_draft_v4.tex` (anonymous, submission) + `tp2_spy_call_acm_draft_v4_1.tex` (named twin).
**Theory:** `AICIF/tp2_american_theory_note.{tex,pdf}` — INTERNAL derivation document only (PI decision 2026-08-05). The submitted paper is self-contained: Props 2.1–2.2, the dividend sandwich, and their proof sketches are inline in paper §2, and the paper must NEVER cite or mention the note or "a companion analysis." The note holds the extended derivations (Doob/Chernoff margins, perturbation lemma, Topkis program) for our own use and for the tasks below.
**Rerun queue:** `docs/REVISION_BRIEF_STATUS.md` — change log for the 2026-08-02 revision brief, the authoritative list of pending empirical reruns (brief items 13–27), and the claim-support table. Paper comments `RERUN-REQUIRED[brief-n]` mark each insertion point.
**Deadline context:** ICAIF submission is Aug 2 AoE. Only Priority 0 is submission-relevant; the v4 text is written to be truthful *without* any rerun. Priorities 1–3 target camera-ready / the extended version.

---

## Ground rules — read before doing anything

0. **Citation policy:** do NOT cite or refer to the two Chin–Ólafsson–Nel problem books anywhere. Classical facts are attributed to Merton (1973) and Black (1975); concentration framework to Vershynin, *High-Dimensional Probability*, 2nd ed. (CUP 2026). The GP theory papers and the SSRN trading paper are the anchors. Check any new bib entry against the actual source.

1. **No number enters the paper without a committed artifact.** Every figure/claim must be reproducible from a file in `TP2MVTransformer/tables/action_list/` (JSON/CSV) or `TP2MVTransformer/tables/`. This paper was rebuilt around that rule after two review rounds found paper-vs-repo mismatches (a DSR reported as 0.85 while the repo held 0.250/0.585; a selectivity claim whose supporting curve was never computed). Do not reintroduce the pattern.
2. **Reproduction gates before new results.** Every retraining task below has a gate: reproduce the committed benchmark numbers first. If the gate fails, stop and reconcile — do not merge downstream numbers.
3. **One metrics implementation.** `metrics()` and `sel_by_thresh()` in `TP2MVTransformer/60_action_list_compute.py` are the single source of truth: daily-grouped Sharpe = `mean/std × √252` over per-day summed `target_norm_pnl`; Sortino uses the std of negative days only; `mean_ret_pct = mean(norm_pnl)/10`; selection is **one action per episode** (highest-prob strategy at `p ≥ thr`). Copy these functions verbatim; never re-derive.
4. **Both tex variants stay in sync** except the author block. After any edit to `v4.tex`, regenerate `v4_1.tex` (snippet in §"Paper mechanics"), compile both, and confirm **Pages = 8** for each. 8 pages including references is a hard ICAIF limit.
5. **Find the paper's open slots** with: `grep -n "TODO-RERUN\|RERUN-REQUIRED" AICIF/tp2_spy_call_acm_draft_v4.tex` (`TODO-RERUN[67]` = submission-relevant P0 slots; `RERUN-REQUIRED[brief-n]` = post-submission rerun queue, specs in `docs/REVISION_BRIEF_STATUS.md`)
6. **Commit hygiene:** artifact files first (JSON/CSV + regenerated figure), then the tex edit, in the same PR/commit; message states which task from this guide it closes.
7. **The paper has NO numbered subsections** (converted to run-in bold headers to save space). Locate every edit slot by grepping the header text or the `TODO-RERUN` markers, never by subsection number. `\ref{sec:...}` labels resolve to the parent section number; that is intended.
8b. **Excised content stays out (PI decisions, cumulative):** never reintroduce any of the following into the paper — the operating-point confound (any naming or discussion of it), matched-selectivity curves or the selectivity figure, seed-dispersion intervals, single-seed Table 2 rows, deflated Sharpe ratios, calibrator-sensitivity discussion, feature-selection-leakage discussion, or references to the companion theory note. The supporting artifacts stay committed as the internal record.
8c. **Do not emphasize logistic-regression comparability** ("linear matches deep" and similar). The logistic row is the *linear floor*, nothing more. Frame model comparisons around each family's strengths: MLP — largest total P&L and, with LightGBM, leading cash-budgeted growth; transformers — highest win rates, best calibration, 6/6 positive years, attention adds discrimination; LightGBM — strongest pooled discrimination, most balanced across denominations. Any sentence you insert about model comparisons follows this framing.
8d. **Never name repo artifact files in the paper text** (no `\texttt{...csv}`/`.json`/script names). Readers cannot access them. The paper's phrase is "part of the released implementation" (or "provided with the released implementation"). Vendor dataset identifiers (e.g., Databento `OPRA.PILLAR`) are data sources, not repo files, and may be named.
8. **Match the paper's voice in every sentence you insert:** short sentences; no evaluative adjectives or adverbs (nothing like "strikingly", "remarkably", "decisively"); plain words an undergraduate can follow. The templates below already follow this style — use them verbatim with numbers filled in.

## Environment

- Python ≥ 3.11: `pandas numpy scipy scikit-learn lightgbm pyarrow matplotlib`
- LaTeX: TeX Live with `acmart` (`texlive-publishers`, `texlive-fonts-extra`, `texlive-plain-generic`)
- Compile: `pdflatex -interaction=nonstopmode <file>.tex` **twice**; check `pdfinfo <file>.pdf | grep Pages` → 8; `grep -c undefined <file>.log` → 0.

---

## Priority 0 — run the script that already exists (hours; submission-relevant)

### Task 0.1 — MLP-5 selectivity curve + DSR fix — **DONE (2026-08-04, commits `fee0d96` + `3afcd53`)**
Gates matched exactly; verdict `mlp5_sweep_verdict.json`: MLP-5 Sharpe 1.477 at N=14,338 vs transformer 2.18. NOTE (2026-08-05): the four-findings paragraph, the selectivity figure, and the entire operating-point/matched-selectivity discussion were subsequently REMOVED from the paper by PI decision (full excision — see Task 0.2 and ground rule 9). The artifacts from this task remain committed as the internal record; do not delete them and do not put them back in the paper.

### Task 0.2 — two-rule evaluation + five-seed re-anchor (inference-only; NO retraining)
**Context (PI decision 2026-08-05):** the operating-point-confound contribution, matched-selectivity figure, seed-dispersion discussion, and every single-seed Table 2 row were removed from the paper. Table 2 now reports one protocol: five-seed ensembles for MLP/transformers, single fits for LightGBM/logistic. To preempt the reviewer question "do the rankings depend on how many trades each model takes?", the paper will report every model under TWO prespecified rules — the committed probability rule (calibrated p ≥ 0.5) and a coverage rule (each model's top N = 14,000 candidates by calibrated probability). Everything runs from committed prediction files.

**Inputs (all in repo):** `data/mlp_5seed_ensemble_test_predictions.parquet`; the transformer plain/ALiBi ensemble prediction files used by scripts 66/67; `lgbm_final_test_predictions.parquet`; logistic predictions (if no file exists, refit from the 60-series config — deterministic, seconds). Copy `metrics()` and `sel_by_thresh()` from `60_action_list_compute.py` verbatim (ground rule 3).

**Steps:**
1. **Gates first** — reproduce each family's committed p≥0.5 row: LightGBM 19,636 / $682K / 1.25 (its selection-file convention); MLP-5 24,977 / $3.43M / 1.66; TF-plain-5 14,338 / $695K / 2.18; TF-ALiBi-5 14,648 / $704K / 2.36; Logistic 13,841 / $2.02M / 2.27. A failed gate stops the task.
2. **Coverage rule:** one action per episode (higher calibrated probability, ties toward T1); rank all candidates by calibrated probability over the full 2021–2026 test span; take the top N = 14,000 per model; compute the full `metrics()` row. Apply the SAME one-action convention to LightGBM here — this simultaneously quantifies the ~5% selection-file difference (subsumes Task 3.1).
3. **Five-seed re-anchor** (paper slots marked `RERUN-REQUIRED[fiveseed-reanchor]`): from the five-seed ensemble predictions recompute (a) the strategy-level T1/K2 breakdown, (b) the year-by-year P&L bands, (c) PR-AUC / Brier / ECE for the MLP ensemble (the classification-metrics table's MLP row is currently single-seed).
4. **Artifacts:** `tables/action_list/two_rule_comparison.json` (per model × rule: N, PnL, sharpe, downside ratio, win rate, mean_ret_pct, skew, pos_years) and `tables/action_list/fiveseed_breakdowns.json`. Same commit as the tex edit.
5. **Tex edits (both variants; ≤8 pages; zero undefined):** fill the `TODO-RERUN[two-rule]` slot in the evaluation paragraph using the template in that comment; add two columns (Sharpe@14K, P&L@14K) to Table 2, or a compact second table if the columns overflow the column width; rewrite the two robustness run-ins and the calibration-table MLP row from `fiveseed_breakdowns.json`, keeping the existing sentence structure; delete the fulfilled comment markers.

**Voice rule binds:** report both rules as two prespecified evaluations, with no commentary about thresholds, operating points, or ranking reversals. See ground rule 9.
**Script (already written):** `TP2MVTransformer/67_mlp5_selectivity_and_dsr_fix.py`
**Why:** the paper currently states, honestly (slot marked `TODO-RERUN[67]`), that the five-seed-MLP vs five-seed-transformer comparison at matched trade count *cannot be adjudicated* because `34_mlp_5seed_ensemble.py` never saved full-population predictions. This task closes that gap. It also repairs `63_market_neutral_dsr_exit.py`'s filename collision (`label[-3] == "("` for both `MLP(1)` and `MLP(5)`, so `dsr_MLP_(.json` holds only MLP(5)).

**What it does:** retrains the 5-seed MLP ensemble exactly as script 34 (seeds 42/101/202/303/404, identical `MLPClassifier` hyperparameters, per-fold `StandardScaler`, average raw probs → one isotonic fit), saves `data/mlp_5seed_ensemble_test_predictions.parquet`, appends `MLP (5)` rows to `tables/action_list/selectivity_sweep.csv`, regenerates `AICIF/figures/selectivity_curves.pdf` (now 5 curves), writes `mlp5_sweep_verdict.json` with the matched-N read-off at N≈14,338, and recomputes DSRs with collision-free filenames (`dsr_MLP_1seed_ntrials{21,25}.json`, `dsr_MLP_5seed_ntrials{21,25}.json`).

**Gates:**
- Rerun @0.5 must reproduce the committed ensemble: `N = 24,977 (±250)`, `PnL ≈ $3,433,936`, `Sharpe ≈ 1.66 (±0.05)`. The script asserts this and refuses otherwise (sklearn-version drift is the usual cause; committed runs used the repo `requirements.txt` env).
- MLP(1) from `data/mlp_baseline_test_predictions.parquet` @0.5 must give `N = 14,258`, `Sharpe 2.60`.
- DSR check: the script's Transformer-plain-5 DSR must reproduce `0.5853507` (already committed in `dsr_Transformer_plain_(.json`).

**Paper edits after the run** (one `TODO-RERUN[67]` slot — the DSR and calibration-sensitivity sections were removed from the paper by PI decision on 2026-08-02; the DSR and Platt JSONs remain repo artifacts only and must NOT be added back to the paper):
- *Matched-Selectivity Curves* run-in (grep `Matched-Selectivity`), final sentence — replace per this decision table:
  - If `mlp5_sharpe_at_N > 2.18`: the reversal **is** an operating-point artifact. Use: *"At the five-seed transformer's operating point (N≈14,338), the five-seed MLP reaches Sharpe X.XX, above 2.18. The Table 2 reversal is an operating-point artifact, not an architecture ranking."* Also extend the abstract clause "and can reverse Sharpe rankings" with ", a reversal that vanishes at matched selectivity".
  - If `mlp5_sharpe_at_N ≤ 2.18`: use: *"At N≈14,338 the five-seed MLP reaches Sharpe X.XX, below 2.18. The ensemble-level ranking survives the operating-point control."* No abstract change.
- The selectivity figure gains the `MLP (5)` curve automatically; update its caption's model list. When regenerating the figure, set the axis labels to "Active-day Sharpe" and "Downside ratio" to match the renamed Table 2 columns.

---

## Priority 1 — removed (PI decision, 2026-08-01)

The F* re-selection control (former Tasks 1.1/1.2) is dropped. Do not run it, and do not add anything about feature selection to the paper beyond the neutral description already there. Task IDs 2.x and 3.x below are unchanged.

## Priority 2 — theory-driven R1/R2/R3 regime partition (new empirics; the big one)

**STATUS (2026-08-05): partially done externally.** A first end-to-end run (full quarterly ex-dividend calendar, FRED DTB3 rate — the panel's parity-rate column is a 0.04 fallback on ~80% of rows, confirming Task 2.1a's prediction) produced pooled and by-year shares, now committed as `tables/action_list/regime_partition_full_by_year.csv` and cited in the paper: R1 59.3% (52.2% no-dividend + 7.1% small-dividend), R2 4.7%, R3 36.0%. **Framing correction that supersedes the template below:** the small-dividend (cond.-ii) share tracks the rate — zero through 2022, 10–19% in 2023–2025 — but overall Europeanization is calendar-driven; do not write "2023–2026 are R1-heavy, tracking the short rate" as a blanket claim. Still open: the per-episode labels artifact (2.2), per-model selected-trade shares/P&L by regime (2.4) — whose numbers exist in the PI report but have no committed artifact yet — plus the sharpened margins (2.3b) and the portfolio bound (2.3c), and committing the classification script itself.

**Why:** the paper's *From European Theory to American Practice* run-in (Props 2.1–2.2) and `tp2_american_theory_note.pdf` define three provable regimes per violation rectangle:
- **R1 certified-European** — no ex-dividend date in `(t, T2]`, or the small-dividend condition (Prop 2.1(ii)) holds for **all four legs**: `δ_i + Σ_{j>i} δ_j e^{−r(t_j−t_i)} ≤ K(1 − e^{−r(T−t_i)})` at every ex-date `t_i ≤ T`. Here `C_am = C_eu` and the Glasserman–Pirjol theory applies verbatim.
- **R2 dividend-robust** — not R1, but the conservative-quote violation premium exceeds `M = D̄2·C(K̃2,T1) + D̄1·C(K̃1,T2)` where `D̄i = Σ_{t_j ≤ T_i} δ_j e^{−r t_j}` (Prop 2.2): a European-theory violation even granting the early-exercise premium its maximal effect.
- **R3 unresolved** — inside the wedge; sign-ambiguous; where ML selection operates.

**The falsifiable prediction (rate-regime discussion in paper §2):** the Europeanization condition is contract-specific; the break-even horizon `T − t_i ≥ −log(1 − δ_i/K)/r` is ≈55 calendar days at r=2% and ≈22 at r=5% for δ/K=0.003, so near-zero-rate 2021 should be R3-heavy and 2023–2026 R1-heavy for dividend-spanning maturities. Do NOT phrase the result as a blanket `r ≳ 2%` rule; report shares by year against the parity rate. Verifying this turns a Remark into a figure.

### Task 2.1 — reference data (CORRECTED: primary source is IN the panel)
The daily panel (`spy_options_daily`, see its README) already carries per row: `next_ex_div_date`, `next_div_amount` (Yahoo realized series), and `risk_free_rate` (**parity-implied** per date×expiry: cross-strike OLS of C−P on K gives the discount factor and forward; fallback rows carry exactly `r == 0.04` — filter on equality). The `forward` used by the detector is the parity-implied one, so market dividend expectations are already inside the strike adjustment whenever the fit succeeded.

- **2.1a (required diagnostic):** report, by year: share of fallback rows (`risk_free_rate == 0.04` exactly) and mean parity-implied r. Sanity: r≈0 in 2021, ≈5% in 2023–24. Artifact: `tables/action_list/panel_rate_diagnostics.json`. If fallback share is material in any year, flag every result that consumes r.
- **External CSVs now only for:** (i) dividend amounts *beyond the next one* (panel stores only the next) — needed when `(t, T2]` spans ≥2 ex-dates; build `data/reference/spy_dividends.csv` (ex_date, amount; quarterly; ≈$0.69 in 2013 → ≈$1.6–1.9 in 2025–26) from SSGA/Yahoo history; (ii) validating parity-r against FRED DTB3.
- **Look-ahead caveat (panel README):** `next_ex_div_date`/`next_div_amount` are realized-series and embed pre-declaration information. Correct for the certificates (an ex-post theory audit wants realized dividends); must never enter model FEATURES — verified: `feature_dictionary.csv` contains zero dividend-derived features.

### Task 2.2 — per-rectangle classifier `69_regime_partition.py`
Inputs (CORRECTED — no raw-data join needed): `TP2MVTransformer/data/tp2_t1k2_token_dataset.parquet` already carries all four legs — `feature_{A,B,C,D}_{bid,ask,mid,strike,delta,implied_volatility,…}`, with A·B the undervalued (ask-side) and C·D the overvalued (bid-side) products exactly as `src/tp2rr2/conservative_bidask.py` defines them. Join only date-level dividend/rate fields (`next_ex_div_date`, `next_div_amount`, parity `risk_free_rate`) from the daily panel on `signal_date` (+ each leg's expiry). **If a joined field is missing for a rectangle, classify it `R_unknown` and report the share — do not impute.**
Per rectangle at detection date `t`:
1. Ex-dates and amounts in `(t, T1]` and `(t, T2]` from the dividend calendar; `r` from the rate file at `t`.
2. R1 check: no-div-window, else Prop 2.1(ii) for each of the four legs (each leg has its own `K`, `T`).
3. Violation premium (conservative quotes) `= C(K̃1,T2)·C(K̃2,T1) − C(K1,T1)·C(K2,T2)`; margin `M` as above; R2 iff `premium > M` and not R1.
4. Else R3.
Output: `tables/action_list/regime_partition_by_year.json` — per year: counts and shares of R1/R2/R3/R_unknown, mean `r`, share of rectangles spanning an ex-date; plus per-episode parquet `data/ml/episode_regime_labels.parquet` (`episode_id, strategy, regime, margin_M, premium, spans_exdiv`).

### Task 2.3 — the test and the figure
- Table: year × {R1, R2, R3} shares. Test the prediction: R3-share in 2021 vs 2023–2026 (report shares and a two-sample proportion z, dividend-spanning rectangles only).
- Figure `AICIF/figures/regime_partition.pdf`: stacked shares by year with the short rate overlaid on a secondary axis. Follow the existing figure style (matplotlib, `figsize=(7.2,3.0)`-family, no seaborn).
- **Paper edit: DONE for the shares (2026-08-05)** — pooled shares + the rate-tracking footnote are in §2's post-certificate paragraph (grep `59.3`). The superseded template ("2023–2026 are R1-heavy, tracking the short rate") must not be used: condition (ii) tracks the rate; the no-dividend share is calendar-driven. The regime figure stays out of the 8-page version (its data is the committed CSV); the "Excluding the partial year" compression was already spent as an offset.

### Task 2.3b — sharpened margins M^quote and M^model (theory note, Corollary "Two sharpened margins")
Alongside `M`, compute TWO sharpened margins per rectangle, both of the form
`M♯ = π̄(K̃1,T2)·C^bid(K̃2,T1) + π̄(K̃2,T1)·C^bid(K̃1,T2)`, with per-leg
`π̄(K,T) = Σ_i w_i e^{−r t_i} q_i(K)`, `w_i = δ_i + Σ_{j>i} δ_j e^{−r(t_j−t_i)}`, differing only in `q_i`:
- **M^quote (rigorous; primary):** `q_i = min{1, Doob bound}`. Doob, from ask quotes and martingality alone: `q ≤ C^ask(λ′, τ) / (K_i − λ)` for a listed maturity `τ ≥ t_i` and any `λ` with `Σ_{t_j≤τ} δ_j < λ < K_i = K + Σ_{t_j<t_i} δ_j`, shifted strike `λ′ = λ − Σ_{t_j≤τ} δ_j > 0`; if `λ′` is not listed, round the strike DOWN (conservative). Optimize over `λ` on the strike grid and the 1–2 nearest listed maturities. The `r>0` form is in the note's appendix ("The Positive-Rate Doob Bound"). Needs detection-date call quotes at lower strikes — same source as Task 2.2's leg quotes.
- **M^model (diagnostic; model-dependent):** `q_i = min{1, Doob, Chernoff}` with `q ≤ exp(−log²(K/F_{t_i}) / (2 σ̄² t_i))` for `K ≥ F_{t_i}`, `t_i` in CALENDAR YEARS (never a 252-day convention). Valid only under an explicit sub-Gaussian assumption on the log-forward return with variance proxy `σ̄²t`; taking `σ̄` as the max over the rectangle's own `feature_{A..D}_implied_volatility` is a modeling approximation, not a consequence of quotes. Vershynin (HDP 2nd ed., §2.6) is the framework citation. Reference magnitudes: q ≈ 0.015 at 12%-OTM/14d/σ̄=20%; q ≈ 0.70 at 5%-OTM/30d.
Classification rule: the primary R2 label uses `M^quote` only. A violation exceeding `M^model` but not `M^quote` is labeled model-assisted/diagnostic in every artifact and sentence — never "certified." Add columns `margin_M_quote, margin_M_model, q_short_leg` to `episode_regime_labels.parquet`; re-run the R2 classification under each margin and report the share growth (expected: order-of-magnitude margin reduction for deep-OTM short-dated rectangles — the violation locus). Artifact: extend `regime_partition_by_year.json` with `share_R2_quote, share_R2_model`.

### Task 2.3c — portfolio-level common-driver bound (theory note, "Common-driver bound")
For each ex-dividend date in 2021–2026 and each model's live book at that date, compute `K_min` = minimum short-leg strike among open positions and the quote-version bound `q(K_min)` from 2.3b. This bounds `P(any position is inside its rational exercise region at t_D)` — one number per ex-date, no union-bound inflation (all legs share the SPY driver). Two caveats bind the wording: on the complement, American and European cash flows coincide only THROUGH that date (future early-exercise optionality may remain), and the bound addresses rational exercise, not operational assignment. Report the time series and its max/mean per model. Artifact: `portfolio_europeanization_bounds.json`.

### Task 2.4 (optional) — selector performance by regime
Join `episode_regime_labels.parquet` to each primary model's selected trades; report Sharpe/PnL within R1/R2/R3. Artifact: `regime_stratified_performance.json`. Interesting either way: theory says R1 violations are "true" anomalies; if the models' edge concentrates in R3, that says selection is pricing the American wedge itself.

---

## Priority 3 — smaller reconciliations (camera-ready polish)

- **3.1 LightGBM N mismatch.** Sweep@0.5 gives 18,665; the selected-trades file has 19,636. Diagnose: count episodes where *both* T1 and K2 crossed 0.5 in `lgbm_final_test_predictions.parquet` (per-strategy classifiers ⇒ likely dual-selection). Artifact: `lgbm_selection_reconciliation.json`. Paper: one clause in Fig. 4's caption if the explanation differs from "one-action-per-episode convention."
- **3.2 Platt-vs-isotonic attribution — RESOLVED (2026-08-04 review).** The `total_n = 20,125` config is a per-fold recalibration of LightGBM. Record the attribution in the artifact JSON; the calibrator-sensitivity sentence no longer exists in the paper and must not return.
- **3.3 Exit-policy paired check.** For the 171 selected trades with a defined true-reversion exit (`exit_policy_comparison.json`), compute the *paired* expiry-vs-reversion PnL on the same trades. Update the labels run-in (grep `171`) only if the paired result is material; n=171 stays labeled a small-sample check.
- **3.4 Beta vs delta reconciliation (upgraded: fully feasible).** The *Market exposure* run-in flags the open tension: trade-level β=+7.2 (MLP-5) vs 75–82% net-short entry delta. Per-leg entry deltas ARE in the token dataset (`feature_{A..D}_delta`, BS spot delta with dividend yield). Compute each trade's net entry delta from its traded legs, regress trade return on (underlying move × net delta) and on raw move; if short-gamma path-dependence explains the positive β, one sentence resolves a disclosed limitation.

---

## Paper mechanics (any task that touches the tex)

```bash
cd AICIF
pdflatex -interaction=nonstopmode tp2_spy_call_acm_draft_v4.tex   # run twice
pdflatex -interaction=nonstopmode tp2_spy_call_acm_draft_v4.tex
pdfinfo tp2_spy_call_acm_draft_v4.pdf | grep Pages                 # must print 8
grep -c undefined tp2_spy_call_acm_draft_v4.log                    # must print 0
```

Regenerate the named twin after every v4 edit (never edit v4_1 directly):

```python
src = open('tp2_spy_call_acm_draft_v4.tex').read()
src = src.replace(r'\documentclass[sigconf,review,anonymous]{acmart}',
                  r'\documentclass[sigconf,review]{acmart}')
anon = open('author_block_anon.txt').read()   # the \author{Anonymous...}...\shortauthors block
named = open('author_block_named.txt').read() # the three-author Stony Brook block
open('tp2_spy_call_acm_draft_v4_1.tex','w').write(src.replace(anon, named))
```

(The two blocks are verbatim in the current files — extract them once into the two txt files above, or inline them.)

**Do not:** exceed 8 pages; renumber Propositions 2.1/2.2 or Equations (1)–(3) (cross-referenced throughout); put author names or Stony Brook anywhere in the anonymous variant; edit committed artifacts under `tables/action_list/` (write new files); reformat unrelated tex; claim any number that has no artifact.

**Git note for agents running inside Cowork cloud sessions:** the mounted repo cannot delete files, so git leaves stale `*.lock` files under `.git/`. Before committing: `find .git -maxdepth 2 -name "*.lock"` and `mv` them into `_to_delete/`; after your commit, tell the human to run `rm -f .git/*.lock && rm -rf _to_delete && git push`. Local (non-Cowork) agents are unaffected.

## Definition of done, per task

1. Artifact file(s) committed under `tables/action_list/` (or `data/reference/`), containing every number used.
2. Reproduction gate passed and its numbers printed in the run log.
3. Tex edited in the designated slot only; both variants recompiled; 8 pages; 0 undefined refs.
4. Commit message: `[guide P<k>.<task>] <what>` + one line stating the gate result.
