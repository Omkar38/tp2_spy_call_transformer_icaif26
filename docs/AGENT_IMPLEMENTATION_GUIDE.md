# Agent Implementation Guide — TP2 ICAIF '26 (paper v4)

**Audience:** coding agents (Claude Code / Codex / etc.) working with Jyotir and Omkar.
**Repo:** `tp2_spy_call_transformer_icaif26`, branch `fresh-files-2026-05-12`.
**Paper:** `AICIF/tp2_spy_call_acm_draft_v4.tex` (anonymous, submission) + `tp2_spy_call_acm_draft_v4_1.tex` (named twin).
**Theory:** `AICIF/tp2_american_theory_note.{tex,pdf}` — Propositions referenced below live there and in paper §2.3.
**Deadline context:** ICAIF submission is Aug 2 AoE. Only Priority 0 is submission-relevant; the v4 text is written to be truthful *without* any rerun. Priorities 1–3 target camera-ready / the extended version.

---

## Ground rules — read before doing anything

0. **Citation policy:** do NOT cite or refer to the two Chin–Ólafsson–Nel problem books anywhere. Classical facts are attributed to Merton (1973) and Black (1975); concentration framework to Vershynin, *High-Dimensional Probability*, 2nd ed. (CUP 2026). The GP theory papers and the SSRN trading paper are the anchors. Check any new bib entry against the actual source.

1. **No number enters the paper without a committed artifact.** Every figure/claim must be reproducible from a file in `TP2MVTransformer/tables/action_list/` (JSON/CSV) or `TP2MVTransformer/tables/`. This paper was rebuilt around that rule after two review rounds found paper-vs-repo mismatches (a DSR reported as 0.85 while the repo held 0.250/0.585; a selectivity claim whose supporting curve was never computed). Do not reintroduce the pattern.
2. **Reproduction gates before new results.** Every retraining task below has a gate: reproduce the committed benchmark numbers first. If the gate fails, stop and reconcile — do not merge downstream numbers.
3. **One metrics implementation.** `metrics()` and `sel_by_thresh()` in `TP2MVTransformer/60_action_list_compute.py` are the single source of truth: daily-grouped Sharpe = `mean/std × √252` over per-day summed `target_norm_pnl`; Sortino uses the std of negative days only; `mean_ret_pct = mean(norm_pnl)/10`; selection is **one action per episode** (highest-prob strategy at `p ≥ thr`). Copy these functions verbatim; never re-derive.
4. **Both tex variants stay in sync** except the author block. After any edit to `v4.tex`, regenerate `v4_1.tex` (snippet in §"Paper mechanics"), compile both, and confirm **Pages = 8** for each. 8 pages including references is a hard ICAIF limit.
5. **Find the paper's open slots** with: `grep -n "TODO-RERUN" AICIF/tp2_spy_call_acm_draft_v4.tex`
6. **Commit hygiene:** artifact files first (JSON/CSV + regenerated figure), then the tex edit, in the same PR/commit; message states which task from this guide it closes.
7. **The paper has NO numbered subsections** (converted to run-in bold headers to save space). Locate every edit slot by grepping the header text or the `TODO-RERUN` markers, never by subsection number. `\ref{sec:...}` labels resolve to the parent section number; that is intended.
8. **Match the paper's voice in every sentence you insert:** short sentences; no evaluative adjectives or adverbs (nothing like "strikingly", "remarkably", "decisively"); plain words an undergraduate can follow. The templates below already follow this style — use them verbatim with numbers filled in.

## Environment

- Python ≥ 3.11: `pandas numpy scipy scikit-learn lightgbm pyarrow matplotlib`
- LaTeX: TeX Live with `acmart` (`texlive-publishers`, `texlive-fonts-extra`, `texlive-plain-generic`)
- Compile: `pdflatex -interaction=nonstopmode <file>.tex` **twice**; check `pdfinfo <file>.pdf | grep Pages` → 8; `grep -c undefined <file>.log` → 0.

---

## Priority 0 — run the script that already exists (hours; submission-relevant)

### Task 0.1 — MLP-5 selectivity curve + DSR fix
**Script (already written):** `TP2MVTransformer/67_mlp5_selectivity_and_dsr_fix.py`
**Why:** the paper's §5.3 currently states, honestly, that the five-seed-MLP vs five-seed-transformer comparison at matched trade count *cannot be adjudicated* because `34_mlp_5seed_ensemble.py` never saved full-population predictions. This task closes that gap. It also repairs `63_market_neutral_dsr_exit.py`'s filename collision (`label[-3] == "("` for both `MLP(1)` and `MLP(5)`, so `dsr_MLP_(.json` holds only MLP(5)).

**What it does:** retrains the 5-seed MLP ensemble exactly as script 34 (seeds 42/101/202/303/404, identical `MLPClassifier` hyperparameters, per-fold `StandardScaler`, average raw probs → one isotonic fit), saves `data/mlp_5seed_ensemble_test_predictions.parquet`, appends `MLP (5)` rows to `tables/action_list/selectivity_sweep.csv`, regenerates `AICIF/figures/selectivity_curves.pdf` (now 5 curves), writes `mlp5_sweep_verdict.json` with the matched-N read-off at N≈14,338, and recomputes DSRs with collision-free filenames (`dsr_MLP_1seed_ntrials{21,25}.json`, `dsr_MLP_5seed_ntrials{21,25}.json`).

**Gates:**
- Rerun @0.5 must reproduce the committed ensemble: `N = 24,977 (±250)`, `PnL ≈ $3,433,936`, `Sharpe ≈ 1.66 (±0.05)`. The script asserts this and refuses otherwise (sklearn-version drift is the usual cause; committed runs used the repo `requirements.txt` env).
- MLP(1) from `data/mlp_baseline_test_predictions.parquet` @0.5 must give `N = 14,258`, `Sharpe 2.60`.
- DSR check: the script's Transformer-plain-5 DSR must reproduce `0.5853507` (already committed in `dsr_Transformer_plain_(.json`).

**Paper edits after the run** (three `TODO-RERUN[67]` slots):
- *Matched-Selectivity Curves* run-in (grep `Matched-Selectivity`), final sentence — replace per this decision table:
  - If `mlp5_sharpe_at_N > 2.18`: the reversal **is** an operating-point artifact. Use: *"At the five-seed transformer's operating point (N≈14,338), the five-seed MLP reaches Sharpe X.XX, above 2.18. The Table 2 reversal is an operating-point artifact, not an architecture ranking."* Also extend the abstract clause "and can reverse Sharpe rankings" with ", a reversal that vanishes at matched selectivity".
  - If `mlp5_sharpe_at_N ≤ 2.18`: use: *"At N≈14,338 the five-seed MLP reaches Sharpe X.XX, below 2.18. The ensemble-level ranking survives the operating-point control."* No abstract change.
- Deflated-Sharpe paragraph (grep `Multiple Testing`) — insert the real MLP(1) DSR (forward calculation predicted ≈0.86–0.91; report whatever the artifact says) and the n_trials=25 sensitivity.
- Calibration run-in `TODO-RERUN[67]` (Platt config attribution; grep `Platt`) — see Task 3.2; same run can settle it.
- Figure 4 gains the `MLP (5)` curve automatically; update its caption's model list.

---

## Priority 1 — feature-selection leakage control (1 new script + CPU reruns)

**Why:** the *Feature Selection* run-in discloses that F* was selected pooling importance over all six folds (train years 2017–2024, val 2020–2025), so test years 2021–2025 leak into the *choice* of features. The paper currently flags this as the main unresolved validity issue. Bounding it is the highest-value camera-ready addition.

### Task 1.1 — `68_fstar_leakage_control.py`
Clone the selection logic of `01_build_feature_dictionary.py` but restrict the importance/stability computation to the **first fold only** (train 2017–2019, val 2020) — clean w.r.t. every test year. Keep the composite score (gain + permutation + stability over the 2 strategy runs available) and take the top 48. Output: `tables/feature_dictionary_2017_2019.csv`, same schema as `feature_dictionary.csv` (`selected_for_transformer`, `selected_rank` columns), plus a printed overlap count with the original F*.

### Task 1.2 — rerun cheap models on the clean F*
Retrain **logistic-L2, MLP(1), MLP(5)** on `feature_dictionary_2017_2019.csv` features (transformer rerun is optional — GPU; if skipped, say so in the artifact). Same folds, calibration, `p ≥ 0.5`.
**Artifact:** `tables/action_list/fstar_control_comparison.json` — for each model: `{sharpe, pnl, n}` under original vs clean F*, plus feature-overlap count.
**Paper edit** (the *Feature Selection* run-in, one sentence, both outcomes templated):
- If headline Sharpe changes by < ~0.3 and signs/rankings hold: *"A control re-selects F* from 2017–2019 only, which is clean with respect to every test year; K of 48 features are shared. Headline results are essentially unchanged (MLP-1 Sharpe X.XX vs 2.60), so the disclosed leakage is not load-bearing."*
- Otherwise: report the degradation plainly and move the leakage from "threat" to "material limitation" in the Discussion — do not soften it.

---

## Priority 2 — theory-driven R1/R2/R3 regime partition (new empirics; the big one)

**Why:** the paper's *From European Theory to American Practice* run-in (Props 2.1–2.2) and `tp2_american_theory_note.pdf` define three provable regimes per violation rectangle:
- **R1 certified-European** — no ex-dividend date in `(t, T2]`, or the small-dividend condition (Prop 2.1(ii)) holds for **all four legs**: `δ_i + Σ_{j>i} δ_j e^{−r(t_j−t_i)} ≤ K(1 − e^{−r(T−t_i)})` at every ex-date `t_i ≤ T`. Here `C_am = C_eu` and the Glasserman–Pirjol theory applies verbatim.
- **R2 dividend-robust** — not R1, but the conservative-quote violation premium exceeds `M = D̄2·C(K̃2,T1) + D̄1·C(K̃1,T2)` where `D̄i = Σ_{t_j ≤ T_i} δ_j e^{−r t_j}` (Prop 2.2): a European-theory violation even granting the early-exercise premium its maximal effect.
- **R3 unresolved** — inside the wedge; sign-ambiguous; where ML selection operates.

**The falsifiable prediction (Remark in §2.3):** at SPY's δ/S ≈ 0.3%/quarter, R1 requires roughly `r ≳ 2%` for dividend-spanning maturities ⇒ 2021 should be R3-heavy, 2023–2026 R1-heavy. Verifying this turns a Remark into a figure.

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
- **Paper edit:** one run-in under *Robustness* — *"Regime partition (Props 2.1–2.2). X% of episodes are certified-European (R1), Y% dividend-robust (R2), Z% unresolved (R3); as predicted by the rate condition, 2021 is R3-heavy (Z₂₀₂₁%) while 2023–2026 are R1-heavy (…), tracking the short rate."* Page budget: adding this + the figure requires an offsetting cut — first candidate is compressing §5.6 "Excluding the partial year" to one clause; second is dropping the regime figure and keeping only the table row inline.

### Task 2.3b — sharpened margins M♯ (concentration bounds; theory note §4)
Alongside `M`, compute the sharpened margin per rectangle:
`M♯ = π̄(K̃1,T2)·C(K̃2,T1) + π̄(K̃2,T1)·C(K̃1,T2)`, with per-leg
`π̄(K,T) = Σ_i w_i e^{−r t_i} q_i(K)`, `w_i = δ_i + Σ_{j>i} δ_j e^{−r(t_j−t_i)}`, and
`q_i(K) = min{1, Doob bound, sub-Gaussian bound}` where:
- **Doob (model-free, from quotes):** `q ≤ C_obs(λ − Σ_{t_j≤τ}δ_j, τ) / (K_i − λ)` for any listed maturity `τ ≥ t_i` and any `λ < K_i = K + Σ_{t_j<t_i}δ_j`; optimize over `λ` on the strike grid and over the 1–2 nearest listed maturities. Needs detection-date call quotes at lower strikes — same source as Task 2.2's leg quotes.
- **Sub-Gaussian (from IV proxy):** `q ≤ exp(−log²(K/F_{t_i}) / (2 σ̄² t_i))` for `K ≥ F_{t_i}`, with `σ̄` a conservative IV upper bound (use max over the rectangle's own `feature_{A..D}_implied_volatility`, already in the token dataset). Vershynin (HDP 2nd ed., §2.6) is the framework citation.
Add columns `margin_M_sharp, q_short_leg` to `episode_regime_labels.parquet`; re-run the R2 classification with `M♯` and report how much the dividend-robust share grows (expected: order-of-magnitude margin reduction for deep-OTM short-dated rectangles — the violation locus). Artifact: extend `regime_partition_by_year.json` with `share_R2_sharp`.

### Task 2.3c — portfolio-level Europeanization check (theory note §4.5)
For each ex-dividend date in 2021–2026 and each model's live book at that date, compute `K_min` = minimum short-leg strike among open positions and the bound `q(K_min)` from 2.3b. Report the time series `P(any early assignment optimal at t_D) ≤ q(K_min)` and its max/mean per model. This quantifies, with high probability, that the realized book coincided with its Europeanized counterpart — one number per ex-date, no union-bound inflation (all legs share the SPY driver). Artifact: `portfolio_europeanization_bounds.json`.

### Task 2.4 (optional) — selector performance by regime
Join `episode_regime_labels.parquet` to each primary model's selected trades; report Sharpe/PnL within R1/R2/R3. Artifact: `regime_stratified_performance.json`. Interesting either way: theory says R1 violations are "true" anomalies; if the models' edge concentrates in R3, that says selection is pricing the American wedge itself.

---

## Priority 3 — smaller reconciliations (camera-ready polish)

- **3.1 LightGBM N mismatch.** Sweep@0.5 gives 18,665; the selected-trades file has 19,636. Diagnose: count episodes where *both* T1 and K2 crossed 0.5 in `lgbm_final_test_predictions.parquet` (per-strategy classifiers ⇒ likely dual-selection). Artifact: `lgbm_selection_reconciliation.json`. Paper: one clause in Fig. 4's caption if the explanation differs from "one-action-per-episode convention."
- **3.2 Platt-vs-isotonic attribution.** `calibration_platt_vs_isotonic.json` has `total_n = 20,125`, matching no Table 2 row. Identify the config inside `65_calibration_robustness.py`, name it in the Calibration run-in.
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

**Do not:** exceed 8 pages; renumber Propositions 2.1/2.2 or Equations (1)–(3) (cross-referenced in §2.3, §3.2, §5.1); put author names or Stony Brook anywhere in the anonymous variant; edit committed artifacts under `tables/action_list/` (write new files); reformat unrelated tex; claim any number that has no artifact.

**Git note for agents running inside Cowork cloud sessions:** the mounted repo cannot delete files, so git leaves stale `*.lock` files under `.git/`. Before committing: `find .git -maxdepth 2 -name "*.lock"` and `mv` them into `_to_delete/`; after your commit, tell the human to run `rm -f .git/*.lock && rm -rf _to_delete && git push`. Local (non-Cowork) agents are unaffected.

## Definition of done, per task

1. Artifact file(s) committed under `tables/action_list/` (or `data/reference/`), containing every number used.
2. Reproduction gate passed and its numbers printed in the run log.
3. Tex edited in the designated slot only; both variants recompiled; 8 pages; 0 undefined refs.
4. Commit message: `[guide P<k>.<task>] <what>` + one line stating the gate result.
