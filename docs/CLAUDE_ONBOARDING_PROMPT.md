# Onboarding Prompt for Another Claude Instance

Paste everything below the line into the other Claude, running inside the freshly-fetched repo.

---

You are picking up an in-progress quantitative finance research project. Read this brief in full, then read the files it points you to **before** doing any work or making any claims. Do not trust your memory over the repo — verify every number against the underlying data/CSV/JSON, never from a report PDF or from assumption. This is a hard rule the project owner enforces.

## 1. What the project is
Research paper for **ICAIF 2026** (ACM conference, Milan/Bocconi; hard **8-page** limit; submission deadline **Aug 2, 2026**). Question: can machine learning select *which* detected **TP2 violations** (a total-positivity-of-order-2 shape violation in option prices) in **SPY American call options** are worth trading? Trading every detected violation loses money (~ -$12.52M over 2021–2026 test folds); the contribution is that a learned, causal (p>=0.5) selection rule turns that into a profit across three independent model families.

**Current headline of the paper** is methodological, not a trading result: *"When Ensembling Reverses Which Model Wins"* — naively comparing a single-seed model against a multi-seed ensemble is an uncontrolled comparison and can reverse which variant looks superior. This is demonstrated within the transformer's own plain-vs-MV-ALiBi attention comparison (single-seed vs five-seed).

## 2. Read these files first, in this order
1. `README.md` — top-level scope, data policy, caveats.
2. `docs/pipeline.md` — plain-English pipeline narrative.
3. `docs/archive2_to_transformer_pipeline_tracker.md` — stage-by-stage tracker (input -> script -> output -> purpose). Last updated 2026-07-12.
4. `icaif26/PIPELINE_FILES.md` — the exact start-to-end file lineage behind the paper's numbers (the "pure" pipeline). Note: it still references the old draft name `tp2_spy_call_acm_draft.tex`; the live paper is the v2 file below.
5. `TP2MVTransformer/README.md` — the transformer sub-stage.
6. `icaif26/tp2_spy_call_acm_draft_v2.tex` — **THE CURRENT, FINAL, WORKING DRAFT.** This is what to edit. Not v1 (`tp2_spy_call_acm_draft.tex`), not v3.

## 3. Repo geography
- `Archive-2/` — professor-provided raw SPY option data (not in git).
- `src/tp2rr2/`, `scripts/` — ingestion, TP2 detection, feature building, LightGBM.
- `data/` — `processed/`, `features/`, `labels/`, `ml/` parquets (the source of truth for every number).
- `ExpiryMLCorrection/` — corrected expiry-label ML pipeline (LightGBM baselines).
- `TP2MVTransformer/` — numbered scripts (`01_...` to `57_...`) for the MLP baseline, the TP2-MV-ALiBi transformer, ablations, capital-scaled backtest, bootstrap, calibration, and report generators. Outputs in its `data/`, `tables/`, `figures/`.
- `icaif26/` — the ICAIF paper (.tex/.pdf), `figures/`, `PIPELINE_FILES.md`.
- `ProfessorMeetingScript/` — speaking scripts + agendas for professor meetings.
- `docs/agent_memory/` — a mirror of the persistent project memory; read `project_icaif_submission_status.md` for the most current state.

## 4. The three models (all compared on the same frozen 48-feature dictionary F*)
- **LightGBM** — tree baseline, also used for feature discovery.
- **MLP** — 48-64-32-1, ~5,249 params. Single-seed only in the paper.
- **Transformer** — historical cross-attention, ~37,086 params, 5 domain distances; **MV-ALiBi** = multivariate ALiBi, learned per-head per-distance-type slopes.

Evaluation: 6-fold walk-forward rolling, test years 2021–2026, causal p>=0.5 selection, isotonic calibration fit on the validation year only. Metrics: Sharpe/Sortino/CAGR/MaxDD/Calmar, PR-AUC/Brier/ECE, paired bootstrap + moving block bootstrap (21-day blocks). Two sizing conventions: flat $1,000/trade (isolates selection quality) and a capital-scaled $1M account with a 1%-of-cash daily cap.

## 5. Where the professor's 5-item checklist lives in v2
- Block bootstrap, economic risk exposures, execution-cost/early-assignment, calibration metrics -> Section 5.5 "Additional Robustness Checks" (`sec:addl-robustness`).
- No-history ablation -> Section 4.6 (`sec:nohistory`).
- Capital-scaled backtest -> Section 5.6 (`sec:capitalscaled`). Key finding: both transformer variants fall far behind LightGBM/MLP under realistic sizing due to a poor payoff ratio, not weaker selection.

## 6. Critical history you must not re-litigate or re-introduce
- **MLP-5-seed was deliberately removed** from the paper (both table and prose) by the owner's explicit decision. The ensembling headline now rests SOLELY on the transformer's plain-vs-ALiBi comparison. Do not add it back.
- An **MLP checkpointing fix** (val-loss-based instead of sklearn `early_stopping=True` accuracy) was investigated, verified correct three ways, and then **rejected by the owner**. The shipped MLP numbers are the ORIGINAL ones (Sharpe 2.60, $1.41M, 6/6 years). `tp2_spy_call_acm_draft_v3.tex` holds the corrected-MLP version and is dead on disk — do not use or edit it.
- `data/mlp_*_FLAWED_CHECKPOINT.parquet` files are the preserved originals that were restored; the live files match them.

## 7. Working rules the owner enforces
- Verify claims/data against source before stating them; never from memory or a report PDF.
- Stage any training run or irreversible change for approval before running it.
- Watch for selection-rule confounds; a finding scoped to one place must not become an unqualified claim elsewhere in the same document.
- Always check the compiled PDF page count — the 8-page limit is hard.
- Explain terminology in plain language when asked; the owner values that.
- The owner makes final authorial calls that can override a technically-correct recommendation. Respect that.

## 8. Git
Two remotes: `origin` = `github.com/Omkar38/option-surface-ml-tp2-rr2-relative-value`, `target` = `github.com/Omkar38/tp2_spy_call_transformer_icaif26` (the ICAIF paper repo). The paper branch is `fresh-files-2026-05-12`, tracking `target`. Commit/push only when asked; keep commits scoped.

## 9. First action
Confirm you've read the files in section 2, then ask what the owner wants to work on. Do not assume the paper needs changes — it is considered final and consistent unless told otherwise.
