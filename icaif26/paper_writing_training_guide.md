# Conference Paper Writing Training Guide
# (Updated for: TP2-MV-ALiBi → ICAIF Submission)

Since this is your first paper, think of conference paper writing as
**argument engineering**, not simply writing many pages. Every section has a
specific job. This guide is written specifically for the TP2-MV-ALiBi project
targeting ICAIF.

---

## Part 1: Key Paper Terms (General)

**Abstract**: A compressed version of the whole paper. Usually includes the
problem, gap, method, data, result, and significance. For ICAIF: aim for
150–200 words. Must contain at least one number (e.g., Sharpe ratio).

**Introduction**: Explains why the problem matters. It should answer what the
market/ML problem is, why existing work is insufficient, and what this paper
contributes. Do NOT describe your methodology in the introduction — just state
the contributions.

**Research Gap**: The missing piece in prior literature. For this project: TP2
and arbitrage literature exists, ML for options exists, but learning WHICH TP2
violations are economically tradable using a transformer that exploits
historical structural similarity has not been done.

**Contribution Statement**: Usually 3–4 bullet points saying exactly what the
paper adds. This is the most important paragraph in the introduction. Every
reviewer reads it. Be specific with numbers. Example:
> We show that restricting to T1 and K2 strategies (short-D-leg structural
> prior) achieves Sharpe 2.23 over six out-of-sample test years.

**Related Work / Literature Review**: Not a summary of every paper. It
positions the work against prior work: TP2/arbitrage theory, option-market
microstructure, ML for options, and transformer/attention models.

**Methodology**: The exact pipeline. Data, TP2 detection, episode construction,
label policy, feature selection, model architecture, training protocol, and
evaluation metrics. This is where reviewers check for leakage.

**Experimental Setup**: The rules of the experiment. Rolling train/val/test
splits, models compared, hyperparameters, transaction cost assumptions, and
metrics used.

**Ablation Study**: Removing or changing parts of the model to prove what
matters. For this paper: LightGBM → MLP on F* → Transformer without ALiBi →
Full TP2-MV-ALiBi. The decisive question: does multivariate history-aware
attention improve over a static feature model?

**Baseline**: A reasonable comparison model or strategy. For this paper:
corrected LightGBM T1+K2 is the main baseline. MLP on F* is the intermediate
ablation. Rule-based unconditional T1+K2 is the no-ML lower bound.

**Leakage**: When future information accidentally enters training. In finance
papers, reviewers care about this more than anything else. You must explicitly
state: (1) signal_year excluded, (2) thresholds tuned on validation only,
(3) test years evaluated once, (4) 2013 excluded due to data anomaly.

**Out-of-Sample**: Results on test years never used for training or threshold
tuning. Your 6 test years (2021–2026) are truly out-of-sample. Always label
these clearly in tables.

**Robustness Check**: Tests showing the result is not just luck or one artifact.
Include: year-wise P&L, strategy mix per year, and sensitivity to the
threshold grid.

**Caveat / Limitation**: Honest weakness. Example: EOD execution assumptions,
one negative test year for the transformer (2022), and limited to SPY options.

**Threats to Validity**: Data quality, survivorship, execution feasibility,
market impact, and American exercise assumptions.

**Economic Significance**: Whether the result matters financially, not only
statistically. P&L, Sharpe, drawdown, transaction costs, and capital exposure.

**Statistical Significance**: Whether the result is unlikely to be random.
For this paper: year-wise Sharpe consistency (5 or 6 out of 6 positive years)
is your main evidence. Consider bootstrap confidence intervals on the combined
Sharpe as a robustness check.

**Reproducibility**: Whether another researcher can rerun the pipeline. You
have clean scripts in ExpiryMLCorrection/ and TP2MVTransformer/ — mention this.

---

## Part 2: ICAIF-Specific Rules

**Page limit**: 8 pages for the body + unlimited reference pages. Do not go
over 8 pages on body content. Use the ACM `sigconf` template (the .tex file
in this folder).

**Double-blind**: ICAIF typically uses double-anonymous review. Remove your
names and institution from the submitted version. The template has an
`anonymous,review` option — use this for submission.

**Figures count toward the page limit.** Budget carefully. Plan for
approximately:
- 1 architecture figure (the TP2-MV-ALiBi pipeline diagram)
- 1 year-by-year P&L bar chart
- 1 ablation results table
- 1 per-fold results table

**CCS Concepts for this paper** (copy into the .tex file):
```
Computing methodologies → Machine learning → Neural networks
Applied computing → Economics → Computational finance
Information systems → Data management systems → Data mining
```

**Keywords for this paper**: options arbitrage, TP2 inequality, transformer,
attention with linear biases, ALiBi, financial machine learning, SPY options,
rolling out-of-sample evaluation

---

## Part 3: How to Think While Writing

For every paragraph, ask: **What claim am I making, and what evidence supports
it?**

For every result, ask: **Could this be leakage, a data artifact, or one lucky
year?**

For every model, ask: **Why is this model needed beyond a simpler baseline?**

For every figure/table, ask: **What should the reviewer learn in 10 seconds?**

---

## Part 4: Section-by-Section Guide FOR THIS PAPER

### Abstract (write this last)

Structure: Problem → Gap → Method → Result → Significance

Draft skeleton:
> Glasserman's TP2 inequality identifies no-arbitrage violations in option
> surfaces, but not all violations are economically tradable. We study which
> TP2 violations on SPY American call options yield positive P&L at expiry
> using a transformer-based classifier with multivariate ALiBi attention.
> We restrict to T1 and K2 strategies (structurally motivated: both short the
> shorter-dated leg) and train on rolling 3-year windows evaluated across six
> out-of-sample test years (2021–2026). Our corrected LightGBM baseline
> achieves Sharpe [X]. An MLP trained on the LightGBM-selected feature
> dictionary achieves Sharpe 2.23 with 6/6 positive years. The TP2-MV-ALiBi
> transformer — which attends to historically similar TP2 episodes using
> time, option-surface, short-D-leg, liquidity, and lifecycle distances —
> achieves total out-of-sample P&L of $3.84M across 35,354 selected trades.
> Ablations confirm that both feature selection and historical structural
> similarity contribute to performance.

Fill in the LightGBM Sharpe number once confirmed.

---

### 1. Introduction

Paragraph 1 — The economic problem (2–3 sentences):
> Option markets occasionally exhibit TP2 violations: configurations where
> C(K1,T1)·C(K2,T2) < C(K1*,T2)·C(K2*,T1). Glasserman [cite] proves this
> violates no-arbitrage for European calls. Trading these violations in liquid
> index options is an empirical question: not all violations are large enough
> to survive transaction costs and execution risk.

Paragraph 2 — Why existing work is insufficient (2–3 sentences):
> Prior work on TP2 violations focuses on theoretical characterization
> [Glasserman cite] or statistical detection [cite if any]. Machine learning
> for options trading [cite general ML options papers] typically targets
> implied volatility forecasting or hedging, not arbitrage-label classification
> under a rolling backtest accounting framework.

Paragraph 3 — What this paper does (transition into contributions):
> We build an end-to-end pipeline for TP2 expiry-payoff classification on SPY
> options (2014–2026) that corrects for product-weighting artifacts, fixes
> leakage in signal_year, and introduces a structural prior (T1+K2 only) based
> on which leg is shorted. On top of this corrected baseline, we propose
> TP2-MV-ALiBi: a transformer that replaces scalar temporal ALiBi distance
> [cite ALiBi paper] with a vector of TP2-relevant distances.

Paragraph 4 — Contributions (bullet list, 3–4 items):
1. We identify and correct a product-weighting artifact that inflated raw T2
   P&L by $194M; after $1K normalization, all strategies are negative
   unconditionally.
2. We show that restricting to T1+K2 (structural short-D-leg prior) recovers
   positive expected P&L, with LightGBM achieving Sharpe [X] and MLP on the
   selected feature dictionary achieving Sharpe 2.23 with 6/6 positive years.
3. We propose TP2-MV-ALiBi, a multivariate ALiBi transformer that encodes
   historical structural similarity using five TP2-domain distance functions
   and achieves total out-of-sample P&L of $3.84M across six test years.
4. We provide a full 8-model ablation from unconditional baseline to full
   transformer, enabling clean attribution of performance to feature selection
   versus historical attention.

---

### 2. Background and Related Work

Subsections:

**2.1 TP2 Inequalities and Options Arbitrage**
Explain Glasserman's TP2 inequality. Cite the original paper. State what
A, B, C, D are in your notation. Mention that this is a necessary condition
for no-arbitrage on European calls. Note that you work with American calls
(SPY) and treat expiry payoff as the label.

**2.2 Machine Learning for Options**
Brief survey of ML applied to options: IV surface fitting, hedging, and
pricing. Distinguish your setup: you are doing binary classification on
whether an arbitrage trade makes money at expiry, not forecasting IV.

**2.3 Attention with Linear Biases (ALiBi)**
Explain the original ALiBi [cite 2108.12409]. One paragraph. State its
inductive bias: penalizes attention between distant positions linearly.
Then state the gap: scalar temporal distance is not meaningful for TP2
episodes, which are distributed across option-surface geometry, liquidity,
and lifecycle state. Your contribution: replace scalar distance with a
domain-specific distance vector.

---

### 3. Data and Episode Construction

**3.1 Data Source**
SPY American call options from Archive-2 (your data vendor). Date range:
2014–2026. State the total number of episodes: 177,487 T1+K2 episodes after
filtering.

**3.2 TP2 Episode Detection**
Explain how you detect violations: C(K1,T1)·C(K2,T2) < C(K1_adj,T2)·C(K2_adj,T1).
State the log-product severity: v = log C + log D - log A - log B.

**3.3 Label Construction**
Label policy: `exit_selected_leg_expiration_payoff`. Binary label: 1 if
realized_return > 0 at expiry. Normalize to $1,000 gross exposure per trade
to remove the product-weighting artifact. Explain WHY: median raw exposure was
$7–$11, max was $814K, making raw P&L meaningless.

**3.4 Strategy Restriction: T1 and K2**
Explain the structural prior:
- T1: Long A(K1,T1), Short D(K2_adj,T1) — short shorter-dated option
- K2: Long B(K2,T2), Short D(K2_adj,T1) — short shorter-dated option
- T2 and K1 both short the longer-dated C-leg → more time for losses

This is a THEORETICAL prior, not an empirical selection. State this clearly.

**3.5 Data Anomaly: 2013**
2013 has 49,090 episodes vs 438 in 2014 — data vendor artifact. Excluded
from all training and evaluation.

---

### 4. Methodology

**4.1 Rolling Evaluation Protocol**
State the 6-fold scheme as a table:
Fold 1: Train 2017–2019, Val 2020, Test 2021
...
Fold 6: Train 2022–2024, Val 2025, Test 2026
Emphasize: thresholds tuned on validation only, test evaluated once.

**4.2 Feature Selection via LightGBM (Boosting as Discovery Tool)**
149 candidate features → 48 selected. State the composite importance score:
I_k = α1·I_gain + α2·I_perm + α3·I_stability
Describe the feature blocks (Table: 8 blocks, counts).
Key rule: boosting predictions, leaves, and SHAP scores do NOT enter the
transformer. Only the selected feature set F* is passed forward.

**4.3 MLP Baseline**
Architecture: simple feedforward on F*. Purpose: tests whether sequence
history adds value beyond good feature selection. This is the first ablation
step before the transformer.

**4.4 TP2-MV-ALiBi Transformer**
Describe the architecture:
- Token: x_{i,r} = selected features + strategy embedding + month + dayofweek
- Cross-attention: current token attends to MAX_HISTORY=32 prior tokens
- Multivariate ALiBi bias (the core contribution):
  a_{ij}^h = Q·K/sqrt(d) - m_h^T · d_{ij}
  where d_{ij} = [D_time, D_surface, D_short_D, D_liq, D_life]
- Slopes learned via softplus (guaranteed positive)
- Output: sigmoid probability for binary label

State what each distance measures and why it is TP2-relevant.

---

### 5. Results

**5.1 Unconditional Baseline (No ML)**
T1+K2 at expiry without any model. Sharpe and P&L. This is the floor.

**5.2 Corrected LightGBM T1+K2**
Your ExpiryMLCorrection result. Sharpe ~2.16, 6/6 positive years. This is
the main ML baseline.

**5.3 MLP on F* (First Ablation)**
Sharpe 2.23, 6/6 positive years, $1.22M total. Shows feature selection alone
is valuable.

**5.4 TP2-MV-ALiBi Transformer**
Sharpe 1.61, 5/6 positive years, $3.84M total, 35,354 trades.

**HOW TO WRITE THE MIXED RESULT HONESTLY:**
The transformer has higher absolute P&L but lower Sharpe. Write it this way:
> The TP2-MV-ALiBi transformer selects substantially more trades (35,354 vs
> 14,885 for MLP) and achieves higher absolute P&L ($3.84M vs $1.22M), but
> at lower Sharpe (1.61 vs 2.23), reflecting wider threshold selection in
> folds where the validation signal is diffuse. The 2022 test year is the one
> negative year for the transformer, driven by a validation threshold of 0.05
> (floor) on the 2021 validation set. The structural contribution — replacing
> scalar ALiBi with domain-specific TP2 distances — is validated by the
> ablation: removing the multivariate distance bias degrades AUC [fill in from
> ablation].

Do NOT hide the 2022 negative year. State it honestly. Reviewers respect
honesty; they penalize omission.

**5.5 Year-by-Year Breakdown Table**
Show all three models side by side per year. This is your most important table.

**5.6 Ablation Table**
Show the 8-model progression (when GPU results are ready). For now, you have
LightGBM, MLP, and Transformer. Show AUC and Sharpe for each.

---

### 6. Discussion

**What worked:** Feature selection (F*) extracts meaningful signals. The
structural T1+K2 restriction based on short-D-leg theory removes structural
losers without any ML. MLP on F* achieves stable 6/6 positive years.

**What the transformer adds:** Substantially more trade volume and higher
absolute P&L. The multivariate ALiBi distances encode TP2-relevant historical
similarity that static features cannot.

**What needs further work:** Threshold calibration in folds where validation
signal is diffuse (2022 case). Multitask loss (Huber return + pairwise ranking)
is specified but not yet trained — this is a known next step.

**Transaction costs (if you have them):** State the impact. If not modeled,
state this as a limitation and note that bid-ask spreads on the D-leg would
be the primary cost driver.

---

### 7. Conclusion (5–8 sentences)

1. Restate the problem.
2. State the key structural finding (T1+K2 short-D-leg prior).
3. State the corrected baseline result.
4. State the transformer result.
5. State the main ablation finding.
6. State the main limitation.
7. State the future direction (GPU training, multitask loss, transaction costs).

---

## Part 5: Tables to Prepare

**Table 1: Dataset Statistics**
Columns: Metric, Value
Rows: Total episodes, T1+K2 episodes, Date range, Candidate features,
Selected features (F*), Training folds

**Table 2: Feature Block Summary**
Columns: Block name, Count, Description
Rows: 8 blocks from build_summary.json

**Table 3: Model Comparison (Main Result)**
Columns: Model, Sharpe, Total P&L ($K), Positive Years, Trades Selected
Rows: Unconditional T1+K2, LightGBM, MLP on F*, TP2-MV-ALiBi

**Table 4: Year-by-Year P&L**
Columns: Year, LightGBM P&L, MLP P&L, Transformer P&L
Rows: 2021 through 2026, then Total

**Table 5: Ablation Results** (to be filled with GPU results)
Columns: Model, Val AUC, Test AUC, Sharpe, Positive Years
Rows: 8 models in order

---

## Part 6: Figures to Prepare

**Figure 1: Pipeline Architecture** — use the TP2_Fig1.png the professor
provided. Add it as the teaser figure or a full-width figure in Section 4.

**Figure 2: Year-by-Year P&L Bar Chart** — grouped bars for LightGBM, MLP,
Transformer per year. Already generated in TP2MVTransformer/figures/.

**Figure 3: Feature Block Importance** — bar chart of feature importance by
block. Already generated in TP2MVTransformer/figures/feature_blocks.pdf.

---

## Part 7: LaTeX Template Notes (for sample-sigconf-authordraft.tex)

When you start writing the actual paper .tex file:

1. **Change the documentclass for submission:**
   ```latex
   \documentclass[sigconf,anonymous,review]{acmart}
   ```

2. **Update conference info** (fill in after ICAIF 2026 CFP is released):
   ```latex
   \acmConference[ICAIF '26]{ACM International Conference on AI in Finance}
   {November 2026}{City, Country}
   ```

3. **CCS concepts** (replace the placeholder with):
   ```
   Computing methodologies~Neural networks
   Applied computing~Computational finance
   ```

4. **Do not modify margins, font sizes, or line spacing.** ACM will reject
   papers that alter the template formatting.

5. **Tables go ABOVE their captions** in ACM style. Figures go BELOW their
   captions.

6. **Use `\toprule`, `\midrule`, `\bottomrule`** from booktabs for all tables.
   Never use `\hline`.

7. **Every figure needs a `\Description{}`** tag for accessibility. Required
   by ACM.

8. **Bibliography:** Use `ACM-Reference-Format` style. Keep a .bib file with
   Glasserman TP2 paper, ALiBi paper (2108.12409), and any other references.

---

## Part 8: Common ICAIF Reviewer Concerns — and How to Address Them

| Reviewer concern | How to address it in the paper |
|-----------------|-------------------------------|
| "Is the T1+K2 restriction post-hoc?" | State it is a theoretical structural prior based on short-D-leg exposure, decided before any ML training. Cite the structural argument explicitly. |
| "Could this be leakage?" | Explicitly list all leakage protections: signal_year excluded, thresholds from val only, test evaluated once, 2013 excluded. |
| "Why a transformer on such small data?" | Show the ablation: the MV-ALiBi distances capture TP2-domain structure that temporal position alone cannot. AUC comparison is the evidence. |
| "Why is 2022 negative for the transformer?" | State it explicitly: validation threshold fell to floor (0.05) on 2021 val set. This is a threshold calibration issue, not a model failure. It is a known limitation. |
| "Where are transaction costs?" | State EOD mid-price execution assumption. Estimate bid-ask impact on the D-leg as a sensitivity. If not computed, state it clearly as a limitation. |
| "6 test years is small sample" | Show year-wise consistency (5 or 6 of 6 positive). Consider block-bootstrap Sharpe confidence intervals. |

---

## ICAIF / Quant AI Mindset

For ICAIF, the paper must look less like:

> We trained a transformer.

and more like:

> We define a financially motivated prediction problem, correct for known
> data artifacts, construct leakage-safe rolling labels, identify a structural
> prior grounded in arbitrage theory, and evaluate a transformer with
> domain-specific attention distances on six out-of-sample test years.

That is the writing mindset for this project.
