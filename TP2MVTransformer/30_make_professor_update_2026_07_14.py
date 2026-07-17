#!/usr/bin/env python3
"""Professor-facing update: everything done since the 2026-07-13 meeting.
Narrative, results-oriented -- companion to v3_technical_changelog.pdf
(code-level detail) and the underlying PDFs this summarizes
(final_comprehensive_report.pdf, shipped_final_report.pdf,
model_comparison_report.pdf).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TP2MVTransformer"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
REPORT_TEX = OUT / "professor_update_2026-07-14.tex"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def param_counts() -> tuple[int, int]:
    """Recomputed fresh each run, not hardcoded."""
    spec = importlib.util.spec_from_file_location("base_for_paramcount", OUT / "03_train_mv_alibi_transformer.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    model = base.MVAlibiClassifier(n_features=48, hidden=64, heads=4, use_return_head=True)
    transformer_params = sum(p.numel() for p in model.parameters())

    from sklearn.neural_network import MLPClassifier
    rng = np.random.RandomState(42)
    X = rng.randn(200, 48)
    y = rng.randint(0, 2, 200)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1, random_state=42)
    mlp.fit(X, y)
    mlp_params = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
    return int(transformer_params), int(mlp_params)


def significance_test(a: np.ndarray, b: np.ndarray, label: str) -> str:
    d = a - b
    n = len(d)
    mean = d.mean()
    n_pos = int((d > 0).sum())
    rng = np.random.default_rng(42)
    boots = np.array([rng.choice(d, size=n, replace=True).mean() for _ in range(20000)])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    p_sign = stats.binomtest(n_pos, n, 0.5, alternative="greater").pvalue
    return f"{label} & {n_pos}/{n} & {money(mean)} [{money(ci_lo)}, {money(ci_hi)}] & {p_sign:.3f} \\\\"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def money(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    sign = "-" if x < 0 else ""
    return f"{sign}\\${abs(x):,.0f}"


def f2(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    return f"{x:.2f}"


def num(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    return f"{x:,.0f}"


def main() -> None:
    v3f = read_json(TABLE_DIR / "mv_alibi_transformer_v3final_summary.json")
    v4f = read_json(TABLE_DIR / "mv_alibi_transformer_v4final_summary.json")
    lgbm = read_json(TABLE_DIR / "lgbm_final_summary.json")
    lgbm_t1k2 = lgbm.get("t1_k2_only", {})
    mlp = read_json(TABLE_DIR / "mlp_final_summary.json")

    transformer_params, mlp_params = param_counts()

    convergence = read_csv(TABLE_DIR / "mlp_vs_transformer_convergence.csv")
    convergence_rows = "\n".join(
        f"{int(r['test_year'])} & {int(r['mlp_best_epoch'])} & {f2(r['mlp_best_val_loss'])} & "
        f"{int(r['transformer_best_epoch']) if pd.notna(r['transformer_best_epoch']) else '--'} & "
        f"{f2(r['transformer_best_val_loss'])} \\\\"
        for _, r in convergence.iterrows()
    ) if not convergence.empty else "\\multicolumn{5}{c}{Pending.} \\\\"

    epoch0 = read_csv(TABLE_DIR / "mlp_epoch0_baseline.csv")
    epoch0_rows = "\n".join(
        f"{int(r['test_year'])} & {f2(r['epoch0_val_loss'])} & {f2(r['epoch1_val_loss'])} & {f2(r['reduction_pct'])}\\% \\\\"
        for _, r in epoch0.iterrows()
    ) if not epoch0.empty else "\\multicolumn{4}{c}{Pending.} \\\\"

    regcomp = read_csv(TABLE_DIR / "mlp_regularization_comparison.csv")
    regcomp_rows = "\n".join(
        f"{int(r['test_year'])} & {f2(r['best_val_loss_alpha1e-4'])} & {f2(r['daily_sharpe_alpha1e-4'])} & "
        f"{f2(r['best_val_loss_alpha1e-2'])} & {f2(r['daily_sharpe_alpha1e-2'])} \\\\"
        for _, r in regcomp.iterrows()
    ) if not regcomp.empty else "\\multicolumn{5}{c}{Pending.} \\\\"

    window = read_csv(TABLE_DIR / "training_window_sensitivity.csv")
    window = window[window["window"] == "all_history"].sort_values("test_year") if not window.empty else window
    v3f_yearly = read_csv(TABLE_DIR / "mv_alibi_v3final_yearly_diagnostics.csv")
    window_rows = []
    total_rolling, total_expanding, pos_rolling, pos_expanding = 0.0, 0.0, 0, 0
    for _, w in window.iterrows():
        y = int(w["test_year"])
        r = v3f_yearly[v3f_yearly["test_year"] == y].iloc[0]
        r_pnl, r_sharpe = r["test_norm_total_pnl"], r["test_daily_sharpe"]
        e_pnl, e_sharpe = w["norm_total_pnl"], w["daily_sharpe"]
        total_rolling += r_pnl; total_expanding += e_pnl
        pos_rolling += int(r_pnl > 0); pos_expanding += int(e_pnl > 0)
        winner = "Expanding" if e_sharpe > r_sharpe else "Rolling"
        window_rows.append(f"{y} & {money(r_pnl)} & {f2(r_sharpe)} & {money(e_pnl)} & {f2(e_sharpe)} & {winner} \\\\")
    window_table = "\n".join(window_rows) if window_rows else "\\multicolumn{6}{c}{Pending.} \\\\"

    mlp_yearly = read_csv(TABLE_DIR / "mlp_final_results.csv")
    v4f_yearly = read_csv(TABLE_DIR / "mv_alibi_v4final_yearly_diagnostics.csv")
    sig_rows = ""
    if not mlp_yearly.empty and not v3f_yearly.empty and not v4f_yearly.empty:
        mlp_pnl = mlp_yearly.sort_values("test_year")["test_norm_total_pnl"].to_numpy()
        v3_pnl = v3f_yearly.sort_values("test_year")["test_norm_total_pnl"].to_numpy()
        v4_pnl = v4f_yearly.sort_values("test_year")["test_norm_total_pnl"].to_numpy()
        sig_rows = (
            significance_test(mlp_pnl, v3_pnl, "MLP $-$ Transformer (v3-final)") + "\n"
            + significance_test(mlp_pnl, v4_pnl, "MLP $-$ Transformer (v4-final, 5-seed)")
        )

    v4_attn = read_csv(TABLE_DIR / "mv_alibi_v4_seed_attention_parameters.csv")
    v4_attn_summary = ""
    if not v4_attn.empty:
        slope_cols = [c for c in v4_attn.columns if c.startswith("slope_")]
        means = v4_attn[slope_cols].mean()
        stds = v4_attn[slope_cols].std()
        v4_attn_summary = "\n".join(
            f"{c.replace('slope_', '').replace('_', chr(92) + chr(95))} & {f2(means[c])} & {f2(stds[c])} \\\\" for c in slope_cols
        )

    t10 = read_json(TABLE_DIR / "mv_alibi_transformer_10seedfinal_summary.json")
    mlp5 = read_json(TABLE_DIR / "mlp_5seed_ensemble_summary.json")
    seedcount_rows = "\n".join([
        f"Transformer, 1 seed & {num(v3f.get('total_selected', np.nan))} & {money(v3f.get('total_norm_pnl', np.nan))} & {f2(v3f.get('combined_daily_sharpe', np.nan))} & {money(v3f.get('combined_max_drawdown', np.nan))} & {v3f.get('positive_years','--')}/{v3f.get('total_years','--')} \\\\",
        f"Transformer, 5 seeds & {num(v4f.get('total_selected', np.nan))} & {money(v4f.get('total_norm_pnl', np.nan))} & {f2(v4f.get('combined_daily_sharpe', np.nan))} & {money(v4f.get('combined_max_drawdown', np.nan))} & {v4f.get('positive_years','--')}/{v4f.get('total_years','--')} \\\\",
        f"Transformer, 10 seeds & {num(t10.get('total_selected', np.nan))} & {money(t10.get('total_norm_pnl', np.nan))} & {f2(t10.get('combined_daily_sharpe', np.nan))} & {money(t10.get('combined_max_drawdown', np.nan))} & {t10.get('positive_years','--')}/{t10.get('total_years','--')} \\\\",
        f"MLP, 1 seed & {num(mlp.get('n_selected', np.nan))} & {money(mlp.get('norm_total_pnl', np.nan))} & {f2(mlp.get('daily_sharpe', np.nan))} & {money(mlp.get('max_drawdown', np.nan))} & {mlp.get('positive_years','--')}/{mlp.get('total_years','--')} \\\\",
        f"MLP, 5 seeds & {num(mlp5.get('n_selected', np.nan))} & {money(mlp5.get('norm_total_pnl', np.nan))} & {f2(mlp5.get('daily_sharpe', np.nan))} & {money(mlp5.get('max_drawdown', np.nan))} & {mlp5.get('positive_years','--')}/{mlp5.get('total_years','--')} \\\\",
    ])

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=0.75in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{float}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}

\title{{Project Update -- TP2-MV-ALiBi Transformer\\\large What Changed Since the 2026-07-13 Meeting}}
\author{{SPY American Call TP2 Project}}
\date{{\today}}

\begin{{document}}
\sloppy
\maketitle

\section*{{Summary}}
Since our last meeting, four things happened, in this order: (1) every remaining item from your checklist
was completed (architecture/weight-decay sensitivity, training-window-length sensitivity, MV-ALiBi
diagnostics re-run); (2) a self-audit of that work caught a real methodological problem before it went
anywhere near a paper, and it was fixed; (3) the fix -- a simple, non-tunable trading rule -- was actually
built into the production training code and validated on held-out data, not just checked after the fact;
(4) the transformer was benchmarked against LightGBM, an MLP, and (for context) the published SPX results
from Glasserman, Li, and Pirjol's TP2 paper. Point (4) surfaced a result that needs your input: a simple
MLP now outperforms the transformer on risk-adjusted return once every model is held to the same standard.

\section*{{1. Checklist Items Completed}}
\begin{{itemize}}
\item \textbf{{Architecture sensitivity:}} tested hidden size (32, 64, 128), attention heads, and history
length independently. A larger hidden size (128) modestly improved results; halving the attention history
length (32 instead of 64) clearly hurt.
\item \textbf{{Weight-decay sensitivity:}} tested independently of learning rate. Found to be essentially
inert at the model's current learning rate (3e-5) -- confirmed mechanistically (AdamW's decoupled decay term
scales with $\mathrm{{lr}}\times\mathrm{{weight\_decay}}$, and at this learning rate even a 1000$\times$ change in
weight decay changes each training step by a negligible amount). Not a bug -- a real, explained null result.
\item \textbf{{Training-window-length sensitivity:}} tested 1yr/2yr/3yr(current)/all-history training windows.
The current 3-year default turned out to be the best or a close second in 2 of 3 sampled years once measured
correctly (see Section 2) -- a reassuring result, not a reason to change the default.
\item \textbf{{MV-ALiBi diagnostics, re-run for v3:}} the five distance types (time, surface, short-leg,
liquidity, lifecycle) have very different raw scales (roughly 5:1 between the largest and smallest), but the
model's learned attention-penalty slopes land in a tight band (0.26--0.27) across all five -- confirming the
mechanism is correctly compensating for the different units rather than being dominated by whichever
distance happens to have the largest raw numbers.
\end{{itemize}}
Full detail: \texttt{{final\_comprehensive\_report.pdf}}.

\section*{{2. Self-Audit: A Real Problem Found and Fixed Before It Reached a Paper}}
Before treating the checklist results as final, they were checked for anything that might make the numbers
look artificially good or bad. Two of the new experiments (architecture/weight-decay sweep, training-window
sensitivity) turned out to be using a trade-selection mechanism already known elsewhere in this project to be
unstable -- it picks whatever probability cutoff maximized raw P\&L on one validation year, with no risk
adjustment. Checking the actual thresholds it picked confirmed the problem directly: for two of the three
sampled years, \emph{{every single architecture/weight-decay variant}} converged to the same floor value of
the search grid, regardless of model size -- meaning the apparent differences between configurations were
partly an artifact of that search, not real signal. Both experiments were corrected (one by re-scoring
already-saved predictions, the other by a full retrain) using a simple, fixed decision rule instead (Section
3). The corrected training-window result changed materially and became more sensible: the current 3-year
default is now the best or a strong runner-up in most years, rather than "no window length wins," which was
itself an artifact of the same flawed mechanism.

\section*{{3. The Fix, Shipped: Trust the Model Directly ($p \ge 0.5$)}}
The corrected decision rule is the simplest possible one: if the model's calibrated probability that a trade
will be profitable is at least 50\%, take it; otherwise skip it. No tuning, no search, no reference to any
other trade's prediction -- a fixed constant, decided before ever looking at a backtest result. This was
first validated post-hoc against already-trained models, then \textbf{{actually built into the training
scripts}} (\texttt{{23\_train\_mv\_alibi\_transformer\_v3\_final.py}},
\texttt{{24\_train\_mv\_alibi\_transformer\_v4\_final.py}}) and re-run end to end on all 6 walk-forward folds.

\textbf{{Before shipping it, the choice itself was stress-tested for the obvious objection}} -- did we just
pick the rule that happened to backtest best? Using only the first half of the test years (2021--2023), on
risk-adjusted metrics (not raw dollars, which one lucky trade cluster can dominate), $p\ge0.5$ already beat a
validation-tuned fixed threshold decisively (Sharpe 1.17 vs.\ 0.53, max drawdown -\$268K vs.\ -\$1.81M, 3/3
vs.\ 2/3 positive years) -- using half the data, before ever touching the other half. Applying that same
fixed rule, unchanged, to the untouched 2024--2026 years held up (Sharpe 2.88, 3/3 positive years).

\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrrrr}}
\toprule
Model & Total P\&L & Sharpe & Sortino & Positive years \\
\midrule
Transformer, single seed (v3-final) & {money(v3f.get('total_norm_pnl', np.nan))} & {f2(v3f.get('combined_daily_sharpe', np.nan))} & {f2(v3f.get('combined_daily_sortino', np.nan))} & {v3f.get('positive_years','--')}/{v3f.get('total_years','--')} \\
Transformer, 5-seed ensemble (v4-final) & {money(v4f.get('total_norm_pnl', np.nan))} & {f2(v4f.get('combined_daily_sharpe', np.nan))} & {f2(v4f.get('combined_daily_sortino', np.nan))} & {v4f.get('positive_years','--')}/{v4f.get('total_years','--')} \\
\bottomrule
\end{{tabular}}
\caption{{Shipped, fully causal, all 6 test years (2021--2026).}}
\end{{table}}
Both are now 6/6 positive years for the first time under a rule that is genuinely live-deployable -- every
earlier "good" number in this project (including numbers you've already seen) was computed under a rule that
either wasn't causal or was prone to this kind of tuning noise. Full detail: \texttt{{shipped\_final\_report.pdf}}.

\section*{{4. Model Comparison: A Result That Needs Your Input}}
LightGBM and the MLP baseline were re-trained from scratch and, in the process, brought up to the exact same
technical standard as the transformer: probability calibration (neither had any), the same $p\ge0.5$ decision
rule (both previously used the same unstable mechanism described in Section 2), and correctly-computed daily
Sharpe/Sortino/drawdown (LightGBM's Sharpe had been computed from trade-level, not daily, returns -- a
different and incorrect statistic; neither model had ever computed Sortino or drawdown at all).

\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{lrrrr}}
\toprule
Model & Total P\&L & Sharpe & Sortino & Positive years \\
\midrule
LightGBM (T1+K2) & {money(lgbm_t1k2.get('norm_total_pnl', np.nan))} & {f2(lgbm_t1k2.get('daily_sharpe', np.nan))} & {f2(lgbm_t1k2.get('daily_sortino', np.nan))} & {lgbm_t1k2.get('positive_years','--')}/{lgbm_t1k2.get('total_years','--')} \\
MLP ($F^*$), 1 seed & {money(mlp.get('norm_total_pnl', np.nan))} & {f2(mlp.get('daily_sharpe', np.nan))} & {f2(mlp.get('daily_sortino', np.nan))} & {mlp.get('positive_years','--')}/{mlp.get('total_years','--')} \\
Transformer, 5-seed ensemble & {money(v4f.get('total_norm_pnl', np.nan))} & {f2(v4f.get('combined_daily_sharpe', np.nan))} & {f2(v4f.get('combined_daily_sortino', np.nan))} & {v4f.get('positive_years','--')}/{v4f.get('total_years','--')} \\
\bottomrule
\end{{tabular}}
\caption{{All models, identical selection rule and metrics, 2021--2026.}}
\end{{table}}
\textbf{{Update (Section 6.6): this table is not apples-to-apples on ensembling, and you were right to flag
it.}} Single-seed MLP beats the 5-seed transformer ensemble here, but that's comparing a 1-seed model
against a 5-seed one. Once both are given identical ensembling treatment, \textbf{{the ranking reverses}} --
see Section 6.6 for the full seed-count sensitivity table and the statistically-tested resolution (Section
6.4: neither direction of the single-seed gap is significant). \textbf{{The honest conclusion is that MLP
and the transformer are close, not that either is clearly superior.}} Full detail, including the LightGBM
T1+K2 structural-prior check (unrestricted 4-strategy LightGBM loses -\$37.2M; T1+K2 alone is profitable)
and the technical audit of what was fixed in each model: \texttt{{model\_comparison\_report.pdf}}.

\section*{{5. External Benchmark: Comparison to Glasserman, Li, and Pirjol (SPX)}}
Per your suggestion, our results were compared against the published SPX-based TP2 violation-trading paper
using their exact return convention: each trade normalized to a fixed unit of exposure, reporting mean
return per trade and the ``hit rate'' (percentage of trades that are profitable).

\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{lrr}}
\toprule
& 2021 & 2022 \\
\midrule
Glasserman et al., SPX, $T_1$-denominated & 23\% (100\% hit rate) & 4\% (99.8\% hit rate) \\
Glasserman et al., SPX, $K_2$-denominated & 74\% (100\% hit rate) & 3\% (99.2\% hit rate) \\
\midrule
This project, SPY, unconditional (T1+K2) & 7.9\% (15\% hit rate) & -19.6\% (16\% hit rate) \\
This project, SPY, transformer (v4-final) & 3.7\% (51\% hit rate) & 3.1\% (82\% hit rate) \\
\bottomrule
\end{{tabular}}
\caption{{Only overlapping years shown -- Glasserman et al.'s published data ends in 2022.}}
\end{{table}}
\textbf{{The gap is large and worth discussing directly.}} Their SPX per-trade returns are roughly
5--30$\times$ larger than anything any model in this project produces, and their hit rates (99--100\%) are
far above ours (36--89\% for the learned models; only 15--16\% for unconditional trading, which is the
number most directly comparable to their unconditional table). Plausible reasons, none verified: SPX options
are European-style, cash-settled, on a larger and more liquid index with no early-assignment or dividend
complications; SPY (used throughout this project) is American-style and dividend-paying. Their exact
violation-detection filters, no-arbitrage checks, and data vendor also have not been cross-checked against
ours. \textbf{{This gap should be understood before citing their result as supporting evidence for this
project's approach -- right now the two read as different-magnitude phenomena, not confirmation of the same
effect on a different underlying.}}

\section*{{6. Follow-Up on Your Specific Requests (2026-07-15/16 meeting)}}

\subsection*{{6.1 Parameter counts -- your hypothesis was right}}
You asked directly for the parameter counts of the transformer versus the MLP, to test your hypothesis that
``MLP might be a relatively small model compared to the transformer, and that is why you get better
results.'' That question got answered with the feature count (48) instead of the parameter count in the
meeting -- here are the real numbers:
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lr}}
\toprule
Model & Parameters \\
\midrule
Transformer (hidden=64, 4 heads, both output heads) & {num(transformer_params)} \\
MLP (48 $\to$ 64 $\to$ 32 $\to$ 1) & {num(mlp_params)} \\
\bottomrule
\end{{tabular}}
\end{{table}}
\textbf{{The transformer is about 7$\times$ larger, not smaller.}} This directly supports your hypothesis.
Also worth correcting directly: ``hidden=64'' is the width of the representation, not depth -- the
transformer has one cross-attention block plus two small output heads (roughly 4--5 linear layers total),
not the ``64 layers'' read from the meeting.

\subsection*{{6.2 MLP vs.\ transformer training/validation loss -- ``how quickly is it training''}}
Per your request, MLP's per-epoch train/validation loss was tracked directly (same loss function, binary
cross-entropy, as the transformer -- genuinely comparable), using the transformer's exact checkpointing
convention (best validation loss, patience 5) rather than sklearn's internal heuristic, so the comparison
is apples-to-apples.
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrrrr}}
\toprule
Test year & MLP best epoch & MLP best val\_loss & Transformer best epoch & Transformer best val\_loss \\
\midrule
{convergence_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
\textbf{{MLP converges in 1--3 epochs in 4 of 6 folds}} (vs.\ the transformer's 2--24) \textbf{{and reaches a
lower validation loss in 4 of 6 folds}} (2023/2024/2025/2026). Combined with 6.1, this is a coherent,
mechanistic explanation, not just a selection-rule artifact: a much smaller model, converging faster,
generalizing better on the amount of data available per fold.

\textbf{{Verified, not assumed: best-epoch=1 is real learning, not a v2-style stall.}} Best epoch is 1 in
three of six folds, which raises a fair question -- is this the same ``validation loss never really
decreases'' problem originally diagnosed for the transformer? Checked directly by computing the model's
TRUE pre-training state (sklearn's own random weight initialization, zero gradient steps -- not a
theoretical guess) and comparing it to epoch 1:
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrrr}}
\toprule
Test year & Epoch-0 (untrained) val\_loss & Epoch-1 val\_loss & Reduction \\
\midrule
{epoch0_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
\textbf{{Five of six folds show a 57--82\% loss reduction from the untrained state to epoch 1}} -- substantial,
unambiguous real learning, not a stall. \textbf{{2021 is a genuine exception, worth flagging honestly:}}
val\_loss is essentially flat-to-slightly-worse from epoch 0 to epoch 1 (0.68 $\to$ 0.73) and only reaches
its actual best at epoch 6 -- this fold's dynamics look more like the transformer's gradual convergence
than the ``fast learner'' story that holds everywhere else, and is the one fold in this table that
deserves the same scrutiny originally applied to the transformer's val\_loss trajectory.

\textbf{{Also tested (not just visually smoothed): does stronger regularization change anything?}} MLP's
current $\alpha=$1e-4 converges so fast that its post-peak overfitting looks dramatic in the loss-curve
figure (Figure 2) even though checkpointing already discards those epochs. Re-ran all 6 folds with
$\alpha=$1e-2 (100$\times$ stronger) to see if this is purely cosmetic or actually changes anything:
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrrrr}}
\toprule
& \multicolumn{{2}}{{c}}{{$\alpha=$1e-4 (shipped)}} & \multicolumn{{2}}{{c}}{{$\alpha=$1e-2 (100$\times$ stronger)}} \\
Test year & Best val\_loss & Sharpe & Best val\_loss & Sharpe \\
\midrule
{regcomp_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
Validation loss barely moves in any fold (the convergence speed is dominated by architecture/data, not this
regularization strength), but \textbf{{stronger regularization is a mild, consistent net positive for
trading results}} -- Sharpe improves or holds in 5 of 6 folds (2023: 1.64$\to$1.69, P\&L +\$91K; 2024:
2.10$\to$2.75, P\&L +\$23K), with only a negligible decrease in 2026 (6.25$\to$6.17). \textbf{{Not a dramatic
fix, but a safe one worth adopting}} -- no fold gets meaningfully worse, several get meaningfully better.

\subsection*{{6.3 Rolling vs.\ expanding window -- all 6 folds, not a 3-fold sample}}
Per your request (``use all the previous years for training, not just the previous year''), the expanding
window (every year from 2014 up to the validation year) was tested against the current rolling 3-year
default on all 6 folds.
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrrrrl}}
\toprule
Year & Rolling P\&L & Rolling Sharpe & Expanding P\&L & Expanding Sharpe & Winner \\
\midrule
{window_table}
\bottomrule
\end{{tabular}}
\caption{{Totals: rolling {money(total_rolling)} ({pos_rolling}/6 positive years), expanding
{money(total_expanding)} ({pos_expanding}/6 positive years).}}
\end{{table}}
\textbf{{The honest answer is mixed, not a clean win either way.}} Expanding wins decisively in 2026
(\$629K vs.\ \$3.8K) and helps slightly in 2021/2022; rolling wins clearly in 2023--2025, especially 2024
where expanding collapses (Sharpe 0.06 vs.\ 2.02). Aggregate P\&L favors expanding, driven almost entirely
by 2026; per-year Sharpe is roughly a 3--3 split. Both stay 6/6 positive years. \textbf{{This means the
sentence you suggested for the paper -- ``results are similar for one, two, or three-year window sizes...
robust to window size'' -- is not accurate as stated; 2024 alone shows a swing from Sharpe -2.44 (2yr) to
+2.02 (3yr).}} The defensible version is: the current 3-year default is competitive, not universally best,
and the model is not especially sensitive to \emph{{which}} reasonable window is chosen, but the specific
``similar/robust'' framing should not go into the paper verbatim.

\subsection*{{6.4 Statistical significance: MLP vs.\ transformer}}
Same bootstrap/sign-test methodology already used elsewhere in this project for v2 vs.\ v3.
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrrr}}
\toprule
Comparison & Years favoring MLP & Mean P\&L delta [95\% CI] & Sign-test $p$ \\
\midrule
{sig_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
\textbf{{Neither comparison is statistically significant.}} Both 95\% confidence intervals include zero,
and both sign-test $p$-values are far above 0.05. One year (2023) drives most of the aggregate MLP
advantage. \textbf{{The honest framing for the paper: MLP's edge is suggestive, not proven, with only 6
test years.}}

\subsection*{{6.5 MV-ALiBi diagnostics, extended to the full 5-seed ensemble}}
The tight-clustering result already shown for single-seed v3 (learned attention slopes landing in a narrow
band despite very different raw distance scales) was re-checked across all 5 seeds and 6 folds (30 models).
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrr}}
\toprule
Distance type & Mean learned slope & Std across 30 models \\
\midrule
{v4_attn_summary}
\bottomrule
\end{{tabular}}
\end{{table}}
The 5 seeds are not learning meaningfully different attention strategies from each other -- they reliably
converge to essentially the same distance-weighting solution regardless of random initialization. This
supports reading the ensemble's benefit (v4-final: Sharpe 2.36, 6/6 positive years) as averaging out noise
in individual predictions, not as combining genuinely different learned representations.

\subsection*{{6.6 Seed-count sensitivity, both models -- your comment ([12:23]) was exactly right}}
You said: ``You can see a little bit of that because you are averaging models and only then you get better
performance.'' That was a direct challenge to the Section 4 comparison (single-seed MLP vs.\ 5-seed
transformer). Tested properly -- both models, 1/5/10 seeds where applicable, identical ensembling
methodology:
\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{lrrrrr}}
\toprule
Model & Trades & Total P\&L & Sharpe & Max DD & Pos. years \\
\midrule
{seedcount_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
\textbf{{You were right -- ensembling explains a meaningful part of the original comparison.}} Single-seed
MLP beats single-seed transformer (Sharpe 2.60 vs.\ 1.51), but the 5-seed-ensembled transformer beats the
5-seed-ensembled MLP (2.36 vs.\ 1.66) -- \textbf{{the ranking reverses once both get the same treatment.}}
Ensembling helps the transformer (1$\to$5 seeds: Sharpe 1.51$\to$2.36, drawdown -\$268K$\to$-\$91K) but
hurts MLP (1$\to$5 seeds: Sharpe 2.60$\to$1.66, positive years 6/6$\to$4/6, win rate 54\%$\to$33\% as
$\sim$75\% more, lower-quality trades cross the $p\ge0.5$ bar after recalibration). \textbf{{Also tested:
10-seed transformer under the current shipped methodology}} (never done before -- the original 10-seed run
predates the causal-selection fix). Result: 5$\to$10 seeds partially \emph{{reverses}} the 1$\to$5 gain
(Sharpe 2.36$\to$1.81, drawdown back out to -\$298K) -- 5 seeds looks like a sweet spot for the transformer,
not a monotonic more-is-better relationship, though total P\&L keeps climbing at every step. \textbf{{Neither
model is unambiguously ``the best'' -- this should replace the Section 4 framing in any paper draft.}}

\section*{{Open Items for Discussion}}
\begin{{enumerate}}
\item \textbf{{Resolved: ``MLP beats the transformer'' is not a defensible unqualified claim.}} It was an
ensembling-confounded comparison (6.6, your own catch) -- single-seed MLP wins, 5-seed-ensembled
transformer wins, and the underlying gap isn't statistically significant either way (6.4). Model-capacity
differences (6.1, 6.2: MLP is 7x smaller, converges faster, reaches lower val\_loss single-seed) remain a
real, partial explanation for the single-seed result, but don't settle which model is ``better'' overall.
Decide how to frame this in the paper -- probably as ``comparable performance, different scaling behavior
under ensembling'' rather than a winner/loser framing.
\item Why is the SPX per-trade return/hit-rate so much larger than SPY's? (Section 5) -- still open.
\item Moneyness/price-ratio features are still absent from the frozen feature dictionary -- flagged since
the original diagnosis, still the most likely source of remaining regime-drift sensitivity, still not
touched because it's a shared artifact across every model in the project.
\item The first-liquid (execution-realistic) exit policy remains a separate, still-negative-results track,
untouched by any of the work described here.
\item The two Glasserman/Pirjol theory papers (references [5], [6] in the SPX paper -- ``Total Positivity
and Relative Convexity of Option Prices'' and ``When Are Option Prices TP2?'') are not yet in our
\texttt{{Literature/}} folder and have not been cross-checked against our methodology, per your explicit
request before anything goes to him or gets submitted.
\end{{enumerate}}

\end{{document}}
"""
    REPORT_TEX.write_text(tex, encoding="utf-8")
    print(f"Wrote {REPORT_TEX}")

    try:
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", REPORT_TEX.name],
                cwd=OUT, check=True, capture_output=True, text=True,
            )
        print(f"Wrote {REPORT_TEX.with_suffix('.pdf')}")
    except subprocess.CalledProcessError as exc:
        print("pdflatex failed:")
        print(exc.stdout[-4000:] if exc.stdout else "")
        print(exc.stderr[-4000:] if exc.stderr else "")
        raise


if __name__ == "__main__":
    main()
