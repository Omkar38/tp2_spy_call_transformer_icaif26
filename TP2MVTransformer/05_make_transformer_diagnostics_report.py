#!/usr/bin/env python3
"""Build the transformer diagnostics report from saved V2 artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TP2MVTransformer"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MPL_DIR = OUT / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def f2(x: float) -> str:
    if pd.isna(x):
        return "--"
    if np.isposinf(x):
        return "$\\infty$"
    if np.isneginf(x):
        return "$-\\infty$"
    return f"{x:.2f}"


def money(x: float) -> str:
    if pd.isna(x):
        return "--"
    sign = "-" if x < 0 else ""
    return f"{sign}\\${abs(x):,.0f}"


def num(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:,.0f}"


def pct(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{100*x:.1f}\\%"


def safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if exists(path) else pd.DataFrame()


def save_training_plot(hist: pd.DataFrame) -> bool:
    if hist.empty:
        return False
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.8), sharex=False)
    for fold, g in hist.groupby("fold"):
        label = str(g["test_year"].iloc[0])
        axes[0].plot(g["epoch"], g["train_loss"], linewidth=1.3, label=f"train {label}")
        axes[0].plot(g["epoch"], g["val_loss"], linestyle="--", linewidth=1.3, label=f"val {label}")
        axes[1].plot(g["epoch"], g["mean_grad_norm_pre_clip"], linewidth=1.3, label=label)
    axes[0].set_title("Training and validation loss by fold")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7, ncols=3)
    axes[1].set_title("Mean pre-clip gradient norm by fold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Gradient norm")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7, ncols=6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mv_alibi_v2_training_diagnostics.pdf")
    plt.close(fig)
    return True


def save_distance_plot(dist: pd.DataFrame) -> bool:
    if dist.empty:
        return False
    test = dist[dist["split"].eq("test")].copy()
    if test.empty:
        test = dist.copy()
    pivot = test.pivot_table(index="distance", values=["mean", "p99"], aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(pivot.index))
    ax.bar(x - 0.18, pivot["mean"], width=0.36, label="Mean")
    ax.bar(x + 0.18, pivot["p99"], width=0.36, label="P99")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel("Distance scale")
    ax.set_title("MV-ALiBi distance scale diagnostics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mv_alibi_v2_distance_scales.pdf")
    plt.close(fig)
    return True


def save_attention_plot(attn: pd.DataFrame) -> bool:
    if attn.empty:
        return False
    slope_cols = [c for c in attn.columns if c.startswith("slope_")]
    mean_slopes = attn[slope_cols].mean().rename(lambda x: x.replace("slope_", ""))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(mean_slopes.index, mean_slopes.values, color="#476a6f")
    ax.set_ylabel("Mean learned slope")
    ax.set_title("Learned MV-ALiBi distance penalties")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mv_alibi_v2_attention_slopes.pdf")
    plt.close(fig)
    return True


def yearly_rows(yearly: pd.DataFrame) -> str:
    if yearly.empty:
        return "Pending GPU run & -- & -- & -- & -- & -- & -- \\\\"
    rows = []
    for _, r in yearly.iterrows():
        rows.append(
            f"{int(r['test_year'])} & {num(r['test_n_selected'])} & "
            f"{money(r['test_norm_total_pnl'])} & {pct(r['test_win_rate'])} & "
            f"{f2(r.get('test_daily_sharpe', np.nan))} & "
            f"{f2(r.get('test_daily_sortino', np.nan))} & "
            f"{money(r.get('test_max_drawdown', np.nan))} \\\\"
        )
    return "\n".join(rows)


def attention_rows(attn: pd.DataFrame) -> str:
    if attn.empty:
        return "Pending GPU run & -- & -- & -- & -- & -- & -- \\\\"
    slope_cols = ["slope_time", "slope_surface", "slope_short_d", "slope_liquidity", "slope_lifecycle"]
    mean = attn[slope_cols + ["same_strategy_bonus", "cross_strategy_bonus"]].mean()
    return (
        f"Mean across heads/folds & {f2(mean['slope_time'])} & {f2(mean['slope_surface'])} & "
        f"{f2(mean['slope_short_d'])} & {f2(mean['slope_liquidity'])} & "
        f"{f2(mean['slope_lifecycle'])} & {f2(mean['same_strategy_bonus'])} / "
        f"{f2(mean['cross_strategy_bonus'])} \\\\"
    )


def main() -> None:
    hist = safe_read_csv(TABLE_DIR / "mv_alibi_v2_training_history.csv")
    dist = safe_read_csv(TABLE_DIR / "mv_alibi_v2_distance_diagnostics.csv")
    attn = safe_read_csv(TABLE_DIR / "mv_alibi_v2_attention_parameters.csv")
    yearly = safe_read_csv(TABLE_DIR / "mv_alibi_v2_yearly_diagnostics.csv")
    summary_path = TABLE_DIR / "mv_alibi_transformer_v2_summary.json"
    summary = {}
    if exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as fh:
            summary = json.load(fh)

    has_training_plot = save_training_plot(hist)
    has_distance_plot = save_distance_plot(dist)
    has_attention_plot = save_attention_plot(attn)

    status = "available" if not hist.empty and not yearly.empty else "pending GPU rerun"
    total_pnl = summary.get("total_norm_pnl", np.nan)
    total_selected = summary.get("total_selected", np.nan)
    combined_sharpe = summary.get("combined_daily_sharpe", np.nan)
    combined_sortino = summary.get("combined_daily_sortino", np.nan)
    max_dd = summary.get("combined_max_drawdown", np.nan)

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=0.78in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}

\title{{TP2-MV-ALiBi Transformer Diagnostics Report}}
\author{{SPY American Call TP2 Project}}
\date{{\today}}

\begin{{document}}
\sloppy
\maketitle

\section{{Purpose}}
This report addresses the transformer diagnostics requested in the latest project meeting: training/validation loss, gradient norms, learned MV-ALiBi distance weights, distance scale checks, Sortino ratio, and drawdown. It is generated from saved run artifacts, so after a GPU run the PDF can be regenerated without manual number entry.

\section{{Run Status}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lr}}
\toprule
Field & Value \\
\midrule
Diagnostic status & {status} \\
Total selected trades & {num(total_selected)} \\
Total normalized P\&L & {money(total_pnl)} \\
Combined Sharpe & {f2(combined_sharpe)} \\
Combined Sortino & {f2(combined_sortino)} \\
Combined max drawdown & {money(max_dd)} \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Yearly Test Diagnostics}}
\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{rrrrrrr}}
\toprule
Year & Trades & P\&L & Win rate & Sharpe & Sortino & Max drawdown \\
\midrule
{yearly_rows(yearly)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Training Stability}}
The training diagnostics check whether the model is learning smoothly. We want training loss and validation loss to decrease or stabilize, gradient norms to remain controlled, and early stopping to occur before validation behavior deteriorates.
"""
    if has_training_plot:
        tex += r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.92\linewidth]{figures/mv_alibi_v2_training_diagnostics.pdf}
\caption{Training loss, validation loss, and mean pre-clip gradient norm by fold.}
\end{figure}
"""
    else:
        tex += r"""
\textbf{Pending:} training-history artifacts are not available yet. Rerun the V2 transformer trainer on GPU to populate this section.
"""

    tex += r"""
\section{Distance Scale Diagnostics}
The MV-ALiBi bias combines five distances: time, surface, short-D-leg, liquidity, and lifecycle. If one distance has a much larger numerical scale than the others, it may dominate the attention penalty unless the learned slope compensates. This section checks whether normalization or clipping may be needed.
"""
    if has_distance_plot:
        tex += r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.86\linewidth]{figures/mv_alibi_v2_distance_scales.pdf}
\caption{Average test-fold distance scales. Large scale differences suggest a normalization experiment.}
\end{figure}
"""
    else:
        tex += r"""
\textbf{Pending:} distance diagnostics are not available yet.
"""

    tex += rf"""
\section{{Learned Attention Bias Parameters}}
\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{lrrrrrr}}
\toprule
Scope & Time & Surface & Short-D & Liquidity & Lifecycle & Same/Cross bonus \\
\midrule
{attention_rows(attn)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    if has_attention_plot:
        tex += r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.82\linewidth]{figures/mv_alibi_v2_attention_slopes.pdf}
\caption{Mean learned distance penalties. Higher slope means the model more strongly discounts historical episodes that are far along that dimension.}
\end{figure}
"""
    else:
        tex += r"""
\textbf{Pending:} learned attention parameters are not available yet.
"""

    tex += r"""
\section{Interpretation and Next Checks}
\begin{itemize}
  \item If validation loss rises while training loss falls, the transformer is overfitting and needs stronger regularization, lower capacity, or earlier stopping.
  \item If gradient norms are unstable or frequently clipped, learning rate should be reduced and the optimizer schedule should be checked.
  \item If one distance channel is orders of magnitude larger than the others, run a distance-normalized transformer variant before claiming the MV-ALiBi weights are economically interpretable.
  \item If the learned liquidity or lifecycle slopes are near zero, the model is not using those distances strongly; this is useful evidence for architecture simplification or feature redesign.
  \item If Sortino is much weaker than Sharpe, the model has downside-tail risk even if total P\&L is positive.
\end{itemize}

\section{GPU Run Command}
Run the trainer on Colab/GPU:
\begin{verbatim}
python TP2MVTransformer/03_train_mv_alibi_transformer.py
python TP2MVTransformer/05_make_transformer_diagnostics_report.py
\end{verbatim}

Bring back the generated \texttt{tables/} CSVs and \texttt{data/} parquet files before regenerating project-level comparison reports.

\end{document}
"""
    out = OUT / "transformer_diagnostics_report.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
