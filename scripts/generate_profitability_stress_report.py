#!/usr/bin/env python3
"""Generate LaTeX profitability stress-test report for rolling ML outputs."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tp2rr2.config import CONFIG


RUNS = {
    "base": {
        "label": "Base",
        "predictions": "data/ml/archive2_ml_lightgbm_rolling_predictions_net_pnl_base.parquet",
    },
    "volume_and_delta": {
        "label": "Volume + Delta",
        "predictions": "data/ml/archive2_ml_lightgbm_rolling_predictions_net_pnl_volume_and_delta.parquet",
    },
    "strict_quality_combo": {
        "label": "Strict Quality",
        "predictions": "data/ml/archive2_ml_lightgbm_rolling_predictions_net_pnl_strict_quality_combo.parquet",
    },
}
STRATEGIES = ["T1", "T2", "K1", "K2"]
BASE_COMMISSION = 0.65
STRESS_COMMISSIONS = [0.65, 1.00, 1.50, 2.00]


def esc(value) -> str:
    text = "" if pd.isna(value) else str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def money(value: float) -> str:
    if pd.isna(value):
        return "NA"
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    if value >= 1e6:
        return f"{sign}${value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{sign}${value / 1e3:.1f}K"
    return f"{sign}${value:.0f}"


def pct(value: float) -> str:
    return "NA" if pd.isna(value) else f"{100 * float(value):.1f}%"


def tex_table(df: pd.DataFrame, columns: list[str], align: str | None = None) -> str:
    align = align or ("l" * len(columns))
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    lines.append(" & ".join(esc(c) for c in columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in df[columns].iterrows():
        lines.append(" & ".join(esc(row[c]) for c in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def include_graphic(path: str, caption: str) -> str:
    return rf"""
\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.92\linewidth]{{{path}}}
  \caption{{{esc(caption)}}}
\end{{figure}}
"""


def compile_tex(tex_path: Path) -> None:
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tex_path.parent,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def load_label_components(root: Path) -> pd.DataFrame:
    labels = pd.read_parquet(
        root / "data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet",
        columns=[
            "episode_id",
            "strategy",
            "exit_policy",
            "is_valid_backtest",
            "price_pnl",
            "transaction_cost",
            "net_pnl",
            "realized_return",
        ],
    )
    labels = labels[
        labels["exit_policy"].eq("exit_first_liquid_on_or_after_reversion")
        & labels["strategy"].isin(STRATEGIES)
    ].copy()
    return labels.drop(columns=["exit_policy"]).rename(columns={"strategy": "predicted_action"})


def attach_components(root: Path, gate: str, spec: dict, labels: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_parquet(root / spec["predictions"])
    pred = pred[pred["split"].eq("test") & pred["selected"]].copy()
    pred["signal_date"] = pd.to_datetime(pred["signal_date"])
    pred["Gate"] = spec["label"]
    merged = pred.merge(
        labels,
        on=["episode_id", "predicted_action"],
        how="left",
        suffixes=("", "_label"),
    )
    merged["component_valid"] = merged["is_valid_backtest"].fillna(False)
    # Invalid selected rows stay counted but contribute zero P&L under stress.
    merged["base_price_pnl"] = np.where(merged["component_valid"], merged["price_pnl"].fillna(0.0), 0.0)
    merged["base_transaction_cost"] = np.where(
        merged["component_valid"], merged["transaction_cost"].fillna(0.0), 0.0
    )
    merged["base_net_pnl_recomputed"] = merged["base_price_pnl"] - merged["base_transaction_cost"]
    merged["year"] = merged["signal_date"].dt.year
    return merged


def max_drawdown(daily_pnl: pd.Series) -> float:
    curve = daily_pnl.cumsum()
    peak = curve.cummax()
    drawdown = curve - peak
    return float(drawdown.min()) if len(drawdown) else 0.0


def summarize_gate(df: pd.DataFrame) -> dict:
    executed = df[df["component_valid"]].copy()
    selected = len(df)
    valid = len(executed)
    pnl = df["base_net_pnl_recomputed"]
    pos = executed["base_net_pnl_recomputed"].gt(0).sum()
    daily = df.groupby("signal_date")["base_net_pnl_recomputed"].sum().sort_index()
    sorted_pos = executed["base_net_pnl_recomputed"].sort_values(ascending=False)
    top1_n = max(1, int(np.ceil(0.01 * len(sorted_pos)))) if len(sorted_pos) else 0
    top5_n = max(1, int(np.ceil(0.05 * len(sorted_pos)))) if len(sorted_pos) else 0
    total = float(pnl.sum())
    return {
        "Gate": df["Gate"].iloc[0],
        "Selected": selected,
        "Executed": valid,
        "Invalid": selected - valid,
        "Hit rate": pos / valid if valid else np.nan,
        "Net P&L": total,
        "Median executed P&L": float(executed["base_net_pnl_recomputed"].median()) if valid else np.nan,
        "Max drawdown": max_drawdown(daily),
        "Top 1% P&L": float(sorted_pos.head(top1_n).sum()) if top1_n else 0.0,
        "Top 5% P&L": float(sorted_pos.head(top5_n).sum()) if top5_n else 0.0,
        "Top 1% share": float(sorted_pos.head(top1_n).sum() / total) if total else np.nan,
        "Top 5% share": float(sorted_pos.head(top5_n).sum() / total) if total else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate profitability stress-test report.")
    parser.add_argument("--tables-dir", default="reports/raw_archive2_eda/tables/profitability_stress")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--tex-output", default="reports/profitability_stress_report.tex")
    parser.add_argument("--pdf-output", default="reports/profitability_stress_report.pdf")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    root = CONFIG.paths.root
    tables_dir = root / args.tables_dir
    figures_dir = root / args.figures_dir
    tex_path = root / args.tex_output
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    labels = load_label_components(root)
    selected = pd.concat(
        [attach_components(root, gate, spec, labels) for gate, spec in RUNS.items()],
        ignore_index=True,
    )
    selected.to_parquet(tables_dir / "profitability_stress_selected_trades.parquet", index=False)

    aggregate = pd.DataFrame([summarize_gate(group) for _, group in selected.groupby("Gate", sort=False)])

    commission_rows = []
    for gate, group in selected.groupby("Gate", sort=False):
        for commission in STRESS_COMMISSIONS:
            ratio = commission / BASE_COMMISSION
            stressed = group["base_price_pnl"] - group["base_transaction_cost"] * ratio
            commission_rows.append(
                {
                    "Gate": gate,
                    "Commission": commission,
                    "Net P&L": float(stressed.sum()),
                    "Executed": int(group["component_valid"].sum()),
                    "Hit rate": float((stressed[group["component_valid"]] > 0).sum() / group["component_valid"].sum())
                    if group["component_valid"].sum()
                    else np.nan,
                }
            )
    commission = pd.DataFrame(commission_rows)

    tail_rows = []
    for gate, group in selected.groupby("Gate", sort=False):
        executed = group[group["component_valid"]].copy()
        pnl = executed["base_net_pnl_recomputed"].to_numpy(float)
        total = float(group["base_net_pnl_recomputed"].sum())
        if len(pnl):
            p99 = np.quantile(pnl, 0.99)
            p95 = np.quantile(pnl, 0.95)
            p01 = np.quantile(pnl, 0.01)
            p05 = np.quantile(pnl, 0.05)
        else:
            p99 = p95 = p01 = p05 = np.nan
        tail_rows.extend(
            [
                {"Gate": gate, "Stress": "Original", "Net P&L": total},
                {
                    "Gate": gate,
                    "Stress": "Remove top 1% executed trades",
                    "Net P&L": float(group["base_net_pnl_recomputed"].sum() - executed[executed["base_net_pnl_recomputed"] >= p99]["base_net_pnl_recomputed"].sum()),
                },
                {
                    "Gate": gate,
                    "Stress": "Remove top 5% executed trades",
                    "Net P&L": float(group["base_net_pnl_recomputed"].sum() - executed[executed["base_net_pnl_recomputed"] >= p95]["base_net_pnl_recomputed"].sum()),
                },
                {
                    "Gate": gate,
                    "Stress": "Winsorize 1/99%",
                    "Net P&L": float(np.clip(pnl, p01, p99).sum()) if len(pnl) else 0.0,
                },
                {
                    "Gate": gate,
                    "Stress": "Winsorize 5/95%",
                    "Net P&L": float(np.clip(pnl, p05, p95).sum()) if len(pnl) else 0.0,
                },
            ]
        )
    tail = pd.DataFrame(tail_rows)

    yearly = (
        selected.groupby(["Gate", "year"])
        .agg(
            selected=("episode_id", "size"),
            executed=("component_valid", "sum"),
            net_pnl=("base_net_pnl_recomputed", "sum"),
        )
        .reset_index()
    )

    aggregate.to_csv(tables_dir / "profitability_stress_aggregate.csv", index=False)
    commission.to_csv(tables_dir / "profitability_stress_commission.csv", index=False)
    tail.to_csv(tables_dir / "profitability_stress_tail.csv", index=False)
    yearly.to_csv(tables_dir / "profitability_stress_yearly.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    pivot = commission.pivot(index="Commission", columns="Gate", values="Net P&L") / 1e6
    pivot.plot(marker="o", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Commission Stress: Net P&L")
    ax.set_ylabel("Net P&L ($M)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    commission_fig = figures_dir / "profitability_stress_commission.png"
    fig.savefig(commission_fig, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    tail_pivot = tail.pivot(index="Stress", columns="Gate", values="Net P&L") / 1e6
    tail_pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Tail Sensitivity: Net P&L")
    ax.set_ylabel("Net P&L ($M)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    tail_fig = figures_dir / "profitability_stress_tail.png"
    fig.savefig(tail_fig, dpi=180)
    plt.close(fig)

    display_agg = aggregate.copy()
    for col in ["Selected", "Executed", "Invalid"]:
        display_agg[col] = display_agg[col].map(lambda x: f"{int(x):,}")
    for col in ["Hit rate", "Top 1% share", "Top 5% share"]:
        display_agg[col] = display_agg[col].map(pct)
    for col in ["Net P&L", "Median executed P&L", "Max drawdown", "Top 1% P&L", "Top 5% P&L"]:
        display_agg[col] = display_agg[col].map(money)

    display_comm = commission.copy()
    display_comm["Commission"] = display_comm["Commission"].map(lambda x: f"${x:.2f}")
    display_comm["Net P&L"] = display_comm["Net P&L"].map(money)
    display_comm["Executed"] = display_comm["Executed"].map(lambda x: f"{int(x):,}")
    display_comm["Hit rate"] = display_comm["Hit rate"].map(pct)

    display_tail = tail.copy()
    display_tail["Net P&L"] = display_tail["Net P&L"].map(money)

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=0.78in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\setlength{{\parskip}}{{0.45em}}
\setlength{{\parindent}}{{0pt}}
\renewcommand{{\arraystretch}}{{1.12}}

\title{{Profitability Stress Tests\\\large Rolling ML TP2 Strategy Selection}}
\author{{ML-Based Trading of TP2-Inspired Violations in SPY American Call Options}}
\date{{June 22, 2026}}

\begin{{document}}
\maketitle

\section{{Purpose}}

This report stress-tests the rolling LightGBM selected trades. It does not
retrain the models. Instead, it takes the selected test trades and checks
commission sensitivity, tail dependence, concentration, and drawdown.

\section{{Aggregate Selected-Trade Diagnostics}}
"""
    tex += tex_table(
        display_agg,
        [
            "Gate",
            "Selected",
            "Executed",
            "Invalid",
            "Hit rate",
            "Net P&L",
            "Median executed P&L",
            "Max drawdown",
            "Top 1% share",
            "Top 5% share",
        ],
        "lrrrrrrrrr",
    )
    tex += r"""

\section{Commission Stress}
"""
    tex += tex_table(display_comm, ["Gate", "Commission", "Executed", "Hit rate", "Net P&L"], "lrrrr")
    tex += include_graphic(
        str(commission_fig.relative_to(tex_path.parent)),
        "Net P&L under higher per-contract per-side commission assumptions.",
    )
    tex += r"""

\section{Tail Stress}
"""
    tex += tex_table(display_tail, ["Gate", "Stress", "Net P&L"], "llr")
    tex += include_graphic(
        str(tail_fig.relative_to(tex_path.parent)),
        "Net P&L after removing or winsorizing top executed-trade tails.",
    )
    tex += r"""

\section{Interpretation}

The main paper result should not rely only on aggregate P\&L. Concentration,
drawdown, and tail sensitivity must be reported next to the rolling ML results.
If top-tail removal destroys the base result, the paper should frame the model
as identifying rare high-convexity TP2 episodes rather than producing a smooth
market-neutral edge. Filtered universes improve hit rate and execution quality,
but their smaller opportunity set naturally reduces total P\&L.

\end{document}
"""
    tex_path.write_text(tex)
    if not args.no_compile:
        compile_tex(tex_path)

    print(f"Wrote {tables_dir / 'profitability_stress_aggregate.csv'}")
    print(f"Wrote {tables_dir / 'profitability_stress_commission.csv'}")
    print(f"Wrote {tables_dir / 'profitability_stress_tail.csv'}")
    print(f"Wrote {tables_dir / 'profitability_stress_yearly.csv'}")
    print(f"Wrote {tex_path}")
    if not args.no_compile:
        print(f"Compiled {root / args.pdf_output}")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
