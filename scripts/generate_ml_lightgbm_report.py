#!/usr/bin/env python3
"""Generate report for rolling LightGBM TP2 strategy-selection results."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from tp2rr2.config import CONFIG


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
        return f"{sign}\\${value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{sign}\\${value / 1e3:.1f}K"
    return f"{sign}\\${value:.0f}"


def pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{100 * float(value):.1f}\\%"


def tex_table(df: pd.DataFrame, columns: list[str], align: str | None = None) -> str:
    align = align or ("l" * len(columns))
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    lines.append(" & ".join(esc(col) for col in columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in df[columns].iterrows():
        lines.append(" & ".join(str(row[col]) for col in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def include_graphic(path: str, caption: str, label: str) -> str:
    return rf"""
\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.92\linewidth]{{{path}}}
  \caption{{{caption}}}
  \label{{{label}}}
\end{{figure}}
"""


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LightGBM rolling ML report.")
    parser.add_argument("--dataset-summary", default="data/ml/archive2_ml_episode_dataset_summary_2013_2026.json")
    parser.add_argument("--summary", default="data/ml/archive2_ml_lightgbm_rolling_summary.csv")
    parser.add_argument("--predictions", default="data/ml/archive2_ml_lightgbm_rolling_predictions.parquet")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--tables-dir", default="reports/raw_archive2_eda/tables/ml_lightgbm")
    parser.add_argument("--tex-output", default="reports/ml_lightgbm_rolling_report.tex")
    parser.add_argument("--pdf-output", default="reports/ml_lightgbm_rolling_report.pdf")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    root = CONFIG.paths.root
    figures_dir = root / args.figures_dir
    tables_dir = root / args.tables_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary = json.loads((root / args.dataset_summary).read_text())
    summary = pd.read_csv(root / args.summary)
    pred = pd.read_parquet(root / args.predictions)
    pred["signal_date"] = pd.to_datetime(pred["signal_date"])
    pred["test_year"] = pred["signal_date"].dt.year

    test_summary = summary[summary["split"].eq("test")].copy()
    ml = test_summary[test_summary["model"].eq("lightgbm_5_action")].copy()
    ml["test_year"] = ml["fold"].str.extract(r"test(\d{4})").astype(int)
    compare = ml.copy()
    compare.to_csv(tables_dir / "lightgbm_test_year_comparison.csv", index=False)

    action_mix = (
        pred[pred["split"].eq("test")]
        .groupby(["test_year", "predicted_action"])
        .size()
        .rename("rows")
        .reset_index()
    )
    action_mix.to_csv(tables_dir / "lightgbm_test_action_mix.csv", index=False)
    action_count_table = (
        action_mix.pivot(index="test_year", columns="predicted_action", values="rows")
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"test_year": "Year"})
    )
    for col in ["skip", "T1", "T2", "K1", "K2"]:
        if col not in action_count_table:
            action_count_table[col] = 0
    action_count_table = action_count_table[["Year", "skip", "T1", "T2", "K1", "K2"]]
    action_count_table.to_csv(tables_dir / "lightgbm_test_action_counts_wide.csv", index=False)
    action_display = action_count_table.copy()
    action_display["Year"] = action_display["Year"].astype(int).astype(str)
    action_display = action_display.rename(
        columns={"skip": "Skip", "T1": "T1", "T2": "T2", "K1": "K1", "K2": "K2"}
    )
    for col in ["Skip", "T1", "T2", "K1", "K2"]:
        action_display[col] = action_display[col].map(lambda x: f"{int(x):,}")

    selected = pred[pred["split"].eq("test") & pred["selected"]].copy()
    daily = selected.groupby("signal_date")["realized_net_pnl"].sum().sort_index().cumsum().reset_index()
    daily.to_csv(tables_dir / "lightgbm_test_cumulative_pnl.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = range(len(compare))
    ax.bar(list(x), compare["net_pnl"] / 1e6, width=0.55, label="LightGBM")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(compare["test_year"].astype(str))
    ax.set_ylabel("Net P&L ($M)")
    ax.set_title("Rolling Test Net P&L By Year")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, figures_dir / "ml_lightgbm_yearly_pnl.png")

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(compare["test_year"], compare["selected"], marker="o", label="selected")
    ax.plot(compare["test_year"], compare["executed"], marker="o", label="executed")
    ax.plot(compare["test_year"], compare["invalid_selected"], marker="o", label="invalid selected")
    ax.set_title("LightGBM Trade Counts By Test Year")
    ax.set_xlabel("Test year")
    ax.set_ylabel("Rows")
    ax.legend()
    ax.grid(alpha=0.25)
    save_fig(fig, figures_dir / "ml_lightgbm_trade_counts.png")

    mix_pivot = action_mix.pivot(index="test_year", columns="predicted_action", values="rows").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    mix_pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Predicted Action Mix In Test Years")
    ax.set_xlabel("Test year")
    ax.set_ylabel("Rows")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    save_fig(fig, figures_dir / "ml_lightgbm_action_mix.png")

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(daily["signal_date"], daily["realized_net_pnl"] / 1e6)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Cumulative Test P&L For Selected LightGBM Trades")
    ax.set_xlabel("Signal date")
    ax.set_ylabel("Cumulative net P&L ($M)")
    ax.grid(alpha=0.25)
    save_fig(fig, figures_dir / "ml_lightgbm_cumulative_pnl.png")

    display = compare[
        [
            "test_year",
            "rows",
            "selected",
            "executed",
            "invalid_selected",
            "hit_rate_executed",
            "net_pnl",
            "threshold",
            "action_label_accuracy",
            "balanced_action_label_accuracy",
        ]
    ].copy()
    display = display.rename(
        columns={
            "test_year": "Year",
            "rows": "Rows",
            "selected": "Selected",
            "executed": "Executed",
            "invalid_selected": "Invalid",
            "hit_rate_executed": "Hit rate",
            "net_pnl": "ML P&L",
            "threshold": "Threshold",
            "action_label_accuracy": "Action-label acc.",
            "balanced_action_label_accuracy": "Balanced action acc.",
        }
    )
    for col in ["Rows", "Selected", "Executed", "Invalid"]:
        display[col] = display[col].map(lambda x: f"{int(x):,}")
    display["Hit rate"] = display["Hit rate"].map(pct)
    display["ML P&L"] = display["ML P&L"].map(money)
    display["Threshold"] = display["Threshold"].map(lambda x: f"{x:.3f}")
    display["Action-label acc."] = display["Action-label acc."].map(pct)
    display["Balanced action acc."] = display["Balanced action acc."].map(pct)

    aggregate = {
        "ml_total_pnl": float(compare["net_pnl"].sum()),
        "ml_selected": int(compare["selected"].sum()),
        "ml_executed": int(compare["executed"].sum()),
        "ml_invalid": int(compare["invalid_selected"].sum()),
        "positive_ml_years": int(compare["net_pnl"].gt(0).sum()),
        "test_years": int(len(compare)),
    }
    (tables_dir / "lightgbm_aggregate_summary.json").write_text(json.dumps(aggregate, indent=2))

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=0.78in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{amsmath}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\setlength{{\parskip}}{{0.45em}}
\setlength{{\parindent}}{{0pt}}
\renewcommand{{\arraystretch}}{{1.12}}

\title{{Rolling LightGBM ML Report\\\large Five-Action TP2 Strategy Selection}}
\author{{ML-Based Trading of TP2-Inspired Violations in SPY American Call Options}}
\date{{June 22, 2026}}

\begin{{document}}
\maketitle

\section{{Purpose}}

This report is the first leakage-safe ML pass. The model chooses among five actions at the first
conservative TP2 violation date:

\[
\text{{skip}},\quad \text{{trade T1}},\quad \text{{trade T2}},\quad \text{{trade K1}},\quad \text{{trade K2}}.
\]

Labels use the first-liquid exit policy. Features are restricted to signal-date values and
lifecycle aggregates from event days \(\le 0\). Future reversion dates, exit dates, realized
P\&L, and future lifecycle rows are excluded from features. The LightGBM decision is per episode:
for each test violation it may choose skip, T1, T2, K1, or K2.

\section{{Dataset}}

\begin{{center}}
\begin{{tabular}}{{lr}}
\toprule
Quantity & Value \\
\midrule
Episode rows & {dataset_summary["rows"]:,} \\
Usable feature columns & {dataset_summary["feature_count"]:,} \\
Primary 2017-forward rows & {dataset_summary["primary_rows_2017_forward"]:,} \\
Early robustness rows & {dataset_summary["early_robustness_rows_2013_2016"]:,} \\
Skip labels & {dataset_summary["action_counts"].get("skip", 0):,} \\
T1 labels & {dataset_summary["action_counts"].get("T1", 0):,} \\
T2 labels & {dataset_summary["action_counts"].get("T2", 0):,} \\
K1 labels & {dataset_summary["action_counts"].get("K1", 0):,} \\
K2 labels & {dataset_summary["action_counts"].get("K2", 0):,} \\
\bottomrule
\end{{tabular}}
\end{{center}}

The removed all-null optional features are:
{", ".join(esc(x) for x in dataset_summary.get("all_null_feature_columns_removed", []))}.

\section{{Rolling Design}}

Primary folds use three training years, one validation year for threshold tuning, and one untouched
test year. The primary evaluation begins at 2017 because 2013 is partial and 2014--2016 have much
smaller conservative episode counts.

\section{{Results}}

\begin{{center}}
\resizebox{{\linewidth}}{{!}}{{%
{tex_table(display, list(display.columns), "rrrrrrrlrrr")}
}}
\end{{center}}

Aggregate LightGBM test P\&L across all primary test years is {money(aggregate["ml_total_pnl"])}.
Every reported decision is made per episode. The report intentionally excludes fixed-strategy
hindsight comparisons so the results cannot be confused with a different decision problem.

{include_graphic("figures/ml_lightgbm_yearly_pnl.png", "Rolling test P\\&L by year for the per-episode LightGBM five-action policy.", "fig:ml-yearly-pnl")}

{include_graphic("figures/ml_lightgbm_trade_counts.png", "Selected, executed, and invalid selected rows by test year.", "fig:ml-trade-counts")}

\section{{Action Usage}}

The next table shows the exact action chosen by the model in each test year. These counts are
per episode, not one strategy per year.

\begin{{center}}
{tex_table(action_display, list(action_display.columns), "rrrrrr")}
\end{{center}}

{include_graphic("figures/ml_lightgbm_action_mix.png", "Predicted action mix by test year.", "fig:ml-action-mix")}

{include_graphic("figures/ml_lightgbm_cumulative_pnl.png", "Cumulative out-of-sample test P\\&L across selected LightGBM trades.", "fig:ml-cumulative-pnl")}

\section{{Interpretation}}

This first ML pass is useful but not yet publishable as a final model. It shows that the lifecycle
and signal-date features contain predictive information in several years, especially 2023--2026,
but performance is unstable in 2021 and near flat in 2022. That instability is important evidence:
the task is not solved by simply applying a classifier to all violations.

The strongest research direction is to improve decision quality rather than increase model size
immediately. Next steps should include expected-P\&L regression per action, capped P\&L targets,
probability calibration, lower invalid-selection penalties, and a Transformer over event days
\(-20\) through 0.

\section{{Next Checks}}

\begin{{enumerate}}
\item Add expected-P\&L LightGBM regressors for T1/T2/K1/K2 and compare against the current multiclass policy.
\item Penalize invalid exits directly in the target, not only in the report.
\item Report capped and winsorized P\&L to reduce concentration risk.
\item Add SHAP or feature importance after the model target is stable.
\end{{enumerate}}

\end{{document}}
"""

    tex_path = root / args.tex_output
    pdf_path = root / args.pdf_output
    tex_path.write_text(tex)
    if args.compile:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=tex_path.parent, check=True)
        generated_pdf = tex_path.with_suffix(".pdf")
        if generated_pdf != pdf_path:
            pdf_path.write_bytes(generated_pdf.read_bytes())
    print(f"Wrote {tex_path}")
    if args.compile:
        print(f"PDF available at {pdf_path}")


if __name__ == "__main__":
    main()
