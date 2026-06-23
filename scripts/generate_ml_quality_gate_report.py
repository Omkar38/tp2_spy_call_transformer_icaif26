#!/usr/bin/env python3
"""Generate LaTeX report comparing rolling ML quality-gate universes."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from tp2rr2.config import CONFIG


RUNS = {
    "base": {
        "summary": "data/ml/archive2_ml_lightgbm_rolling_summary_net_pnl_base.csv",
        "predictions": "data/ml/archive2_ml_lightgbm_rolling_predictions_net_pnl_base.parquet",
        "label": "Base",
    },
    "volume_and_delta": {
        "summary": "data/ml/archive2_ml_lightgbm_rolling_summary_net_pnl_volume_and_delta.csv",
        "predictions": "data/ml/archive2_ml_lightgbm_rolling_predictions_net_pnl_volume_and_delta.parquet",
        "label": "Volume + Delta",
    },
    "strict_quality_combo": {
        "summary": "data/ml/archive2_ml_lightgbm_rolling_summary_net_pnl_strict_quality_combo.csv",
        "predictions": "data/ml/archive2_ml_lightgbm_rolling_predictions_net_pnl_strict_quality_combo.parquet",
        "label": "Strict Quality",
    },
}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ML quality-gate comparison report.")
    parser.add_argument("--tables-dir", default="reports/raw_archive2_eda/tables/ml_quality_gate")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--tex-output", default="reports/ml_quality_gate_comparison_report.tex")
    parser.add_argument("--pdf-output", default="reports/ml_quality_gate_comparison_report.pdf")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    root = CONFIG.paths.root
    tables_dir = root / args.tables_dir
    figures_dir = root / args.figures_dir
    tex_path = root / args.tex_output
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    year_rows = []
    action_rows = []

    for gate, spec in RUNS.items():
        summary = pd.read_csv(root / spec["summary"])
        test = summary[summary["split"].eq("test")].copy()
        test["Gate"] = spec["label"]
        test["Year"] = test["test_years"].astype(int)
        year_rows.append(test)

        selected = int(test["selected"].sum())
        executed = int(test["executed"].sum())
        invalid = int(test["invalid_selected"].sum())
        pnl = float(test["net_pnl"].sum())
        hit = (
            (test["hit_rate_executed"] * test["executed"]).sum() / executed
            if executed
            else float("nan")
        )
        summaries.append(
            {
                "Gate": spec["label"],
                "Rows after gate": int(test["dataset_rows_after_gate"].iloc[0]),
                "Selected": selected,
                "Executed": executed,
                "Invalid": invalid,
                "Exec rate": executed / selected if selected else float("nan"),
                "Hit rate": hit,
                "Net P&L": pnl,
                "Positive years": int((test["net_pnl"] > 0).sum()),
                "Test years": int(len(test)),
            }
        )

        pred = pd.read_parquet(root / spec["predictions"])
        pred = pred[pred["split"].eq("test")].copy()
        mix = pred.groupby("predicted_action").size().rename("Rows").reset_index()
        mix["Gate"] = spec["label"]
        action_rows.append(mix)

    comparison = pd.DataFrame(summaries)
    year = pd.concat(year_rows, ignore_index=True)
    action_mix = pd.concat(action_rows, ignore_index=True)

    comparison.to_csv(tables_dir / "ml_quality_gate_aggregate_comparison.csv", index=False)
    year.to_csv(tables_dir / "ml_quality_gate_year_comparison.csv", index=False)
    action_mix.to_csv(tables_dir / "ml_quality_gate_action_mix.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    pivot = year.pivot(index="Year", columns="Gate", values="net_pnl").sort_index()
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Rolling ML Net P&L By Quality Gate")
    ax.set_ylabel("Net P&L ($)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    pnl_fig = figures_dir / "ml_quality_gate_yearly_pnl.png"
    fig.savefig(pnl_fig, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = range(len(comparison))
    ax.bar(x, comparison["Selected"], label="Selected")
    ax.bar(x, comparison["Executed"], label="Executed")
    ax.set_xticks(list(x))
    ax.set_xticklabels(comparison["Gate"], rotation=15, ha="right")
    ax.set_title("Selected And Executed Trades By Quality Gate")
    ax.set_ylabel("Rows")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    count_fig = figures_dir / "ml_quality_gate_trade_counts.png"
    fig.savefig(count_fig, dpi=180)
    plt.close(fig)

    display = comparison.copy()
    for col in ["Rows after gate", "Selected", "Executed", "Invalid", "Positive years", "Test years"]:
        display[col] = display[col].map(lambda x: f"{int(x):,}")
    display["Exec rate"] = display["Exec rate"].map(pct)
    display["Hit rate"] = display["Hit rate"].map(pct)
    display["Net P&L"] = display["Net P&L"].map(money)

    year_display = year[
        ["Gate", "Year", "rows", "selected", "executed", "invalid_selected", "hit_rate_executed", "net_pnl", "threshold"]
    ].copy()
    year_display = year_display.rename(
        columns={
            "rows": "Rows",
            "selected": "Selected",
            "executed": "Executed",
            "invalid_selected": "Invalid",
            "hit_rate_executed": "Hit rate",
            "net_pnl": "Net P&L",
            "threshold": "Threshold",
        }
    )
    for col in ["Rows", "Selected", "Executed", "Invalid"]:
        year_display[col] = year_display[col].map(lambda x: f"{int(x):,}")
    year_display["Hit rate"] = year_display["Hit rate"].map(pct)
    year_display["Net P&L"] = year_display["Net P&L"].map(money)
    year_display["Threshold"] = year_display["Threshold"].map(lambda x: f"{float(x):.3f}")

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

\title{{Rolling ML Quality-Gate Comparison\\\large Base vs Filtered TP2 Universes}}
\author{{ML-Based Trading of TP2-Inspired Violations in SPY American Call Options}}
\date{{June 22, 2026}}

\begin{{document}}
\maketitle

\section{{Purpose}}

This report compares the current rolling LightGBM five-action model across
three point-in-time candidate universes: the base conservative bid/ask TP2
universe, the volume-and-delta universe, and the strict market-quality universe.
All runs use the same 3-year train, 1-year validation, 1-year test folds and the
same net-P\&L target.

\section{{Aggregate Results}}
"""
    tex += tex_table(
        display,
        ["Gate", "Rows after gate", "Selected", "Executed", "Invalid", "Exec rate", "Hit rate", "Net P&L", "Positive years"],
        "lrrrrrrrr",
    )
    tex += include_graphic(
        str(pnl_fig.relative_to(tex_path.parent)),
        "Test-year rolling net P&L by quality gate.",
    )
    tex += include_graphic(
        str(count_fig.relative_to(tex_path.parent)),
        "Selected and executed trade counts by quality gate.",
    )
    tex += r"""

\section{Yearly Detail}
"""
    tex += tex_table(
        year_display,
        ["Gate", "Year", "Rows", "Selected", "Executed", "Invalid", "Hit rate", "Net P&L", "Threshold"],
        "lrrrrrrrr",
    )
    tex += r"""

\section{Interpretation}

The base universe remains the primary sample because it has broad coverage and
positive aggregate out-of-sample P\&L, but it includes many low-quality episodes.
The volume-and-delta universe is cleaner but much smaller and produces mixed
rolling results. The strict-quality universe has the cleanest label universe but
is too sparse in recent years, especially 2026, so it should be treated as a
robustness diagnostic rather than a main model sample.

For the paper, this supports a layered empirical design: report the base model
as the main rolling result, then use volume/delta and strict-quality filtered
runs to show how sensitive the ML edge is to market-quality assumptions.

\end{document}
"""
    tex_path.write_text(tex)
    if not args.no_compile:
        compile_tex(tex_path)

    print(f"Wrote {tables_dir / 'ml_quality_gate_aggregate_comparison.csv'}")
    print(f"Wrote {tables_dir / 'ml_quality_gate_year_comparison.csv'}")
    print(f"Wrote {tables_dir / 'ml_quality_gate_action_mix.csv'}")
    print(f"Wrote {tex_path}")
    if not args.no_compile:
        print(f"Compiled {root / args.pdf_output}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
