#!/usr/bin/env python3
"""Generate LaTeX-sourced robustness reports."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

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
    if value >= 1e9:
        return f"{sign}${value / 1e9:.2f}B"
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


def figure(path: str, caption: str) -> str:
    return rf"""
\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.92\linewidth]{{{path}}}
  \caption{{{esc(caption)}}}
\end{{figure}}
"""


def preamble(title: str, subtitle: str) -> str:
    return rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=0.78in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{tabularx}}
\usepackage{{array}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{amsmath}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\setlength{{\parskip}}{{0.45em}}
\setlength{{\parindent}}{{0pt}}
\renewcommand{{\arraystretch}}{{1.12}}

\title{{{esc(title)}\\\large {esc(subtitle)}}}
\author{{ML-Based Trading of TP2-Inspired Violations in SPY American Call Options}}
\date{{June 22, 2026}}

\begin{{document}}
\maketitle
"""


def compile_tex(tex_path: Path) -> None:
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tex_path.parent,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def generate_strategy_report(root: Path, compile_pdf: bool) -> None:
    tables = root / "reports/raw_archive2_eda/tables/strategy_exit_robustness"
    figures = root / "reports/figures"
    out = root / "reports/strategy_exit_robustness_report.tex"

    summary = pd.read_csv(tables / "strategy_exit_policy_summary.csv")
    rankings = pd.read_csv(tables / "strategy_exit_policy_rankings.csv")

    display = summary.copy()
    display["Exit"] = display["exit_policy_label"]
    display["Strategy"] = display["strategy"]
    display["Valid"] = display["valid"].map(lambda x: f"{int(x):,}")
    display["Valid rate"] = display["valid_rate"].map(pct)
    display["Hit rate"] = display["hit_rate"].map(pct)
    display["Net P&L"] = display["net_pnl"].map(money)
    display["Median return"] = display["median_return"].map(pct)
    display["Mean hold"] = display["mean_holding_days"].map(lambda x: f"{x:.2f}")

    rank = rankings.sort_values(["exit_policy", "pnl_rank_within_policy"]).copy()
    rank["Exit"] = rank["exit_policy_label"]
    rank["Strategy"] = rank["strategy"]
    rank["P&L rank"] = rank["pnl_rank_within_policy"].astype(int).astype(str)
    rank["Return rank"] = rank["median_return_rank_within_policy"].astype(int).astype(str)
    rank["Net P&L"] = rank["net_pnl"].map(money)
    rank["Median return"] = rank["median_return"].map(pct)

    tex = preamble("Strategy Exit-Policy Robustness", "LaTeX Source Report")
    tex += r"""
\section{Purpose}

This report checks whether the four TP2 strategy denominations produce stable
conclusions across exit definitions. It uses the existing liquid-exit strategy
labels and does not recalculate raw TP2 detection.

\section{Exit Policies}

The policies are: reversion-day exit, first-liquid-on-or-after-reversion exit,
reversion plus 1/3/5 trading sessions, and selected-leg expiration payoff.

\section{Policy Strategy Summary}
"""
    tex += tex_table(
        display,
        ["Exit", "Strategy", "Valid", "Valid rate", "Hit rate", "Net P&L", "Median return", "Mean hold"],
        "llrrrrrr",
    )
    tex += r"""

\section{Within-Policy Rankings}
"""
    tex += tex_table(
        rank,
        ["Exit", "Strategy", "P&L rank", "Return rank", "Net P&L", "Median return"],
        "llrrrr",
    )
    tex += figure(
        str((figures / "strategy_robustness_policy_pnl.png").relative_to(out.parent)),
        "Strategy net P&L by exit policy.",
    )
    tex += figure(
        str((figures / "strategy_robustness_median_return.png").relative_to(out.parent)),
        "Median realized return by exit policy.",
    )
    tex += figure(
        str((figures / "strategy_robustness_first_liquid_year_heatmap.png").relative_to(out.parent)),
        "First-liquid exit year-by-strategy net P&L heatmap.",
    )
    tex += r"""

\section{Interpretation}

Only two of the twenty-four policy-strategy cells are positive in aggregate.
Unconditional premium-resale trading is fragile. Expiration payoff behaves
differently from first-liquid premium resale, so the paper should report those
as separate benchmarks. The defensible claim is ML-based selection of tradeable
TP2 episodes, not universal arbitrage from every violation.

\end{document}
"""
    out.write_text(tex)
    if compile_pdf:
        compile_tex(out)


def generate_market_quality_report(root: Path, compile_pdf: bool) -> None:
    tables = root / "reports/raw_archive2_eda/tables/market_quality_robustness"
    figures = root / "reports/figures"
    out = root / "reports/market_quality_robustness_report.tex"

    gate = pd.read_csv(tables / "market_quality_gate_summary.csv")
    strat = pd.read_csv(tables / "market_quality_strategy_summary.csv")

    gate_display = gate.copy()
    gate_display["Episodes"] = gate_display["episodes"].map(lambda x: f"{int(x):,}")
    gate_display["Kept"] = gate_display["kept_share"].map(pct)
    gate_display["Positive labels"] = gate_display["positive_label_share"].map(pct)
    gate_display["Med max spread"] = gate_display["median_max_spread_pct"].map(pct)
    gate_display["Med min volume"] = gate_display["median_min_volume"].map(lambda x: "NA" if pd.isna(x) else f"{x:,.0f}")
    gate_display["Med min OI"] = gate_display["median_min_open_interest"].map(lambda x: "NA" if pd.isna(x) else f"{x:,.0f}")
    gate_display["Med round dist"] = gate_display["median_max_round_distance"].map(lambda x: "NA" if pd.isna(x) else f"{x:.2f}")

    key_gates = [
        "base_conservative_bidask",
        "all_legs_volume_positive",
        "paper_delta_01_99",
        "volume_and_delta",
        "strict_quality_combo",
    ]
    strat_display = strat[strat["gate"].isin(key_gates)].copy()
    strat_display["Valid"] = strat_display["valid"].map(lambda x: f"{int(x):,}")
    strat_display["Valid rate"] = strat_display["valid_rate"].map(pct)
    strat_display["Hit rate"] = strat_display["hit_rate"].map(pct)
    strat_display["Net P&L"] = strat_display["net_pnl"].map(money)
    strat_display["Median return"] = strat_display["median_return"].map(pct)

    tex = preamble("Market-Quality Robustness", "LaTeX Source Report")
    tex += r"""
\section{Purpose}

This report tests whether the conservative TP2 episode universe is dominated by
weak market-quality rows. The gates are sensitivity diagnostics, not permanent
filters yet.

\section{Gate Retention}
"""
    tex += tex_table(
        gate_display,
        ["gate", "Episodes", "Kept", "Positive labels", "Med max spread", "Med min volume", "Med min OI", "Med round dist"],
        "lrrrrrrr",
    )
    tex += figure(
        str((figures / "market_quality_gate_retention.png").relative_to(out.parent)),
        "Retention share for each market-quality gate.",
    )
    tex += figure(
        str((figures / "market_quality_strategy_pnl.png").relative_to(out.parent)),
        "First-liquid strategy net P&L under selected market-quality gates.",
    )
    tex += r"""

\section{Key Strategy Results}
"""
    tex += tex_table(
        strat_display,
        ["gate", "strategy", "Valid", "Valid rate", "Hit rate", "Net P&L", "Median return"],
        "llrrrrr",
    )
    tex += r"""

\section{Interpretation}

Market-quality gates materially improve the economic label universe. The
volume-and-delta gate keeps a smaller but much cleaner subset, while the strict
quality combo keeps only about four percent of episodes and should initially be
treated as a robustness universe rather than the main sample. The next ML step
should compare the base universe against volume-and-delta and strict-quality
filtered universes.

\end{document}
"""
    out.write_text(tex)
    if compile_pdf:
        compile_tex(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX-sourced robustness reports.")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    root = CONFIG.paths.root
    compile_pdf = not args.no_compile
    generate_strategy_report(root, compile_pdf)
    generate_market_quality_report(root, compile_pdf)
    print("Wrote reports/strategy_exit_robustness_report.tex")
    print("Wrote reports/market_quality_robustness_report.tex")
    if compile_pdf:
        print("Compiled reports/strategy_exit_robustness_report.pdf")
        print("Compiled reports/market_quality_robustness_report.pdf")


if __name__ == "__main__":
    main()
