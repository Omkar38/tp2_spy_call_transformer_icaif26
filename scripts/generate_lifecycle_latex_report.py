#!/usr/bin/env python3
"""Generate a polished LaTeX report for TP2 lifecycle tracking."""
from __future__ import annotations

import argparse
import json
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


def pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{100 * float(value):.2f}\\%"


def num(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def money(value: float) -> str:
    if pd.isna(value):
        return "NA"
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    if value >= 1e9:
        return f"{sign}\\${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{sign}\\${value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{sign}\\${value / 1e3:.1f}K"
    return f"{sign}\\${value:.0f}"


def int_or_na(value) -> str:
    if value is None or value == "NA" or pd.isna(value):
        return "NA"
    return f"{int(value):,}"


def tex_table(df: pd.DataFrame, columns: list[str], align: str | None = None) -> str:
    align = align or ("l" * len(columns))
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    lines.append(" & ".join(esc(col) for col in columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in df[columns].iterrows():
        lines.append(" & ".join(str(row[col]) for col in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def include_graphic(path: str, caption: str, label: str, width: str = "0.92\\linewidth") -> str:
    return rf"""
\begin{{figure}}[H]
  \centering
  \includegraphics[width={width}]{{{path}}}
  \caption{{{caption}}}
  \label{{{label}}}
\end{{figure}}
"""


def nearest_event_rows(event: pd.DataFrame, requested: list[int]) -> pd.DataFrame:
    rows = []
    available = sorted(event["event_day_index"].unique())
    for day in requested:
        if day not in available:
            continue
        rows.append(event[event["event_day_index"].eq(day)].iloc[0])
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX lifecycle tracking report.")
    parser.add_argument(
        "--lifecycle",
        default="data/features/archive2_lifecycle/tp2_conservative_episode_lifecycle_2013_2026.parquet",
    )
    parser.add_argument(
        "--summary",
        default="data/features/archive2_lifecycle/tp2_conservative_episode_lifecycle_summary_2013_2026.json",
    )
    parser.add_argument(
        "--clean-summary",
        default="data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026_summary.json",
    )
    parser.add_argument(
        "--episode-summary",
        default="data/features/archive2_episodes/tp2_call_episode_summary_2013_2026.json",
    )
    parser.add_argument(
        "--conservative-summary",
        default="data/labels/archive2_episode_start_labels_conservative_bidask_summary_2013_2026.json",
    )
    parser.add_argument(
        "--tables-dir",
        default="reports/raw_archive2_eda/tables/lifecycle_tracking",
    )
    parser.add_argument("--tex-output", default="reports/lifecycle_tracking_report.tex")
    parser.add_argument("--pdf-output", default="reports/lifecycle_tracking_report.pdf")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    root = CONFIG.paths.root
    tables_dir = root / args.tables_dir
    tex_path = root / args.tex_output
    pdf_path = root / args.pdf_output

    lifecycle_summary = json.loads((root / args.summary).read_text())
    clean_summary = json.loads((root / args.clean_summary).read_text())
    episode_summary = json.loads((root / args.episode_summary).read_text())
    conservative_summary = json.loads((root / args.conservative_summary).read_text())
    event = pd.read_csv(tables_dir / "lifecycle_event_time_pressure_summary.csv")
    status = pd.read_csv(tables_dir / "lifecycle_status_by_event_day.csv")
    episode_level = json.loads((tables_dir / "lifecycle_episode_level_summary.json").read_text())

    checkpoints = nearest_event_rows(event, [-20, -10, -5, -1, 0, 1, 3, 5, 10, 20, 40, 53])
    checkpoint_display = checkpoints[
        [
            "event_day_index",
            "unique_episodes",
            "pressure_rows",
            "median_pressure",
            "p25_pressure",
            "p75_pressure",
            "violation_share",
            "conservative_violation_share",
            "tradable_50pct_share",
        ]
    ].copy()
    checkpoint_display = checkpoint_display.rename(
        columns={
            "event_day_index": "Day",
            "unique_episodes": "Episodes",
            "pressure_rows": "Evaluable",
            "median_pressure": "Median pressure",
            "p25_pressure": "P25",
            "p75_pressure": "P75",
            "violation_share": "Mid violation",
            "conservative_violation_share": "Bid/ask violation",
            "tradable_50pct_share": "All legs liquid",
        }
    )
    checkpoint_display["Day"] = checkpoint_display["Day"].astype(int).astype(str)
    checkpoint_display["Episodes"] = checkpoint_display["Episodes"].map(lambda x: f"{int(x):,}")
    checkpoint_display["Evaluable"] = checkpoint_display["Evaluable"].map(lambda x: f"{int(x):,}")
    for col in ["Median pressure", "P25", "P75"]:
        checkpoint_display[col] = checkpoint_display[col].map(lambda x: num(x, 4))
    for col in ["Mid violation", "Bid/ask violation", "All legs liquid"]:
        checkpoint_display[col] = checkpoint_display[col].map(pct)

    status_days = status[status["event_day_index"].isin([-20, -10, -5, -1, 0, 1, 3, 5, 10, 20, 40, 53])]
    status_pivot = (
        status_days.pivot_table(index="event_day_index", columns="status", values="share", fill_value=0)
        .reset_index()
        .rename(columns={"event_day_index": "Day"})
    )
    status_cols = [
        "observable",
        "liquidity_failure_wide_spread",
        "missing_A_or_B",
        "missing_C_or_D",
        "missing_forward",
    ]
    for col in status_cols:
        if col not in status_pivot:
            status_pivot[col] = 0.0
    status_display = status_pivot[["Day", *status_cols]].copy()
    status_display["Day"] = status_display["Day"].astype(int).astype(str)
    status_display = status_display.rename(
        columns={
            "observable": "Observable",
            "liquidity_failure_wide_spread": "Wide spread",
            "missing_A_or_B": "Missing A/B",
            "missing_C_or_D": "Missing C/D",
            "missing_forward": "Missing F",
        }
    )
    for col in ["Observable", "Wide spread", "Missing A/B", "Missing C/D", "Missing F"]:
        status_display[col] = status_display[col].map(pct)

    price_bound_rows = pd.DataFrame(clean_summary["years"])
    price_bound_display = price_bound_rows[
        price_bound_rows["call_price_bound_fail_rows"].gt(0)
    ][["year", "clean_rows", "call_price_bound_fail_rows"]].copy()
    price_bound_display = price_bound_display.rename(
        columns={
            "year": "Year",
            "clean_rows": "Clean rows",
            "call_price_bound_fail_rows": "Sanity-filtered rows",
        }
    )
    price_bound_display["Year"] = price_bound_display["Year"].astype(str)
    price_bound_display["Clean rows"] = price_bound_display["Clean rows"].map(lambda x: f"{int(x):,}")
    price_bound_display["Sanity-filtered rows"] = price_bound_display["Sanity-filtered rows"].map(
        lambda x: f"{int(x):,}"
    )

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=0.78in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{tabularx}}
\usepackage{{array}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{amsmath}}
\usepackage{{tikz}}
\usetikzlibrary{{arrows.meta,positioning,shapes.geometric}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\setlength{{\parskip}}{{0.45em}}
\setlength{{\parindent}}{{0pt}}
\renewcommand{{\arraystretch}}{{1.13}}

\title{{TP2 Violation Lifecycle Tracking Report\\\large SPY Call Option Research Pipeline}}
\author{{ML-Based Trading of TP2-Inspired Violations in SPY American Call Options}}
\date{{June 22, 2026}}

\begin{{document}}
\maketitle

\section{{Purpose}}

This report documents the lifecycle layer used before machine learning. The goal is to track each
conservative bid/ask TP2 episode from the pre-violation window through the T1 expiration boundary,
then understand whether the violation is sudden, persistent, tradable, and useful for later
strategy-label construction.

The report reflects the corrected pipeline after the raw-data sanity screen for impossible call
quotes. The clean-data layer now removes call quotes that fail the broad static bound:

\[
\max(S-K,0)-1 \le C_{{mid}} \le 1.05S+1.
\]

This screen removes raw vendor artifacts such as the March 2017 deep-in-the-money SPY call quote
that was far below intrinsic on one date and above the underlying on another date. It is a data
quality filter, not a Black-Scholes pricing assumption.

\section{{Current Pipeline}}

\begin{{figure}}[H]
\centering
\begin{{tikzpicture}}[
  node distance=0.86cm and 0.72cm,
  box/.style={{rectangle, rounded corners=2pt, draw=black!65, fill=blue!6, text width=2.75cm, align=center, minimum height=0.86cm}},
  good/.style={{rectangle, rounded corners=2pt, draw=black!65, fill=green!8, text width=2.75cm, align=center, minimum height=0.86cm}},
  warn/.style={{rectangle, rounded corners=2pt, draw=black!65, fill=orange!10, text width=2.75cm, align=center, minimum height=0.86cm}},
  arrow/.style={{-{{Latex[length=2.1mm]}}, thick, draw=black!70}}
]
\node[box] (raw) {{Professor raw\\Archive-2 SPY calls\\2013--2026}};
\node[box, right=of raw] (clean) {{Clean quote layer\\last available quote\\price sanity bound}};
\node[box, right=of clean] (detect) {{Forward TP2 detector\\A/B/C/D rectangles\\all moneyness}};
\node[good, below=of detect] (episodes) {{Episode tracking\\same A/B pair\\one continuing event}};
\node[good, left=of episodes] (conservative) {{Conservative entry\\Aask Bask < Cbid Dbid}};
\node[warn, left=of conservative] (lifecycle) {{Lifecycle panel\\event days -20 to T1\\wide quotes retained}};
\node[good, below=of lifecycle] (exit) {{Liquid-exit labels\\entry/exit liquidity\\strategy P\&L}};
\node[box, right=of exit] (ml) {{Next phase\\rolling ML selection\\episode-level actions}};
\draw[arrow] (raw) -- (clean);
\draw[arrow] (clean) -- (detect);
\draw[arrow] (detect) -- (episodes);
\draw[arrow] (episodes) -- (conservative);
\draw[arrow] (conservative) -- (lifecycle);
\draw[arrow] (lifecycle) -- (exit);
\draw[arrow] (exit) -- (ml);
\end{{tikzpicture}}
\caption{{Final pre-ML pipeline. The lifecycle report covers the highlighted lifecycle layer and its connection to liquid-exit labels.}}
\end{{figure}}

\section{{Core Counts}}

\begin{{center}}
\begin{{tabular}}{{lr}}
\toprule
Quantity & Value \\
\midrule
Raw rows & {int_or_na(clean_summary["total_raw_rows"])} \\
Deduplicated rows & {int_or_na(clean_summary["total_deduped_rows"])} \\
Clean call rows & {int_or_na(clean_summary["total_clean_rows"])} \\
Daily TP2 violation rows & {int_or_na(episode_summary.get("daily_violation_rows"))} \\
Pair episodes & {int_or_na(episode_summary.get("episode_rows"))} \\
Conservative bid/ask entry episodes & {int_or_na(conservative_summary["conservative_bidask_rows"])} \\
Lifecycle rows & {int_or_na(lifecycle_summary["lifecycle_rows"])} \\
Lifecycle event-day range & {lifecycle_summary["event_day_min"]} to {lifecycle_summary["event_day_max"]} \\
Observable lifecycle rows & {int_or_na(lifecycle_summary["observable_rows"])} \\
Conservative bid/ask lifecycle violation rows & {int_or_na(lifecycle_summary["conservative_bidask_violation_rows"])} \\
\bottomrule
\end{{tabular}}
\end{{center}}

\section{{Lifecycle Definition}}

Event day 0 is the first conservative bid/ask TP2 violation date for an episode. Negative event
days are pre-violation observations of the same left-side pair; positive event days track the same
economic pair forward until T1 expiration. On each lifecycle date, the pipeline reconstructs the
right-side C/D legs using the current forward-adjusted round-up rule.

Normalized pressure is:

\[
\text{{pressure}} = \frac{{C_{{mid}}D_{{mid}} - A_{{mid}}B_{{mid}}}}{{C_{{mid}}D_{{mid}} + A_{{mid}}B_{{mid}}}}.
\]

Positive pressure means a mid-price TP2 violation. Conservative bid/ask violation is stricter:

\[
A_{{ask}}B_{{ask}} < C_{{bid}}D_{{bid}}.
\]

Wide-spread rows are kept for behavior tracking, but they are not executable trade exits unless
the selected legs pass the liquidity rule at the entry/exit point.

\section{{Event-Time Summary}}

\begin{{center}}
\resizebox{{\linewidth}}{{!}}{{%
{tex_table(checkpoint_display, list(checkpoint_display.columns), "rrrrrrrrr")}
}}
\end{{center}}

Interpretation: the violation begins at day 0 by construction, but the pre-event rows show whether
the same pair was already near the TP2 boundary. Fast decay after day 0 supports reversion-style
labels; persistent positive pressure indicates a harder trade because the dislocation does not
correct quickly.

{include_graphic("figures/lifecycle_normalized_pressure_band.png", "Median normalized pressure with 25--75 and 10--90 percentile bands. The dashed vertical line is the first violation day.", "fig:lifecycle-pressure-band")}

{include_graphic("figures/lifecycle_violation_tradability_share.png", "Share of lifecycle rows that remain mid-price violations, conservative bid/ask violations, and all-leg liquid under the 50\\% spread rule.", "fig:lifecycle-tradability")}

\section{{Status And Observability}}

\begin{{center}}
\resizebox{{0.98\linewidth}}{{!}}{{%
{tex_table(status_display, list(status_display.columns), "rrrrrr")}
}}
\end{{center}}

Observable means all reconstructed A/B/C/D legs and forward inputs are available on that lifecycle
date. Missing A/B usually appears in pre-violation lookback when one of the left-pair contracts was
not listed yet. Missing C/D means the forward-adjusted right leg could not be reconstructed on that
date. Wide spread means the rectangle is observable but at least one leg fails the 50\% spread
liquidity rule.

{include_graphic("figures/lifecycle_status_mix_by_event_day.png", "Lifecycle row status mix by event day. This separates economic non-reversion from quote/contract availability.", "fig:lifecycle-status")}

\section{{Outcome Split}}

The lifecycle path is not itself a live trading label, but it helps diagnose whether entry-time
features might predict profitable exits. The next figure splits median pressure paths by T1
product-weighted outcome. Only event days at or before 0 can become ML features; future paths are
used for labels and diagnostics only.

{include_graphic("figures/lifecycle_pressure_by_T1_outcome.png", "Median pressure path split by T1 product-weighted backtest outcome.", "fig:lifecycle-outcome")}

{include_graphic("figures/lifecycle_pressure_heatmap.png", "Distribution of normalized pressure by event day. This shows concentration near zero versus extreme dislocations.", "fig:lifecycle-heatmap")}

\section{{Data Quality Correction}}

\begin{{center}}
{tex_table(price_bound_display, list(price_bound_display.columns), "rrr")}
\end{{center}}

These are otherwise eligible clean-layer rows removed by the call-price sanity bound. The 2017
artifact that created the previous abnormal P\&L spike is removed here; the corrected liquid-exit
labels contain zero rows involving that bad contract in the strategy legs.

\section{{What Is Left Before ML}}

\begin{{enumerate}}
\item Lock the target definition: choose whether the first ML target is T1/K1 first-liquid P\&L,
positive return classification, or a three-way action label of skip/T1/K1.
\item Build only point-in-time features: signal-date surface features, bid/ask severity,
moneyness, DTE gaps, liquidity flags, prior pressure path summaries from event days \(\le 0\),
and no future lifecycle columns.
\item Add concentration controls: report results with winsorized P\&L, capped position size, and
top-episode contribution so one date cannot dominate conference results.
\item Define rolling evaluation: train on past years, validate on the next block, test on the
following block, then roll forward. No random row split.
\item Add transaction-cost and execution assumptions: keep bid/ask fills, contract multiplier,
and option fee assumptions in the labels and report both gross and net performance.
\end{{enumerate}}

\section{{Conclusion}}

The pre-ML pipeline is now conceptually ready for model construction. The major remaining work is
not more cleaning; it is label choice, leakage-proof feature construction, rolling evaluation, and
robustness reporting. The lifecycle layer provides the key bridge: it tells us how violations
behave before and after first detection while preserving a clear separation between observable
history, future labels, and executable liquidity.

\end{{document}}
"""

    tex_path.write_text(tex)

    if args.compile:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=tex_path.parent,
            check=True,
        )
        generated_pdf = tex_path.with_suffix(".pdf")
        if generated_pdf != pdf_path:
            pdf_path.write_bytes(generated_pdf.read_bytes())

    print(f"Wrote {tex_path}")
    if args.compile:
        print(f"PDF available at {pdf_path}")


if __name__ == "__main__":
    main()
