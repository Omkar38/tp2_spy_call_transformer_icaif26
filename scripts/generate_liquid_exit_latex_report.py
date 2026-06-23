#!/usr/bin/env python3
"""Generate a structured LaTeX report for the liquid-exit TP2 rebuild."""
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
        return f"{sign}\\${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{sign}\\${value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{sign}\\${value / 1e3:.1f}K"
    return f"{sign}\\${value:.0f}"


def pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{100 * float(value):.2f}\\%"


def tex_table(df: pd.DataFrame, columns: list[str], align: str | None = None) -> str:
    align = align or ("l" * len(columns))
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    lines.append(" & ".join(esc(col) for col in columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in df[columns].iterrows():
        lines.append(" & ".join(str(row[col]) for col in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


POLICY_SHORT = {
    "exit_at_reversion_day": "rev-day",
    "exit_first_liquid_on_or_after_reversion": "first-liquid",
    "exit_reversion_plus_1_session": "rev+1",
    "exit_reversion_plus_3_sessions": "rev+3",
    "exit_reversion_plus_5_sessions": "rev+5",
    "exit_selected_leg_expiration_payoff": "expiration",
}


def include_graphic(path: str, caption: str, label: str) -> str:
    return rf"""
\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.92\linewidth]{{{path}}}
  \caption{{{caption}}}
  \label{{{label}}}
\end{{figure}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX liquid-exit report.")
    parser.add_argument("--labels", default="data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet")
    parser.add_argument("--tables-dir", default="reports/raw_archive2_eda/tables/liquid_exit_rebuild")
    parser.add_argument("--tex-output", default="reports/liquid_exit_rebuild_report.tex")
    parser.add_argument("--pdf-output", default="reports/liquid_exit_rebuild_report.pdf")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    root = CONFIG.paths.root
    tables_dir = root / args.tables_dir
    tex_path = root / args.tex_output
    pdf_path = root / args.pdf_output

    labels = pd.read_parquet(
        root / args.labels,
        columns=[
            "episode_id",
            "strategy",
            "exit_policy",
            "exit_reason",
            "is_valid_backtest",
            "net_pnl",
            "realized_return",
            "actual_holding_days",
        ],
    )
    by_policy = pd.read_csv(tables_dir / "liquid_exit_by_policy_strategy.csv")
    by_year = pd.read_csv(tables_dir / "liquid_exit_by_year_policy_strategy.csv")
    reversion = pd.read_csv(tables_dir / "liquid_exit_reversion_by_year.csv")
    wait = pd.read_csv(tables_dir / "liquid_exit_wait_summary.csv")

    total_episodes = labels["episode_id"].nunique()
    total_rows = len(labels)
    valid_rows = int(labels["is_valid_backtest"].sum())
    total_valid_pnl = labels["net_pnl"].sum()

    policy_display = by_policy.copy()
    policy_display["selected"] = policy_display["selected"].map(lambda x: f"{int(x):,}")
    policy_display["valid"] = policy_display["valid"].map(lambda x: f"{int(x):,}")
    policy_display["valid_rate"] = policy_display["valid_rate"].map(pct)
    policy_display["hit_rate"] = policy_display["hit_rate"].map(pct)
    policy_display["net_pnl"] = policy_display["net_pnl"].map(money)
    policy_display["median_return"] = policy_display["median_return"].map(pct)
    policy_display["mean_holding_days"] = policy_display["mean_holding_days"].map(lambda x: f"{x:.2f}")
    policy_display["policy"] = policy_display["exit_policy"].map(POLICY_SHORT).map(esc)
    policy_display["exit_policy"] = policy_display["exit_policy"].map(esc)
    policy_display["strategy"] = policy_display["strategy"].map(esc)

    primary = by_policy[by_policy["exit_policy"].eq("exit_first_liquid_on_or_after_reversion")].copy()
    primary_display = primary.copy()
    primary_display["valid"] = primary_display["valid"].map(lambda x: f"{int(x):,}")
    primary_display["valid_rate"] = primary_display["valid_rate"].map(pct)
    primary_display["hit_rate"] = primary_display["hit_rate"].map(pct)
    primary_display["net_pnl"] = primary_display["net_pnl"].map(money)
    primary_display["median_return"] = primary_display["median_return"].map(pct)
    primary_display["mean_holding_days"] = primary_display["mean_holding_days"].map(lambda x: f"{x:.2f}")

    exp = by_policy[by_policy["exit_policy"].eq("exit_selected_leg_expiration_payoff")].copy()
    exp_display = exp.copy()
    exp_display["valid"] = exp_display["valid"].map(lambda x: f"{int(x):,}")
    exp_display["hit_rate"] = exp_display["hit_rate"].map(pct)
    exp_display["net_pnl"] = exp_display["net_pnl"].map(money)
    exp_display["median_return"] = exp_display["median_return"].map(pct)
    exp_display["mean_holding_days"] = exp_display["mean_holding_days"].map(lambda x: f"{x:.2f}")

    rev_display = reversion.copy()
    rev_display["episodes"] = rev_display["episodes"].map(lambda x: f"{int(x):,}")
    rev_display["reversion_episodes"] = rev_display["reversion_episodes"].map(lambda x: f"{int(x):,}")
    rev_display["reversion_share"] = rev_display["reversion_share"].map(pct)
    rev_display["median_reversion_event_day"] = rev_display["median_reversion_event_day"].map(lambda x: "NA" if pd.isna(x) else f"{x:.1f}")
    rev_display["p75_reversion_event_day"] = rev_display["p75_reversion_event_day"].map(lambda x: "NA" if pd.isna(x) else f"{x:.1f}")
    rev_display = rev_display.rename(
        columns={
            "year": "Year",
            "episodes": "Episodes",
            "reversion_episodes": "Reverted",
            "reversion_share": "Revert share",
            "median_reversion_event_day": "Med day",
            "p75_reversion_event_day": "P75 day",
        }
    )

    wait_display = wait.copy()
    wait_display["valid"] = wait_display["valid"].map(lambda x: f"{int(x):,}")
    wait_display["same_day_exit"] = wait_display["same_day_exit"].map(lambda x: f"{int(x):,}")
    wait_display["same_day_exit_share"] = wait_display["same_day_exit_share"].map(pct)
    wait_display["median_wait_days"] = wait_display["median_wait_days"].map(lambda x: f"{x:.1f}")
    wait_display["p90_wait_days"] = wait_display["p90_wait_days"].map(lambda x: f"{x:.1f}")

    primary_year = by_year[
        by_year["exit_policy"].eq("exit_first_liquid_on_or_after_reversion")
        & by_year["strategy"].isin(["T1", "K1"])
    ].copy()
    primary_year_raw = primary_year.pivot(index="year", columns="strategy", values="net_pnl").reset_index()
    primary_valid_raw = primary_year.pivot(index="year", columns="strategy", values="valid").reset_index()
    primary_hit_raw = primary_year.pivot(index="year", columns="strategy", values="hit_rate").reset_index()
    primary_year = primary_year_raw.copy()
    for col in ["K1", "T1"]:
        if col in primary_year:
            primary_year[col] = primary_year[col].map(money)
    primary_year["year"] = primary_year["year"].astype(str)

    year_notes = []
    for _, row in primary_year_raw.iterrows():
        year = int(row["year"])
        k1 = float(row.get("K1", 0.0))
        t1 = float(row.get("T1", 0.0))
        valid_row = primary_valid_raw[primary_valid_raw["year"].eq(year)]
        hit_row = primary_hit_raw[primary_hit_raw["year"].eq(year)]
        k1_valid = int(valid_row["K1"].iloc[0]) if not valid_row.empty and "K1" in valid_row else 0
        t1_valid = int(valid_row["T1"].iloc[0]) if not valid_row.empty and "T1" in valid_row else 0
        k1_hit = float(hit_row["K1"].iloc[0]) if not hit_row.empty and "K1" in hit_row else float("nan")
        t1_hit = float(hit_row["T1"].iloc[0]) if not hit_row.empty and "T1" in hit_row else float("nan")
        total = k1 + t1
        if year == 2020 and total < 0:
            note = "High reversion coverage but negative T1/K1 P&L, consistent with stressed COVID-period option repricing overwhelming simple reversion exits."
        elif year in {2024, 2025} and total <= 0:
            note = "Many valid episodes; 2025 turns positive while 2024 is mildly negative, so recent data still needs ML selection rather than unconditional trading."
        elif abs(total) < 1_000_000:
            note = "Small economic contribution; useful mainly for sample coverage and robustness."
        elif total > 50_000_000:
            note = "Large positive aggregate P&L; inspect concentration because unconditional product-weighted trading can be dominated by a small number of episodes."
        elif total < 0:
            note = "Negative aggregate year; unconditional trading is weak and supports ML filtering."
        else:
            note = "Positive aggregate year, but still heterogeneous across strategies."
        year_notes.append(
            {
                "Year": str(year),
                "K1 P&L": money(k1),
                "T1 P&L": money(t1),
                "Valid K1/T1": f"{k1_valid:,}/{t1_valid:,}",
                "Hit K1/T1": f"{100*k1_hit:.1f}\\%/{100*t1_hit:.1f}\\%",
                "Interpretation": esc(note),
            }
        )
    year_notes_display = pd.DataFrame(year_notes)

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
\renewcommand{{\arraystretch}}{{1.15}}

\title{{Liquid-Exit TP2 Rebuild Report\\\large SPY Call Option TP2 Research Pipeline}}
\author{{ML-Based Trading of TP2-Inspired Violations in SPY American Call Options}}
\date{{June 21, 2026}}

\begin{{document}}
\maketitle

\section{{Purpose}}

This report documents the liquid-exit rebuild of the TP2-inspired SPY call option pipeline. The
goal is to preserve the full TP2 lifecycle path while enforcing liquidity only where it matters
economically: at trade entry and trade exit. This was motivated by the observation that a TP2
violation may revert before the selected option legs are cleanly executable. The prior strict
reversion-day mini-backtest could therefore mark a trade invalid even though a realistic trader
could close it on the first later liquid date before expiration.

The rebuild is implemented as a parallel layer. It does not overwrite the earlier episode,
lifecycle, or conservative bid/ask outputs.

\section{{High-Level Result}}

\begin{{center}}
\begin{{tabular}}{{lr}}
\toprule
Quantity & Value \\
\midrule
Entry episodes & {total_episodes:,} \\
Strategy-exit-policy rows & {total_rows:,} \\
Valid strategy-exit-policy rows & {valid_rows:,} \\
Total valid net P\&L across all implemented policies & {money(total_valid_pnl)} \\
Implemented strategies & T1, T2, K1, K2 \\
Implemented exit policies & 6 \\
\bottomrule
\end{{tabular}}
\end{{center}}

The main live-style policy is \texttt{{exit\_first\_liquid\_on\_or\_after\_reversion}}. The
expiration-payoff policy is included as a Glasserman-style benchmark, not as the same execution
question as premium resale.

\section{{Pipeline From Zero}}

\begin{{figure}}[H]
\centering
\begin{{tikzpicture}}[
  node distance=0.92cm and 0.95cm,
  box/.style={{rectangle, rounded corners=2pt, draw=black!65, fill=blue!6, text width=3.2cm, align=center, minimum height=0.9cm}},
  good/.style={{rectangle, rounded corners=2pt, draw=black!65, fill=green!8, text width=3.2cm, align=center, minimum height=0.9cm}},
  warn/.style={{rectangle, rounded corners=2pt, draw=black!65, fill=orange!10, text width=3.2cm, align=center, minimum height=0.9cm}},
  arrow/.style={{-{{Latex[length=2.1mm]}}, thick, draw=black!70}}
]
\node[box] (raw) {{Professor Archive-2\\SPY option data\\2013--2026}};
\node[box, right=of raw] (quote) {{No-spread quote\\availability layer\\valid bid/ask only}};
\node[box, right=of quote] (tp2) {{Forward-adjusted\\TP2 detector\\A/B/C/D rectangles}};
\node[good, below=of tp2] (episode) {{Episode tracking\\same A/B pair counted\\as one event}};
\node[good, left=of episode] (entry) {{Conservative entry\\Aask Bask < Cbid Dbid\\and liquid entry}};
\node[warn, left=of entry] (life) {{Lifecycle tracking\\wide quotes retained\\as observable/illiquid}};
\node[good, below=of life] (rev) {{First true reversion\\mid lhs >= rhs\\before T1}};
\node[good, right=of rev] (exit) {{Exit-policy labels\\reversion day, first liquid,\\+1/+3/+5, expiration}};
\node[box, right=of exit] (ml) {{ML-ready actions\\episode + strategy\\+ exit policy}};

\draw[arrow] (raw) -- (quote);
\draw[arrow] (quote) -- (tp2);
\draw[arrow] (tp2) -- (episode);
\draw[arrow] (episode) -- (entry);
\draw[arrow] (entry) -- (life);
\draw[arrow] (life) -- (rev);
\draw[arrow] (rev) -- (exit);
\draw[arrow] (exit) -- (ml);
\end{{tikzpicture}}
\caption{{Liquid-exit rebuild architecture. Liquidity is stored throughout, but enforced only when opening or closing a trade.}}
\end{{figure}}

\subsection{{Data Layer}}

The rebuild uses the no-spread quote availability file:

\begin{{quote}}\small
\texttt{{data/processed/archive2\_spy\_call\_quote\_availability\_nospread\_2013\_2026.parquet}}
\end{{quote}}

This file keeps SPY calls with valid bid/ask, positive bid, positive mid, and the latest available
contract-day quote. It also applies a broad call-price sanity filter:

\[
\max(S-K,0)-1 \le C_{{mid}} \le 1.05S+1.
\]

This removes raw vendor artifacts where a deep in-the-money call is priced far below intrinsic
value or above the underlying itself. It is not a model fit; it is a static no-arbitrage data
quality screen. The file does not delete rows solely because the bid/ask spread exceeds 50\%.
Instead it stores \texttt{{spread\_pct}}, \texttt{{tradable\_under\_50pct\_spread}}, and
\texttt{{wide\_spread\_over\_50pct}}. This is the key structural correction: wide quotes remain
available for behavior tracking but cannot be used as executable fills.

\subsection{{TP2 Rectangle And Violation Definitions}}

For calls, each rectangle contains:

\begin{{align*}}
A &= C(K_1,T_1), & B &= C(K_2,T_2),\\
C &= C(\tilde K_1,T_2), & D &= C(\tilde K_2,T_1),
\end{{align*}}

where the right-side strikes are forward-adjusted and rounded up to available strikes. The
mid-price TP2 pressure is:

\[
\text{{pressure}} = C_{{mid}}D_{{mid}} - A_{{mid}}B_{{mid}}.
\]

Positive pressure means a TP2 violation. The conservative tradable entry condition is stricter:

\[
A_{{ask}}B_{{ask}} < C_{{bid}}D_{{bid}}.
\]

This means the violation survives the unfavorable side of the bid/ask spread.

\subsection{{Episode Tracking}}

Daily violations are grouped by the same left-side economic pair:

\begin{{quote}}\small
\texttt{{pair\_key = A\_expiration | A\_strike | B\_expiration | B\_strike}}
\end{{quote}}

The same continuing violation is therefore counted once as an episode, not repeatedly every day.
The liquid-exit rebuild starts from conservative bid/ask episode starts that were executable at
entry.

\subsection{{Lifecycle And Reversion Tracking}}

For each entry episode, the lifecycle panel tracks the same A/B pair from 20 sessions before the
first violation through T1 expiration. On each lifecycle date, the pipeline reconstructs C/D using
the current forward adjustment and round-up strike rule. Reversion is defined as the first later
date where:

\[
A_{{mid}}B_{{mid}} \ge C_{{mid}}D_{{mid}}.
\]

The important point is that reversion is measured in the no-spread quote availability universe.
If a row is observable but wide, it is still useful for identifying the TP2 path. It is not
automatically an executable exit.

\subsection{{Strategy Sizing And Execution}}

The rebuild uses product-weighted Glasserman-style quantities. For calls:

\begin{{center}}
\begin{{tabular}}{{lll}}
\toprule
Strategy & Long leg & Short leg \\
\midrule
T1 & B\_mid units of A & C\_mid units of D \\
T2 & A\_mid units of B & D\_mid units of C \\
K1 & B\_mid units of A & D\_mid units of C \\
K2 & A\_mid units of B & C\_mid units of D \\
\bottomrule
\end{{tabular}}
\end{{center}}

Execution is bid/ask based. Long positions are opened at ask and closed at bid. Short positions
are opened at bid and closed at ask. Transaction cost is \$0.65 per contract per side.

\section{{Exit Policies Implemented}}

\begin{{center}}
\begin{{tabularx}}{{0.96\linewidth}}{{lX}}
\toprule
Exit policy & Interpretation \\
\midrule
\texttt{{exit\_at\_reversion\_day}} & Strict close on the first TP2 reversion date. Selected legs must be liquid that same day. \\
\texttt{{exit\_first\_liquid\_on\_or\_after\_reversion}} & Main live-style policy. If reversion day is not executable, wait for the first later liquid selected-leg close before selected-leg expiration. \\
\texttt{{exit\_reversion\_plus\_1\_session}} & Robustness policy: exit one trading session after reversion. \\
\texttt{{exit\_reversion\_plus\_3\_sessions}} & Robustness policy: exit three trading sessions after reversion. \\
\texttt{{exit\_reversion\_plus\_5\_sessions}} & Robustness policy: exit five trading sessions after reversion. \\
\texttt{{exit\_selected\_leg\_expiration\_payoff}} & Glasserman-style benchmark using expiration payoff of selected legs. This is a benchmark, not the same premium-resale execution question. \\
\bottomrule
\end{{tabularx}}
\end{{center}}

\section{{Main Live-Style Policy Results}}

\begin{{center}}
{tex_table(primary_display, ["strategy", "valid", "valid_rate", "hit_rate", "net_pnl", "median_return", "mean_holding_days"], "lrrrrrr")}
\end{{center}}

For the first-liquid-after-reversion policy, T1 and K1 remain the main positive aggregate
strategies. The valid row count is slightly higher than strict reversion-day exit, confirming that
some episodes revert before the selected legs are immediately executable. However, the total P\&L
is close to the strict rule, so the main value of the liquid-exit correction is methodological
cleanliness and executable coverage rather than a dramatic profit change.

\subsection{{Why Median Return Is Often Negative}}

Median return is the middle trade return after sorting all valid strategy-policy rows. If the median
return is negative, more than half of the individual trades lose money. This can happen at the
same time that total net P\&L is strongly positive because product-weighted option strategies have
highly skewed payoffs: many small/medium losses can be outweighed by a smaller number of very
large winners.

That is exactly why the unconditional pipeline should not be the final trading rule. TP2 gives a
theoretically motivated candidate set, but the ML layer is needed to select which episodes, strategy
denominations, and exit policies are worth trading. In short: positive total P\&L with negative median
return means the signal has useful tails, but needs selection.

\section{{Year-Wise Interpretation For The Main Live-Style Policy}}

\small
\begin{{longtable}}{{llllp{{5.2cm}}}}
\toprule
Year & K1 P\&L & T1 P\&L & Valid K1/T1 & Interpretation \\
\midrule
\endfirsthead
\toprule
Year & K1 P\&L & T1 P\&L & Valid K1/T1 & Interpretation \\
\midrule
\endhead
"""

    for _, row in year_notes_display.iterrows():
        tex += (
            f"{row['Year']} & {row['K1 P&L']} & {row['T1 P&L']} & "
            f"{row['Valid K1/T1']} & {row['Interpretation']} \\\\\n"
        )

    tex += rf"""\bottomrule
\end{{longtable}}
\normalsize

The year-wise pattern confirms that unconditional TP2 trading is not uniformly profitable. The
main research value is the labeled action dataset: it lets ML learn when the TP2 dislocation is
tradable and which strategy/exit rule should be selected.

\section{{All Exit Policies}}

\begin{{center}}
\begin{{tabularx}}{{0.96\linewidth}}{{lX}}
\toprule
Short label & Full policy \\
\midrule
\texttt{{rev-day}} & \texttt{{exit\_at\_reversion\_day}} \\
\texttt{{first-liquid}} & \texttt{{exit\_first\_liquid\_on\_or\_after\_reversion}} \\
\texttt{{rev+1}} & \texttt{{exit\_reversion\_plus\_1\_session}} \\
\texttt{{rev+3}} & \texttt{{exit\_reversion\_plus\_3\_sessions}} \\
\texttt{{rev+5}} & \texttt{{exit\_reversion\_plus\_5\_sessions}} \\
\texttt{{expiration}} & \texttt{{exit\_selected\_leg\_expiration\_payoff}} \\
\bottomrule
\end{{tabularx}}
\end{{center}}

\begin{{center}}
\footnotesize
\begin{{tabular}}{{llrrrrr}}
\toprule
Policy & Strat. & Valid & Hit rate & Net P\&L & Med. ret. & Hold days \\
\midrule
"""

    for _, row in policy_display.iterrows():
        tex += (
            f"{row['policy']} & {row['strategy']} & {row['valid']} & "
            f"{row['hit_rate']} & {row['net_pnl']} & "
            f"{row['median_return']} & {row['mean_holding_days']} \\\\\n"
        )

    tex += rf"""\bottomrule
\end{{tabular}}
\normalsize
\end{{center}}

Every policy-strategy row starts from the same {total_episodes:,} selected entry episodes; the table reports how many rows remain executable under each exit rule.

\section{{Expiration-Payoff Benchmark}}

\begin{{center}}
{tex_table(exp_display, ["strategy", "valid", "hit_rate", "net_pnl", "median_return", "mean_holding_days"], "lrrrrr")}
\end{{center}}

The expiration payoff benchmark answers a different question from the liquid-exit premium-resale
rules. It is useful because Glasserman-style strategy evaluation often holds option legs to
expiration, while the live-style pipeline asks whether the premium trade can be closed after the
violation reverts.

\section{{Reversion And Exit Availability}}

\begin{{center}}
{tex_table(wait_display, ["strategy", "valid", "same_day_exit", "same_day_exit_share", "median_wait_days", "p90_wait_days"], "lrrrrr")}
\end{{center}}

Most first-liquid exits occur on the same calendar day as reversion. T1 and K2 still gain thousands
of additional executable rows relative to strict reversion-day exit, which is exactly why the
liquid-exit rule is worth keeping: it avoids rejecting trades only because the exact reversion day
was not executable.

\begin{{center}}
{tex_table(rev_display, ["Year", "Episodes", "Reverted", "Revert share", "Med day", "P75 day"], "rrrrrr")}
\end{{center}}

\section{{Visual Diagnostics}}

{include_graphic("figures/liquid_exit_valid_rows_by_policy_strategy.png", "Executable rows by exit policy and strategy. First-liquid-after-reversion increases valid rows over strict reversion-day exit for several strategies.", "fig:valid_rows")}

{include_graphic("figures/liquid_exit_net_pnl_by_policy_strategy.png", "Net P\\&L by exit policy and strategy. T1 and K1 dominate the live-style premium-resale policies in aggregate.", "fig:pnl_policy")}

{include_graphic("figures/liquid_exit_yearly_primary_pnl.png", "Yearly net P\\&L for T1 and K1 under strict reversion-day and first-liquid-after-reversion exits.", "fig:yearly_pnl")}

{include_graphic("figures/liquid_exit_reason_mix_by_policy.png", "Exit reason mix by policy. No-reversion and invalid/late exits are retained in accounting rather than silently removed.", "fig:reason_mix")}

{include_graphic("figures/liquid_exit_reversion_share_by_year.png", "Share of entry episodes with a mid-price reversion before T1 expiration.", "fig:reversion_share")}

{include_graphic("figures/liquid_exit_wait_days_distribution.png", "Wait-time distribution after reversion for first-liquid exit. Most valid exits happen at reversion, with a small but important tail.", "fig:wait_days")}

\section{{How This Becomes ML}}

The ML observation unit should be:

\begin{{quote}}\small
\texttt{{episode\_id + strategy + exit\_policy}}
\end{{quote}}

Features must be available at entry time only: entry severity, bid/ask severity, spreads, volume,
open interest, DTE, term gap, moneyness, delta if available, and pre-event lifecycle pressure
features from days \(\le 0\). The model must not use first reversion date, future liquidity, exit
date, realized P\&L, or end-state labels as inputs.

The target can be binary profitability, realized return, or a finite action choice over strategy and
exit policy. The best publishable framing is not ``AI detects arbitrage.'' The stronger framing is:
TP2 gives a theoretically motivated anomaly candidate, and ML learns which tradable episodes,
strategy denominations, and exit policies are economically useful.

\section{{Current Interpretation}}

\begin{{itemize}}
  \item The liquid-exit correction is conceptually right: it separates TP2 reversion from immediate
  exit liquidity.
  \item The correction increases executable rows compared with strict reversion-day exit.
  \item T1 and K1 remain the main aggregate premium-resale candidates.
  \item Fixed delays after reversion reduce aggregate P\&L, so waiting blindly is not attractive.
  \item Expiration payoff remains useful as a benchmark, but it should not be mixed with premium
  resale labels in the ML target without explicitly defining the action space.
\end{{itemize}}

\section{{Generated Files}}

\begin{{quote}}\small
\texttt{{scripts/build\_liquid\_exit\_labels.py}}\\
\texttt{{scripts/generate\_liquid\_exit\_latex\_report.py}}\\
\texttt{{data/labels/archive2\_liquid\_exit\_strategy\_labels\_2013\_2026.parquet}}\\
\texttt{{data/labels/archive2\_liquid\_exit\_strategy\_labels\_summary\_2013\_2026.json}}\\
\texttt{{reports/liquid\_exit\_rebuild\_report.pdf}}\\
\texttt{{reports/raw\_archive2\_eda/tables/liquid\_exit\_rebuild/}}\\
\texttt{{reports/figures/liquid\_exit\_*.png}}
\end{{quote}}

\end{{document}}
"""

    tex_path.write_text(tex)
    if args.compile:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            check=True,
        )
        built_pdf = tex_path.with_suffix(".pdf")
        if built_pdf != pdf_path and built_pdf.exists():
            built_pdf.replace(pdf_path)
    print(f"Wrote {tex_path}")
    if pdf_path.exists():
        print(f"PDF available at {pdf_path}")


if __name__ == "__main__":
    main()
