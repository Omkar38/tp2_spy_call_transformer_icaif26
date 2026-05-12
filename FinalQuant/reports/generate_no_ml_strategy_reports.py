#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "labels" / "strategy_training_rows.parquet"
TARGET_GROSS_EXPOSURE = 1000.0

STRATEGIES = {
    "full4": {
        "folder": "full4_all_violations",
        "title": "Full4 All 2024 Violations",
        "label": "full4",
    },
    "T2": {
        "folder": "t2_all_violations",
        "title": "T2 All 2024 Violations",
        "label": "T2",
    },
    "K1": {
        "folder": "k1_all_violations",
        "title": "K1 All 2024 Violations",
        "label": "K1",
    },
    "K2": {
        "folder": "k2_all_violations",
        "title": "K2 All 2024 Violations",
        "label": "K2",
    },
}


def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}\\${abs(value):,.0f}"


def pct(value: float) -> str:
    return f"{value * 100:.2f}\\%"


def summarize(trades: pd.DataFrame, strategy: str, total_candidates: int) -> dict:
    invalid = total_candidates - len(trades)
    if trades.empty:
        return {
            "strategy": strategy,
            "policy": "all_valid_violations_no_ml",
            "target_gross_exposure": TARGET_GROSS_EXPOSURE,
            "candidate_2024_violations": int(total_candidates),
            "valid_traded_2024_violations": 0,
            "invalid_skipped_2024_violations": int(invalid),
            "trades": 0,
            "net_pnl": 0.0,
            "avg_return": np.nan,
            "median_return": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown": np.nan,
            "daily_sharpe_like": np.nan,
            "avg_actual_holding_days": np.nan,
        }

    daily = trades.groupby("date")["net_pnl"].sum().sort_index()
    equity = daily.cumsum()
    drawdown = equity - equity.cummax()
    gross_profit = trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum()
    gross_loss = -trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum()
    daily_sharpe_like = np.nan
    if len(daily) > 1 and daily.std(ddof=1) > 0:
        daily_sharpe_like = daily.mean() / daily.std(ddof=1) * np.sqrt(252)

    return {
        "strategy": strategy,
        "policy": "all_valid_violations_no_ml",
        "target_gross_exposure": TARGET_GROSS_EXPOSURE,
        "candidate_2024_violations": int(total_candidates),
        "valid_traded_2024_violations": int(len(trades)),
        "invalid_skipped_2024_violations": int(invalid),
        "trades": int(len(trades)),
        "net_pnl": float(trades["net_pnl"].sum()),
        "avg_return": float(trades["realized_return"].mean()),
        "median_return": float(trades["realized_return"].median()),
        "win_rate": float((trades["net_pnl"] > 0).mean()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.inf,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
        "daily_sharpe_like": float(daily_sharpe_like),
        "avg_actual_holding_days": float(trades["actual_holding_days"].mean()),
        "exit_reasons": trades["exit_reason"].value_counts(dropna=False).to_dict(),
        "by_option_type": {
            key: {
                "trades": int(len(group)),
                "net_pnl": float(group["net_pnl"].sum()),
                "avg_return": float(group["realized_return"].mean()),
                "win_rate": float((group["net_pnl"] > 0).mean()),
            }
            for key, group in trades.groupby("option_type")
        },
    }


def monthly(trades: pd.DataFrame) -> pd.DataFrame:
    out = (
        trades.groupby(trades["date"].dt.to_period("M"))
        .agg(
            trades=("net_pnl", "size"),
            net_pnl=("net_pnl", "sum"),
            avg_return=("realized_return", "mean"),
            win_rate=("net_pnl", lambda s: (s > 0).mean()),
        )
        .reset_index()
    )
    out["month"] = out["date"].astype(str)
    out["cum_pnl"] = out["net_pnl"].cumsum()
    return out


def bar(value: float, max_abs: float, color_pos: str = "green!50!black") -> str:
    width = 0.2 if max_abs <= 0 else max(0.2, min(10.0, abs(value) / max_abs * 10.0))
    color = color_pos if value >= 0 else "red"
    return f"\\textcolor{{{color}}}{{\\rule{{{width:.1f}cm}}{{7pt}}}}"


def write_readme(out_dir: Path, display: str, summary: dict) -> None:
    readme = f"""# {display} All 2024 Violations, No ML

This folder is separate from the main pipeline outputs.

Policy:

- Test period: 2024 only
- Strategy: `{summary["strategy"]}`
- Candidate rule: include every 2024 `{summary["strategy"]}` violation row
- Trade rule: trade every valid/labeled row; invalid rows are reported as skipped
- Gross exposure: fixed at `$1,000` per trade
- Input: `data/labels/strategy_training_rows.parquet`

| Metric | Value |
|---|---:|
| 2024 candidates | {summary["candidate_2024_violations"]:,} |
| Valid/traded rows | {summary["valid_traded_2024_violations"]:,} |
| Invalid/skipped rows | {summary["invalid_skipped_2024_violations"]:,} |
| Trades | {summary["trades"]:,} |
| Net PnL | ${summary["net_pnl"]:,.2f} |
| Average return | {summary["avg_return"] * 100:.2f}% |
| Median return | {summary["median_return"] * 100:.2f}% |
| Win rate | {summary["win_rate"] * 100:.2f}% |
| Profit factor | {summary["profit_factor"]:.4f} |
| Max drawdown | ${summary["max_drawdown"]:,.2f} |
| Daily Sharpe-like ratio | {summary["daily_sharpe_like"]:.4f} |
| Average actual holding days | {summary["avg_actual_holding_days"]:.2f} |
"""
    (out_dir / "README.md").write_text(readme)


def write_tex(out_dir: Path, display: str, summary: dict, month_df: pd.DataFrame) -> None:
    max_month = month_df["net_pnl"].abs().max()
    max_cum = month_df["cum_pnl"].abs().max()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    table_rows = []
    monthly_bar_rows = []
    cumulative_bar_rows = []
    for idx, row in month_df.reset_index(drop=True).iterrows():
        name = month_names[idx]
        table_rows.append(
            f"{name} & {int(row['trades']):,} & {money(row['net_pnl'])} & {money(row['cum_pnl'])} & {pct(row['avg_return'])} \\\\"
        )
        monthly_bar_rows.append(f"{name} & {money(row['net_pnl'])} & {bar(row['net_pnl'], max_month)}\\\\")
        cumulative_bar_rows.append(f"{name} & {money(row['cum_pnl'])} & {bar(row['cum_pnl'], max_cum)}\\\\")

    tex = rf"""\documentclass[11pt]{{article}}

\usepackage[margin=0.85in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}

\title{{{display}: Monthly Fixed-Gross Backtest}}
\author{{FinalQuant}}
\date{{May 2026}}

\begin{{document}}
\maketitle

\section{{Setup}}

\begin{{itemize}}
  \item Test year: 2024 only
  \item Strategy: \texttt{{{summary["strategy"]}}}
  \item Trade rule: trade every valid/labeled 2024 row for this strategy
  \item ML selection: none
  \item Gross exposure: \$1,000 per accepted trade
\end{{itemize}}

\section{{Overall Result}}

\begin{{center}}
\begin{{tabular}}{{@{{}}lr@{{}}}}
\toprule
Metric & Value \\
\midrule
2024 candidates & {summary["candidate_2024_violations"]:,} \\
Valid/traded rows & {summary["valid_traded_2024_violations"]:,} \\
Invalid/skipped rows & {summary["invalid_skipped_2024_violations"]:,} \\
Net PnL & {money(summary["net_pnl"])} \\
Average return & {pct(summary["avg_return"])} \\
Median return & {pct(summary["median_return"])} \\
Win rate & {pct(summary["win_rate"])} \\
Profit factor & {summary["profit_factor"]:.4f} \\
Max drawdown & {money(summary["max_drawdown"])} \\
Daily Sharpe-like ratio & {summary["daily_sharpe_like"]:.4f} \\
\bottomrule
\end{{tabular}}
\end{{center}}

\section{{Month-Wise Result}}

\begin{{center}}
\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\toprule
Month & Trades & Monthly PnL & Cumulative PnL & Avg Return \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}

\begin{{figure}}[h]
\centering
\begin{{tabular}}{{@{{}}lrl@{{}}}}
\toprule
Month & Monthly PnL & Relative scale \\
\midrule
{chr(10).join(monthly_bar_rows)}
\bottomrule
\end{{tabular}}
\caption{{Month-wise fixed-gross PnL for {display}.}}
\end{{figure}}

\begin{{figure}}[h]
\centering
\begin{{tabular}}{{@{{}}lrl@{{}}}}
\toprule
Month & Cumulative PnL & Relative scale \\
\midrule
{chr(10).join(cumulative_bar_rows)}
\bottomrule
\end{{tabular}}
\caption{{Month-end cumulative PnL path for {display}.}}
\end{{figure}}

\section{{Comparison Note}}

This is a no-ML all-violation comparison. It should be compared against the ML-selected fixed-gross benchmark and the other no-ML strategy reports using the same 2024 test period and \$1,000 gross exposure per accepted trade.

\end{{document}}
"""
    (out_dir / "monthly_summary.tex").write_text(tex)


def build_strategy(strategy: str, meta: dict) -> None:
    out_dir = ROOT / "reports" / meta["folder"]
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "date",
        "violation_id",
        "strategy",
        "option_type",
        "is_valid_label",
        "net_pnl",
        "entry_gross_exposure",
        "realized_return",
        "actual_holding_days",
        "exit_reason",
    ]
    candidates = pd.read_parquet(
        INPUT,
        columns=columns,
        filters=[
            ("strategy", "=", strategy),
            ("date", ">=", pd.Timestamp("2024-01-01")),
            ("date", "<=", pd.Timestamp("2024-12-31")),
        ],
    )
    trades = candidates[candidates["is_valid_label"]].copy()
    trades["unscaled_net_pnl"] = trades["net_pnl"].astype(float)
    trades["unscaled_entry_gross_exposure"] = trades["entry_gross_exposure"].astype(float)
    trades["scale_factor"] = TARGET_GROSS_EXPOSURE / trades["unscaled_entry_gross_exposure"]
    trades["entry_gross_exposure"] = TARGET_GROSS_EXPOSURE
    trades["net_pnl"] = trades["unscaled_net_pnl"] * trades["scale_factor"]
    trades["realized_return"] = trades["net_pnl"] / trades["entry_gross_exposure"]

    trades_out = out_dir / f"{strategy.lower()}_all_2024_valid_trades.csv"
    summary_out = out_dir / f"{strategy.lower()}_all_2024_summary.json"
    trades.to_csv(trades_out, index=False)
    summary = summarize(trades, strategy, len(candidates))
    summary_out.write_text(json.dumps(summary, indent=2))
    month_df = monthly(trades)
    month_df.to_csv(out_dir / f"{strategy.lower()}_monthly_2024.csv", index=False)
    write_readme(out_dir, meta["title"], summary)
    write_tex(out_dir, meta["title"], summary, month_df)
    print(json.dumps(summary, indent=2))


def main() -> None:
    for strategy, meta in STRATEGIES.items():
        build_strategy(strategy, meta)


if __name__ == "__main__":
    main()
