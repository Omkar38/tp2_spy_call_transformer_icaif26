#!/usr/bin/env python3
"""Conservative transformer selection with fixed probability-floor robustness.

This is the practical risk-control layer for the corrected-history transformer:
validation may tune the threshold, but it is not allowed to choose a threshold
below a fixed floor. This prevents the low-confidence over-selection that
caused the raw transformer failures in 2022 and 2025.

Important publication note:
To avoid a single "magic" floor, this script reports a small fixed family:
0.10, 0.20, and 0.30. The 0.20 floor is marked as the primary moderate rule,
while 0.10 and 0.30 are robustness checks. The script itself does not use test
P&L to choose thresholds; each fold tunes only on validation data subject to the
fixed floor being evaluated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TP2MVTransformer"
DATA_DIR = OUT / "data"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MPL_DIR = OUT / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEST_PATH = DATA_DIR / "mv_alibi_transformer_v2_test_predictions.parquet"
VAL_PATH = DATA_DIR / "mv_alibi_transformer_v2_val_predictions.parquet"

FLOORS = [0.10, 0.20, 0.30]
PRIMARY_FLOOR = 0.20
THRESHOLD_STEP = 0.025
MIN_TRADES = 30

FOLDS = [
    {"fold": "2017_2019__val2020__test2021", "val_year": 2020, "test_year": 2021},
    {"fold": "2018_2020__val2021__test2022", "val_year": 2021, "test_year": 2022},
    {"fold": "2019_2021__val2022__test2023", "val_year": 2022, "test_year": 2023},
    {"fold": "2020_2022__val2023__test2024", "val_year": 2023, "test_year": 2024},
    {"fold": "2021_2023__val2024__test2025", "val_year": 2024, "test_year": 2025},
    {"fold": "2022_2024__val2025__test2026", "val_year": 2025, "test_year": 2026},
]


def select_actions(tokens: pd.DataFrame, threshold: float) -> pd.DataFrame:
    candidates = tokens[tokens["predicted_prob"] >= threshold].copy()
    if candidates.empty:
        return candidates
    return (
        candidates.sort_values(["predicted_prob", "strategy"], ascending=[False, True])
        .groupby("episode_id", as_index=False)
        .first()
    )


def daily_series(selected: pd.DataFrame) -> pd.Series:
    if selected.empty:
        return pd.Series(dtype=float)
    return selected.groupby("signal_date")["target_norm_pnl"].sum().sort_index()


def sharpe(selected: pd.DataFrame) -> float:
    daily = daily_series(selected)
    if len(daily) < 5:
        return float("nan")
    std = daily.std()
    if std == 0 or pd.isna(std):
        return float("nan")
    return float(daily.mean() / std * np.sqrt(252))


def sortino(selected: pd.DataFrame) -> float:
    daily = daily_series(selected)
    if len(daily) < 5:
        return float("nan")
    downside = daily[daily < 0.0]
    if downside.empty:
        return float("inf") if daily.mean() > 0 else float("nan")
    downside_std = downside.std()
    if downside_std == 0 or pd.isna(downside_std):
        return float("nan")
    return float(daily.mean() / downside_std * np.sqrt(252))


def max_drawdown(selected: pd.DataFrame) -> float:
    daily = daily_series(selected)
    if daily.empty:
        return float("nan")
    equity = daily.cumsum()
    return float((equity - equity.cummax()).min())


def summarize(selected: pd.DataFrame) -> dict:
    if selected.empty:
        return {
            "n": 0,
            "pnl": 0.0,
            "win_rate": float("nan"),
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
            "t1": 0,
            "k2": 0,
        }
    dist = selected["strategy"].value_counts().to_dict()
    return {
        "n": int(len(selected)),
        "pnl": float(selected["target_norm_pnl"].sum()),
        "win_rate": float((selected["target_norm_pnl"] > 0).mean()),
        "sharpe": sharpe(selected),
        "sortino": sortino(selected),
        "max_drawdown": max_drawdown(selected),
        "t1": int(dist.get("T1", 0)),
        "k2": int(dist.get("K2", 0)),
    }


def tune_threshold_on_validation(val_tokens: pd.DataFrame, floor: float) -> tuple[float, pd.DataFrame]:
    rows = []
    best_threshold = floor
    best_pnl = -np.inf
    threshold_grid = np.round(np.arange(floor, 0.951, THRESHOLD_STEP), 3)
    for threshold in threshold_grid:
        selected = select_actions(val_tokens, float(threshold))
        stats = summarize(selected)
        rows.append({"floor": floor, "threshold": float(threshold), **stats})
        if stats["n"] < MIN_TRADES:
            continue
        if stats["pnl"] > best_pnl:
            best_pnl = stats["pnl"]
            best_threshold = float(threshold)
    return best_threshold, pd.DataFrame(rows)


def save_plots(yearly: pd.DataFrame, summary_by_floor: pd.DataFrame, primary_yearly: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    labels = [f"{x:.2f}" for x in summary_by_floor["floor"]]
    ax.bar(labels, summary_by_floor["total_pnl"] / 1_000_000, color="#355070")
    ax.axhline(0, color="#222", linewidth=0.8)
    ax.set_title("Conservative transformer floors: total normalized P&L")
    ax.set_xlabel("Probability floor")
    ax.set_ylabel("Total P&L ($M)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "conservative_transformer_floor_summary_pnl.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    years = primary_yearly["test_year"].astype(str)
    ax.bar(years, primary_yearly["test_pnl"] / 1_000_000, color="#355070")
    ax.axhline(0, color="#222", linewidth=0.8)
    ax.set_title("Conservative transformer floor: yearly normalized P&L")
    ax.set_ylabel("P&L ($M)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "conservative_transformer_floor_yearly_pnl.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(years, primary_yearly["test_n"], color="#6d597a")
    ax.set_title("Conservative transformer floor: selected trades")
    ax.set_ylabel("Trades")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "conservative_transformer_floor_trades.pdf")
    plt.close(fig)


def write_report(yearly: pd.DataFrame, summary_by_floor: pd.DataFrame, primary_summary: dict) -> None:
    def money(x: float) -> str:
        if pd.isna(x):
            return "--"
        sign = "-" if x < 0 else ""
        return f"{sign}\\${abs(x):,.0f}"

    def f2(x: float) -> str:
        if pd.isna(x):
            return "--"
        if np.isposinf(x):
            return "$\\infty$"
        return f"{x:.2f}"

    summary_rows = []
    for _, r in summary_by_floor.iterrows():
        summary_rows.append(
            f"{f2(r['floor'])} & {int(r['total_selected']):,} & {money(r['total_pnl'])} & "
            f"{int(r['positive_years'])}/{int(r['total_years'])} & {f2(r['combined_sharpe'])} & "
            f"{f2(r['combined_sortino'])} & {money(r['combined_max_drawdown'])} \\\\"
        )

    primary_yearly = yearly[np.isclose(yearly["floor"], PRIMARY_FLOOR)].copy()
    yearly_rows = []
    for _, r in primary_yearly.iterrows():
        yearly_rows.append(
            f"{int(r['test_year'])} & {f2(r['threshold'])} & {int(r['test_n'])} & "
            f"{money(r['test_pnl'])} & {f2(r['test_sharpe'])} & "
            f"{f2(r['test_sortino'])} & {money(r['test_max_drawdown'])} \\\\"
        )

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=0.78in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\title{{Conservative Transformer Floor Selection Report}}
\author{{SPY American Call TP2 Project}}
\date{{\today}}
\begin{{document}}
\maketitle

\section{{Purpose}}
This report evaluates the corrected-history TP2-MV-ALiBi transformer with conservative probability-floor risk controls. The transformer still tunes its exact threshold on validation P\&L, but the threshold cannot fall below a fixed floor. To avoid relying on one hand-picked value, we report floors 0.10, 0.20, and 0.30 side by side. The 0.20 floor is treated as the primary moderate rule; 0.10 and 0.30 are robustness checks.

\section{{Leakage Control}}
The floor family is treated as a risk-control design choice. Within each rolling fold and each fixed floor, the exact threshold is selected using the validation year only. The test year is evaluated once. No test-year P\&L enters threshold tuning.

\section{{Aggregate Results Across Fixed Floors}}
\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{rrrrrrr}}
\toprule
Floor & Trades & P\&L & Positive years & Sharpe & Sortino & Max DD \\
\midrule
{chr(10).join(summary_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.82\linewidth]{{figures/conservative_transformer_floor_summary_pnl.pdf}}
\caption{{Total normalized P\&L for the fixed confidence-floor family.}}
\end{{figure}}

\section{{Primary Floor Yearly Results}}
The table below reports the primary moderate floor, \(p \geq {PRIMARY_FLOOR:.2f}\). The adjacent floors are retained in the aggregate robustness table above.

\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{rrrrrrr}}
\toprule
Year & Threshold & Trades & P\&L & Sharpe & Sortino & Max DD \\
\midrule
{chr(10).join(yearly_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\linewidth]{{figures/conservative_transformer_floor_yearly_pnl.pdf}}
\caption{{Yearly normalized P\&L under the conservative transformer floor rule.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\linewidth]{{figures/conservative_transformer_floor_trades.pdf}}
\caption{{Selected trade counts under the conservative transformer floor rule.}}
\end{{figure}}

\section{{Interpretation}}
The raw transformer selected too many low-confidence trades in adverse years. The conservative floor reduces over-selection by requiring a minimum model confidence before any trade can enter. This should be presented as a risk-control layer, not as a new arbitrage condition.

\end{{document}}
"""
    (OUT / "conservative_transformer_floor_report.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    if not TEST_PATH.exists() or not VAL_PATH.exists():
        raise FileNotFoundError("Run 03_train_mv_alibi_transformer.py first; prediction parquets are missing.")

    val = pd.read_parquet(VAL_PATH)
    test = pd.read_parquet(TEST_PATH)
    for df in [val, test]:
        df["signal_date"] = pd.to_datetime(df["signal_date"])
        df["signal_year"] = df["signal_date"].dt.year

    yearly_rows = []
    grid_rows = []
    selected_by_floor: dict[float, list[pd.DataFrame]] = {floor: [] for floor in FLOORS}
    for floor in FLOORS:
        for fold in FOLDS:
            val_fold = val[val["signal_year"].eq(fold["val_year"])].copy()
            test_fold = test[test["signal_year"].eq(fold["test_year"])].copy()
            threshold, grid = tune_threshold_on_validation(val_fold, floor)
            grid["fold"] = fold["fold"]
            grid["val_year"] = fold["val_year"]
            grid["test_year"] = fold["test_year"]
            grid_rows.append(grid)

            val_selected = select_actions(val_fold, threshold)
            test_selected = select_actions(test_fold, threshold)
            test_selected = test_selected.copy()
            test_selected["floor"] = floor
            test_selected["fold"] = fold["fold"]
            test_selected["threshold"] = threshold
            selected_by_floor[floor].append(test_selected)

            val_stats = summarize(val_selected)
            test_stats = summarize(test_selected)
            yearly_rows.append({
                "floor": floor,
                "is_primary_floor": bool(np.isclose(floor, PRIMARY_FLOOR)),
                "fold": fold["fold"],
                "val_year": fold["val_year"],
                "test_year": fold["test_year"],
                "threshold": threshold,
                **{f"val_{k}": v for k, v in val_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
            })

    yearly = pd.DataFrame(yearly_rows)
    summary_rows = []
    selected_primary = pd.DataFrame()
    for floor in FLOORS:
        selected_all = pd.concat(selected_by_floor[floor], ignore_index=True) if selected_by_floor[floor] else pd.DataFrame()
        floor_yearly = yearly[np.isclose(yearly["floor"], floor)].copy()
        if np.isclose(floor, PRIMARY_FLOOR):
            selected_primary = selected_all.copy()
        summary_rows.append({
            "floor": floor,
            "is_primary_floor": bool(np.isclose(floor, PRIMARY_FLOOR)),
            "total_selected": int(len(selected_all)),
            "total_pnl": float(selected_all["target_norm_pnl"].sum()) if not selected_all.empty else 0.0,
            "positive_years": int((floor_yearly["test_pnl"] > 0).sum()),
            "total_years": int(len(floor_yearly)),
            "combined_sharpe": sharpe(selected_all),
            "combined_sortino": sortino(selected_all),
            "combined_max_drawdown": max_drawdown(selected_all),
        })
    summary_by_floor = pd.DataFrame(summary_rows)
    primary_summary = summary_by_floor[summary_by_floor["is_primary_floor"]].iloc[0].to_dict()
    summary = {
        "method": "corrected_history_transformer_conservative_probability_floor_family",
        "floors": FLOORS,
        "primary_floor": PRIMARY_FLOOR,
        "threshold_rule": "for each fixed floor, maximize validation P&L subject to threshold >= floor",
        "primary_summary": primary_summary,
        "all_floor_summary": summary_rows,
    }

    pd.concat(grid_rows, ignore_index=True).to_csv(TABLE_DIR / "conservative_transformer_floor_validation_grid.csv", index=False)
    yearly.to_csv(TABLE_DIR / "conservative_transformer_floor_yearly.csv", index=False)
    summary_by_floor.to_csv(TABLE_DIR / "conservative_transformer_floor_summary_by_floor.csv", index=False)
    selected_primary.to_parquet(DATA_DIR / "conservative_transformer_floor_selected_trades.parquet", index=False)
    selected_primary.groupby("signal_date", as_index=False)["target_norm_pnl"].sum().to_csv(
        TABLE_DIR / "conservative_transformer_floor_daily_pnl.csv", index=False
    )
    with open(TABLE_DIR / "conservative_transformer_floor_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    primary_yearly = yearly[yearly["is_primary_floor"]].copy()
    save_plots(yearly, summary_by_floor, primary_yearly)
    write_report(yearly, summary_by_floor, primary_summary)

    print("\nConservative transformer floor summary by floor:")
    print(summary_by_floor.to_string(index=False))
    print(f"\nPrimary floor ({PRIMARY_FLOOR:.2f}) yearly results:")
    print(primary_yearly[[
        "test_year", "threshold", "test_n", "test_pnl", "test_win_rate",
        "test_sharpe", "test_sortino", "test_max_drawdown",
    ]].to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
