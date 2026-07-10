#!/usr/bin/env python3
"""Leakage-safe risk-gated selection for corrected transformer predictions.

This stage does not retrain the transformer. It consumes validation/test
prediction parquets produced by 03_train_mv_alibi_transformer.py and chooses a
selection policy using validation data only for each fold.

Why separate this from 04_selection_policy_sweep.py?
- 04 is diagnostic: it reports many policies so we can understand failure modes.
- 06 is the publishable-style candidate: it predefines a policy family, chooses
  the policy per fold using validation only, then evaluates the test year once.
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

FOLDS = [
    {"fold": "2017_2019__val2020__test2021", "val_year": 2020, "test_year": 2021},
    {"fold": "2018_2020__val2021__test2022", "val_year": 2021, "test_year": 2022},
    {"fold": "2019_2021__val2022__test2023", "val_year": 2022, "test_year": 2023},
    {"fold": "2020_2022__val2023__test2024", "val_year": 2023, "test_year": 2024},
    {"fold": "2021_2023__val2024__test2025", "val_year": 2024, "test_year": 2025},
    {"fold": "2022_2024__val2025__test2026", "val_year": 2025, "test_year": 2026},
]

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.025), 3)
MIN_TRADES = 30

# Fixed before looking at any test year in this stage.
POLICY_FAMILY = [
    {"policy": "floor05_pnl", "objective": "pnl", "floor": 0.05, "max_trades_per_day": None, "min_win_rate": 0.00},
    {"policy": "floor10_pnl", "objective": "pnl", "floor": 0.10, "max_trades_per_day": None, "min_win_rate": 0.00},
    {"policy": "floor20_pnl", "objective": "pnl", "floor": 0.20, "max_trades_per_day": None, "min_win_rate": 0.00},
    {"policy": "floor30_pnl", "objective": "pnl", "floor": 0.30, "max_trades_per_day": None, "min_win_rate": 0.00},
    {"policy": "floor20_sharpe", "objective": "sharpe", "floor": 0.20, "max_trades_per_day": None, "min_win_rate": 0.00},
    {"policy": "floor20_sortino", "objective": "sortino", "floor": 0.20, "max_trades_per_day": None, "min_win_rate": 0.00},
    {"policy": "floor20_pnl_tpd50", "objective": "pnl", "floor": 0.20, "max_trades_per_day": 50, "min_win_rate": 0.00},
    {"policy": "floor20_pnl_tpd20", "objective": "pnl", "floor": 0.20, "max_trades_per_day": 20, "min_win_rate": 0.00},
    {"policy": "floor30_pnl_tpd50", "objective": "pnl", "floor": 0.30, "max_trades_per_day": 50, "min_win_rate": 0.00},
]


def select_actions(tokens: pd.DataFrame, threshold: float, max_trades_per_day: int | None) -> pd.DataFrame:
    candidates = tokens[tokens["predicted_prob"] >= threshold].copy()
    if candidates.empty:
        return candidates

    if max_trades_per_day is not None:
        candidates = (
            candidates.sort_values("predicted_prob", ascending=False)
            .groupby("signal_date", group_keys=False)
            .head(max_trades_per_day)
        )

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


def metric_value(summary: dict, objective: str) -> float:
    if objective == "pnl":
        return summary["pnl"]
    if objective == "sharpe":
        return summary["sharpe"]
    if objective == "sortino":
        val = summary["sortino"]
        return 1e6 if np.isposinf(val) else val
    raise ValueError(f"Unknown objective: {objective}")


def evaluate_policy_grid(tokens: pd.DataFrame, policy: dict, split: str) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLD_GRID:
        if threshold < policy["floor"] - 1e-9:
            continue
        selected = select_actions(tokens, float(threshold), policy["max_trades_per_day"])
        stats = summarize(selected)
        row = {
            "split": split,
            "policy": policy["policy"],
            "objective": policy["objective"],
            "floor": policy["floor"],
            "max_trades_per_day": policy["max_trades_per_day"] if policy["max_trades_per_day"] else "none",
            "min_win_rate": policy["min_win_rate"],
            "threshold": float(threshold),
            **stats,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def choose_validation_candidate(val_grid: pd.DataFrame) -> pd.Series:
    candidates = val_grid[
        (val_grid["n"] >= MIN_TRADES)
        & (val_grid["pnl"] > 0)
        & (val_grid["win_rate"].fillna(0.0) >= val_grid["min_win_rate"])
    ].copy()
    if candidates.empty:
        # Abstain in test if validation offers no positive, sufficiently active candidate.
        return pd.Series({
            "policy": "abstain",
            "objective": "none",
            "floor": np.nan,
            "max_trades_per_day": "none",
            "threshold": np.inf,
            "selection_score": -np.inf,
            "selection_reason": "no_positive_validation_candidate",
        })

    scores = []
    for _, row in candidates.iterrows():
        # Validation-only robust score:
        # 1. Primary: risk-adjusted score if objective is risk metric, otherwise P&L.
        # 2. Penalize drawdown magnitude mildly so validation blow-ups are less attractive.
        # 3. Avoid choosing tiny trade counts when alternatives are materially active.
        score = metric_value(row, row["objective"])
        dd_penalty = 0.0001 * abs(row["max_drawdown"]) if pd.notna(row["max_drawdown"]) else 0.0
        activity_bonus = 0.001 * min(row["n"], 5000)
        scores.append(score - dd_penalty + activity_bonus)
    candidates["selection_score"] = scores

    return candidates.sort_values(
        ["selection_score", "pnl", "n"],
        ascending=[False, False, False],
    ).iloc[0]


def save_plots(yearly: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    years = yearly["test_year"].astype(str)
    ax.bar(years, yearly["test_pnl"] / 1_000_000, color="#496f5d")
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_title("Risk-gated transformer: test P&L by year")
    ax.set_ylabel("Normalized P&L ($M)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "risk_gated_transformer_yearly_pnl.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.bar(years, yearly["test_n"], color="#6b705c")
    ax.set_title("Risk-gated transformer: selected trades by year")
    ax.set_ylabel("Selected trades")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "risk_gated_transformer_trade_counts.pdf")
    plt.close(fig)


def write_report(yearly: pd.DataFrame, summary: dict) -> None:
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

    rows = []
    for _, r in yearly.iterrows():
        rows.append(
            f"{int(r['test_year'])} & {r['chosen_policy']} & {f2(r['chosen_threshold'])} & "
            f"{int(r['test_n'])} & {money(r['test_pnl'])} & {f2(r['test_sharpe'])} & "
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
\title{{Risk-Gated Transformer Selection Report}}
\author{{SPY American Call TP2 Project}}
\date{{\today}}
\begin{{document}}
\maketitle

\section{{Purpose}}
This report evaluates a leakage-safe risk-gated selection layer on top of the corrected-history TP2-MV-ALiBi transformer. The transformer predictions are fixed. For each rolling fold, the policy and threshold are chosen using the validation year only. The test year is evaluated once.

\section{{Selection Rule}}
The candidate family includes probability floors, risk-aware objectives, and trade-count caps. A candidate is eligible only if validation P\&L is positive, validation trade count is at least {MIN_TRADES}, and its win-rate constraint is satisfied. The chosen candidate maximizes a validation-only robust score. No test-year P\&L is used to choose the policy or threshold.

\section{{Aggregate Result}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lr}}
\toprule
Metric & Value \\
\midrule
Total selected trades & {int(summary['total_selected']):,} \\
Total normalized P\&L & {money(summary['total_pnl'])} \\
Positive years & {summary['positive_years']} / {summary['total_years']} \\
Combined Sharpe & {f2(summary['combined_sharpe'])} \\
Combined Sortino & {f2(summary['combined_sortino'])} \\
Combined max drawdown & {money(summary['combined_max_drawdown'])} \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Yearly Results}}
\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{llrrrrrr}}
\toprule
Year & Policy & Threshold & Trades & P\&L & Sharpe & Sortino & Max DD \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\linewidth]{{figures/risk_gated_transformer_yearly_pnl.pdf}}
\caption{{Risk-gated transformer normalized P\&L by test year.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\linewidth]{{figures/risk_gated_transformer_trade_counts.pdf}}
\caption{{Risk-gated transformer selected trades by test year.}}
\end{{figure}}

\section{{Interpretation}}
This stage directly addresses the raw transformer's over-selection problem. If it improves worst-year behavior, the contribution is not merely the architecture; it is the combination of TP2-aware representation and validation-only risk-gated trade selection. If it remains weak, the next step should be feature/regime improvement rather than increasing transformer complexity.

\end{{document}}
"""
    (OUT / "risk_gated_transformer_report.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    if not TEST_PATH.exists() or not VAL_PATH.exists():
        raise FileNotFoundError("Run 03_train_mv_alibi_transformer.py first; prediction parquets are missing.")

    val = pd.read_parquet(VAL_PATH)
    test = pd.read_parquet(TEST_PATH)
    for df in [val, test]:
        df["signal_date"] = pd.to_datetime(df["signal_date"])
        df["signal_year"] = df["signal_date"].dt.year

    all_val_grid = []
    yearly_rows = []
    selected_rows = []

    for fold in FOLDS:
        val_fold = val[val["signal_year"].eq(fold["val_year"])].copy()
        test_fold = test[test["signal_year"].eq(fold["test_year"])].copy()
        if val_fold.empty or test_fold.empty:
            continue

        val_grids = [evaluate_policy_grid(val_fold, p, "validation") for p in POLICY_FAMILY]
        val_grid = pd.concat(val_grids, ignore_index=True)
        val_grid["fold"] = fold["fold"]
        val_grid["val_year"] = fold["val_year"]
        val_grid["test_year"] = fold["test_year"]
        all_val_grid.append(val_grid)

        chosen = choose_validation_candidate(val_grid)
        if chosen["policy"] == "abstain":
            selected = pd.DataFrame(columns=test_fold.columns)
        else:
            max_tpd = chosen["max_trades_per_day"]
            max_tpd = None if max_tpd == "none" or pd.isna(max_tpd) else int(max_tpd)
            selected = select_actions(test_fold, float(chosen["threshold"]), max_tpd)
            selected = selected.copy()
            selected["chosen_policy"] = chosen["policy"]
            selected["chosen_threshold"] = float(chosen["threshold"])
            selected["fold"] = fold["fold"]
            selected_rows.append(selected)

        val_stats = summarize(
            select_actions(
                val_fold,
                float(chosen["threshold"]) if chosen["policy"] != "abstain" else np.inf,
                None if chosen.get("max_trades_per_day", "none") == "none" else int(chosen["max_trades_per_day"]),
            )
        )
        test_stats = summarize(selected)
        yearly_rows.append({
            "fold": fold["fold"],
            "val_year": fold["val_year"],
            "test_year": fold["test_year"],
            "chosen_policy": chosen["policy"],
            "chosen_objective": chosen["objective"],
            "chosen_floor": chosen["floor"],
            "chosen_threshold": chosen["threshold"],
            "chosen_max_trades_per_day": chosen["max_trades_per_day"],
            "selection_score": chosen.get("selection_score", np.nan),
            **{f"val_{k}": v for k, v in val_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
        })

    val_grid_all = pd.concat(all_val_grid, ignore_index=True) if all_val_grid else pd.DataFrame()
    yearly = pd.DataFrame(yearly_rows)
    selected_all = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()

    summary = {
        "method": "risk_gated_transformer_validation_only_policy_selection",
        "policy_family": POLICY_FAMILY,
        "total_selected": int(len(selected_all)),
        "total_pnl": float(selected_all["target_norm_pnl"].sum()) if not selected_all.empty else 0.0,
        "positive_years": int((yearly["test_pnl"] > 0).sum()) if not yearly.empty else 0,
        "total_years": int(len(yearly)),
        "combined_sharpe": sharpe(selected_all),
        "combined_sortino": sortino(selected_all),
        "combined_max_drawdown": max_drawdown(selected_all),
        "notes": [
            "Policy and threshold are selected using validation-year predictions only.",
            "Test-year results are evaluated once after validation selection.",
            "This stage is intended as the official risk-gated transformer selection candidate.",
        ],
    }

    val_grid_all.to_csv(TABLE_DIR / "risk_gated_validation_policy_grid.csv", index=False)
    yearly.to_csv(TABLE_DIR / "risk_gated_transformer_yearly.csv", index=False)
    if not selected_all.empty:
        selected_all.to_parquet(DATA_DIR / "risk_gated_transformer_selected_trades.parquet", index=False)
        selected_all.groupby("signal_date", as_index=False)["target_norm_pnl"].sum().to_csv(
            TABLE_DIR / "risk_gated_transformer_daily_pnl.csv", index=False
        )
    with open(TABLE_DIR / "risk_gated_transformer_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    save_plots(yearly)
    write_report(yearly, summary)

    print("\nRisk-gated transformer yearly results:")
    print(yearly[[
        "test_year", "chosen_policy", "chosen_threshold", "test_n", "test_pnl",
        "test_win_rate", "test_sharpe", "test_sortino", "test_max_drawdown",
    ]].to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {TABLE_DIR / 'risk_gated_transformer_yearly.csv'}")
    print(f"Wrote {TABLE_DIR / 'risk_gated_transformer_summary.json'}")
    print(f"Wrote {OUT / 'risk_gated_transformer_report.tex'}")


if __name__ == "__main__":
    main()
