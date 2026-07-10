#!/usr/bin/env python3
"""Generate a report comparing first-liquid and expiry TP2 exits."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports"
TABLE_DIR = REPORT / "raw_archive2_eda" / "tables" / "exit_policy_comparison"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2017, 2027))
TEST_YEARS = list(range(2021, 2027))


def money(x: float) -> str:
    if pd.isna(x):
        return "--"
    sign = "-" if x < 0 else ""
    return f"{sign}\\${abs(x):,.0f}"


def pct(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{100*x:.1f}\\%"


def num(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:,.0f}"


def f2(x: float) -> str:
    if pd.isna(x):
        return "--"
    if np.isposinf(x):
        return "$\\infty$"
    if np.isneginf(x):
        return "$-\\infty$"
    return f"{x:.2f}"


def esc(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("$", "\\$")
        .replace("#", "\\#")
    )


def daily_sharpe(df: pd.DataFrame, pnl_col: str = "norm_pnl") -> float:
    if df.empty:
        return float("nan")
    daily = df.groupby("signal_date")[pnl_col].sum()
    std = daily.std()
    if len(daily) < 2 or std == 0 or pd.isna(std):
        return float("nan")
    return float(daily.mean() / std * np.sqrt(252))


def daily_sortino(df: pd.DataFrame, pnl_col: str = "norm_pnl") -> float:
    if df.empty:
        return float("nan")
    daily = df.groupby("signal_date")[pnl_col].sum()
    if len(daily) < 2:
        return float("nan")
    downside = daily[daily < 0.0]
    if downside.empty:
        return float("inf") if daily.mean() > 0 else float("nan")
    downside_std = downside.std()
    if downside_std == 0 or pd.isna(downside_std):
        return float("nan")
    return float(daily.mean() / downside_std * np.sqrt(252))


def load_all_trade() -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_parquet(
        ROOT / "data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet",
        columns=[
            "signal_date",
            "strategy",
            "exit_policy",
            "is_valid_backtest",
            "realized_return",
        ],
    )
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    labels["year"] = labels["signal_date"].dt.year
    labels = labels[
        labels["year"].isin(YEARS)
        & labels["strategy"].isin(["T1", "K2"])
        & labels["exit_policy"].isin(
            [
                "exit_selected_leg_expiration_payoff",
                "exit_first_liquid_on_or_after_reversion",
            ]
        )
        & labels["is_valid_backtest"].astype(bool)
    ].copy()
    labels["norm_pnl"] = pd.to_numeric(labels["realized_return"], errors="coerce").fillna(0.0) * 1000.0

    rows = []
    for policy, policy_name in [
        ("exit_selected_leg_expiration_payoff", "Expiration payoff"),
        ("exit_first_liquid_on_or_after_reversion", "First-liquid exit"),
    ]:
        p = labels[labels["exit_policy"].eq(policy)]
        for year in YEARS:
            for strategy in ["T1", "K2"]:
                d = p[(p["year"].eq(year)) & (p["strategy"].eq(strategy))]
                rows.append(
                    {
                        "exit_policy": policy_name,
                        "year": year,
                        "strategy": strategy,
                        "trades": len(d),
                        "norm_pnl": float(d["norm_pnl"].sum()),
                        "win_rate": float((d["norm_pnl"] > 0).mean()) if len(d) else float("nan"),
                        "mean_return": float(d["realized_return"].mean()) if len(d) else float("nan"),
                        "sharpe": daily_sharpe(d),
                        "sortino": daily_sortino(d),
                    }
                )
    all_trade = pd.DataFrame(rows)

    combined_rows = []
    for policy_name in all_trade["exit_policy"].unique():
        for year in YEARS:
            part = labels[
                labels["year"].eq(year)
                & labels["strategy"].isin(["T1", "K2"])
                & labels["exit_policy"].map(
                    {
                        "exit_selected_leg_expiration_payoff": "Expiration payoff",
                        "exit_first_liquid_on_or_after_reversion": "First-liquid exit",
                    }
                ).eq(policy_name)
            ].copy()
            combined_rows.append(
                {
                    "exit_policy": policy_name,
                    "year": year,
                    "trades": len(part),
                    "norm_pnl": float(part["norm_pnl"].sum()),
                    "win_rate": float((part["norm_pnl"] > 0).mean()) if len(part) else float("nan"),
                    "mean_return": float(part["realized_return"].mean()) if len(part) else float("nan"),
                    "sharpe": daily_sharpe(part),
                    "sortino": daily_sortino(part),
                }
            )
    combined = pd.DataFrame(combined_rows)
    all_trade.to_csv(TABLE_DIR / "all_trade_by_strategy.csv", index=False)
    combined.to_csv(TABLE_DIR / "all_trade_combined.csv", index=False)
    return all_trade, combined


def normalize_result_table(path: Path, model: str, exit_policy: str, kind: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            [
                {
                    "model": model,
                    "exit_policy": exit_policy,
                    "year": np.nan,
                    "trades": np.nan,
                    "norm_pnl": np.nan,
                    "win_rate": np.nan,
                    "sharpe": np.nan,
                    "sortino": np.nan,
                    "status": "pending",
                }
            ]
        )
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        if kind == "lightgbm_expiry":
            year = int(r["test_year"])
            trades = int(r["n_selected"])
            pnl = float(r["norm_total_pnl"])
            win = float(r["win_rate"])
            sharpe = float(r["daily_sharpe"])
        elif kind == "v2":
            year = int(r["test_year"])
            trades = int(r["test_n_selected"])
            pnl = float(r["test_norm_total_pnl"])
            win = float(r["test_win_rate"])
            sharpe = float(r["test_daily_sharpe"])
        else:
            year = int(r["test_year"])
            trades = int(r["test_n_selected"])
            pnl = float(r["test_norm_total_pnl"])
            win = float(r["test_win_rate"])
            sharpe = float(r["test_daily_sharpe"])
        rows.append(
            {
                "model": model,
                "exit_policy": exit_policy,
                "year": year,
                "trades": trades,
                "norm_pnl": pnl,
                "win_rate": win,
                "sharpe": sharpe,
                "sortino": np.nan,
                "status": "available",
            }
        )
    return pd.DataFrame(rows)


def select_episode_actions(token_rows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    candidates = token_rows[token_rows["predicted_prob"] >= threshold].copy()
    if candidates.empty:
        return candidates
    return (
        candidates.sort_values(["predicted_prob", "strategy"], ascending=[False, True])
        .groupby("episode_id", as_index=False)
        .first()
    )


def selected_from_generic_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    preds = pd.read_parquet(path)
    preds["signal_date"] = pd.to_datetime(preds["signal_date"])
    selected = []
    for fold, g in preds.groupby("fold"):
        threshold = float(g["threshold"].iloc[0]) if "threshold" in g.columns else np.nan
        if pd.isna(threshold):
            if "selected" in g.columns:
                selected.append(g[g["selected"].astype(bool)].copy())
            continue
        selected.append(select_episode_actions(g, threshold))
    if not selected:
        return pd.DataFrame()
    out = pd.concat(selected, ignore_index=True)
    out["norm_pnl"] = pd.to_numeric(out["target_norm_pnl"], errors="coerce").fillna(0.0)
    return out


def selected_from_saved_selected(path_options: list[Path]) -> pd.DataFrame:
    for path in path_options:
        if not path.exists():
            continue
        selected = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        if selected.empty:
            return pd.DataFrame()
        selected["signal_date"] = pd.to_datetime(selected["signal_date"])
        pnl_col = "target_norm_pnl" if "target_norm_pnl" in selected.columns else "norm_pnl"
        selected["norm_pnl"] = pd.to_numeric(selected[pnl_col], errors="coerce").fillna(0.0)
        return selected
    return pd.DataFrame()


def selected_from_clean_lightgbm() -> pd.DataFrame:
    """Reconstruct the clean T1/K2 LightGBM selector from saved predictions."""
    pred_path = ROOT / "ExpiryMLCorrection/tables/parallel_binary_lgbm_predictions.parquet"
    val_path = ROOT / "ExpiryMLCorrection/tables/parallel_binary_lgbm_summary.csv"
    label_path = ROOT / "data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet"
    if not pred_path.exists() or not val_path.exists() or not label_path.exists():
        return pd.DataFrame()

    strategies = ["T1", "K2"]
    folds = [
        {"fold": "2017_2019__val2020__test2021", "train": [2017, 2018, 2019], "test": [2021]},
        {"fold": "2018_2020__val2021__test2022", "train": [2018, 2019, 2020], "test": [2022]},
        {"fold": "2019_2021__val2022__test2023", "train": [2019, 2020, 2021], "test": [2023]},
        {"fold": "2020_2022__val2023__test2024", "train": [2020, 2021, 2022], "test": [2024]},
        {"fold": "2021_2023__val2024__test2025", "train": [2021, 2022, 2023], "test": [2025]},
        {"fold": "2022_2024__val2025__test2026", "train": [2022, 2023, 2024], "test": [2026]},
    ]

    preds = pd.read_parquet(pred_path)
    preds["signal_date"] = pd.to_datetime(preds["signal_date"])
    preds["signal_year"] = preds["signal_date"].dt.year
    own_pnl = pd.Series(np.nan, index=preds.index)
    for strategy in ["T1", "T2", "K1", "K2"]:
        col = f"norm_pnl_{strategy}"
        if col in preds.columns:
            own_pnl.loc[preds["strategy"].eq(strategy)] = preds.loc[preds["strategy"].eq(strategy), col]
    preds["own_norm_pnl"] = own_pnl

    raw = pd.read_parquet(
        label_path,
        columns=["strategy", "exit_policy", "signal_date", "realized_return", "is_valid_backtest"],
    )
    raw = raw[raw["exit_policy"].eq("exit_selected_leg_expiration_payoff")].copy()
    raw["signal_date"] = pd.to_datetime(raw["signal_date"])
    raw["signal_year"] = raw["signal_date"].dt.year

    selected = []
    for fold in folds:
        stats = {}
        for strategy in strategies:
            sub = raw[
                raw["strategy"].eq(strategy)
                & raw["signal_year"].isin(fold["train"])
                & raw["is_valid_backtest"].astype(bool)
            ]["realized_return"].dropna()
            wins = sub[sub > 0]
            losses = sub[sub <= 0]
            stats[strategy] = {
                "avg_win": float(wins.mean()) if len(wins) else 0.0,
                "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            }

        te = preds[
            preds["signal_year"].isin(fold["test"])
            & preds["strategy"].isin(strategies)
            & preds["selected"].astype(bool)
        ].copy()
        if te.empty:
            continue

        te["expected_return"] = te.apply(
            lambda r: r["predicted_prob"] * stats[r["strategy"]]["avg_win"]
            + (1.0 - r["predicted_prob"]) * stats[r["strategy"]]["avg_loss"],
            axis=1,
        )
        te = te[te["expected_return"] > 0].copy()
        if te.empty:
            continue
        best = (
            te.sort_values("expected_return", ascending=False)
            .groupby("episode_id", as_index=False)
            .first()
        )
        best["target_norm_pnl"] = best["own_norm_pnl"]
        selected.append(best)

    if not selected:
        return pd.DataFrame()
    out = pd.concat(selected, ignore_index=True)
    out["norm_pnl"] = pd.to_numeric(out["target_norm_pnl"], errors="coerce").fillna(0.0)
    return out


def sortino_lookup_from_selected(selected: pd.DataFrame) -> tuple[dict[int, float], float]:
    if selected.empty:
        return {}, float("nan")
    selected = selected.copy()
    selected["signal_date"] = pd.to_datetime(selected["signal_date"])
    selected["year"] = selected["signal_date"].dt.year
    by_year = {
        int(year): daily_sortino(g, "norm_pnl")
        for year, g in selected.groupby("year")
    }
    return by_year, daily_sortino(selected, "norm_pnl")


def load_model_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = [
        normalize_result_table(
            ROOT / "ExpiryMLCorrection/tables/meta_selector_clean_t1k2_only_results.csv",
            "LightGBM",
            "Expiration payoff",
            "lightgbm_expiry",
        ),
        normalize_result_table(
            ROOT / "TP2MVTransformer/tables/mlp_baseline_results.csv",
            "MLP",
            "Expiration payoff",
            "generic",
        ),
        normalize_result_table(
            ROOT / "TP2MVTransformer/tables/mv_alibi_transformer_results.csv",
            "Transformer V1",
            "Expiration payoff",
            "generic",
        ),
        normalize_result_table(
            ROOT / "TP2MVTransformer/ColabRunResults/mv_alibi_v2_yearly_diagnostics.csv",
            "Transformer V2 GPU",
            "Expiration payoff",
            "v2",
        ),
        normalize_result_table(
            ROOT / "TP2MVTransformerFirstLiquid/tables/lightgbm_first_liquid_results.csv",
            "LightGBM",
            "First-liquid exit",
            "generic",
        ),
        normalize_result_table(
            ROOT / "TP2MVTransformerFirstLiquid/tables/mlp_baseline_results.csv",
            "MLP",
            "First-liquid exit",
            "generic",
        ),
        normalize_result_table(
            ROOT / "TP2MVTransformerFirstLiquid/tables/mv_alibi_transformer_v2_results.csv",
            "Transformer",
            "First-liquid exit",
            "generic",
        ),
    ]
    yearly = pd.concat(pieces, ignore_index=True)

    selected_sources = {
        ("Expiration payoff", "LightGBM"): selected_from_clean_lightgbm(),
        ("Expiration payoff", "MLP"): selected_from_generic_predictions(ROOT / "TP2MVTransformer/data/mlp_baseline_test_predictions.parquet"),
        ("Expiration payoff", "Transformer V1"): selected_from_generic_predictions(ROOT / "TP2MVTransformer/data/mv_alibi_transformer_test_predictions.parquet"),
        ("Expiration payoff", "Transformer V2 GPU"): selected_from_saved_selected([
            ROOT / "TP2MVTransformer/data/mv_alibi_transformer_v2_selected_trades.parquet",
            ROOT / "TP2MVTransformer/ColabRunResults/mv_alibi_transformer_v2_selected_trades.parquet",
            ROOT / "TP2MVTransformer/ColabRunResults/mv_alibi_transformer_v2_selected_trades.csv",
        ]),
        ("First-liquid exit", "LightGBM"): selected_from_generic_predictions(ROOT / "TP2MVTransformerFirstLiquid/data/lightgbm_first_liquid_test_predictions.parquet"),
        ("First-liquid exit", "MLP"): selected_from_generic_predictions(ROOT / "TP2MVTransformerFirstLiquid/data/mlp_baseline_test_predictions.parquet"),
        ("First-liquid exit", "Transformer"): selected_from_generic_predictions(ROOT / "TP2MVTransformerFirstLiquid/data/mv_alibi_transformer_v2_test_predictions.parquet"),
    }
    combined_sortino = {}
    selected_daily_rows = []
    for key, selected in selected_sources.items():
        by_year, combined = sortino_lookup_from_selected(selected)
        combined_sortino[key] = combined
        if not selected.empty:
            daily = selected.groupby("signal_date", as_index=False)["norm_pnl"].sum()
            daily["exit_policy"] = key[0]
            daily["model"] = key[1]
            selected_daily_rows.append(daily)
        mask = yearly["exit_policy"].eq(key[0]) & yearly["model"].eq(key[1])
        yearly.loc[mask, "sortino"] = yearly.loc[mask, "year"].map(by_year)

    if selected_daily_rows:
        pd.concat(selected_daily_rows, ignore_index=True).to_csv(
            TABLE_DIR / "selected_daily_pnl_for_sortino.csv", index=False
        )
    yearly.to_csv(TABLE_DIR / "model_yearly_results.csv", index=False)

    rows = []
    for (exit_policy, model), g in yearly.groupby(["exit_policy", "model"], dropna=False):
        available = g[g["status"].eq("available")].copy()
        if available.empty:
            rows.append(
                {
                    "exit_policy": exit_policy,
                    "model": model,
                    "total_pnl": np.nan,
                    "trades": np.nan,
                    "win_rate": np.nan,
                    "mean_yearly_sharpe": np.nan,
                    "mean_yearly_sortino": np.nan,
                    "combined_sortino": np.nan,
                    "positive_years": np.nan,
                    "status": "pending",
                }
            )
            continue
        rows.append(
            {
                "exit_policy": exit_policy,
                "model": model,
                "total_pnl": float(available["norm_pnl"].sum()),
                "trades": int(available["trades"].sum()),
                "win_rate": float((available["win_rate"] * available["trades"]).sum() / available["trades"].sum()),
                "mean_yearly_sharpe": float(available["sharpe"].mean()),
                "mean_yearly_sortino": float(available["sortino"].replace([np.inf, -np.inf], np.nan).mean()),
                "combined_sortino": float(combined_sortino.get((exit_policy, model), np.nan)),
                "positive_years": int((available["norm_pnl"] > 0).sum()),
                "status": "available",
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE_DIR / "model_summary_results.csv", index=False)
    return yearly, summary


def table_all_trade(all_trade: pd.DataFrame, policy: str) -> str:
    rows = []
    for year in YEARS:
        t1 = all_trade[(all_trade["exit_policy"].eq(policy)) & (all_trade["year"].eq(year)) & (all_trade["strategy"].eq("T1"))].iloc[0]
        k2 = all_trade[(all_trade["exit_policy"].eq(policy)) & (all_trade["year"].eq(year)) & (all_trade["strategy"].eq("K2"))].iloc[0]
        rows.append(
            f"{year} & {num(t1.trades)} & {money(t1.norm_pnl)} & {pct(t1.win_rate)} & {f2(t1.sortino)} & "
            f"{num(k2.trades)} & {money(k2.norm_pnl)} & {pct(k2.win_rate)} & {f2(k2.sortino)} \\\\"
        )
    return "\n".join(rows)


def table_model_summary(summary: pd.DataFrame, policy: str) -> str:
    order = ["LightGBM", "MLP", "Transformer V1", "Transformer V2 GPU", "Transformer"]
    sub = summary[summary["exit_policy"].eq(policy)].copy()
    sub["ord"] = sub["model"].map({m: i for i, m in enumerate(order)}).fillna(99)
    sub = sub.sort_values("ord")
    rows = []
    for _, r in sub.iterrows():
        if r["status"] != "available":
            rows.append(f"{esc(r.model)} & pending & -- & -- & -- & -- & -- & -- \\\\")
        else:
            rows.append(
                f"{esc(r.model)} & {money(r.total_pnl)} & {num(r.trades)} & "
                f"{pct(r.win_rate)} & {f2(r.mean_yearly_sharpe)} & {f2(r.mean_yearly_sortino)} & "
                f"{f2(r.combined_sortino)} & {int(r.positive_years)}/6 \\\\"
            )
    return "\n".join(rows)


def table_yearly_model(yearly: pd.DataFrame, policy: str) -> str:
    sub = yearly[(yearly["exit_policy"].eq(policy)) & yearly["status"].eq("available")].copy()
    models = [m for m in ["LightGBM", "MLP", "Transformer V1", "Transformer V2 GPU", "Transformer"] if m in set(sub["model"])]
    rows = []
    for year in TEST_YEARS:
        vals = []
        for model in models:
            r = sub[(sub["year"].eq(year)) & (sub["model"].eq(model))]
            vals.append(money(float(r.iloc[0]["norm_pnl"])) if not r.empty else "--")
        rows.append(f"{year} & " + " & ".join(vals) + " \\\\")
    header = "Year & " + " & ".join(esc(m) for m in models) + " \\\\"
    cols = "l" + "r" * len(models)
    return cols, header, "\n".join(rows)


def main() -> None:
    all_trade, combined = load_all_trade()
    yearly, summary = load_model_results()

    exp_cols, exp_header, exp_year_rows = table_yearly_model(yearly, "Expiration payoff")
    fl_cols, fl_header, fl_year_rows = table_yearly_model(yearly, "First-liquid exit")

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=0.78in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\title{{TP2 Exit Policy Comparison: First-Liquid vs Expiration Payoff}}
\author{{SPY American Call TP2 Project}}
\date{{\today}}
\begin{{document}}
\maketitle

\section{{Purpose}}
This report separates two distinct TP2 backtests. The entry episode is the same in both cases: a conservative bid/ask TP2 violation in SPY calls. The difference is the post-entry exit accounting.

\textbf{{First-liquid exit}} closes the selected legs at the first liquid quote date on or after TP2 reversion. It is more execution-realistic because it uses future tradable bid/ask quotes, liquidity checks, and bid/ask costs. It is not guaranteed to be more profitable.

\textbf{{Expiration payoff}} holds selected legs to their selected expiration payoff and uses intrinsic call payoff, \(\max(S_T-K,0)\). It is less quote-dependent and cleaner theoretically, but it is not a full assignment-aware live trading simulator for American calls.

All P\&L below uses the \(\$1{{,}}000\) normalized exposure convention: \(\mathrm{{target\_norm\_pnl}}=\mathrm{{realized\_return}}\times1000\).

\textbf{{Sharpe and Sortino definitions.}} Sharpe is annualized from daily selected normalized P\&L. Sortino uses the same daily series but replaces total daily volatility with downside-only volatility, so it penalizes negative daily P\&L variation rather than large positive jumps. The risk-free rate is set to zero for these diagnostic ratios.

\section{{All-Trade Baselines}}
These tables trade every valid T1 or K2 episode without ML selection. They show why the ML problem is needed.

\subsection{{Expiration Payoff: All T1 and K2}}
\begin{{table}}[H]
\centering
\resizebox{{\textwidth}}{{!}}{{%
\scriptsize
\begin{{tabular}}{{rrrrrrrrr}}
\toprule
Year & T1 trades & T1 P\&L & T1 win & T1 Sortino & K2 trades & K2 P\&L & K2 win & K2 Sortino \\
\midrule
{table_all_trade(all_trade, "Expiration payoff")}
\bottomrule
\end{{tabular}}
}}
\end{{table}}

\subsection{{First-Liquid Exit: All T1 and K2}}
\begin{{table}}[H]
\centering
\resizebox{{\textwidth}}{{!}}{{%
\scriptsize
\begin{{tabular}}{{rrrrrrrrr}}
\toprule
Year & T1 trades & T1 P\&L & T1 win & T1 Sortino & K2 trades & K2 P\&L & K2 win & K2 Sortino \\
\midrule
{table_all_trade(all_trade, "First-liquid exit")}
\bottomrule
\end{{tabular}}
}}
\end{{table}}

\section{{Model Summary, Test Years 2021--2026}}
\subsection{{Expiration Payoff Models}}
\begin{{table}}[H]
\centering
\resizebox{{\textwidth}}{{!}}{{%
\small
\begin{{tabular}}{{lrrrrrrr}}
\toprule
Model & Total P\&L & Trades & Win rate & Mean yearly Sharpe & Mean yearly Sortino & Combined Sortino & Positive years \\
\midrule
{table_model_summary(summary, "Expiration payoff")}
\bottomrule
\end{{tabular}}
}}
\end{{table}}

\subsection{{First-Liquid Exit Models}}
\begin{{table}}[H]
\centering
\resizebox{{\textwidth}}{{!}}{{%
\small
\begin{{tabular}}{{lrrrrrrr}}
\toprule
Model & Total P\&L & Trades & Win rate & Mean yearly Sharpe & Mean yearly Sortino & Combined Sortino & Positive years \\
\midrule
{table_model_summary(summary, "First-liquid exit")}
\bottomrule
\end{{tabular}}
}}
\end{{table}}

\section{{Year-Wise Model P\&L}}
\subsection{{Expiration Payoff}}
\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{{exp_cols}}}
\toprule
{exp_header}
\midrule
{exp_year_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{First-Liquid Exit}}
\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{{fl_cols}}}
\toprule
{fl_header}
\midrule
{fl_year_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Interpretation}}
The first-liquid exit is more execution-realistic, but it is much harder in this dataset. The all-trade first-liquid T1/K2 portfolio is negative in every test year from 2021 to 2026. LightGBM and MLP also remain negative overall, although they reduce the unconditional loss substantially.

The expiration-payoff target is materially stronger. Blindly trading T1 and K2 from 2017 to 2026 is still negative in aggregate because T1 loses heavily, but K2 contains strong positive payoff-skew years. The ML and transformer models convert the noisy unconditional opportunity into selected portfolios with positive out-of-sample P\&L.

The first-liquid transformer has not yet been trained on GPU. Its row is marked pending. The correct next experiment is to run \texttt{{TP2MVTransformerFirstLiquid/03\_train\_mv\_alibi\_transformer\_first\_liquid.py}} on Colab/GPU and then regenerate this report.

The Transformer V2 GPU expiration row is loaded from Colab year-level diagnostics. Because the local Colab artifact does not include the selected episode-level prediction parquet, its Sortino cannot be independently recomputed from daily selected P\&L in this report and is shown as unavailable. Once the selected V2 prediction file is exported from Colab, the same generator can compute its Sortino directly.

\section{{Caveats}}
\begin{{itemize}}
  \item Normalized P\&L is return-based. It controls exposure per trade, but rare high-return option outcomes can dominate total P\&L.
  \item Expiration payoff is terminal-payoff accounting. For SPY American calls, assignment and dividend risk are not fully modeled.
  \item First-liquid exits depend on future option quote availability and bid/ask quality, so they are execution-realistic but quote-noise-sensitive.
  \item Model threshold choices are validation-year choices only; test years are evaluated once.
\end{{itemize}}

\end{{document}}
"""
    out = REPORT / "tp2_exit_policy_comparison_report.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
