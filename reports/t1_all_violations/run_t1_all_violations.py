#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "labels" / "strategy_training_rows.parquet"
OUT_DIR = ROOT / "reports" / "t1_all_violations"
TRADES_OUT = OUT_DIR / "t1_all_2024_valid_trades.csv"
SUMMARY_OUT = OUT_DIR / "t1_all_2024_summary.json"
TARGET_GROSS_EXPOSURE = 1000.0


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "strategy": "T1",
            "policy": "all_valid_violations_no_ml",
            "target_gross_exposure": TARGET_GROSS_EXPOSURE,
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
        "strategy": "T1",
        "policy": "all_valid_violations_no_ml",
        "target_gross_exposure": TARGET_GROSS_EXPOSURE,
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
            ("strategy", "=", "T1"),
            ("date", ">=", pd.Timestamp("2024-01-01")),
            ("date", "<=", pd.Timestamp("2024-12-31")),
        ],
    )
    total_candidates = len(candidates)
    trades = candidates[candidates["is_valid_label"]].copy()
    trades = trades.copy()
    trades["unscaled_net_pnl"] = trades["net_pnl"].astype(float)
    trades["unscaled_entry_gross_exposure"] = trades["entry_gross_exposure"].astype(float)
    trades["scale_factor"] = TARGET_GROSS_EXPOSURE / trades["unscaled_entry_gross_exposure"]
    trades["entry_gross_exposure"] = TARGET_GROSS_EXPOSURE
    trades["net_pnl"] = trades["unscaled_net_pnl"] * trades["scale_factor"]
    trades["realized_return"] = trades["net_pnl"] / trades["entry_gross_exposure"]

    trades.to_csv(TRADES_OUT, index=False)
    summary = summarize(trades)
    summary["candidate_2024_t1_violations"] = int(total_candidates)
    summary["valid_traded_2024_t1_violations"] = int(len(trades))
    summary["invalid_skipped_2024_t1_violations"] = int(total_candidates - len(trades))
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
