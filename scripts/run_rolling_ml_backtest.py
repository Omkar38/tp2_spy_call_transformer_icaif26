#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from tp2rr2.backtest import select_trades, summarize_trades
from tp2rr2.config import CONFIG
from tp2rr2.io import write_table
from tp2rr2.model import MODEL_TYPES, make_pipeline


def _load_training_rows(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _quarter_windows(start: pd.Timestamp, end: pd.Timestamp, months: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows = []
    current = pd.Timestamp(start).normalize()
    while current <= end:
        stop = current + pd.DateOffset(months=months) - pd.Timedelta(days=1)
        windows.append((current, min(stop, end)))
        current = current + pd.DateOffset(months=months)
    return windows


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run expanding-window ML backtests in rolling calendar windows.")
    parser.add_argument("--input", default="data/labels/archive2_forward_strategy_training_rows_2013_2026.parquet")
    parser.add_argument("--output-dir", default="reports/rolling_3m_lightgbm")
    parser.add_argument("--model-type", choices=sorted(MODEL_TYPES), default="lightgbm")
    parser.add_argument("--window-months", type=int, default=3)
    parser.add_argument("--first-test-start", default="2014-01-01")
    parser.add_argument("--last-test-end", default=None)
    parser.add_argument("--min-train-valid", type=int, default=500)
    parser.add_argument("--prediction-threshold", type=float, default=None)
    parser.add_argument("--min-normalized-severity", type=float, default=None)
    parser.add_argument("--max-spread-pct", type=float, default=None)
    parser.add_argument("--max-avg-spread-pct", type=float, default=None)
    parser.add_argument("--max-trades-per-day", type=int, default=None)
    parser.add_argument("--min-volume", type=int, default=None)
    parser.add_argument("--min-open-interest", type=int, default=None)
    args = parser.parse_args()

    input_path = CONFIG.paths.root / args.input
    output_dir = CONFIG.paths.root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_training_rows(input_path)

    last_end = pd.Timestamp(args.last_test_end) if args.last_test_end else df["date"].max()
    windows = _quarter_windows(pd.Timestamp(args.first_test_start), last_end, args.window_months)

    backtest_config = CONFIG.backtest
    updates = {}
    if args.prediction_threshold is not None:
        updates["prediction_threshold"] = args.prediction_threshold
    if args.min_normalized_severity is not None:
        updates["min_normalized_severity"] = args.min_normalized_severity
    if args.max_spread_pct is not None:
        updates["max_spread_pct"] = args.max_spread_pct
    if args.max_avg_spread_pct is not None:
        updates["max_avg_spread_pct"] = args.max_avg_spread_pct
    if args.max_trades_per_day is not None:
        updates["max_trades_per_day"] = args.max_trades_per_day
    if args.min_volume is not None:
        updates["min_volume"] = args.min_volume
    if args.min_open_interest is not None:
        updates["min_open_interest"] = args.min_open_interest
    if updates:
        backtest_config = replace(backtest_config, **updates)

    summaries = []
    all_trades = []

    for idx, (test_start, test_end) in enumerate(windows, 1):
        train = df[(df["date"] < test_start) & df["is_valid_label"].fillna(False)].copy()
        test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        if len(train) < args.min_train_valid or test.empty:
            summary = {
                "window": idx,
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "status": "skipped",
                "train_valid_rows": int(len(train)),
                "test_rows": int(len(test)),
                "reason": "insufficient_train_or_empty_test",
            }
            summaries.append(summary)
            print(f"{idx:02d} {test_start.date()}..{test_end.date()} skipped train={len(train):,} test={len(test):,}", flush=True)
            continue

        y_train = train["realized_return"].astype(float)
        model = make_pipeline(train, model_type=args.model_type)
        model.fit(train, y_train)

        test = test.copy()
        test["predicted_return"] = model.predict(test)
        trades = select_trades(test, backtest_config)
        if not trades.empty:
            trades = trades.copy()
            trades["rolling_window"] = idx
            trades["window_start"] = test_start
            trades["window_end"] = test_end
            all_trades.append(trades)

        perf = summarize_trades(trades)
        summary = {
            "window": idx,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "status": "completed",
            "train_valid_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_valid_rows": int(test["is_valid_label"].fillna(False).sum()),
            **perf,
        }
        summaries.append(summary)
        print(
            f"{idx:02d} {test_start.date()}..{test_end.date()} "
            f"train={len(train):,} test={len(test):,} selected={summary.get('selected_trades', 0):,} "
            f"exec={summary.get('executable_trades', 0):,} pnl={summary.get('net_pnl', 0.0):,.2f}",
            flush=True,
        )

    summary_df = pd.DataFrame(summaries)
    summary_csv = output_dir / "rolling_3m_summary.csv"
    summary_json = output_dir / "rolling_3m_summary.json"
    trades_path = output_dir / "rolling_3m_selected_trades.parquet"

    summary_df.to_csv(summary_csv, index=False)
    if all_trades:
        write_table(pd.concat(all_trades, ignore_index=True), trades_path)
    else:
        write_table(pd.DataFrame(), trades_path)

    completed = summary_df[summary_df["status"].eq("completed")]
    aggregate = {
        "input": args.input,
        "output_dir": args.output_dir,
        "model_type": args.model_type,
        "window_months": args.window_months,
        "first_test_start": args.first_test_start,
        "last_test_end": str(last_end.date()),
        "min_train_valid": args.min_train_valid,
        "backtest_config": asdict(backtest_config),
        "windows_total": int(len(summary_df)),
        "windows_completed": int(len(completed)),
        "selected_trades": int(completed["selected_trades"].fillna(0).sum()) if not completed.empty else 0,
        "executable_trades": int(completed["executable_trades"].fillna(0).sum()) if not completed.empty else 0,
        "invalid_selected_trades": int(completed["invalid_selected_trades"].fillna(0).sum()) if not completed.empty else 0,
        "net_pnl": float(completed["net_pnl"].fillna(0).sum()) if not completed.empty else 0.0,
        "summary_csv": str(summary_csv),
        "trades_path": str(trades_path),
    }
    summary_json.write_text(json.dumps(_json_ready({"aggregate": aggregate, "windows": summaries}), indent=2))
    print(json.dumps(_json_ready(aggregate), indent=2))


if __name__ == "__main__":
    main()
