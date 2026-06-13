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


def _load_rows(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period("M").to_timestamp()


def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(ts).to_period("M").to_timestamp() + pd.offsets.MonthEnd(0)).normalize()


def _windows(
    first_train_start: pd.Timestamp,
    last_test_end: pd.Timestamp,
    train_months: int,
    validation_months: int,
    test_months: int,
    step_months: int,
) -> list[dict]:
    out = []
    current = _month_floor(first_train_start)
    idx = 1
    while current <= last_test_end:
        train_start = current
        train_end = current + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        validation_start = train_end + pd.Timedelta(days=1)
        validation_end = validation_start + pd.DateOffset(months=validation_months) - pd.Timedelta(days=1)
        test_start = validation_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        if test_start > last_test_end:
            break
        out.append(
            {
                "window": idx,
                "train_start": train_start,
                "train_end": train_end,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "test_start": test_start,
                "test_end": min(test_end, last_test_end),
            }
        )
        current = current + pd.DateOffset(months=step_months)
        idx += 1
    return out


def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()


def _threshold_candidates(pred: np.ndarray) -> list[float]:
    finite = pred[np.isfinite(pred)]
    values = [0.0]
    if len(finite):
        for q in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            values.append(float(np.quantile(finite, q)))
    return sorted(set(values))


def _choose_threshold(validation: pd.DataFrame, backtest_config) -> tuple[float, dict]:
    best_threshold = float(backtest_config.prediction_threshold)
    best_summary = None
    for threshold in _threshold_candidates(validation["predicted_return"].to_numpy(dtype=float)):
        config = replace(backtest_config, prediction_threshold=threshold)
        trades = select_trades(validation, config)
        summary = summarize_trades(trades)
        summary["_threshold"] = threshold
        summary["_score_tuple"] = (
            float(summary.get("net_pnl") or 0.0),
            int(summary.get("executable_trades") or 0),
            -int(summary.get("invalid_selected_trades") or 0),
        )
        if best_summary is None or summary["_score_tuple"] > best_summary["_score_tuple"]:
            best_summary = summary
            best_threshold = threshold
    assert best_summary is not None
    best_summary.pop("_score_tuple", None)
    return best_threshold, best_summary


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if pd.isna(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3m train / 1m validation / 2m test walk-forward ML backtest.")
    parser.add_argument("--input", default="data/labels/archive2_forward_strategy_training_rows_2013_2026.parquet")
    parser.add_argument("--output-dir", default="reports/walkforward_3_1_2_lightgbm_all_cleaned_gates")
    parser.add_argument("--model-type", choices=sorted(MODEL_TYPES), default="lightgbm")
    parser.add_argument("--first-train-start", default=None)
    parser.add_argument("--last-test-end", default=None)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--validation-months", type=int, default=1)
    parser.add_argument("--test-months", type=int, default=2)
    parser.add_argument("--step-months", type=int, default=2)
    parser.add_argument("--min-train-valid", type=int, default=20)
    parser.add_argument("--min-validation-valid", type=int, default=1)
    parser.add_argument("--prediction-threshold", type=float, default=0.0)
    parser.add_argument("--min-normalized-severity", type=float, default=0.0001)
    parser.add_argument("--max-spread-pct", type=float, default=0.50)
    parser.add_argument("--max-avg-spread-pct", type=float, default=0.50)
    parser.add_argument("--min-volume", type=int, default=0)
    parser.add_argument("--min-open-interest", type=int, default=0)
    parser.add_argument("--max-trades-per-day", type=int, default=25)
    args = parser.parse_args()

    output_dir = CONFIG.paths.root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _load_rows(CONFIG.paths.root / args.input)

    first_train_start = pd.Timestamp(args.first_train_start) if args.first_train_start else _month_floor(df["date"].min())
    last_test_end = pd.Timestamp(args.last_test_end) if args.last_test_end else df["date"].max()
    windows = _windows(
        first_train_start=first_train_start,
        last_test_end=last_test_end,
        train_months=args.train_months,
        validation_months=args.validation_months,
        test_months=args.test_months,
        step_months=args.step_months,
    )

    base_config = replace(
        CONFIG.backtest,
        prediction_threshold=args.prediction_threshold,
        min_normalized_severity=args.min_normalized_severity,
        max_spread_pct=args.max_spread_pct,
        max_avg_spread_pct=args.max_avg_spread_pct,
        min_volume=args.min_volume,
        min_open_interest=args.min_open_interest,
        max_trades_per_day=args.max_trades_per_day,
    )

    summaries = []
    all_test_trades = []
    all_validation_summaries = []

    for window in windows:
        train = _slice(df, window["train_start"], window["train_end"])
        validation = _slice(df, window["validation_start"], window["validation_end"])
        test = _slice(df, window["test_start"], window["test_end"])
        train_valid = train[train["is_valid_label"].fillna(False)].copy()
        validation_valid_count = int(validation["is_valid_label"].fillna(False).sum()) if not validation.empty else 0

        if len(train_valid) < args.min_train_valid or validation_valid_count < args.min_validation_valid or test.empty:
            reason = []
            if len(train_valid) < args.min_train_valid:
                reason.append("insufficient_train")
            if validation_valid_count < args.min_validation_valid:
                reason.append("insufficient_validation")
            if test.empty:
                reason.append("empty_test")
            summary = {
                **{k: str(v.date()) if isinstance(v, pd.Timestamp) else v for k, v in window.items()},
                "status": "skipped",
                "reason": ",".join(reason),
                "train_rows": int(len(train)),
                "train_valid_rows": int(len(train_valid)),
                "validation_rows": int(len(validation)),
                "validation_valid_rows": validation_valid_count,
                "test_rows": int(len(test)),
            }
            summaries.append(summary)
            print(
                f"{window['window']:02d} {window['train_start'].date()} train -> "
                f"{window['test_start'].date()}..{window['test_end'].date()} skipped {summary['reason']}",
                flush=True,
            )
            continue

        model = make_pipeline(train_valid, model_type=args.model_type)
        model.fit(train_valid, train_valid["realized_return"].astype(float))

        validation = validation.copy()
        validation["predicted_return"] = model.predict(validation)
        threshold, validation_summary = _choose_threshold(validation, base_config)
        selected_config = replace(base_config, prediction_threshold=threshold)

        test = test.copy()
        test["predicted_return"] = model.predict(test)
        test_trades = select_trades(test, selected_config)
        if not test_trades.empty:
            test_trades = test_trades.copy()
            test_trades["walkforward_window"] = window["window"]
            test_trades["train_start"] = window["train_start"]
            test_trades["train_end"] = window["train_end"]
            test_trades["validation_start"] = window["validation_start"]
            test_trades["validation_end"] = window["validation_end"]
            test_trades["test_start"] = window["test_start"]
            test_trades["test_end"] = window["test_end"]
            test_trades["selected_threshold"] = threshold
            all_test_trades.append(test_trades)

        test_summary = summarize_trades(test_trades)
        summary = {
            **{k: str(v.date()) if isinstance(v, pd.Timestamp) else v for k, v in window.items()},
            "status": "completed",
            "train_rows": int(len(train)),
            "train_valid_rows": int(len(train_valid)),
            "validation_rows": int(len(validation)),
            "validation_valid_rows": validation_valid_count,
            "test_rows": int(len(test)),
            "test_valid_rows": int(test["is_valid_label"].fillna(False).sum()),
            "selected_threshold": float(threshold),
            **test_summary,
        }
        summaries.append(summary)
        validation_summary = {
            **{k: str(v.date()) if isinstance(v, pd.Timestamp) else v for k, v in window.items()},
            **{f"validation_{k}": v for k, v in validation_summary.items()},
        }
        all_validation_summaries.append(validation_summary)
        print(
            f"{window['window']:02d} train={window['train_start'].date()}..{window['train_end'].date()} "
            f"val={window['validation_start'].date()}..{window['validation_end'].date()} "
            f"test={window['test_start'].date()}..{window['test_end'].date()} "
            f"train_valid={len(train_valid):,} test={len(test):,} selected={summary.get('selected_trades', 0):,} "
            f"exec={summary.get('executable_trades', 0):,} pnl={summary.get('net_pnl', 0.0):,.2f}",
            flush=True,
        )

    summary_df = pd.DataFrame(summaries)
    validation_df = pd.DataFrame(all_validation_summaries)
    summary_csv = output_dir / "walkforward_3_1_2_summary.csv"
    validation_csv = output_dir / "walkforward_3_1_2_validation_thresholds.csv"
    trades_path = output_dir / "walkforward_3_1_2_selected_trades.parquet"
    summary_df.to_csv(summary_csv, index=False)
    validation_df.to_csv(validation_csv, index=False)
    write_table(pd.concat(all_test_trades, ignore_index=True) if all_test_trades else pd.DataFrame(), trades_path)

    completed = summary_df[summary_df["status"].eq("completed")]
    aggregate = {
        "input": args.input,
        "output_dir": args.output_dir,
        "model_type": args.model_type,
        "protocol": {
            "train_months": args.train_months,
            "validation_months": args.validation_months,
            "test_months": args.test_months,
            "step_months": args.step_months,
            "min_train_valid": args.min_train_valid,
            "min_validation_valid": args.min_validation_valid,
        },
        "backtest_config_base": asdict(base_config),
        "windows_total": int(len(summary_df)),
        "windows_completed": int(len(completed)),
        "test_rows": int(completed["test_rows"].fillna(0).sum()) if not completed.empty else 0,
        "selected_trades": int(completed["selected_trades"].fillna(0).sum()) if not completed.empty else 0,
        "executable_trades": int(completed["executable_trades"].fillna(0).sum()) if not completed.empty else 0,
        "invalid_selected_trades": int(completed["invalid_selected_trades"].fillna(0).sum()) if not completed.empty else 0,
        "net_pnl": float(completed["net_pnl"].fillna(0).sum()) if not completed.empty else 0.0,
        "summary_csv": str(summary_csv),
        "validation_csv": str(validation_csv),
        "trades_path": str(trades_path),
    }
    (output_dir / "walkforward_3_1_2_summary.json").write_text(
        json.dumps(_json_ready({"aggregate": aggregate, "windows": summaries}), indent=2)
    )
    print(json.dumps(_json_ready(aggregate), indent=2))


if __name__ == "__main__":
    main()
