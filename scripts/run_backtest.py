#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import pandas as pd
import pyarrow.parquet as pq

from tp2rr2.backtest import select_trades, summarize_trades
from tp2rr2.config import CONFIG
from tp2rr2.io import write_table
from tp2rr2.model import load_model


def _model_columns(model) -> list[str]:
    pre = model.named_steps["preprocess"]
    columns: list[str] = []
    for _, _, transformer_columns in pre.transformers_:
        if isinstance(transformer_columns, list):
            columns.extend(transformer_columns)
    return columns


def _backtest_columns(path, model) -> list[str]:
    required = {
        "date",
        "violation_id",
        "strategy",
        "is_valid_label",
        "realized_return",
        "net_pnl",
        "normalized_severity",
        "max_spread_pct",
        "avg_spread_pct",
        "min_volume",
        "min_open_interest",
        "option_type",
        "exit_reason",
        "actual_holding_days",
    }
    required.update(f"{leg}_contract_id" for leg in ["A", "B", "C", "D"])
    required.update(f"{leg}_{side}_size" for leg in ["A", "B", "C", "D"] for side in ["bid", "ask"])
    schema_columns = set(pq.read_schema(path).names)
    return [column for column in sorted(required.union(_model_columns(model))) if column in schema_columns]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/labels/strategy_training_rows.parquet")
    parser.add_argument("--model", default="data/models/lightgbm_strategy_return.joblib")
    parser.add_argument("--trades-output", default="reports/tables/test_selected_trades.csv")
    parser.add_argument("--summary-output", default="reports/tables/test_summary.json")
    args = parser.parse_args()

    model = load_model(CONFIG.paths.root / args.model)
    input_path = CONFIG.paths.root / args.input
    test = pd.read_parquet(
        input_path,
        columns=_backtest_columns(input_path, model),
        filters=[
            ("is_valid_label", "=", True),
            ("date", ">=", pd.Timestamp(CONFIG.splits.test_start)),
            ("date", "<=", pd.Timestamp(CONFIG.splits.test_end)),
        ],
    )
    test = test.copy()
    test["predicted_return"] = model.predict(test)
    trades = select_trades(test, CONFIG.backtest)
    summary = summarize_trades(trades)
    write_table(trades, CONFIG.paths.root / args.trades_output)
    summary_path = CONFIG.paths.root / args.summary_output
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
