#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd
import pyarrow.parquet as pq

from tp2rr2.config import CONFIG
from tp2rr2.model import DROP_COLUMNS, ID_COLUMNS, split_by_date, train_model


def _training_columns(path) -> list[str]:
    columns = pq.read_schema(path).names
    keep = []
    for column in columns:
        if column in {"date", "is_valid_label", "realized_return"}:
            keep.append(column)
        elif column not in DROP_COLUMNS and column not in ID_COLUMNS:
            keep.append(column)
    return keep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/labels/strategy_training_rows.parquet")
    parser.add_argument("--model-output", default="data/models/lightgbm_strategy_return.joblib")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_path = CONFIG.paths.root / args.input
    df = pd.read_parquet(
        input_path,
        columns=_training_columns(input_path),
        filters=[("is_valid_label", "=", True)],
    )
    train = split_by_date(df, CONFIG.splits.train_start, CONFIG.splits.train_end)
    val = split_by_date(df, CONFIG.splits.validation_start, CONFIG.splits.validation_end)
    if args.max_train_rows and len(train) > args.max_train_rows:
        train = train.sample(n=args.max_train_rows, random_state=args.random_state)
    if args.max_validation_rows and len(val) > args.max_validation_rows:
        val = val.sample(n=args.max_validation_rows, random_state=args.random_state)
    metrics = train_model(train, val, CONFIG.paths.root / args.model_output)
    print(metrics)


if __name__ == "__main__":
    main()
