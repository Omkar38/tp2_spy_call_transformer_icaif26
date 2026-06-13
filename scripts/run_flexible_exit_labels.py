#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pyarrow as pa
import pyarrow.parquet as pq

from tp2rr2.config import CONFIG
from tp2rr2.flexible_labels import build_flexible_exit_strategy_training_rows
from tp2rr2.io import read_table, write_table
from tp2rr2.labels import prepare_quote_frame


def _write_chunked_training_rows(features, quote_frame, trading_days, args) -> int:
    output_path = CONFIG.paths.root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    writer = None
    total_rows = 0
    try:
        for start in range(0, len(features), args.chunk_size):
            stop = min(start + args.chunk_size, len(features))
            chunk = features.iloc[start:stop].copy()
            final = build_flexible_exit_strategy_training_rows(
                features=chunk,
                quote_frame=quote_frame,
                trading_days=trading_days,
                config=CONFIG.labels,
                reversion_threshold=args.reversion_threshold,
                max_holding_days=args.max_holding_days,
            )
            table = pa.Table.from_pandas(final, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            total_rows += len(final)
            if args.progress:
                print(
                    f"Wrote {total_rows:,} flexible-exit strategy rows from {stop:,}/{len(features):,} violations",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
    return total_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build flexible reversion-exit labels for TP2 strategy rows.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--options", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-violations", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--reversion-threshold", type=float, default=0.0001)
    parser.add_argument("--max-holding-days", type=int, default=None)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    features = read_table(CONFIG.paths.root / args.features)
    if args.max_violations:
        features = features.head(args.max_violations)
    options = read_table(CONFIG.paths.root / args.options)
    quote_frame = prepare_quote_frame(options)
    trading_days = sorted(options["date"].drop_duplicates())

    if args.chunk_size and len(features) > args.chunk_size:
        total_rows = _write_chunked_training_rows(features, quote_frame, trading_days, args)
    else:
        final = build_flexible_exit_strategy_training_rows(
            features=features,
            quote_frame=quote_frame,
            trading_days=trading_days,
            config=CONFIG.labels,
            reversion_threshold=args.reversion_threshold,
            max_holding_days=args.max_holding_days,
        )
        write_table(final, CONFIG.paths.root / args.output)
        total_rows = len(final)
    print(f"Wrote {total_rows:,} flexible-exit strategy training rows to {args.output}")


if __name__ == "__main__":
    main()
