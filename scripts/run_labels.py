#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pyarrow as pa
import pyarrow.parquet as pq

from tp2rr2.config import CONFIG
from tp2rr2.io import read_table, write_table
from tp2rr2.labels import build_strategy_training_rows, prepare_quote_frame


def _write_chunked_training_rows(features, options, args) -> int:
    output_path = CONFIG.paths.root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    quote_frame = prepare_quote_frame(options)
    trading_days = sorted(options["date"].drop_duplicates())
    writer = None
    total_rows = 0
    try:
        for start in range(0, len(features), args.chunk_size):
            stop = min(start + args.chunk_size, len(features))
            chunk = features.iloc[start:stop].copy()
            final = build_strategy_training_rows(
                chunk,
                quote_frame,
                trading_days,
                CONFIG.labels,
            )
            table = pa.Table.from_pandas(final, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            total_rows += len(final)
            if args.progress:
                print(
                    f"Wrote {total_rows:,} strategy rows from {stop:,}/{len(features):,} violations",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
    return total_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features/violations_features.parquet")
    parser.add_argument("--options", default="data/processed/options_clean.parquet")
    parser.add_argument("--max-violations", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", default="data/labels/strategy_training_rows.parquet")
    args = parser.parse_args()

    features = read_table(CONFIG.paths.root / args.features)
    if args.max_violations:
        features = features.head(args.max_violations)
    options = read_table(CONFIG.paths.root / args.options)
    if args.chunk_size and len(features) > args.chunk_size:
        total_rows = _write_chunked_training_rows(features, options, args)
    else:
        quote_frame = prepare_quote_frame(options)
        trading_days = sorted(options["date"].drop_duplicates())
        final = build_strategy_training_rows(features, quote_frame, trading_days, CONFIG.labels)
        write_table(final, CONFIG.paths.root / args.output)
        total_rows = len(final)
    print(f"Wrote {total_rows:,} strategy training rows to {args.output}")


if __name__ == "__main__":
    main()
