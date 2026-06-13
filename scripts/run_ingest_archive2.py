#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tp2rr2.archive2_ingest import Archive2CleanConfig, clean_archive2_year
from tp2rr2.config import CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean professor-provided Archive-2 SPY option panel.")
    parser.add_argument("--archive2-root", default="Archive-2/spy_options_daily")
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2025, 2026])
    parser.add_argument("--output", default="data/processed/archive2_spy_call_otm_clean_2024_2026.parquet")
    parser.add_argument("--summary-output", default="data/processed/archive2_spy_call_otm_clean_2024_2026_summary.json")
    parser.add_argument("--min-dte", type=int, default=1)
    parser.add_argument("--max-dte", type=int, default=90)
    parser.add_argument("--min-quote-time", default="15:55")
    parser.add_argument(
        "--use-last-available-quote",
        action="store_true",
        help="Do not require a minimum quote time; after raw duplicate collapse, keep the latest available quote for each contract-day.",
    )
    parser.add_argument("--max-spread-pct", type=float, default=0.50)
    parser.add_argument("--include-atm-itm", action="store_true")
    parser.add_argument("--allow-puts", action="store_true")
    parser.add_argument("--require-positive-size", action="store_true")
    parser.add_argument("--require-positive-open-interest", action="store_true")
    parser.add_argument("--require-positive-volume", action="store_true")
    parser.add_argument("--exclude-rate-fallback", action="store_true")
    args = parser.parse_args()

    hour, minute = (int(part) for part in args.min_quote_time.split(":", 1))
    config = Archive2CleanConfig(
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        calls_only=not args.allow_puts,
        otm_calls_only=not args.include_atm_itm,
        use_last_available_quote=args.use_last_available_quote,
        min_quote_hour=hour,
        min_quote_minute=minute,
        max_spread_pct=args.max_spread_pct,
        require_positive_size=args.require_positive_size,
        require_positive_open_interest=args.require_positive_open_interest,
        require_positive_volume=args.require_positive_volume,
        exclude_rate_fallback=args.exclude_rate_fallback,
    )

    archive_root = CONFIG.paths.root / args.archive2_root
    output_path = CONFIG.paths.root / args.output
    summary_path = CONFIG.paths.root / args.summary_output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    summaries = []
    total_rows = 0
    try:
        for year in args.years:
            cleaned, summary = clean_archive2_year(archive_root, year, config)
            summaries.append(summary)
            total_rows += len(cleaned)
            table = pa.Table.from_pandas(cleaned, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            if len(cleaned):
                writer.write_table(table)
            print(
                f"{year}: raw={summary['raw_rows']:,} deduped={summary['deduped_rows']:,} "
                f"clean={summary['clean_rows']:,} sessions={summary['clean_sessions']:,}",
                flush=True,
            )
    finally:
        if writer is not None:
            writer.close()

    report = {
        "archive2_root": str(archive_root),
        "output": str(output_path),
        "config": asdict(config),
        "years": summaries,
        "total_clean_rows": int(total_rows),
        "total_raw_rows": int(sum(item["raw_rows"] for item in summaries)),
        "total_deduped_rows": int(sum(item["deduped_rows"] for item in summaries)),
        "total_duplicate_rows_removed": int(sum(item["duplicate_rows_removed"] for item in summaries)),
    }
    summary_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {total_rows:,} cleaned rows to {output_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
