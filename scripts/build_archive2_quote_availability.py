#!/usr/bin/env python3
"""Build call quote-availability panel without the 50% spread cutoff.

This derived dataset is for lifecycle/behavior analysis. It keeps otherwise
valid call quotes even when the bid/ask spread is wider than the active trading
clean-data threshold, and stores a tradability flag instead of dropping them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tp2rr2.archive2_ingest import (
    Archive2CleanConfig,
    apply_research_filters,
    clean_archive2_year,
)
from tp2rr2.config import CONFIG


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Archive-2 call quote availability without spread cutoff."
    )
    parser.add_argument(
        "--archive-root",
        default="Archive-2/spy_options_daily",
    )
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2013, 2027)))
    parser.add_argument("--min-dte", type=int, default=1)
    parser.add_argument(
        "--max-dte",
        type=int,
        default=120,
        help="Use >90 so lifecycle lookback can see pre-signal T1/T2 quotes.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/archive2_spy_call_quote_availability_nospread_2013_2026.parquet",
    )
    parser.add_argument(
        "--summary-output",
        default="data/processed/archive2_spy_call_quote_availability_nospread_2013_2026_summary.json",
    )
    args = parser.parse_args()

    root = CONFIG.paths.root
    archive_root = root / args.archive_root
    output_path = root / args.output
    summary_path = root / args.summary_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    config = Archive2CleanConfig(
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        calls_only=True,
        otm_calls_only=False,
        use_last_available_quote=True,
        max_spread_pct=None,
        require_bid_ask=True,
        require_positive_bid=True,
        require_positive_size=False,
        require_positive_open_interest=False,
        require_positive_volume=False,
    )

    writer: pq.ParquetWriter | None = None
    summaries = []
    total_rows = 0
    try:
        for year in args.years:
            cleaned, summary = clean_archive2_year(archive_root, year, config)
            cleaned["tradable_under_50pct_spread"] = cleaned["spread_pct"] <= 0.50
            cleaned["wide_spread_over_50pct"] = cleaned["spread_pct"] > 0.50
            cleaned["quote_availability_universe"] = "valid_bidask_positive_bid_no_spread_cutoff"

            keep_cols = [
                "date",
                "expiration",
                "contractID",
                "contract_id",
                "option_symbol",
                "type",
                "strike",
                "bid",
                "ask",
                "mid",
                "spread",
                "spread_pct",
                "tradable_under_50pct_spread",
                "wide_spread_over_50pct",
                "bid_size",
                "ask_size",
                "volume",
                "open_interest",
                "implied_volatility",
                "delta",
                "theta",
                "vega",
                "forward",
                "risk_free_rate",
                "underlying_price",
                "dte",
                "quote_time",
                "quote_availability_universe",
            ]
            cleaned = cleaned[[c for c in keep_cols if c in cleaned.columns]].copy()
            table = pa.Table.from_pandas(cleaned, preserve_index=False)
            if writer is None:
                if output_path.exists():
                    output_path.unlink()
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)

            year_summary = {
                **summary,
                "quote_availability_rows": int(len(cleaned)),
                "wide_spread_over_50pct_rows": int(cleaned["wide_spread_over_50pct"].sum()),
                "wide_spread_over_50pct_share": float(cleaned["wide_spread_over_50pct"].mean())
                if len(cleaned)
                else 0.0,
                "min_dte": args.min_dte,
                "max_dte": args.max_dte,
            }
            summaries.append(year_summary)
            total_rows += len(cleaned)
            print(
                f"{year}: quote_availability={len(cleaned):,} "
                f"wide_spread>{0.50:.0%}={year_summary['wide_spread_over_50pct_rows']:,}",
                flush=True,
            )
    finally:
        if writer is not None:
            writer.close()

    report = {
        "input": args.archive_root,
        "output": args.output,
        "summary_output": args.summary_output,
        "purpose": "Lifecycle quote availability without hard 50% spread deletion.",
        "config": {
            "min_dte": args.min_dte,
            "max_dte": args.max_dte,
            "calls_only": True,
            "use_last_available_quote": True,
            "require_bid_ask": True,
            "require_positive_bid": True,
            "max_spread_pct": None,
        },
        "total_rows": int(total_rows),
        "wide_spread_over_50pct_rows": int(
            sum(item["wide_spread_over_50pct_rows"] for item in summaries)
        ),
        "years": summaries,
    }
    summary_path.write_text(json.dumps(_json_ready(report), indent=2))
    print(f"Wrote {total_rows:,} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
