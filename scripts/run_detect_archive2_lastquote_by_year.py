#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tp2rr2.config import CONFIG, DetectionConfig
from tp2rr2.detect_forward_tp2 import detect_forward_tp2
from tp2rr2.features import build_features
from tp2rr2.io import read_table, write_table


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
        description="Run forward-adjusted TP2 detection year by year on Archive-2 last-available quotes."
    )
    parser.add_argument(
        "--input",
        default="data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet",
    )
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2013, 2027)))
    parser.add_argument("--output-dir", default="data/features/archive2_lastquote_forward")
    parser.add_argument(
        "--combined-output",
        default="data/features/archive2_lastquote_forward/tp2_call_forward_features_lastquote_2013_2026.parquet",
    )
    parser.add_argument(
        "--summary-output",
        default="data/features/archive2_lastquote_forward/tp2_call_forward_summary_lastquote_2013_2026.json",
    )
    parser.add_argument("--n-jobs", type=int, default=CONFIG.detection.n_jobs)
    parser.add_argument("--min-normalized-severity", type=float, default=CONFIG.detection.min_normalized_severity)
    parser.add_argument("--max-strikes-per-expiration", type=int, default=CONFIG.detection.max_strikes_per_expiration)
    parser.add_argument("--no-combined-output", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    input_path = CONFIG.paths.root / args.input
    output_dir = CONFIG.paths.root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    options = read_table(input_path)
    options["date"] = pd.to_datetime(options["date"])
    options["expiration"] = pd.to_datetime(options["expiration"])
    options["year"] = options["date"].dt.year

    detection_config = DetectionConfig(
        min_normalized_severity=args.min_normalized_severity,
        max_expirations_per_day=CONFIG.detection.max_expirations_per_day,
        max_strikes_per_expiration=args.max_strikes_per_expiration,
        strike_selection=CONFIG.detection.strike_selection,
        n_jobs=args.n_jobs,
    )

    combined_writer: pq.ParquetWriter | None = None
    combined_path = CONFIG.paths.root / args.combined_output
    if not args.no_combined_output:
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        if combined_path.exists():
            combined_path.unlink()

    summaries = []
    total_rows = 0
    try:
        for year in args.years:
            year_options = options[options["year"].eq(year)].drop(columns=["year"]).copy()
            if year_options.empty:
                summary = {
                    "year": year,
                    "rows_input": 0,
                    "sessions_input": 0,
                    "rows_features": 0,
                    "sessions_with_violations": 0,
                    "output": None,
                    "status": "empty_input",
                }
                summaries.append(summary)
                print(f"{year}: empty input", flush=True)
                continue

            violations = detect_forward_tp2(year_options, detection_config, progress=args.progress)
            features = build_features(violations, year_options)
            year_output = output_dir / f"tp2_call_forward_features_lastquote_{year}.parquet"
            write_table(features, year_output)
            if not args.no_combined_output and not features.empty:
                table = pa.Table.from_pandas(features, preserve_index=False)
                if combined_writer is None:
                    combined_writer = pq.ParquetWriter(combined_path, table.schema)
                combined_writer.write_table(table)

            summary = {
                "year": year,
                "rows_input": int(len(year_options)),
                "sessions_input": int(year_options["date"].nunique()),
                "rows_violations": int(len(violations)),
                "rows_features": int(len(features)),
                "sessions_with_violations": int(features["date"].nunique()) if not features.empty else 0,
                "output": str(year_output.relative_to(CONFIG.paths.root)),
                "status": "completed",
            }
            summaries.append(summary)
            total_rows += len(features)
            print(
                f"{year}: input={len(year_options):,} sessions={summary['sessions_input']:,} "
                f"features={len(features):,} violation_sessions={summary['sessions_with_violations']:,}",
                flush=True,
            )
    finally:
        if combined_writer is not None:
            combined_writer.close()

    report = {
        "input": args.input,
        "output_dir": args.output_dir,
        "combined_output": None if args.no_combined_output else args.combined_output,
        "summary_output": args.summary_output,
        "detection_method": "forward_adjusted_round_up",
        "min_normalized_severity": args.min_normalized_severity,
        "max_strikes_per_expiration": args.max_strikes_per_expiration,
        "years": summaries,
        "total_rows_features": int(total_rows),
        "total_sessions_input": int(options[options["year"].isin(args.years)]["date"].nunique()),
        "total_sessions_with_violations": int(
            sum(item.get("sessions_with_violations", 0) for item in summaries)
        ),
    }
    summary_path = CONFIG.paths.root / args.summary_output
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_json_ready(report), indent=2))
    print(f"Wrote {total_rows:,} total feature rows")
    print(f"Wrote summary to {args.summary_output}")


if __name__ == "__main__":
    main()
