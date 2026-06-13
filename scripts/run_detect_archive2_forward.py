#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import pandas as pd

from tp2rr2.config import CONFIG, DetectionConfig
from tp2rr2.detect_forward_tp2 import detect_forward_tp2
from tp2rr2.features import build_features
from tp2rr2.io import read_table, write_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect forward-adjusted TP2 call violations for Archive-2 data.")
    parser.add_argument("--input", default="data/processed/archive2_spy_call_otm_clean_2024_2026.parquet")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=CONFIG.detection.n_jobs)
    parser.add_argument("--min-normalized-severity", type=float, default=CONFIG.detection.min_normalized_severity)
    parser.add_argument("--max-strikes-per-expiration", type=int, default=CONFIG.detection.max_strikes_per_expiration)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", default="data/features/archive2_forward/tp2_call_forward_features_2024_2026.parquet")
    parser.add_argument("--summary-output", default="data/features/archive2_forward/tp2_call_forward_summary_2024_2026.json")
    args = parser.parse_args()

    options = read_table(CONFIG.paths.root / args.input)
    if args.start_date:
        options = options[options["date"] >= pd.Timestamp(args.start_date)]
    if args.end_date:
        options = options[options["date"] <= pd.Timestamp(args.end_date)]
    if args.max_days:
        keep_dates = sorted(options["date"].drop_duplicates())[: args.max_days]
        options = options[options["date"].isin(keep_dates)]

    detection_config = DetectionConfig(
        min_normalized_severity=args.min_normalized_severity,
        max_expirations_per_day=CONFIG.detection.max_expirations_per_day,
        max_strikes_per_expiration=args.max_strikes_per_expiration,
        strike_selection=CONFIG.detection.strike_selection,
        n_jobs=args.n_jobs,
    )
    violations = detect_forward_tp2(options, detection_config, progress=args.progress)
    features = build_features(violations, options)
    write_table(features, CONFIG.paths.root / args.output)

    summary = {
        "input": args.input,
        "output": args.output,
        "rows_input": int(len(options)),
        "sessions_input": int(options["date"].nunique()) if not options.empty else 0,
        "rows_violations": int(len(violations)),
        "rows_features": int(len(features)),
        "sessions_with_violations": int(features["date"].nunique()) if not features.empty else 0,
        "min_normalized_severity": args.min_normalized_severity,
        "max_strikes_per_expiration": args.max_strikes_per_expiration,
        "detection_method": "forward_adjusted_round_up",
    }
    summary_path = CONFIG.paths.root / args.summary_output
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {len(features):,} forward-adjusted violation feature rows to {args.output}")
    print(f"Wrote summary to {args.summary_output}")


if __name__ == "__main__":
    main()
