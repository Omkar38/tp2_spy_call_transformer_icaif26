#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from tp2rr2.config import CONFIG
from tp2rr2.config import DetectionConfig
from tp2rr2.detect_violations import detect_violations
from tp2rr2.features import build_features
from tp2rr2.io import read_table, write_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/options_clean.parquet")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=CONFIG.detection.n_jobs)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", default="data/features/violations_features.parquet")
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
        min_normalized_severity=CONFIG.detection.min_normalized_severity,
        max_expirations_per_day=CONFIG.detection.max_expirations_per_day,
        max_strikes_per_expiration=CONFIG.detection.max_strikes_per_expiration,
        strike_selection=CONFIG.detection.strike_selection,
        n_jobs=args.n_jobs,
    )
    violations = detect_violations(options, detection_config, progress=args.progress)
    features = build_features(violations, options)
    write_table(features, CONFIG.paths.root / args.output)
    print(f"Wrote {len(features):,} violation feature rows to {args.output}")


if __name__ == "__main__":
    main()
