#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from tp2rr2.clean_quotes import clean_options
from tp2rr2.config import CONFIG
from tp2rr2.data_loader import load_archive, normalize_raw_options
from tp2rr2.io import ensure_dirs, write_table
from tp2rr2.underlying import attach_underlying, load_spy_close


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="*", type=int, default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output", default="data/processed/options_clean.parquet")
    args = parser.parse_args()

    ensure_dirs([CONFIG.paths.processed, CONFIG.paths.raw])
    raw = load_archive(CONFIG.paths.archive, years=args.years)
    normalized = normalize_raw_options(raw)
    if args.start_date:
        normalized = normalized[normalized["date"] >= pd.Timestamp(args.start_date)]
    if args.end_date:
        normalized = normalized[normalized["date"] <= pd.Timestamp(args.end_date)]
    cleaned = clean_options(normalized, CONFIG.clean)
    official_close = load_spy_close(CONFIG.paths.spy_close)
    with_underlying = attach_underlying(cleaned, official_close)
    write_table(with_underlying, CONFIG.paths.root / args.output)
    print(f"Wrote {len(with_underlying):,} clean option rows to {args.output}")


if __name__ == "__main__":
    main()
