#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from tp2rr2.config import CONFIG
from tp2rr2.conservative_bidask import build_conservative_bidask_subset
from tp2rr2.io import write_table


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def _write_diagnostics(rows: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    by_year = (
        rows.assign(year=pd.to_datetime(rows["signal_date"]).dt.year)
        .groupby("year", dropna=False)
        .agg(
            conservative_bidask_starts=("episode_id", "size"),
            median_mid_normalized_severity=("feature_normalized_severity", "median"),
            mean_mid_normalized_severity=("feature_normalized_severity", "mean"),
            median_bidask_normalized_severity=("conservative_bidask_normalized_severity", "median"),
            mean_bidask_normalized_severity=("conservative_bidask_normalized_severity", "mean"),
            median_avg_spread_pct=("feature_avg_spread_pct", "median"),
            median_max_spread_pct=("feature_max_spread_pct", "median"),
            median_total_volume=("feature_total_volume", "median"),
            median_total_open_interest=("feature_total_open_interest", "median"),
        )
        .reset_index()
        .sort_values("year")
    )
    path = output_dir / "conservative_bidask_subset_by_year.csv"
    by_year.to_csv(path, index=False)
    paths["by_year"] = str(path)

    severity = rows[
        [
            "episode_id",
            "signal_date",
            "feature_normalized_severity",
            "conservative_bidask_normalized_severity",
            "conservative_bidask_lhs",
            "conservative_bidask_rhs",
            "feature_avg_spread_pct",
            "feature_max_spread_pct",
            "label_end_state",
        ]
    ].sort_values("conservative_bidask_normalized_severity", ascending=False)
    path = output_dir / "top_conservative_bidask_episodes.csv"
    severity.head(500).to_csv(path, index=False)
    paths["top_episodes"] = str(path)

    counts = rows["label_end_state"].value_counts(dropna=False).rename_axis("label_end_state").reset_index(name="episodes")
    counts["share"] = counts["episodes"] / len(rows)
    path = output_dir / "conservative_bidask_end_state_counts.csv"
    counts.to_csv(path, index=False)
    paths["end_state_counts"] = str(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build conservative bid/ask TP2 episode-start subset.")
    parser.add_argument(
        "--input",
        default="data/labels/archive2_episode_start_labels_2013_2026.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/labels/archive2_episode_start_labels_conservative_bidask_2013_2026.parquet",
    )
    parser.add_argument(
        "--summary-output",
        default="data/labels/archive2_episode_start_labels_conservative_bidask_summary_2013_2026.json",
    )
    parser.add_argument(
        "--diagnostics-dir",
        default="reports/raw_archive2_eda/tables/conservative_bidask_subset",
    )
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--no-diagnostics", action="store_true")
    args = parser.parse_args()

    root = CONFIG.paths.root
    input_path = root / args.input
    print(f"Reading episode-start labels from {args.input}", flush=True)
    labels = pd.read_parquet(input_path)
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    if args.years:
        labels = labels[labels["signal_date"].dt.year.isin(args.years)].copy()
    print(f"Loaded {len(labels):,} episode-start rows", flush=True)

    result = build_conservative_bidask_subset(labels)
    print(
        f"Kept {len(result.rows):,} conservative bid/ask TP2 rows "
        f"({result.summary['conservative_bidask_share']:.2%})",
        flush=True,
    )

    output_path = root / args.output
    write_table(result.rows, output_path)
    summary = result.summary | {
        "input": args.input,
        "output": args.output,
        "summary_output": args.summary_output,
        "years_filter": args.years,
    }
    if not args.no_diagnostics:
        summary["diagnostic_outputs"] = _write_diagnostics(result.rows, root / args.diagnostics_dir)

    summary_path = root / args.summary_output
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2))
    print(json.dumps(_json_ready(summary), indent=2), flush=True)
    print(f"Output rows from metadata: {pq.read_metadata(output_path).num_rows:,}", flush=True)


if __name__ == "__main__":
    main()
