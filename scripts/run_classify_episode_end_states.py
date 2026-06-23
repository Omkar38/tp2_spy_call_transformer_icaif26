#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from tp2rr2.config import CONFIG
from tp2rr2.episode_end_states import CLEAN_COLUMNS, EPISODE_COLUMNS, classify_episode_end_states
from tp2rr2.io import write_table


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def _read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    available = pq.read_schema(path).names
    missing = sorted(set(columns).difference(available))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return pd.read_parquet(path, columns=columns)


def _write_diagnostics(end_states: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    counts = end_states["end_state"].value_counts(dropna=False).rename_axis("end_state").reset_index(name="episode_count")
    counts["episode_share"] = counts["episode_count"] / len(end_states)
    path = output_dir / "end_state_counts.csv"
    counts.to_csv(path, index=False)
    paths["end_state_counts"] = str(path)

    by_year = (
        end_states.groupby(["end_year", "end_state"], dropna=False)
        .size()
        .rename("episode_count")
        .reset_index()
        .rename(columns={"end_year": "year"})
        .sort_values(["year", "end_state"])
    )
    totals = by_year.groupby("year")["episode_count"].transform("sum")
    by_year["episode_share_in_year"] = by_year["episode_count"] / totals
    path = output_dir / "end_state_counts_by_year.csv"
    by_year.to_csv(path, index=False)
    paths["end_state_counts_by_year"] = str(path)

    true_reversion = end_states[end_states["end_state"].eq("true_reversion")].copy()
    if not true_reversion.empty:
        severity = true_reversion[
            [
                "episode_id",
                "episode_end_date",
                "next_trading_date",
                "end_normalized_severity",
                "next_normalized_severity",
                "next_lhs",
                "next_rhs",
                "pair_key",
            ]
        ].sort_values(["episode_end_date", "episode_id"]).head(200)
    else:
        severity = pd.DataFrame()
    path = output_dir / "true_reversion_examples.csv"
    severity.to_csv(path, index=False)
    paths["true_reversion_examples"] = str(path)

    failures = end_states[
        end_states["end_state"].isin(
            ["left_leg_expired_or_missing", "right_leg_missing_or_expired", "liquidity_filter_failure"]
        )
    ][["episode_id", "episode_end_date", "next_trading_date", "end_state", "end_state_detail", "pair_key"]].head(500)
    path = output_dir / "non_reversion_examples.csv"
    failures.to_csv(path, index=False)
    paths["non_reversion_examples"] = str(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify next-session end states for pair-level TP2 episodes.")
    parser.add_argument(
        "--episodes-input",
        default="data/features/archive2_episodes/tp2_call_pair_episodes_2013_2026.parquet",
    )
    parser.add_argument(
        "--clean-input",
        default="data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet",
    )
    parser.add_argument("--archive-root", default="Archive-2/spy_options_daily")
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--max-spread-pct", type=float, default=0.50)
    parser.add_argument("--min-normalized-severity", type=float, default=0.0001)
    parser.add_argument(
        "--output",
        default="data/features/archive2_episodes/tp2_call_pair_episode_end_states_2013_2026.parquet",
    )
    parser.add_argument(
        "--summary-output",
        default="data/features/archive2_episodes/tp2_call_pair_episode_end_state_summary_2013_2026.json",
    )
    parser.add_argument(
        "--diagnostics-dir",
        default="reports/raw_archive2_eda/tables/episode_end_states",
    )
    parser.add_argument("--no-diagnostics", action="store_true")
    args = parser.parse_args()

    episodes_path = CONFIG.paths.root / args.episodes_input
    clean_path = CONFIG.paths.root / args.clean_input
    archive_root = CONFIG.paths.root / args.archive_root

    print(f"Reading episodes from {args.episodes_input}", flush=True)
    episodes = _read_columns(episodes_path, EPISODE_COLUMNS)
    for col in ["episode_start_date", "episode_end_date", "next_trading_date"]:
        episodes[col] = pd.to_datetime(episodes[col])
    if args.years:
        episodes = episodes[episodes["episode_end_date"].dt.year.isin(args.years)].copy()
    print(f"Loaded {len(episodes):,} episodes", flush=True)

    print(f"Reading clean options from {args.clean_input}", flush=True)
    clean = _read_columns(clean_path, CLEAN_COLUMNS)
    clean["date"] = pd.to_datetime(clean["date"])
    if args.years:
        needed_years = set(episodes["next_trading_date"].dropna().dt.year.astype(int).unique())
        clean = clean[clean["date"].dt.year.isin(needed_years)].copy()
    print(f"Loaded {len(clean):,} clean option rows for required dates", flush=True)

    print("Classifying episode end states", flush=True)
    result = classify_episode_end_states(
        episodes,
        clean,
        archive_root=archive_root,
        max_spread_pct=args.max_spread_pct,
        min_normalized_severity=args.min_normalized_severity,
    )
    print(f"Classified {len(result.end_states):,} episodes", flush=True)

    output_path = CONFIG.paths.root / args.output
    summary_path = CONFIG.paths.root / args.summary_output
    print(f"Writing end states to {args.output}", flush=True)
    write_table(result.end_states, output_path)

    summary = result.summary | {
        "episodes_input": args.episodes_input,
        "clean_input": args.clean_input,
        "archive_root": args.archive_root,
        "output": args.output,
        "summary_output": args.summary_output,
        "years_filter": args.years,
    }
    if not args.no_diagnostics:
        print(f"Writing diagnostics to {args.diagnostics_dir}", flush=True)
        summary["diagnostic_outputs"] = _write_diagnostics(
            result.end_states, CONFIG.paths.root / args.diagnostics_dir
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2))
    print(f"Writing summary to {args.summary_output}", flush=True)
    print(json.dumps(_json_ready(result.summary), indent=2))


if __name__ == "__main__":
    main()
