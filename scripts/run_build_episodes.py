#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from tp2rr2.config import CONFIG
from tp2rr2.episodes import EPISODE_INPUT_COLUMNS, build_pair_episodes
from tp2rr2.io import write_table


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def _read_existing_columns(path: Path, wanted: list[str]) -> pd.DataFrame:
    available = pq.read_schema(path).names
    missing = sorted(set(wanted).difference(available))
    if missing:
        raise ValueError(f"Input feature file is missing required episode columns: {missing}")
    return pd.read_parquet(path, columns=wanted)


def _write_diagnostics(episodes: pd.DataFrame, members: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    by_year = (
        episodes.groupby("start_year", dropna=False)
        .agg(
            episode_starts=("episode_id", "size"),
            daily_rows_in_started_episodes=("episode_row_count", "sum"),
            median_duration_sessions=("duration_sessions", "median"),
            mean_duration_sessions=("duration_sessions", "mean"),
            single_day_episodes=("duration_sessions", lambda s: int((s == 1).sum())),
            exact_rectangle_change_episodes=("has_exact_rectangle_change", "sum"),
        )
        .reset_index()
        .rename(columns={"start_year": "year"})
    )
    member_rows_by_year = members["date"].dt.year.value_counts().sort_index().rename_axis("year").reset_index(name="daily_violation_rows")
    by_year = by_year.merge(member_rows_by_year, on="year", how="outer").sort_values("year")
    by_year["daily_rows_per_episode_start"] = by_year["daily_violation_rows"] / by_year["episode_starts"]
    path = output_dir / "episode_starts_by_year.csv"
    by_year.to_csv(path, index=False)
    paths["episode_starts_by_year"] = str(path)

    by_month = (
        episodes.groupby("start_month", dropna=False)
        .agg(
            episode_starts=("episode_id", "size"),
            daily_rows_in_started_episodes=("episode_row_count", "sum"),
            median_duration_sessions=("duration_sessions", "median"),
            exact_rectangle_change_episodes=("has_exact_rectangle_change", "sum"),
        )
        .reset_index()
        .rename(columns={"start_month": "month"})
        .sort_values("month")
    )
    path = output_dir / "episode_starts_by_month.csv"
    by_month.to_csv(path, index=False)
    paths["episode_starts_by_month"] = str(path)

    duration = episodes["duration_sessions"].value_counts().sort_index().rename_axis("duration_sessions").reset_index(name="episode_count")
    path = output_dir / "episode_duration_distribution.csv"
    duration.to_csv(path, index=False)
    paths["episode_duration_distribution"] = str(path)

    top_long = episodes.sort_values(
        ["duration_sessions", "episode_row_count", "max_normalized_severity"],
        ascending=[False, False, False],
    ).head(100)
    path = output_dir / "top_longest_episodes.csv"
    top_long.to_csv(path, index=False)
    paths["top_longest_episodes"] = str(path)

    repeat_pairs = (
        episodes.groupby("pair_key")
        .agg(
            episode_count=("episode_id", "size"),
            first_episode_start=("episode_start_date", "min"),
            last_episode_end=("episode_end_date", "max"),
            total_episode_rows=("episode_row_count", "sum"),
        )
        .query("episode_count > 1")
        .sort_values(["episode_count", "total_episode_rows"], ascending=[False, False])
        .head(100)
        .reset_index()
    )
    path = output_dir / "top_reappearing_pair_keys.csv"
    repeat_pairs.to_csv(path, index=False)
    paths["top_reappearing_pair_keys"] = str(path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pair-level TP2 violation episodes from daily detector rows.")
    parser.add_argument(
        "--input",
        default="data/features/archive2_lastquote_forward/tp2_call_forward_features_lastquote_2013_2026.parquet",
    )
    parser.add_argument(
        "--clean-input",
        default="data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet",
    )
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument(
        "--episodes-output",
        default="data/features/archive2_episodes/tp2_call_pair_episodes_2013_2026.parquet",
    )
    parser.add_argument(
        "--members-output",
        default="data/features/archive2_episodes/tp2_call_episode_members_2013_2026.parquet",
    )
    parser.add_argument(
        "--summary-output",
        default="data/features/archive2_episodes/tp2_call_episode_summary_2013_2026.json",
    )
    parser.add_argument(
        "--diagnostics-dir",
        default="reports/raw_archive2_eda/tables/episodes",
    )
    parser.add_argument("--no-diagnostics", action="store_true")
    args = parser.parse_args()

    input_path = CONFIG.paths.root / args.input
    clean_path = CONFIG.paths.root / args.clean_input
    print(f"Reading violation features from {args.input}", flush=True)
    violations = _read_existing_columns(input_path, EPISODE_INPUT_COLUMNS)
    violations["date"] = pd.to_datetime(violations["date"])
    if args.years:
        violations = violations[violations["date"].dt.year.isin(args.years)].copy()
    print(f"Loaded {len(violations):,} daily violation rows", flush=True)

    print(f"Reading clean trading calendar from {args.clean_input}", flush=True)
    clean = pd.read_parquet(clean_path, columns=["date"])
    clean["date"] = pd.to_datetime(clean["date"])
    print(f"Loaded {clean['date'].nunique():,} clean trading sessions", flush=True)

    print("Building pair episodes", flush=True)
    result = build_pair_episodes(violations, clean, input_path=args.input)
    print(f"Built {len(result.episodes):,} pair episodes", flush=True)

    episodes_path = CONFIG.paths.root / args.episodes_output
    members_path = CONFIG.paths.root / args.members_output
    summary_path = CONFIG.paths.root / args.summary_output
    print(f"Writing episodes to {args.episodes_output}", flush=True)
    write_table(result.episodes, episodes_path)
    print(f"Writing episode members to {args.members_output}", flush=True)
    write_table(result.members, members_path)

    summary = result.summary | {
        "clean_input": args.clean_input,
        "episodes_output": args.episodes_output,
        "members_output": args.members_output,
        "summary_output": args.summary_output,
        "years_filter": args.years,
        "primary_episode_key": "A_expiration|A_strike|B_expiration|B_strike",
        "diagnostic_rectangle_key": "A/B/C/D expirations and strikes",
    }
    if not args.no_diagnostics:
        print(f"Writing diagnostics to {args.diagnostics_dir}", flush=True)
        diagnostics = _write_diagnostics(result.episodes, result.members, CONFIG.paths.root / args.diagnostics_dir)
        summary["diagnostic_outputs"] = diagnostics

    print(f"Writing summary to {args.summary_output}", flush=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2))

    print(f"Input daily violation rows: {len(result.members):,}")
    print(f"Pair episodes: {len(result.episodes):,}")
    print(f"Wrote episodes to {args.episodes_output}")
    print(f"Wrote episode members to {args.members_output}")
    print(f"Wrote summary to {args.summary_output}")


if __name__ == "__main__":
    main()
