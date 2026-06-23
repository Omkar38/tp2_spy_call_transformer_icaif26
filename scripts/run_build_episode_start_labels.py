#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from tp2rr2.config import CONFIG
from tp2rr2.episode_start_labels import build_episode_start_labels
from tp2rr2.io import write_table


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def _read_required(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is not None:
        available = pq.read_schema(path).names
        missing = sorted(set(columns).difference(available))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
    return pd.read_parquet(path, columns=columns)


def _write_diagnostics(labels: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    counts = labels["label_end_state"].value_counts(dropna=False).rename_axis("label_end_state").reset_index(name="episode_count")
    counts["episode_share"] = counts["episode_count"] / len(labels)
    path = output_dir / "episode_start_label_counts.csv"
    counts.to_csv(path, index=False)
    paths["episode_start_label_counts"] = str(path)

    by_year = (
        labels.groupby(["start_year", "label_end_state"], dropna=False)
        .size()
        .rename("episode_count")
        .reset_index()
        .sort_values(["start_year", "label_end_state"])
    )
    totals = by_year.groupby("start_year")["episode_count"].transform("sum")
    by_year["episode_share_in_year"] = by_year["episode_count"] / totals
    path = output_dir / "episode_start_labels_by_year.csv"
    by_year.to_csv(path, index=False)
    paths["episode_start_labels_by_year"] = str(path)

    readiness = (
        labels.groupby("start_year", dropna=False)
        .agg(
            episode_starts=("episode_id", "size"),
            observable_next_state=("label_observable_next_state", "sum"),
            true_reversion=("label_true_reversion_next_session", "sum"),
            still_violation=("label_still_violation_next_session", "sum"),
            invalid_or_unobservable=("label_invalid_or_unobservable", "sum"),
            sample_censored=("label_sample_censored", "sum"),
            median_duration_sessions=("duration_sessions", "median"),
            mean_duration_sessions=("duration_sessions", "mean"),
        )
        .reset_index()
        .rename(columns={"start_year": "year"})
        .sort_values("year")
    )
    readiness["observable_share"] = readiness["observable_next_state"] / readiness["episode_starts"]
    readiness["true_reversion_share_all"] = readiness["true_reversion"] / readiness["episode_starts"]
    readiness["true_reversion_share_observable"] = readiness["true_reversion"] / readiness["observable_next_state"].replace(0, pd.NA)
    path = output_dir / "episode_start_ml_readiness_by_year.csv"
    readiness.to_csv(path, index=False)
    paths["episode_start_ml_readiness_by_year"] = str(path)

    by_month = (
        labels.groupby(["start_month", "label_quality_group"], dropna=False)
        .size()
        .rename("episode_count")
        .reset_index()
        .sort_values(["start_month", "label_quality_group"])
    )
    path = output_dir / "episode_start_label_quality_by_month.csv"
    by_month.to_csv(path, index=False)
    paths["episode_start_label_quality_by_month"] = str(path)

    split_counts = (
        labels.groupby(["split_by_signal_date", "label_quality_group"], dropna=False)
        .size()
        .rename("episode_count")
        .reset_index()
        .sort_values(["split_by_signal_date", "label_quality_group"])
    )
    path = output_dir / "episode_start_split_counts.csv"
    split_counts.to_csv(path, index=False)
    paths["episode_start_split_counts"] = str(path)

    examples = labels[
        [
            "episode_id",
            "signal_date",
            "start_violation_id",
            "feature_K1",
            "feature_K2",
            "feature_T1_expiration",
            "feature_T2_expiration",
            "feature_normalized_severity",
            "duration_sessions",
            "label_end_state",
            "label_end_state_detail",
            "label_next_trading_date",
        ]
    ].head(500)
    path = output_dir / "episode_start_label_examples.csv"
    examples.to_csv(path, index=False)
    paths["episode_start_label_examples"] = str(path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one ML-ready label row per TP2 pair episode start.")
    parser.add_argument(
        "--features-input",
        default="data/features/archive2_lastquote_forward/tp2_call_forward_features_lastquote_2013_2026.parquet",
    )
    parser.add_argument(
        "--episodes-input",
        default="data/features/archive2_episodes/tp2_call_pair_episodes_2013_2026.parquet",
    )
    parser.add_argument(
        "--end-states-input",
        default="data/features/archive2_episodes/tp2_call_pair_episode_end_states_2013_2026.parquet",
    )
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument(
        "--output",
        default="data/labels/archive2_episode_start_labels_2013_2026.parquet",
    )
    parser.add_argument(
        "--summary-output",
        default="data/labels/archive2_episode_start_label_summary_2013_2026.json",
    )
    parser.add_argument(
        "--diagnostics-dir",
        default="reports/raw_archive2_eda/tables/episode_start_labels",
    )
    parser.add_argument("--no-diagnostics", action="store_true")
    args = parser.parse_args()

    root = CONFIG.paths.root
    features_path = root / args.features_input
    episodes_path = root / args.episodes_input
    end_states_path = root / args.end_states_input

    print(f"Reading episodes from {args.episodes_input}", flush=True)
    episodes = _read_required(episodes_path)
    episodes["episode_start_date"] = pd.to_datetime(episodes["episode_start_date"])
    if args.years:
        episodes = episodes[episodes["episode_start_date"].dt.year.isin(args.years)].copy()
    print(f"Loaded {len(episodes):,} episode starts", flush=True)

    print(f"Reading end states from {args.end_states_input}", flush=True)
    end_states = _read_required(end_states_path)
    if args.years:
        end_states = end_states[end_states["episode_id"].isin(episodes["episode_id"])].copy()
    print(f"Loaded {len(end_states):,} end-state rows", flush=True)

    print(f"Reading detector start features from {args.features_input}", flush=True)
    features = _read_required(features_path)
    if args.years:
        needed = set(episodes["representative_violation_id"])
        features = features[features["violation_id"].isin(needed)].copy()
    print(f"Loaded {len(features):,} detector feature rows", flush=True)

    print("Building episode-start label dataset", flush=True)
    result = build_episode_start_labels(features, episodes, end_states)
    print(f"Built {len(result.labels):,} episode-start label rows", flush=True)

    output_path = root / args.output
    summary_path = root / args.summary_output
    print(f"Writing labels to {args.output}", flush=True)
    write_table(result.labels, output_path)

    summary = result.summary | {
        "features_input": args.features_input,
        "episodes_input": args.episodes_input,
        "end_states_input": args.end_states_input,
        "output": args.output,
        "summary_output": args.summary_output,
        "years_filter": args.years,
    }
    if not args.no_diagnostics:
        print(f"Writing diagnostics to {args.diagnostics_dir}", flush=True)
        summary["diagnostic_outputs"] = _write_diagnostics(result.labels, root / args.diagnostics_dir)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2))
    print(f"Writing summary to {args.summary_output}", flush=True)
    print(json.dumps(_json_ready(result.summary), indent=2))


if __name__ == "__main__":
    main()
