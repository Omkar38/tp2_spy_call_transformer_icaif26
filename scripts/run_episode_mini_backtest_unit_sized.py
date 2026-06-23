#!/usr/bin/env python3
"""Runner for unit-sized mini-backtest (1 contract per selected leg).

Reads from the conservative bid/ask episode-start labels and the clean quote
frame. Writes a new parquet + JSON summary + diagnostics to NEW paths only.
Nothing in the existing pipeline is overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tp2rr2.config import CONFIG
from tp2rr2.episode_mini_backtest import (
    prepare_episode_quote_frame,
    required_episode_label_columns,
    STRATEGIES,
)
from tp2rr2.episode_mini_backtest_unit import build_unit_sized_backtest_rows


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


def _write_diagnostics(rows: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    counts = (
        rows["exit_reason"]
        .value_counts(dropna=False)
        .rename_axis("exit_reason")
        .reset_index(name="strategy_rows")
    )
    counts["row_share"] = counts["strategy_rows"] / len(rows)
    path = output_dir / "unit_backtest_exit_reason_counts.csv"
    counts.to_csv(path, index=False)
    paths["exit_reason_counts"] = str(path)

    by_strategy = (
        rows.groupby("strategy", dropna=False)
        .agg(
            candidate_rows=("episode_id", "size"),
            valid_rows=("is_valid_backtest", "sum"),
            valid_share=("is_valid_backtest", "mean"),
            mean_return=("realized_return", "mean"),
            median_return=("realized_return", "median"),
            net_pnl=("net_pnl", "sum"),
            transaction_cost=("transaction_cost", "sum"),
            mean_holding_days=("actual_holding_days", "mean"),
        )
        .reset_index()
        .sort_values("strategy")
    )
    path = output_dir / "unit_backtest_by_strategy.csv"
    by_strategy.to_csv(path, index=False)
    paths["by_strategy"] = str(path)

    by_year = (
        rows.groupby(
            [rows["signal_date"].dt.year.rename("year"), "strategy"], dropna=False
        )
        .agg(
            candidate_rows=("episode_id", "size"),
            valid_rows=("is_valid_backtest", "sum"),
            valid_share=("is_valid_backtest", "mean"),
            net_pnl=("net_pnl", "sum"),
            mean_return=("realized_return", "mean"),
            median_return=("realized_return", "median"),
            mean_entry_gross=("entry_gross_exposure", "mean"),
            median_entry_gross=("entry_gross_exposure", "median"),
        )
        .reset_index()
        .sort_values(["year", "strategy"])
    )
    path = output_dir / "unit_backtest_by_year_strategy.csv"
    by_year.to_csv(path, index=False)
    paths["by_year_strategy"] = str(path)

    by_month = (
        rows.assign(month=rows["signal_date"].dt.to_period("M").astype(str))
        .groupby(["month", "strategy"], dropna=False)
        .agg(
            candidate_rows=("episode_id", "size"),
            valid_rows=("is_valid_backtest", "sum"),
            net_pnl=("net_pnl", "sum"),
            mean_return=("realized_return", "mean"),
        )
        .reset_index()
        .sort_values(["month", "strategy"])
    )
    path = output_dir / "unit_backtest_by_month_strategy.csv"
    by_month.to_csv(path, index=False)
    paths["by_month_strategy"] = str(path)

    valid = rows[rows["is_valid_backtest"]].copy()
    if not valid.empty:
        top = valid.sort_values("net_pnl", ascending=False).head(500)
        bottom = valid.sort_values("net_pnl", ascending=True).head(500)
    else:
        top = pd.DataFrame()
        bottom = pd.DataFrame()
    path = output_dir / "unit_backtest_top_valid_pnl.csv"
    top.to_csv(path, index=False)
    paths["top_valid_pnl"] = str(path)
    path = output_dir / "unit_backtest_bottom_valid_pnl.csv"
    bottom.to_csv(path, index=False)
    paths["bottom_valid_pnl"] = str(path)

    return paths


def _combine_summaries(summaries: list[dict], output: str, labels_input: str, clean_input: str) -> dict:
    total_rows = sum(item["strategy_rows"] for item in summaries)
    valid_rows = sum(item["valid_backtest_rows"] for item in summaries)
    exit_counts: dict[str, int] = {}
    by_strategy: dict[str, dict] = {}
    for item in summaries:
        for reason, count in item["exit_reason_counts"].items():
            exit_counts[reason] = exit_counts.get(reason, 0) + int(count)
        for strategy, stats in item["by_strategy"].items():
            target = by_strategy.setdefault(
                strategy,
                {"candidate_rows": 0, "valid_rows": 0, "net_pnl": 0.0, "transaction_cost": 0.0},
            )
            target["candidate_rows"] += int(stats["candidate_rows"])
            target["valid_rows"]     += int(stats["valid_rows"])
            target["net_pnl"]        += float(stats["net_pnl"])
            target["transaction_cost"] += float(stats["transaction_cost"])

    return {
        "output": output,
        "labels_input": labels_input,
        "clean_input": clean_input,
        "sizing": "unit_contract_per_leg",
        "strategy_rows": int(total_rows),
        "strategies": STRATEGIES,
        "strategies_per_episode": len(STRATEGIES),
        "unique_episodes_expected": int(total_rows // len(STRATEGIES)) if STRATEGIES else 0,
        "valid_backtest_rows": int(valid_rows),
        "valid_backtest_share": float(valid_rows / total_rows) if total_rows else 0.0,
        "exit_reason_counts": exit_counts,
        "by_strategy": by_strategy,
        "policy_note": (
            "Unit-sized backtest: 1 contract per selected leg. "
            "Exit at true-reversion next-state. Commission $0.65/side."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build unit-sized episode mini-backtest (1 contract per leg)."
    )
    parser.add_argument(
        "--labels-input",
        default="data/labels/archive2_episode_start_labels_conservative_bidask_2013_2026.parquet",
    )
    parser.add_argument(
        "--clean-input",
        default="data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet",
    )
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--output",
        default="data/labels/archive2_episode_strategy_mini_backtest_unit_conservative_bidask_2013_2026.parquet",
    )
    parser.add_argument(
        "--summary-output",
        default="data/labels/archive2_episode_strategy_mini_backtest_unit_conservative_bidask_summary_2013_2026.json",
    )
    parser.add_argument(
        "--diagnostics-dir",
        default="reports/raw_archive2_eda/tables/episode_mini_backtest_unit_conservative_bidask",
    )
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--progress", action="store_true")
    # Optional filters
    parser.add_argument("--min-abs-delta", type=float, default=0.0,
                        help="Minimum abs(delta) for all selected legs (0=no filter).")
    parser.add_argument("--max-abs-delta", type=float, default=1.0,
                        help="Maximum abs(delta) for all selected legs (1=no filter).")
    parser.add_argument("--max-entry-gross", type=float, default=None,
                        help="Drop episodes where entry gross > this value (USD). Default: no cap.")
    args = parser.parse_args()

    root = CONFIG.paths.root
    labels_path = root / args.labels_input
    clean_path  = root / args.clean_input

    print(f"Reading episode-start labels from {args.labels_input}", flush=True)
    labels = _read_columns(labels_path, required_episode_label_columns())
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    if args.years:
        labels = labels[labels["signal_date"].dt.year.isin(args.years)].copy()
    if args.max_episodes:
        labels = labels.head(args.max_episodes).copy()
    print(f"Loaded {len(labels):,} conservative episode starts", flush=True)

    print(f"Reading clean option quotes from {args.clean_input}", flush=True)
    clean_cols = ["date", "contractID", "option_symbol", "bid", "ask", "mid", "spread_pct"]
    clean = _read_columns(clean_path, clean_cols)
    quote_frame = prepare_episode_quote_frame(clean)
    print(f"Loaded {len(quote_frame):,} clean quote rows", flush=True)

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"WARNING: output already exists, removing: {args.output}", flush=True)
        output_path.unlink()

    writer = None
    summaries: list[dict] = []
    try:
        for start in range(0, len(labels), args.chunk_size):
            stop = min(start + args.chunk_size, len(labels))
            chunk = labels.iloc[start:stop].copy()
            result = build_unit_sized_backtest_rows(
                chunk,
                quote_frame,
                CONFIG.labels,
                min_abs_delta=args.min_abs_delta,
                max_abs_delta=args.max_abs_delta,
                max_entry_gross=args.max_entry_gross,
            )
            summaries.append(result.summary)
            table = pa.Table.from_pandas(result.rows, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            if args.progress:
                done_episodes = stop
                done_strat_rows = sum(item["strategy_rows"] for item in summaries)
                print(
                    f"  [{done_episodes:,}/{len(labels):,} episodes] {done_strat_rows:,} strategy rows",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()

    summary = _combine_summaries(summaries, args.output, args.labels_input, args.clean_input)

    if not args.no_diagnostics:
        print(f"Writing diagnostics to {args.diagnostics_dir}", flush=True)
        diag_cols = [
            "episode_id", "signal_date", "strategy",
            "label_end_state", "exit_reason",
            "is_valid_backtest", "entry_gross_exposure",
            "net_pnl", "transaction_cost", "realized_return", "actual_holding_days",
        ]
        diag_rows = pd.read_parquet(output_path, columns=diag_cols)
        diag_rows["signal_date"] = pd.to_datetime(diag_rows["signal_date"])
        summary["diagnostic_outputs"] = _write_diagnostics(
            diag_rows, root / args.diagnostics_dir
        )

    summary_path = root / args.summary_output
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2))

    print(f"\nWrote {summary['strategy_rows']:,} strategy rows to {args.output}", flush=True)
    print(f"Valid backtest rows: {summary['valid_backtest_rows']:,} ({summary['valid_backtest_share']:.1%})", flush=True)
    print(json.dumps(_json_ready(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
