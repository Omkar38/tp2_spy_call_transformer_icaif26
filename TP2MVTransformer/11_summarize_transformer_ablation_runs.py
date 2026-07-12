#!/usr/bin/env python3
"""Summarize all TP2 transformer ablation runs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ABLATION_DIR = ROOT / "TP2MVTransformer" / "ablation_runs"
OUT_TABLE = ABLATION_DIR / "ablation_run_comparison.csv"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    rows = []
    for run_dir in sorted(ABLATION_DIR.glob("*")):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        summary = read_json(run_dir / "tables" / "ablation_summary.json")
        yearly_path = run_dir / "tables" / "ablation_yearly_results.csv"
        if not summary or not yearly_path.exists():
            continue
        yearly = pd.read_csv(yearly_path)
        args = summary.get("args", {})
        rows.append({
            "tag": run_dir.name,
            "total_selected": summary.get("total_selected"),
            "total_norm_pnl": summary.get("total_norm_pnl"),
            "positive_years": summary.get("positive_years"),
            "total_years": summary.get("total_years"),
            "combined_daily_sharpe": summary.get("combined_daily_sharpe"),
            "combined_daily_sortino": summary.get("combined_daily_sortino"),
            "combined_max_drawdown": summary.get("combined_max_drawdown"),
            "combined_win_rate": summary.get("combined_win_rate"),
            "mean_test_brier": yearly["test_brier"].mean() if "test_brier" in yearly else float("nan"),
            "mean_test_ece": yearly["test_ece"].mean() if "test_ece" in yearly else float("nan"),
            "mean_best_epoch": yearly["best_epoch"].mean() if "best_epoch" in yearly else float("nan"),
            "lr": args.get("lr"),
            "hidden": args.get("hidden"),
            "heads": args.get("heads"),
            "max_history": args.get("max_history"),
            "checkpoint_metric": args.get("checkpoint_metric"),
            "scheduler": args.get("scheduler"),
            "scheduler_metric": args.get("scheduler_metric"),
            "input_dropout": args.get("input_dropout"),
            "head_dropout": args.get("head_dropout"),
            "weight_decay": args.get("weight_decay"),
            "seed": args.get("seed"),
        })
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        print(f"No completed ablation runs found under {ABLATION_DIR}")
        return
    comparison = comparison.sort_values(
        ["positive_years", "combined_daily_sharpe", "total_norm_pnl"],
        ascending=[False, False, False],
    )
    comparison.to_csv(OUT_TABLE, index=False)
    print(comparison.to_string(index=False))
    print(f"Wrote {OUT_TABLE}")


if __name__ == "__main__":
    main()
