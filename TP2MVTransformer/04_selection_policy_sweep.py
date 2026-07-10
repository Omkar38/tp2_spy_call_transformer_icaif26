"""
Selection policy sweep on V2 transformer predictions.

Uses saved val and test predictions to compare different threshold/selection
policies WITHOUT retraining. This isolates whether year-to-year fragility is
coming from weak probabilities or from overly aggressive validation-time
selection logic.

Policies tested:
  P1  P&L-max threshold                          (current baseline)
  P2  Sharpe-max threshold
  P3  Constrained Sharpe: val_pnl>0, n>=min, wr>=0.20
  P4  Constrained Sharpe: val_pnl>0, n>=min, wr>=0.30
  P5  Constrained Sharpe: val_pnl>0, n>=min, wr>=0.40
  P6  Constrained Sharpe + max trades per day=50
  P7  Constrained Sharpe + max trades per day=20
  P8  Hard floor threshold=0.10  (P&L-max above floor)
  P9  Hard floor threshold=0.20
  P10 Hard floor threshold=0.30

For each policy, threshold is tuned on val year only.
Test year is evaluated once with that threshold.
Paper-friendly framing: risk-aware validation-time selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TP2MVTransformer"
DATA_DIR = OUT / "data"
TABLE_DIR = OUT / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

TEST_PATH = DATA_DIR / "mv_alibi_transformer_v2_test_predictions.parquet"
VAL_PATH  = DATA_DIR / "mv_alibi_transformer_v2_val_predictions.parquet"

THRESHOLD_GRID = np.arange(0.05, 0.95, 0.025)
MIN_TRADES = 30

FOLDS = [
    {"fold": "2017_2019__val2020__test2021", "val_year": 2020, "test_year": 2021},
    {"fold": "2018_2020__val2021__test2022", "val_year": 2021, "test_year": 2022},
    {"fold": "2019_2021__val2022__test2023", "val_year": 2022, "test_year": 2023},
    {"fold": "2020_2022__val2023__test2024", "val_year": 2023, "test_year": 2024},
    {"fold": "2021_2023__val2024__test2025", "val_year": 2024, "test_year": 2025},
    {"fold": "2022_2024__val2025__test2026", "val_year": 2025, "test_year": 2026},
]

POLICIES = {
    "P1_pnl_max":           {"objective": "pnl",    "min_wr": 0.00, "max_tpd": None, "floor": 0.05},
    "P2_sharpe_max":        {"objective": "sharpe",  "min_wr": 0.00, "max_tpd": None, "floor": 0.05},
    "P3_sharpe_wr20":       {"objective": "sharpe",  "min_wr": 0.20, "max_tpd": None, "floor": 0.05},
    "P4_sharpe_wr30":       {"objective": "sharpe",  "min_wr": 0.30, "max_tpd": None, "floor": 0.05},
    "P5_sharpe_wr40":       {"objective": "sharpe",  "min_wr": 0.40, "max_tpd": None, "floor": 0.05},
    "P6_sharpe_tpd50":      {"objective": "sharpe",  "min_wr": 0.20, "max_tpd": 50,   "floor": 0.05},
    "P7_sharpe_tpd20":      {"objective": "sharpe",  "min_wr": 0.20, "max_tpd": 20,   "floor": 0.05},
    "P8_floor10":           {"objective": "pnl",    "min_wr": 0.00, "max_tpd": None, "floor": 0.10},
    "P9_floor20":           {"objective": "pnl",    "min_wr": 0.00, "max_tpd": None, "floor": 0.20},
    "P10_floor30":          {"objective": "pnl",    "min_wr": 0.00, "max_tpd": None, "floor": 0.30},
}


# ── selection helpers ──────────────────────────────────────────────────────────

def select_actions(tokens: pd.DataFrame, threshold: float, max_tpd: int | None) -> pd.DataFrame:
    candidates = tokens[tokens["predicted_prob"] >= threshold].copy()
    if candidates.empty:
        return candidates

    if max_tpd is not None:
        # per day: take top-N by predicted_prob
        candidates = (
            candidates.sort_values("predicted_prob", ascending=False)
            .groupby("signal_date", group_keys=False)
            .head(max_tpd)
        )

    # per episode: take highest-prob strategy
    return (
        candidates.sort_values(["predicted_prob", "strategy"], ascending=[False, True])
        .groupby("episode_id", as_index=False)
        .first()
    )


def daily_sharpe(selected: pd.DataFrame) -> float:
    if selected.empty or len(selected) < 5:
        return float("nan")
    daily = selected.groupby("signal_date")["target_norm_pnl"].sum()
    if len(daily) < 5:
        return float("nan")
    std = daily.std()
    if std == 0 or pd.isna(std):
        return float("nan")
    return float(daily.mean() / std * np.sqrt(252))


def daily_sortino(selected: pd.DataFrame) -> float:
    if selected.empty or len(selected) < 5:
        return float("nan")
    daily = selected.groupby("signal_date")["target_norm_pnl"].sum()
    if len(daily) < 5:
        return float("nan")
    downside = daily[daily < 0.0]
    if downside.empty:
        return float("inf") if daily.mean() > 0 else float("nan")
    downside_std = downside.std()
    if downside_std == 0 or pd.isna(downside_std):
        return float("nan")
    return float(daily.mean() / downside_std * np.sqrt(252))


def tune_threshold(val_tokens: pd.DataFrame, policy: dict) -> float:
    objective  = policy["objective"]
    min_wr     = policy["min_wr"]
    max_tpd    = policy["max_tpd"]
    floor      = policy["floor"]

    best_threshold = floor
    best_metric = -np.inf

    # require val_pnl > 0 for all Sharpe-based objectives
    require_positive_pnl = (objective == "sharpe")

    for threshold in THRESHOLD_GRID:
        if threshold < floor - 1e-9:
            continue
        selected = select_actions(val_tokens, float(threshold), max_tpd)
        n = len(selected)
        if n < MIN_TRADES:
            continue

        pnl = float(selected["target_norm_pnl"].sum())
        wr  = float((selected["target_norm_pnl"] > 0).mean())

        # hard constraints
        if wr < min_wr:
            continue
        if require_positive_pnl and pnl <= 0:
            continue

        # objective
        if objective == "pnl":
            metric = pnl
        else:  # sharpe
            sh = daily_sharpe(selected)
            if not np.isfinite(sh):
                continue
            metric = sh

        if metric > best_metric:
            best_metric = metric
            best_threshold = float(threshold)

    return best_threshold


def eval_selection(selected: pd.DataFrame, threshold: float) -> dict:
    if selected.empty:
        return {
            "threshold": threshold, "n": 0, "pnl": 0.0,
            "win_rate": float("nan"), "sharpe": float("nan"),
            "sortino": float("nan"),
            "t1": 0, "k2": 0,
        }
    dist = selected["strategy"].value_counts().to_dict()
    return {
        "threshold": threshold,
        "n": int(len(selected)),
        "pnl": float(selected["target_norm_pnl"].sum()),
        "win_rate": float((selected["target_norm_pnl"] > 0).mean()),
        "sharpe": daily_sharpe(selected),
        "sortino": daily_sortino(selected),
        "t1": int(dist.get("T1", 0)),
        "k2": int(dist.get("K2", 0)),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading predictions...", flush=True)

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test predictions not found: {TEST_PATH}\n"
            "Download mv_alibi_transformer_v2_test_predictions.parquet from Colab "
            "and place it in TP2MVTransformer/data/"
        )
    if not VAL_PATH.exists():
        raise FileNotFoundError(
            f"Val predictions not found: {VAL_PATH}\n"
            "Re-run 03_train_mv_alibi_transformer.py (now saves val predictions) "
            "and download mv_alibi_transformer_v2_val_predictions.parquet from Colab."
        )

    test_preds = pd.read_parquet(TEST_PATH)
    val_preds  = pd.read_parquet(VAL_PATH)

    test_preds["signal_date"] = pd.to_datetime(test_preds["signal_date"])
    val_preds["signal_date"]  = pd.to_datetime(val_preds["signal_date"])
    test_preds["signal_year"] = test_preds["signal_date"].dt.year
    val_preds["signal_year"]  = val_preds["signal_date"].dt.year

    print(f"  Test rows: {len(test_preds):,} | Val rows: {len(val_preds):,}", flush=True)

    all_rows = []

    for fold in FOLDS:
        fold_name = fold["fold"]
        val_year  = fold["val_year"]
        test_year = fold["test_year"]

        val_fold  = val_preds[val_preds["signal_year"] == val_year].copy()
        test_fold = test_preds[test_preds["signal_year"] == test_year].copy()

        if val_fold.empty or test_fold.empty:
            print(f"  [{fold_name}] MISSING val or test data — skipping", flush=True)
            continue

        print(f"\n  Fold {fold_name} | val={val_year} n={len(val_fold):,} | test={test_year} n={len(test_fold):,}", flush=True)

        for policy_name, policy in POLICIES.items():
            threshold = tune_threshold(val_fold, policy)

            # val evaluation at tuned threshold
            val_sel   = select_actions(val_fold,  threshold, policy["max_tpd"])
            val_info  = eval_selection(val_sel, threshold)

            # test evaluation — one shot
            test_sel  = select_actions(test_fold, threshold, policy["max_tpd"])
            test_info = eval_selection(test_sel, threshold)

            row = {
                "fold":      fold_name,
                "val_year":  val_year,
                "test_year": test_year,
                "policy":    policy_name,
                "objective": policy["objective"],
                "min_wr":    policy["min_wr"],
                "max_tpd":   policy["max_tpd"] if policy["max_tpd"] else "none",
                "floor":     policy["floor"],
                **{f"val_{k}":  v for k, v in val_info.items()},
                **{f"test_{k}": v for k, v in test_info.items()},
            }
            all_rows.append(row)

            print(
                f"    {policy_name:<22} thresh={threshold:.3f} "
                f"val_n={val_info['n']:>5,} val_wr={val_info['win_rate']:.2f} "
                f"| test_pnl=${test_info['pnl']:>10,.0f} "
                f"test_wr={test_info['win_rate']:.2f} "
                f"test_sharpe={test_info['sharpe']:>6.2f} "
                f"test_sortino={test_info['sortino']:>6.2f}",
                flush=True,
            )

    if not all_rows:
        print("No results — check that predictions files contain expected years.")
        return

    results = pd.DataFrame(all_rows)
    results.to_csv(TABLE_DIR / "policy_sweep_full.csv", index=False)

    # ── summary table: total P&L and positive years per policy ──────────────
    print("\n" + "="*80)
    print("POLICY SWEEP SUMMARY (test years 2021–2026)")
    print("="*80)

    pivot_pnl = results.pivot_table(
        index="policy", columns="test_year", values="test_pnl", aggfunc="sum"
    )
    pivot_pnl["TOTAL"]     = pivot_pnl.sum(axis=1)
    pivot_pnl["POS_YEARS"] = (pivot_pnl.drop(columns="TOTAL") > 0).sum(axis=1)

    pivot_sharpe = results.pivot_table(
        index="policy", columns="test_year", values="test_sharpe", aggfunc="mean"
    )
    pivot_sortino = results.pivot_table(
        index="policy", columns="test_year", values="test_sortino", aggfunc="mean"
    )

    print("\nPer-year P&L ($):")
    print(pivot_pnl.to_string(float_format=lambda x: f"{x:,.0f}"))

    print("\nPer-year Sharpe:")
    print(pivot_sharpe.to_string(float_format=lambda x: f"{x:.2f}"))

    print("\nPer-year Sortino:")
    print(pivot_sortino.to_string(float_format=lambda x: f"{x:.2f}"))

    pivot_pnl.to_csv(TABLE_DIR / "policy_sweep_pnl_pivot.csv")
    pivot_sharpe.to_csv(TABLE_DIR / "policy_sweep_sharpe_pivot.csv")
    pivot_sortino.to_csv(TABLE_DIR / "policy_sweep_sortino_pivot.csv")

    # ── robust policy diagnostics, not single-year targeting ────────────────
    print("\n" + "="*80)
    print("ROBUST POLICY DIAGNOSTICS:")
    year_cols = [c for c in pivot_pnl.columns if isinstance(c, (int, np.integer))]
    robust = pd.DataFrame(index=pivot_pnl.index)
    robust["TOTAL"] = pivot_pnl["TOTAL"]
    robust["POS_YEARS"] = pivot_pnl["POS_YEARS"]
    robust["WORST_YEAR_PNL"] = pivot_pnl[year_cols].min(axis=1)
    robust["AVG_YEAR_PNL"] = pivot_pnl[year_cols].mean(axis=1)
    robust["MEAN_YEAR_SHARPE"] = pivot_sharpe[year_cols].mean(axis=1)
    robust["MEAN_YEAR_SORTINO"] = pivot_sortino[year_cols].replace([np.inf, -np.inf], np.nan).mean(axis=1)
    robust = robust.sort_values(
        ["POS_YEARS", "WORST_YEAR_PNL", "TOTAL"],
        ascending=[False, False, False],
    )
    print(robust.to_string(float_format=lambda x: f"{x:,.2f}"))
    robust.to_csv(TABLE_DIR / "policy_sweep_robust_summary.csv")
    print()
    print("Interpretation rule:")
    print("  Do not choose a policy because it fixes one specific test year.")
    print("  Prefer policies that improve worst-year loss, positive-year count,")
    print("  and risk-adjusted behavior while preserving out-of-sample total P&L.")

    print(f"\nWrote {TABLE_DIR / 'policy_sweep_full.csv'}")
    print(f"Wrote {TABLE_DIR / 'policy_sweep_pnl_pivot.csv'}")
    print(f"Wrote {TABLE_DIR / 'policy_sweep_sharpe_pivot.csv'}")
    print(f"Wrote {TABLE_DIR / 'policy_sweep_sortino_pivot.csv'}")
    print(f"Wrote {TABLE_DIR / 'policy_sweep_robust_summary.csv'}")


if __name__ == "__main__":
    main()
