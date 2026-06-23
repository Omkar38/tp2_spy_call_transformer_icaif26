#!/usr/bin/env python3
"""Build leakage-safe episode-level ML dataset for TP2 strategy selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tp2rr2.config import CONFIG


STRATEGIES = ["T1", "T2", "K1", "K2"]
FIRST_LIQUID_POLICY = "exit_first_liquid_on_or_after_reversion"
ACTION_MAP = {"skip": 0, "T1": 1, "T2": 2, "K1": 3, "K2": 4}


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    forbidden_tokens = [
        "contract_id",
        "option_symbol",
        "expiration",
    ]
    allowed_prefixes = ("feature_", "start_")
    forbidden_exact = {
        "start_violation_id",
        "start_trading_day_index",
        "start_year",
        "start_month",
        "start_rectangle_key",
        "start_normalized_severity",  # added back below with explicit name
    }
    future_or_episode_summary = {
        "end_trading_day_index",
        "next_trading_day_index",
        "duration_sessions",
        "episode_row_count",
        "unique_violation_days",
        "unique_rectangle_count",
        "has_exact_rectangle_change",
        "end_year",
        "end_month",
        "max_normalized_severity",
        "mean_normalized_severity",
        "end_normalized_severity",
        "end_lhs",
        "end_rhs",
    }

    cols: list[str] = []
    for col in df.columns:
        lower = col.lower()
        if col in forbidden_exact or col in future_or_episode_summary:
            continue
        if not col.startswith(allowed_prefixes):
            continue
        if any(tok in lower for tok in forbidden_tokens):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)

    explicit_safe = ["start_normalized_severity", "start_lhs", "start_rhs"]
    for col in explicit_safe:
        if col in df and pd.api.types.is_numeric_dtype(df[col]) and col not in cols:
            cols.append(col)
    return sorted(cols)


def _add_signal_date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    signal = pd.to_datetime(out["signal_date"])
    out["signal_year"] = signal.dt.year
    out["signal_month"] = signal.dt.month
    out["signal_dayofweek"] = signal.dt.dayofweek

    eps = 1e-9
    if {"feature_lhs", "feature_rhs"}.issubset(out.columns):
        out["feature_product_ratio"] = out["feature_rhs"] / (out["feature_lhs"] + eps)
    if {"feature_A_ask", "feature_B_ask", "feature_C_bid", "feature_D_bid"}.issubset(out.columns):
        out["feature_bidask_executable_ratio"] = (
            out["feature_C_bid"] * out["feature_D_bid"]
        ) / (out["feature_A_ask"] * out["feature_B_ask"] + eps)
    spread_cols = [f"feature_{leg}_spread_pct" for leg in ["A", "B", "C", "D"]]
    if all(col in out.columns for col in spread_cols):
        out["feature_max_leg_spread_pct"] = out[spread_cols].max(axis=1)
        out["feature_mean_leg_spread_pct"] = out[spread_cols].mean(axis=1)
    volume_cols = [f"feature_{leg}_volume" for leg in ["A", "B", "C", "D"]]
    if all(col in out.columns for col in volume_cols):
        out["feature_min_leg_volume"] = out[volume_cols].min(axis=1)
        out["feature_total_leg_volume"] = out[volume_cols].sum(axis=1)
        out["feature_zero_volume_legs"] = out[volume_cols].fillna(0).le(0).sum(axis=1)
    oi_cols = [f"feature_{leg}_open_interest" for leg in ["A", "B", "C", "D"]]
    if all(col in out.columns for col in oi_cols):
        out["feature_min_leg_open_interest"] = out[oi_cols].min(axis=1)
        out["feature_total_leg_open_interest"] = out[oi_cols].sum(axis=1)
        out["feature_zero_oi_legs"] = out[oi_cols].fillna(0).le(0).sum(axis=1)
    return out


def _build_lifecycle_features(path: Path) -> pd.DataFrame:
    cols = [
        "episode_id",
        "event_day_index",
        "status",
        "normalized_pressure",
        "is_violation",
        "is_conservative_bidask_violation",
        "all_legs_tradable_under_50pct_spread",
        "any_leg_wide_spread_over_50pct",
    ]
    life = pd.read_parquet(path, columns=cols)
    life = life[life["event_day_index"].le(0)].copy()
    life["has_pressure"] = life["normalized_pressure"].notna()
    life["observable"] = life["status"].eq("observable")
    life["wide_spread_status"] = life["status"].eq("liquidity_failure_wide_spread")
    life["missing_ab"] = life["status"].eq("missing_A_or_B")
    life["missing_cd"] = life["status"].eq("missing_C_or_D")
    life["missing_forward"] = life["status"].eq("missing_forward")
    life["event0_pressure"] = np.where(
        life["event_day_index"].eq(0), life["normalized_pressure"], np.nan
    )
    life["pre_pressure"] = np.where(
        life["event_day_index"].lt(0), life["normalized_pressure"], np.nan
    )
    life["pre_violation"] = np.where(life["event_day_index"].lt(0), life["is_violation"], np.nan)

    grouped = life.groupby("episode_id", sort=False)
    features = grouped.agg(
        lc_rows_le0=("episode_id", "size"),
        lc_observable_share_le0=("observable", "mean"),
        lc_wide_spread_share_le0=("wide_spread_status", "mean"),
        lc_missing_ab_share_le0=("missing_ab", "mean"),
        lc_missing_cd_share_le0=("missing_cd", "mean"),
        lc_missing_forward_share_le0=("missing_forward", "mean"),
        lc_pressure_rows_le0=("has_pressure", "sum"),
        lc_pressure_mean_le0=("normalized_pressure", "mean"),
        lc_pressure_median_le0=("normalized_pressure", "median"),
        lc_pressure_std_le0=("normalized_pressure", "std"),
        lc_pressure_min_le0=("normalized_pressure", "min"),
        lc_pressure_max_le0=("normalized_pressure", "max"),
        lc_event0_pressure=("event0_pressure", "max"),
        lc_pre_pressure_mean=("pre_pressure", "mean"),
        lc_pre_pressure_max=("pre_pressure", "max"),
        lc_pre_violation_share=("pre_violation", "mean"),
        lc_violation_share_le0=("is_violation", "mean"),
        lc_conservative_violation_share_le0=("is_conservative_bidask_violation", "mean"),
        lc_all_legs_liquid_share_le0=("all_legs_tradable_under_50pct_spread", "mean"),
        lc_any_wide_spread_share_le0=("any_leg_wide_spread_over_50pct", "mean"),
    ).reset_index()

    # Simple slope from earliest evaluable pre/signal pressure to event-0 pressure.
    ordered = life[life["normalized_pressure"].notna()].sort_values(["episode_id", "event_day_index"])
    first = ordered.groupby("episode_id", sort=False).first()[["event_day_index", "normalized_pressure"]]
    last = ordered.groupby("episode_id", sort=False).last()[["event_day_index", "normalized_pressure"]]
    slope = (last["normalized_pressure"] - first["normalized_pressure"]) / (
        (last["event_day_index"] - first["event_day_index"]).replace(0, np.nan)
    )
    slope = slope.rename("lc_pressure_slope_le0").reset_index()
    features = features.merge(slope, on="episode_id", how="left")
    return features


def _best_action_from_matrix(
    out: pd.DataFrame,
    value_cols: list[str],
    valid_cols: list[str],
    *,
    prefix: str,
) -> pd.DataFrame:
    value_matrix = out[value_cols].where(out[valid_cols].to_numpy(), np.nan)
    value_values = value_matrix.to_numpy(dtype=float)
    safe_values = np.where(np.isnan(value_values), -np.inf, value_values)
    best_idx = np.nanargmax(safe_values, axis=1)
    all_nan = value_matrix.isna().all(axis=1).to_numpy()
    best_value = np.max(safe_values, axis=1)
    best_value[all_nan] = 0.0
    best_action = np.array([STRATEGIES[i] for i in best_idx], dtype=object)
    best_action[all_nan | ~(best_value > 0)] = "skip"
    best_value[all_nan] = 0.0

    out[f"{prefix}_action"] = best_action
    out[f"{prefix}_action_id"] = out[f"{prefix}_action"].map(ACTION_MAP).astype("int64")
    out[f"{prefix}_value"] = best_value
    out[f"{prefix}_any_positive"] = value_matrix.gt(0).any(axis=1)
    return out


def _build_strategy_labels(labels: pd.DataFrame, capped_pnl_abs: float) -> pd.DataFrame:
    sub = labels[
        labels["exit_policy"].eq(FIRST_LIQUID_POLICY)
        & labels["strategy"].isin(STRATEGIES)
    ].copy()
    pnl = sub.pivot_table(
        index="episode_id", columns="strategy", values="net_pnl", aggfunc="first", dropna=False
    )
    valid = sub.pivot_table(
        index="episode_id", columns="strategy", values="is_valid_backtest", aggfunc="first", dropna=False
    )
    ret = sub.pivot_table(
        index="episode_id", columns="strategy", values="realized_return", aggfunc="first", dropna=False
    )
    hold = sub.pivot_table(
        index="episode_id", columns="strategy", values="actual_holding_days", aggfunc="first", dropna=False
    )
    exposure = sub.pivot_table(
        index="episode_id", columns="strategy", values="entry_gross_exposure", aggfunc="first", dropna=False
    )

    out = pd.DataFrame({"episode_id": pnl.index.to_numpy()})
    for strat in STRATEGIES:
        out[f"label_net_pnl_{strat}"] = pnl.get(strat, pd.Series(np.nan, index=pnl.index)).to_numpy()
        out[f"label_is_valid_{strat}"] = (
            valid.get(strat, pd.Series(False, index=valid.index)).fillna(False).astype(bool).to_numpy()
        )
        out[f"label_realized_return_{strat}"] = ret.get(strat, pd.Series(np.nan, index=ret.index)).to_numpy()
        out[f"label_holding_days_{strat}"] = hold.get(strat, pd.Series(np.nan, index=hold.index)).to_numpy()
        out[f"label_entry_gross_exposure_{strat}"] = exposure.get(
            strat, pd.Series(np.nan, index=exposure.index)
        ).to_numpy()
        out[f"label_capped_net_pnl_{strat}"] = out[f"label_net_pnl_{strat}"].clip(
            lower=-capped_pnl_abs,
            upper=capped_pnl_abs,
        )

    pnl_cols = [f"label_net_pnl_{s}" for s in STRATEGIES]
    capped_cols = [f"label_capped_net_pnl_{s}" for s in STRATEGIES]
    return_cols = [f"label_realized_return_{s}" for s in STRATEGIES]
    valid_cols = [f"label_is_valid_{s}" for s in STRATEGIES]

    out = _best_action_from_matrix(out, pnl_cols, valid_cols, prefix="label_best")
    out = _best_action_from_matrix(out, return_cols, valid_cols, prefix="label_best_return")
    out = _best_action_from_matrix(out, capped_cols, valid_cols, prefix="label_best_capped_pnl")
    out["label_best_net_pnl"] = out["label_best_value"]
    out["label_best_action"] = out["label_best_action"]
    out["label_best_action_id"] = out["label_best_action_id"]
    out["label_any_valid"] = out[valid_cols].any(axis=1)
    out["label_any_positive"] = out["label_best_any_positive"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML episode dataset.")
    parser.add_argument(
        "--episode-labels",
        default="data/labels/archive2_episode_start_labels_conservative_bidask_2013_2026.parquet",
    )
    parser.add_argument(
        "--lifecycle",
        default="data/features/archive2_lifecycle/tp2_conservative_episode_lifecycle_2013_2026.parquet",
    )
    parser.add_argument(
        "--strategy-labels",
        default="data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet",
    )
    parser.add_argument("--output", default="data/ml/archive2_ml_episode_dataset_2013_2026.parquet")
    parser.add_argument("--summary-output", default="data/ml/archive2_ml_episode_dataset_summary_2013_2026.json")
    parser.add_argument(
        "--capped-pnl-abs",
        type=float,
        default=10000.0,
        help="Fixed symmetric dollar cap for capped-P&L action labels.",
    )
    args = parser.parse_args()

    root = CONFIG.paths.root
    out_path = root / args.output
    summary_path = root / args.summary_output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_parquet(root / args.episode_labels)
    episodes = _add_signal_date_features(episodes)
    safe_feature_cols = _numeric_feature_columns(episodes)
    engineered = [
        "signal_year",
        "signal_month",
        "signal_dayofweek",
        "feature_product_ratio",
        "feature_bidask_executable_ratio",
        "feature_max_leg_spread_pct",
        "feature_mean_leg_spread_pct",
        "feature_min_leg_volume",
        "feature_total_leg_volume",
        "feature_zero_volume_legs",
        "feature_min_leg_open_interest",
        "feature_total_leg_open_interest",
        "feature_zero_oi_legs",
    ]
    safe_feature_cols = sorted(set(safe_feature_cols + [c for c in engineered if c in episodes]))

    base = episodes[["episode_id", "signal_date", "start_year", "start_month", *safe_feature_cols]].copy()
    life_features = _build_lifecycle_features(root / args.lifecycle)
    strategy_labels = _build_strategy_labels(
        pd.read_parquet(root / args.strategy_labels),
        capped_pnl_abs=args.capped_pnl_abs,
    )

    dataset = base.merge(life_features, on="episode_id", how="left").merge(
        strategy_labels, on="episode_id", how="inner"
    )
    dataset["signal_date"] = pd.to_datetime(dataset["signal_date"])
    dataset["sample_role"] = np.where(
        dataset["signal_date"].dt.year.lt(2017),
        "early_robustness",
        "primary_rolling",
    )

    feature_cols = [
        c
        for c in dataset.columns
        if (
            c.startswith("feature_")
            or c.startswith("start_")
            or c.startswith("signal_")
            or c.startswith("lc_")
        )
        and c not in {"signal_date"}
        and pd.api.types.is_numeric_dtype(dataset[c])
        and not c.startswith("label_")
    ]
    feature_cols = sorted(set(feature_cols))
    all_null_feature_cols = [col for col in feature_cols if dataset[col].notna().sum() == 0]
    feature_cols = [col for col in feature_cols if col not in all_null_feature_cols]

    dataset.to_parquet(out_path, index=False)
    summary = {
        "rows": int(len(dataset)),
        "episodes": int(dataset["episode_id"].nunique()),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "all_null_feature_columns_removed": all_null_feature_cols,
        "action_counts": dataset["label_best_action"].value_counts().to_dict(),
        "return_action_counts": dataset["label_best_return_action"].value_counts().to_dict(),
        "capped_pnl_action_counts": dataset["label_best_capped_pnl_action"].value_counts().to_dict(),
        "capped_pnl_abs": float(args.capped_pnl_abs),
        "year_counts": dataset["signal_date"].dt.year.value_counts().sort_index().to_dict(),
        "primary_rows_2017_forward": int(dataset["sample_role"].eq("primary_rolling").sum()),
        "early_robustness_rows_2013_2016": int(dataset["sample_role"].eq("early_robustness").sum()),
        "exit_policy": FIRST_LIQUID_POLICY,
        "actions": ["skip", *STRATEGIES],
        "leakage_rule": "Feature columns are signal-date or event_day<=0 lifecycle aggregates only.",
        "output": str(out_path.relative_to(root)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
