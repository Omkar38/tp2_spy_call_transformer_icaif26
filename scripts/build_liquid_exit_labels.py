#!/usr/bin/env python3
"""Build product-weighted TP2 strategy labels with liquid-after-reversion exits."""
from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd

from tp2rr2.config import CONFIG


STRATEGIES = {
    "T1": ["A", "D"],
    "T2": ["B", "C"],
    "K1": ["A", "C"],
    "K2": ["B", "D"],
}

FIXED_OFFSETS = {
    "exit_at_reversion_day": 0,
    "exit_reversion_plus_1_session": 1,
    "exit_reversion_plus_3_sessions": 3,
    "exit_reversion_plus_5_sessions": 5,
}


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _quantity_and_side(rows: pd.DataFrame, leg: str) -> tuple[np.ndarray, np.ndarray]:
    if leg == "A":
        return rows["feature_B_mid"].to_numpy(float), np.ones(len(rows))
    if leg == "B":
        return rows["feature_A_mid"].to_numpy(float), np.ones(len(rows))
    if leg == "C":
        return rows["feature_D_mid"].to_numpy(float), -np.ones(len(rows))
    if leg == "D":
        return rows["feature_C_mid"].to_numpy(float), -np.ones(len(rows))
    raise ValueError(f"Unknown leg: {leg}")


def _quote_key(series_symbol: pd.Series, series_contract: pd.Series) -> pd.Series:
    return series_symbol.where(series_symbol.notna(), series_contract).astype(str)


def _build_calendar(quotes: pd.DataFrame) -> pd.DataFrame:
    cal = pd.DataFrame({"date": sorted(pd.to_datetime(quotes["date"]).drop_duplicates())})
    cal["date_idx"] = np.arange(len(cal), dtype=np.int32)
    return cal


def _build_tradable_index(quotes: pd.DataFrame, max_spread_pct: float) -> dict[str, np.ndarray]:
    q = quotes[
        (quotes["bid"] > 0)
        & (quotes["ask"] > quotes["bid"])
        & (quotes["mid"] > 0)
        & np.isfinite(quotes["spread_pct"])
        & (quotes["spread_pct"] <= max_spread_pct)
    ][["quote_key", "date_idx"]].copy()
    q = q.drop_duplicates(["quote_key", "date_idx"]).sort_values(["quote_key", "date_idx"])
    return {str(key): grp["date_idx"].to_numpy(np.int32) for key, grp in q.groupby("quote_key", sort=False)}


def _first_common_tradable(
    left: str,
    right: str,
    start_idx: int,
    end_before_idx: int,
    tradable: dict[str, np.ndarray],
) -> int:
    a = tradable.get(str(left))
    b = tradable.get(str(right))
    if a is None or b is None:
        return -1
    i = bisect_left(a, start_idx)
    j = bisect_left(b, start_idx)
    while i < len(a) and j < len(b):
        av = int(a[i])
        bv = int(b[j])
        if av >= end_before_idx or bv >= end_before_idx:
            return -1
        if av == bv:
            return av
        if av < bv:
            i += 1
        else:
            j += 1
    return -1


def _prepare_reversion(lifecycle: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    candidates = lifecycle[
        lifecycle["event_day_index"].ge(1)
        & lifecycle["lhs"].notna()
        & lifecycle["rhs"].notna()
        & lifecycle["lhs"].ge(lifecycle["rhs"])
    ][["episode_id", "lifecycle_date", "event_day_index", "normalized_pressure"]].copy()
    if candidates.empty:
        return pd.DataFrame(
            columns=["episode_id", "first_reversion_date", "first_reversion_event_day", "reversion_pressure"]
        )
    idx = candidates.sort_values(["episode_id", "lifecycle_date"]).groupby("episode_id", as_index=False).first()
    idx = idx.rename(
        columns={
            "lifecycle_date": "first_reversion_date",
            "event_day_index": "first_reversion_event_day",
            "normalized_pressure": "reversion_pressure",
        }
    )
    idx["first_reversion_date"] = pd.to_datetime(idx["first_reversion_date"])
    idx = idx.merge(calendar.rename(columns={"date": "first_reversion_date", "date_idx": "first_reversion_idx"}), on="first_reversion_date", how="left")
    return idx


def _base_strategy_rows(labels: pd.DataFrame, reversion: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for strategy, legs in STRATEGIES.items():
        chunk = labels.copy()
        chunk["strategy"] = strategy
        chunk["selected_legs"] = ",".join(legs)
        pieces.append(chunk)
    rows = pd.concat(pieces, ignore_index=True)
    rows = rows.merge(reversion, on="episode_id", how="left")
    rows["signal_date"] = pd.to_datetime(rows["signal_date"])
    rows = rows.merge(calendar.rename(columns={"date": "signal_date", "date_idx": "signal_idx"}), on="signal_date", how="left")
    for leg in ["A", "B", "C", "D"]:
        rows[f"feature_{leg}_expiration"] = pd.to_datetime(rows[f"feature_{leg}_expiration"])
        rows[f"feature_{leg}_quote_key"] = _quote_key(
            rows[f"feature_{leg}_option_symbol"], rows[f"feature_{leg}_contract_id"]
        )
        rows = rows.merge(
            calendar.rename(columns={"date": f"feature_{leg}_expiration", "date_idx": f"{leg}_expiration_idx"}),
            on=f"feature_{leg}_expiration",
            how="left",
        )
    return rows


def _assign_exit_dates(rows: pd.DataFrame, calendar: pd.DataFrame, tradable: dict[str, np.ndarray]) -> pd.DataFrame:
    date_by_idx = dict(zip(calendar["date_idx"].astype(int), pd.to_datetime(calendar["date"])))
    rows = rows.copy()
    rows["selected_expiration_idx"] = np.nan
    rows["first_liquid_exit_idx"] = -1

    for strategy, legs in STRATEGIES.items():
        mask = rows["strategy"].eq(strategy)
        if not mask.any():
            continue
        exp_cols = [f"{leg}_expiration_idx" for leg in legs]
        min_exp = rows.loc[mask, exp_cols].min(axis=1)
        rows.loc[mask, "selected_expiration_idx"] = min_exp
        left_keys = rows.loc[mask, f"feature_{legs[0]}_quote_key"].to_numpy(str)
        right_keys = rows.loc[mask, f"feature_{legs[1]}_quote_key"].to_numpy(str)
        rev_idx = rows.loc[mask, "first_reversion_idx"].fillna(-1).to_numpy(int)
        end_idx = min_exp.fillna(-1).to_numpy(int)
        out = np.full(mask.sum(), -1, dtype=np.int32)
        for i, (left, right, start, end) in enumerate(zip(left_keys, right_keys, rev_idx, end_idx)):
            if start >= 0 and end > start:
                out[i] = _first_common_tradable(left, right, int(start), int(end), tradable)
        rows.loc[mask, "first_liquid_exit_idx"] = out

    rows["first_liquid_exit_date"] = rows["first_liquid_exit_idx"].map(date_by_idx)
    return rows


def _expand_exit_policies(rows: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    date_by_idx = dict(zip(calendar["date_idx"].astype(int), pd.to_datetime(calendar["date"])))
    pieces = []
    for policy, offset in FIXED_OFFSETS.items():
        chunk = rows.copy()
        chunk["exit_policy"] = policy
        chunk["exit_idx"] = chunk["first_reversion_idx"] + offset
        chunk.loc[chunk["first_reversion_idx"].isna(), "exit_idx"] = -1
        chunk["exit_idx"] = chunk["exit_idx"].fillna(-1).astype(int)
        chunk["exit_date"] = chunk["exit_idx"].map(date_by_idx)
        pieces.append(chunk)

    chunk = rows.copy()
    chunk["exit_policy"] = "exit_first_liquid_on_or_after_reversion"
    chunk["exit_idx"] = chunk["first_liquid_exit_idx"].astype(int)
    chunk["exit_date"] = chunk["first_liquid_exit_date"]
    pieces.append(chunk)
    return pd.concat(pieces, ignore_index=True)


def _merge_exit_quotes(rows: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    quote_lookup = quotes[["date_idx", "quote_key", "bid", "ask", "mid", "spread_pct"]].copy()
    quote_lookup = quote_lookup.rename(
        columns={"bid": "exit_bid", "ask": "exit_ask", "mid": "exit_mid", "spread_pct": "exit_spread_pct"}
    )
    rows = rows.copy()
    rows["_row"] = np.arange(len(rows), dtype=np.int64)
    for leg in ["A", "B", "C", "D"]:
        selected = rows["strategy"].map(lambda name, leg=leg: leg in STRATEGIES[name])
        keys = rows.loc[selected, ["_row", "exit_idx", f"feature_{leg}_quote_key"]].rename(
            columns={"exit_idx": "date_idx", f"feature_{leg}_quote_key": "quote_key"}
        )
        merged = keys.merge(quote_lookup, on=["date_idx", "quote_key"], how="left")
        for col in ["exit_bid", "exit_ask", "exit_mid", "exit_spread_pct"]:
            rows[f"{leg}_{col}"] = np.nan
        rows.loc[merged["_row"].to_numpy(int), f"{leg}_exit_bid"] = merged["exit_bid"].to_numpy(float)
        rows.loc[merged["_row"].to_numpy(int), f"{leg}_exit_ask"] = merged["exit_ask"].to_numpy(float)
        rows.loc[merged["_row"].to_numpy(int), f"{leg}_exit_mid"] = merged["exit_mid"].to_numpy(float)
        rows.loc[merged["_row"].to_numpy(int), f"{leg}_exit_spread_pct"] = merged["exit_spread_pct"].to_numpy(float)
    return rows.drop(columns=["_row"])


def _compute_pnl(rows: pd.DataFrame, max_spread_pct: float) -> pd.DataFrame:
    rows = rows.copy()
    n = len(rows)
    gross = np.zeros(n)
    price_pnl = np.zeros(n)
    txn = np.zeros(n)
    valid = rows["first_reversion_date"].notna().to_numpy().copy()
    valid &= rows["exit_idx"].to_numpy(int) >= 0
    valid &= rows["exit_idx"].to_numpy(float) < rows["selected_expiration_idx"].to_numpy(float)

    for leg in ["A", "B", "C", "D"]:
        selected = rows["strategy"].map(lambda name, leg=leg: leg in STRATEGIES[name]).to_numpy()
        qty, side = _quantity_and_side(rows, leg)
        entry_bid = rows[f"feature_{leg}_bid"].to_numpy(float)
        entry_ask = rows[f"feature_{leg}_ask"].to_numpy(float)
        entry_mid = rows[f"feature_{leg}_mid"].to_numpy(float)
        entry_spread = rows[f"feature_{leg}_spread_pct"].to_numpy(float)
        exit_bid = rows[f"{leg}_exit_bid"].to_numpy(float)
        exit_ask = rows[f"{leg}_exit_ask"].to_numpy(float)
        exit_mid = rows[f"{leg}_exit_mid"].to_numpy(float)
        exit_spread = rows[f"{leg}_exit_spread_pct"].to_numpy(float)

        entry_ok = (entry_bid > 0) & (entry_ask > entry_bid) & (entry_mid > 0) & np.isfinite(entry_spread) & (entry_spread <= max_spread_pct)
        exit_ok = (exit_bid > 0) & (exit_ask > exit_bid) & (exit_mid > 0) & np.isfinite(exit_spread) & (exit_spread <= max_spread_pct)
        valid &= ~selected | (entry_ok & exit_ok)
        gross += np.where(selected, np.abs(entry_mid * qty * CONFIG.labels.option_multiplier), 0.0)
        long_pnl = (exit_bid - entry_ask) * qty * CONFIG.labels.option_multiplier
        short_pnl = (entry_bid - exit_ask) * qty * CONFIG.labels.option_multiplier
        price_pnl += np.where(selected, np.where(side == 1.0, long_pnl, short_pnl), 0.0)
        txn += np.where(selected, 2.0 * CONFIG.labels.commission_per_contract * np.abs(qty), 0.0)

    gross_ok = np.isfinite(gross) & (gross > 0)
    valid &= gross_ok
    rows["is_valid_backtest"] = valid
    rows["entry_gross_exposure"] = gross
    rows["price_pnl"] = np.where(valid, price_pnl, np.nan)
    rows["transaction_cost"] = np.where(valid, txn, np.nan)
    rows["net_pnl"] = np.where(valid, price_pnl - txn, np.nan)
    rows["realized_return"] = np.where(valid, rows["net_pnl"] / gross, np.nan)
    rows["actual_holding_days"] = (
        pd.to_datetime(rows["exit_date"]) - pd.to_datetime(rows["signal_date"])
    ).dt.days
    rows.loc[~valid, "actual_holding_days"] = np.nan

    rows["exit_reason"] = "valid_exit"
    rows.loc[rows["first_reversion_date"].isna(), "exit_reason"] = "no_reversion_before_T1"
    rows.loc[rows["exit_idx"].lt(0) & rows["first_reversion_date"].notna(), "exit_reason"] = "no_liquid_exit_after_reversion"
    rows.loc[
        rows["exit_idx"].ge(rows["selected_expiration_idx"].fillna(-1)) & rows["first_reversion_date"].notna(),
        "exit_reason",
    ] = "exit_on_or_after_selected_expiration"
    rows.loc[rows["exit_reason"].eq("valid_exit") & ~rows["is_valid_backtest"], "exit_reason"] = "invalid_entry_or_exit_quote"
    return rows


def _build_expiration_payoff_rows(
    base_rows: pd.DataFrame,
    underlying_by_date: pd.DataFrame,
    max_spread_pct: float,
) -> pd.DataFrame:
    rows = base_rows.copy()
    rows["exit_policy"] = "exit_selected_leg_expiration_payoff"
    rows["exit_date"] = pd.NaT
    rows["exit_reason"] = "valid_expiration_payoff"
    rows["is_valid_backtest"] = True
    rows["entry_gross_exposure"] = 0.0
    rows["price_pnl"] = 0.0
    rows["transaction_cost"] = 0.0
    rows["net_pnl"] = np.nan
    rows["realized_return"] = np.nan
    rows["actual_holding_days"] = np.nan

    for leg in ["A", "B", "C", "D"]:
        for col in ["exit_bid", "exit_ask", "exit_mid", "exit_spread_pct"]:
            rows[f"{leg}_{col}"] = np.nan
        exp_col = f"feature_{leg}_expiration"
        rows = rows.merge(
            underlying_by_date.rename(
                columns={
                    "date": exp_col,
                    "underlying_price": f"{leg}_expiration_underlying_price",
                }
            ),
            on=exp_col,
            how="left",
        )

    selected_exp_max = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns]")
    for leg in ["A", "B", "C", "D"]:
        selected = rows["strategy"].map(lambda name, leg=leg: leg in STRATEGIES[name]).to_numpy()
        qty, side = _quantity_and_side(rows, leg)
        entry_bid = rows[f"feature_{leg}_bid"].to_numpy(float)
        entry_ask = rows[f"feature_{leg}_ask"].to_numpy(float)
        entry_mid = rows[f"feature_{leg}_mid"].to_numpy(float)
        entry_spread = rows[f"feature_{leg}_spread_pct"].to_numpy(float)
        strike = rows[f"feature_{leg}_strike"].to_numpy(float)
        spot = rows[f"{leg}_expiration_underlying_price"].to_numpy(float)
        exp = pd.to_datetime(rows[f"feature_{leg}_expiration"])
        selected_exp_max = selected_exp_max.mask(selected & (selected_exp_max.isna() | (exp > selected_exp_max)), exp)

        entry_ok = (entry_bid > 0) & (entry_ask > entry_bid) & (entry_mid > 0) & np.isfinite(entry_spread) & (entry_spread <= max_spread_pct)
        payoff = np.maximum(spot - strike, 0.0)
        payoff_ok = np.isfinite(payoff)
        rows["is_valid_backtest"] &= ~selected | (entry_ok & payoff_ok)
        rows["entry_gross_exposure"] += np.where(selected, np.abs(entry_mid * qty * CONFIG.labels.option_multiplier), 0.0)

        long_pnl = (payoff - entry_ask) * qty * CONFIG.labels.option_multiplier
        short_pnl = (entry_bid - payoff) * qty * CONFIG.labels.option_multiplier
        leg_pnl = np.where(side == 1.0, long_pnl, short_pnl)
        rows["price_pnl"] += np.where(selected, leg_pnl, 0.0)
        rows["transaction_cost"] += np.where(
            selected,
            2.0 * CONFIG.labels.commission_per_contract * np.abs(qty),
            0.0,
        )

    gross_ok = rows["entry_gross_exposure"].gt(0) & np.isfinite(rows["entry_gross_exposure"])
    rows["is_valid_backtest"] &= gross_ok
    rows.loc[~gross_ok, "exit_reason"] = "bad_entry_gross"
    rows.loc[rows["is_valid_backtest"] & selected_exp_max.notna(), "exit_date"] = selected_exp_max[rows["is_valid_backtest"] & selected_exp_max.notna()]
    rows.loc[rows["is_valid_backtest"], "net_pnl"] = (
        rows.loc[rows["is_valid_backtest"], "price_pnl"]
        - rows.loc[rows["is_valid_backtest"], "transaction_cost"]
    )
    rows.loc[rows["is_valid_backtest"], "realized_return"] = (
        rows.loc[rows["is_valid_backtest"], "net_pnl"]
        / rows.loc[rows["is_valid_backtest"], "entry_gross_exposure"]
    )
    rows.loc[rows["is_valid_backtest"], "actual_holding_days"] = (
        pd.to_datetime(rows.loc[rows["is_valid_backtest"], "exit_date"])
        - pd.to_datetime(rows.loc[rows["is_valid_backtest"], "signal_date"])
    ).dt.days
    rows.loc[~rows["is_valid_backtest"] & rows["exit_reason"].eq("valid_expiration_payoff"), "exit_reason"] = "missing_expiration_underlying_or_bad_entry"
    return rows


def _summarize(rows: pd.DataFrame) -> dict:
    valid = rows[rows["is_valid_backtest"]]
    by_policy = (
        rows.groupby(["exit_policy", "strategy"])
        .agg(
            selected=("episode_id", "count"),
            valid=("is_valid_backtest", "sum"),
            profitable=("net_pnl", lambda x: (x > 0).sum()),
            net_pnl=("net_pnl", "sum"),
            median_return=("realized_return", "median"),
            mean_holding_days=("actual_holding_days", "mean"),
        )
        .reset_index()
    )
    by_policy["hit_rate"] = by_policy["profitable"] / by_policy["valid"].replace(0, np.nan)
    return {
        "rows": int(len(rows)),
        "episodes": int(rows["episode_id"].nunique()),
        "strategies": sorted(rows["strategy"].unique().tolist()),
        "exit_policies": sorted(rows["exit_policy"].unique().tolist()),
        "valid_rows": int(rows["is_valid_backtest"].sum()),
        "valid_net_pnl": float(valid["net_pnl"].sum()) if len(valid) else 0.0,
        "by_policy_strategy": by_policy.to_dict(orient="records"),
        "exit_reason_counts": rows["exit_reason"].value_counts(dropna=False).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build liquid-exit TP2 labels.")
    parser.add_argument("--labels-input", default="data/labels/archive2_episode_start_labels_conservative_bidask_2013_2026.parquet")
    parser.add_argument("--lifecycle-input", default="data/features/archive2_lifecycle/tp2_conservative_episode_lifecycle_2013_2026.parquet")
    parser.add_argument("--quote-input", default="data/processed/archive2_spy_call_quote_availability_nospread_2013_2026.parquet")
    parser.add_argument("--output", default="data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet")
    parser.add_argument("--summary-output", default="data/labels/archive2_liquid_exit_strategy_labels_summary_2013_2026.json")
    parser.add_argument("--max-spread-pct", type=float, default=0.50)
    args = parser.parse_args()

    root = CONFIG.paths.root
    labels = pd.read_parquet(root / args.labels_input)
    lifecycle_cols = [
        "episode_id",
        "lifecycle_date",
        "event_day_index",
        "lhs",
        "rhs",
        "normalized_pressure",
    ]
    lifecycle = pd.read_parquet(root / args.lifecycle_input, columns=lifecycle_cols)
    quote_cols = ["date", "contractID", "option_symbol", "bid", "ask", "mid", "spread_pct", "underlying_price"]
    quotes = pd.read_parquet(root / args.quote_input, columns=quote_cols)
    quotes["date"] = pd.to_datetime(quotes["date"])
    quotes["quote_key"] = _quote_key(quotes["option_symbol"], quotes["contractID"])
    calendar = _build_calendar(quotes)
    quotes = quotes.merge(calendar, on="date", how="left")

    reversion = _prepare_reversion(lifecycle, calendar)
    base = _base_strategy_rows(labels, reversion, calendar)
    tradable = _build_tradable_index(quotes, args.max_spread_pct)
    base = _assign_exit_dates(base, calendar, tradable)
    rows = _expand_exit_policies(base, calendar)
    rows = _merge_exit_quotes(rows, quotes)
    rows = _compute_pnl(rows, args.max_spread_pct)
    underlying_by_date = (
        quotes.groupby("date", as_index=False)["underlying_price"]
        .median()
        .dropna(subset=["underlying_price"])
    )
    expiration_rows = _build_expiration_payoff_rows(base, underlying_by_date, args.max_spread_pct)
    rows = pd.concat([rows, expiration_rows], ignore_index=True, sort=False)

    keep = [
        "episode_id",
        "start_violation_id",
        "signal_date",
        "strategy",
        "selected_legs",
        "exit_policy",
        "first_reversion_date",
        "first_reversion_event_day",
        "reversion_pressure",
        "exit_date",
        "exit_reason",
        "is_valid_backtest",
        "entry_gross_exposure",
        "price_pnl",
        "transaction_cost",
        "net_pnl",
        "realized_return",
        "actual_holding_days",
        "feature_normalized_severity",
        "feature_max_spread_pct",
        "feature_total_volume",
        "feature_total_open_interest",
    ]
    for leg in ["A", "B", "C", "D"]:
        keep.extend(
            [
                f"feature_{leg}_contract_id",
                f"feature_{leg}_option_symbol",
                f"feature_{leg}_expiration",
                f"feature_{leg}_strike",
                f"feature_{leg}_bid",
                f"feature_{leg}_ask",
                f"feature_{leg}_mid",
                f"feature_{leg}_spread_pct",
                f"{leg}_exit_bid",
                f"{leg}_exit_ask",
                f"{leg}_exit_mid",
                f"{leg}_exit_spread_pct",
            ]
        )

    out = rows[[c for c in keep if c in rows.columns]].copy()
    output = root / args.output
    summary_output = root / args.summary_output
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    summary = _summarize(out)
    summary.update(
        {
            "labels_input": args.labels_input,
            "lifecycle_input": args.lifecycle_input,
            "quote_input": args.quote_input,
            "output": args.output,
            "max_spread_pct": args.max_spread_pct,
        }
    )
    summary_output.write_text(json.dumps(_json_ready(summary), indent=2))
    print(json.dumps(_json_ready(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
