#!/usr/bin/env python3
"""Build event-time lifecycle panel for conservative TP2 episode starts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tp2rr2.config import CONFIG
from tp2rr2.detect_violations import _select_strikes


QUOTE_COLS = [
    "date",
    "expiration",
    "contractID",
    "option_symbol",
    "strike",
    "bid",
    "ask",
    "mid",
    "spread_pct",
    "tradable_under_50pct_spread",
    "volume",
    "open_interest",
    "delta",
    "forward",
    "dte",
]


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def _leg_quote_columns(prefix: str) -> dict[str, str]:
    return {
        "contractID": f"{prefix}_contract_id",
        "option_symbol": f"{prefix}_option_symbol",
        "strike": f"{prefix}_strike",
        "bid": f"{prefix}_bid",
        "ask": f"{prefix}_ask",
        "mid": f"{prefix}_mid",
        "spread_pct": f"{prefix}_spread_pct",
        "tradable_under_50pct_spread": f"{prefix}_tradable_under_50pct_spread",
        "volume": f"{prefix}_volume",
        "open_interest": f"{prefix}_open_interest",
        "delta": f"{prefix}_delta",
    }


def _prepare_episode_frame(labels: pd.DataFrame, calendar: pd.DataFrame, lookback: int) -> pd.DataFrame:
    labels = labels.copy()
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    labels["feature_T1_expiration"] = pd.to_datetime(labels["feature_T1_expiration"])
    labels["feature_T2_expiration"] = pd.to_datetime(labels["feature_T2_expiration"])

    calendar = calendar.reset_index(drop=True).copy()
    calendar["trading_day_index"] = np.arange(len(calendar), dtype=np.int32)
    cal = calendar.rename(columns={"date": "signal_date"})
    episodes = labels.merge(cal, on="signal_date", how="left")
    if episodes["trading_day_index"].isna().any():
        missing = int(episodes["trading_day_index"].isna().sum())
        raise ValueError(f"{missing} episode starts have no calendar index")

    episodes["start_lifecycle_index"] = (episodes["trading_day_index"] - lookback).clip(lower=0)
    t1_cal = calendar.rename(columns={"date": "feature_T1_expiration", "trading_day_index": "t1_index"})
    episodes = episodes.merge(t1_cal, on="feature_T1_expiration", how="left")
    episodes["t1_index"] = episodes["t1_index"].fillna(episodes["trading_day_index"]).astype(int)
    episodes["end_lifecycle_index"] = episodes[["t1_index", "trading_day_index"]].max(axis=1).astype(int)
    return episodes


def _build_lifecycle_skeleton(episodes: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for row in episodes[
        [
            "episode_id",
            "start_violation_id",
            "signal_date",
            "pair_key",
            "trading_day_index",
            "start_lifecycle_index",
            "end_lifecycle_index",
            "feature_A_contract_id",
            "feature_B_contract_id",
            "feature_C_contract_id",
            "feature_D_contract_id",
            "feature_A_strike",
            "feature_B_strike",
            "feature_C_strike",
            "feature_D_strike",
            "feature_T1_expiration",
            "feature_T2_expiration",
            "feature_K1",
            "feature_K2",
            "feature_normalized_severity",
            "label_end_state",
        ]
    ].itertuples(index=False):
        idx = np.arange(row.start_lifecycle_index, row.end_lifecycle_index + 1, dtype=np.int32)
        frame = pd.DataFrame(
            {
                "episode_id": row.episode_id,
                "start_violation_id": row.start_violation_id,
                "pair_key": row.pair_key,
                "signal_date": row.signal_date,
                "episode_start_index": row.trading_day_index,
                "lifecycle_index": idx,
                "event_day_index": idx - int(row.trading_day_index),
                "A_entry_contract_id": str(row.feature_A_contract_id),
                "B_entry_contract_id": str(row.feature_B_contract_id),
                "C_entry_contract_id": str(row.feature_C_contract_id),
                "D_entry_contract_id": str(row.feature_D_contract_id),
                "A_entry_strike": float(row.feature_A_strike),
                "B_entry_strike": float(row.feature_B_strike),
                "C_entry_strike": float(row.feature_C_strike),
                "D_entry_strike": float(row.feature_D_strike),
                "T1_expiration": row.feature_T1_expiration,
                "T2_expiration": row.feature_T2_expiration,
                "K1": float(row.feature_K1),
                "K2": float(row.feature_K2),
                "entry_normalized_severity": float(row.feature_normalized_severity),
                "label_end_state": row.label_end_state,
            }
        )
        pieces.append(frame)
    out = pd.concat(pieces, ignore_index=True)
    calendar = calendar.reset_index(drop=True).copy()
    calendar["lifecycle_index"] = np.arange(len(calendar), dtype=np.int32)
    out = out.merge(calendar.rename(columns={"date": "lifecycle_date"}), on="lifecycle_index", how="left")
    out["days_to_T1"] = (pd.to_datetime(out["T1_expiration"]) - pd.to_datetime(out["lifecycle_date"])).dt.days
    out["days_to_T2"] = (pd.to_datetime(out["T2_expiration"]) - pd.to_datetime(out["lifecycle_date"])).dt.days
    return out


def _quote_frame_for_date(quotes: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    q = quotes[quotes["date"].eq(date)].copy()
    q["contractID"] = q["contractID"].astype(str)
    return q


def _detector_equivalent_chains(quotes: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, chain in quotes.groupby("expiration", sort=False):
        pieces.append(_select_strikes(chain, CONFIG.detection))
    if not pieces:
        return quotes.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def _merge_leg_by_contract(rows: pd.DataFrame, quotes: pd.DataFrame, leg: str, key_col: str) -> pd.DataFrame:
    leg_cols = ["contractID", "option_symbol", "strike", "bid", "ask", "mid", "spread_pct",
                "tradable_under_50pct_spread", "volume", "open_interest", "delta"]
    leg_quotes = quotes[leg_cols].rename(columns=_leg_quote_columns(leg))
    out = rows.merge(
        leg_quotes,
        left_on=key_col,
        right_on=f"{leg}_contract_id",
        how="left",
    )
    return out


def _roundup_leg(
    rows: pd.DataFrame,
    quotes: pd.DataFrame,
    target_exp_col: str,
    target_strike: np.ndarray,
    leg: str,
) -> pd.DataFrame:
    rows = rows.copy()
    for source, col in _leg_quote_columns(leg).items():
        if source in {"contractID", "option_symbol"}:
            rows[col] = pd.NA
        elif source == "tradable_under_50pct_spread":
            rows[col] = pd.NA
        else:
            rows[col] = np.nan
    rows[f"{leg}_target_strike"] = target_strike

    result_pieces = []
    row_idx_name = "_row_order"
    rows[row_idx_name] = np.arange(len(rows))
    for exp, sub in rows.groupby(target_exp_col, sort=False):
        chain = quotes[quotes["expiration"].eq(exp)].sort_values("strike").reset_index(drop=True)
        if chain.empty:
            result_pieces.append(sub)
            continue
        strikes = chain["strike"].to_numpy(dtype=float)
        targets = sub[f"{leg}_target_strike"].to_numpy(dtype=float)
        idx = np.searchsorted(strikes, targets, side="left")
        ok = idx < len(strikes)
        if ok.any():
            selected = chain.iloc[idx[ok]].reset_index(drop=True)
            selected = selected[
                [
                    "contractID",
                    "option_symbol",
                    "strike",
                    "bid",
                    "ask",
                    "mid",
                    "spread_pct",
                    "tradable_under_50pct_spread",
                    "volume",
                    "open_interest",
                    "delta",
                ]
            ].rename(columns=_leg_quote_columns(leg))
            sub_ok_index = sub.index[ok]
            for col in selected.columns:
                sub.loc[sub_ok_index, col] = selected[col].to_numpy()
        result_pieces.append(sub)
    out = pd.concat(result_pieces, ignore_index=True).sort_values(row_idx_name)
    return out.drop(columns=[row_idx_name])


def build_lifecycle(
    labels: pd.DataFrame,
    quotes: pd.DataFrame,
    lookback: int,
    max_episodes: int | None = None,
) -> pd.DataFrame:
    quotes = quotes.copy()
    quotes["date"] = pd.to_datetime(quotes["date"])
    quotes["expiration"] = pd.to_datetime(quotes["expiration"])
    quotes["contractID"] = quotes["contractID"].astype(str)
    quotes = quotes.sort_values(["date", "expiration", "strike"]).reset_index(drop=True)
    calendar = pd.DataFrame({"date": sorted(quotes["date"].drop_duplicates())})

    if max_episodes is not None:
        labels = labels.head(max_episodes).copy()
    episodes = _prepare_episode_frame(labels, calendar, lookback)
    skeleton = _build_lifecycle_skeleton(episodes, calendar)

    outputs = []
    for i, (date, day_rows) in enumerate(skeleton.groupby("lifecycle_date", sort=True), 1):
        q = _quote_frame_for_date(quotes, date)
        q_round = _detector_equivalent_chains(q)
        rows = day_rows.copy()
        rows["A_entry_contract_id"] = rows["A_entry_contract_id"].astype(str)
        rows["B_entry_contract_id"] = rows["B_entry_contract_id"].astype(str)
        rows["C_entry_contract_id"] = rows["C_entry_contract_id"].astype(str)
        rows["D_entry_contract_id"] = rows["D_entry_contract_id"].astype(str)
        rows = _merge_leg_by_contract(rows, q, "A", "A_entry_contract_id")
        rows = _merge_leg_by_contract(rows, q, "B", "B_entry_contract_id")

        fwd = (
            q_round.groupby("expiration", as_index=False)["forward"]
            .median()
            .rename(columns={"forward": "chain_forward"})
        )
        rows = rows.merge(
            fwd.rename(columns={"expiration": "T1_expiration", "chain_forward": "F_T1_lifecycle"}),
            on="T1_expiration",
            how="left",
        )
        rows = rows.merge(
            fwd.rename(columns={"expiration": "T2_expiration", "chain_forward": "F_T2_lifecycle"}),
            on="T2_expiration",
            how="left",
        )

        rows["status"] = "observable"
        rows.loc[rows["lifecycle_date"].gt(rows["T1_expiration"]), "status"] = "expired_T1"
        missing_ab = rows["A_mid"].isna() | rows["B_mid"].isna()
        rows.loc[missing_ab & rows["status"].eq("observable"), "status"] = "missing_A_or_B"
        bad_forward = (
            rows["F_T1_lifecycle"].isna()
            | rows["F_T2_lifecycle"].isna()
            | (rows["F_T1_lifecycle"] <= 0)
            | (rows["F_T2_lifecycle"] <= 0)
        )
        rows.loc[bad_forward & rows["status"].eq("observable"), "status"] = "missing_forward"
        forward_order = (rows["K1"] / rows["F_T1_lifecycle"]) < (rows["K2"] / rows["F_T2_lifecycle"])
        rows.loc[(~forward_order) & rows["status"].eq("observable"), "status"] = "not_forward_ordered"

        rows["C_target_strike"] = rows["K1"] * rows["F_T2_lifecycle"] / rows["F_T1_lifecycle"]
        rows["D_target_strike"] = rows["K2"] * rows["F_T1_lifecycle"] / rows["F_T2_lifecycle"]
        for leg in ["C", "D"]:
            for source, col in _leg_quote_columns(leg).items():
                if col not in rows.columns:
                    if source in {"contractID", "option_symbol", "tradable_under_50pct_spread"}:
                        rows[col] = pd.NA
                    else:
                        rows[col] = np.nan
        candidates = rows["status"].eq("observable")
        event_day = candidates & rows["event_day_index"].eq(0)
        if event_day.any():
            sub0 = rows.loc[event_day].copy()
            sub0 = sub0.drop(columns=list(_leg_quote_columns("C").values()), errors="ignore")
            sub0 = sub0.drop(columns=list(_leg_quote_columns("D").values()), errors="ignore")
            sub0 = _merge_leg_by_contract(sub0, q, "C", "C_entry_contract_id")
            sub0 = _merge_leg_by_contract(sub0, q, "D", "D_entry_contract_id")
            for col in sub0.columns:
                rows.loc[event_day, col] = sub0[col].to_numpy()

        dynamic_candidates = candidates & ~rows["event_day_index"].eq(0)
        if dynamic_candidates.any():
            sub = rows.loc[dynamic_candidates].copy()
            sub = _roundup_leg(sub, q_round, "T2_expiration", sub["C_target_strike"].to_numpy(float), "C")
            sub = _roundup_leg(sub, q_round, "T1_expiration", sub["D_target_strike"].to_numpy(float), "D")
            for col in sub.columns:
                rows.loc[dynamic_candidates, col] = sub[col].to_numpy()

        missing_cd = rows["C_mid"].isna() | rows["D_mid"].isna()
        rows.loc[missing_cd & rows["status"].eq("observable"), "status"] = "missing_C_or_D"

        trade_cols = [f"{leg}_tradable_under_50pct_spread" for leg in ["A", "B", "C", "D"]]
        rows["all_legs_tradable_under_50pct_spread"] = rows[trade_cols].fillna(False).all(axis=1)
        rows["any_leg_wide_spread_over_50pct"] = ~rows["all_legs_tradable_under_50pct_spread"]
        rows.loc[
            rows["status"].eq("observable") & (~rows["all_legs_tradable_under_50pct_spread"]),
            "status",
        ] = "liquidity_failure_wide_spread"

        rows["lhs"] = rows["A_mid"] * rows["B_mid"]
        rows["rhs"] = rows["C_mid"] * rows["D_mid"]
        rows["violation_pressure"] = rows["rhs"] - rows["lhs"]
        gross = rows["rhs"] + rows["lhs"]
        rows["normalized_pressure"] = np.where(gross > 0, rows["violation_pressure"] / gross, np.nan)
        rows["log_pressure"] = np.log(rows["rhs"] / rows["lhs"])
        rows["is_violation"] = rows["violation_pressure"] > 0
        rows["is_conservative_bidask_violation"] = (
            (rows["A_ask"] * rows["B_ask"]) < (rows["C_bid"] * rows["D_bid"])
        )

        keep = [
            "episode_id",
            "start_violation_id",
            "pair_key",
            "signal_date",
            "lifecycle_date",
            "event_day_index",
            "days_to_T1",
            "days_to_T2",
            "T1_expiration",
            "T2_expiration",
            "K1",
            "K2",
            "status",
            "label_end_state",
            "entry_normalized_severity",
            "F_T1_lifecycle",
            "F_T2_lifecycle",
            "C_target_strike",
            "D_target_strike",
            "lhs",
            "rhs",
            "violation_pressure",
            "normalized_pressure",
            "log_pressure",
            "is_violation",
            "is_conservative_bidask_violation",
            "all_legs_tradable_under_50pct_spread",
            "any_leg_wide_spread_over_50pct",
        ]
        for leg in ["A", "B", "C", "D"]:
            keep.extend(
                [
                    f"{leg}_contract_id",
                    f"{leg}_option_symbol",
                    f"{leg}_strike",
                    f"{leg}_bid",
                    f"{leg}_ask",
                    f"{leg}_mid",
                    f"{leg}_spread_pct",
                    f"{leg}_tradable_under_50pct_spread",
                    f"{leg}_volume",
                    f"{leg}_open_interest",
                    f"{leg}_delta",
                ]
            )
        outputs.append(rows[[c for c in keep if c in rows.columns]].copy())
        if i % 250 == 0:
            print(f"Processed lifecycle date {i:,}/{skeleton['lifecycle_date'].nunique():,}", flush=True)

    return pd.concat(outputs, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TP2 violation lifecycle panel.")
    parser.add_argument(
        "--labels-input",
        default="data/labels/archive2_episode_start_labels_conservative_bidask_2013_2026.parquet",
    )
    parser.add_argument(
        "--quote-input",
        default="data/processed/archive2_spy_call_quote_availability_nospread_2013_2026.parquet",
    )
    parser.add_argument(
        "--unit-backtest-input",
        default="data/labels/archive2_episode_strategy_mini_backtest_unit_conservative_bidask_2013_2026.parquet",
    )
    parser.add_argument(
        "--product-backtest-input",
        default="data/labels/archive2_episode_strategy_mini_backtest_conservative_bidask_2013_2026.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/features/archive2_lifecycle/tp2_conservative_episode_lifecycle_2013_2026.parquet",
    )
    parser.add_argument(
        "--summary-output",
        default="data/features/archive2_lifecycle/tp2_conservative_episode_lifecycle_summary_2013_2026.json",
    )
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    root = CONFIG.paths.root
    labels_path = root / args.labels_input
    quote_path = root / args.quote_input
    output_path = root / args.output
    summary_path = root / args.summary_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    label_cols = [
        "episode_id",
        "start_violation_id",
        "signal_date",
        "pair_key",
        "feature_A_contract_id",
        "feature_B_contract_id",
        "feature_C_contract_id",
        "feature_D_contract_id",
        "feature_A_strike",
        "feature_B_strike",
        "feature_C_strike",
        "feature_D_strike",
        "feature_T1_expiration",
        "feature_T2_expiration",
        "feature_K1",
        "feature_K2",
        "feature_normalized_severity",
        "label_end_state",
    ]
    labels = pd.read_parquet(labels_path, columns=label_cols)
    quotes = pd.read_parquet(quote_path, columns=QUOTE_COLS)
    print(f"Loaded {len(labels):,} episode starts and {len(quotes):,} quote rows", flush=True)

    lifecycle = build_lifecycle(labels, quotes, args.lookback, args.max_episodes)

    for prefix, backtest_arg in [
        ("product", args.product_backtest_input),
        ("unit", args.unit_backtest_input),
    ]:
        backtest_path = root / backtest_arg
        if not backtest_path.exists():
            continue
        backtest = pd.read_parquet(
            backtest_path,
            columns=["episode_id", "strategy", "is_valid_backtest", "net_pnl", "realized_return"],
        )
        pivot = backtest.pivot_table(
            index="episode_id",
            columns="strategy",
            values=["is_valid_backtest", "net_pnl", "realized_return"],
            aggfunc="first",
        )
        pivot.columns = [f"{prefix}_{metric}_{strategy}" for metric, strategy in pivot.columns]
        lifecycle = lifecycle.merge(pivot.reset_index(), on="episode_id", how="left")

    lifecycle.to_parquet(output_path, index=False)

    summary = {
        "labels_input": args.labels_input,
        "quote_input": args.quote_input,
        "product_backtest_input": args.product_backtest_input,
        "unit_backtest_input": args.unit_backtest_input,
        "output": args.output,
        "lookback_sessions": args.lookback,
        "max_episodes": args.max_episodes,
        "episode_count": int(lifecycle["episode_id"].nunique()),
        "lifecycle_rows": int(len(lifecycle)),
        "event_day_min": int(lifecycle["event_day_index"].min()),
        "event_day_max": int(lifecycle["event_day_index"].max()),
        "status_counts": {
            str(k): int(v) for k, v in lifecycle["status"].value_counts(dropna=False).items()
        },
        "observable_rows": int(lifecycle["status"].eq("observable").sum()),
        "liquidity_failure_wide_spread_rows": int(
            lifecycle["status"].eq("liquidity_failure_wide_spread").sum()
        ),
        "violation_rows": int(lifecycle["is_violation"].fillna(False).sum()),
        "conservative_bidask_violation_rows": int(
            lifecycle["is_conservative_bidask_violation"].fillna(False).sum()
        ),
    }
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2))
    print(json.dumps(_json_ready(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
