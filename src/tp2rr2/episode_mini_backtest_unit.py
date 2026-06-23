"""Unit-sized episode mini-backtest.

Identical exit logic to episode_mini_backtest.py but uses qty=1 contract per
selected leg instead of the product-replication weighting (qty = opposing leg's
mid price). This makes position sizes comparable across years and across option
price levels, and directly replicable in practice.

Key differences from episode_mini_backtest.py:
  - qty = 1.0 for every selected leg (unit contract sizing)
  - gross = sum(|leg_mid| * 100) for selected legs only
  - fees = 2 * commission * 1.0 per selected leg (not * abs(B_mid))
  - optional max_delta filter: drops rows where any selected leg has
    abs(delta) outside [min_abs_delta, max_abs_delta] when those columns exist

Nothing in the existing pipeline is overwritten by this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LabelConfig


STRATEGIES = ["T1", "T2", "K1", "K2"]
SELECTED_LEGS = {
    "T1": ["A", "D"],
    "T2": ["B", "C"],
    "K1": ["A", "C"],
    "K2": ["B", "D"],
}

_LEG_SIDE_CALL = {"A": 1.0, "B": 1.0, "C": -1.0, "D": -1.0}
_LEG_SIDE_PUT  = {"A": -1.0, "B": -1.0, "C": 1.0, "D": 1.0}


@dataclass(frozen=True)
class UnitBacktestResult:
    rows: pd.DataFrame
    summary: dict


def _entry_valid(rows: pd.DataFrame, leg: str, max_spread_pct: float) -> np.ndarray:
    bid    = rows[f"feature_{leg}_bid"].to_numpy(dtype=float)
    ask    = rows[f"feature_{leg}_ask"].to_numpy(dtype=float)
    mid    = rows[f"feature_{leg}_mid"].to_numpy(dtype=float)
    spread = rows[f"feature_{leg}_spread_pct"].to_numpy(dtype=float)
    return (bid > 0) & (ask > bid) & (mid > 0) & np.isfinite(spread) & (spread <= max_spread_pct)


def _exit_valid(exit_bid: np.ndarray, exit_ask: np.ndarray,
                exit_mid: np.ndarray, exit_spread: np.ndarray,
                max_spread_pct: float) -> np.ndarray:
    return (
        (exit_bid > 0) & (exit_ask > exit_bid) & (exit_mid > 0)
        & np.isfinite(exit_spread) & (exit_spread <= max_spread_pct)
    )


def _delta_filter_mask(rows: pd.DataFrame, legs: list[str],
                       min_delta: float, max_delta: float) -> np.ndarray:
    """Return True for rows where ALL selected legs have abs(delta) in bounds.

    Silently passes (all True) if delta columns are not present in the frame.
    """
    n = len(rows)
    mask = np.ones(n, dtype=bool)
    for leg in legs:
        col = f"feature_{leg}_delta"
        if col not in rows.columns:
            continue
        delta = pd.to_numeric(rows[col], errors="coerce").to_numpy(dtype=float)
        abs_delta = np.abs(delta)
        valid = np.isfinite(abs_delta) & (abs_delta >= min_delta) & (abs_delta <= max_delta)
        mask &= valid
    return mask


def build_unit_sized_backtest_rows(
    episode_labels: pd.DataFrame,
    quote_frame: pd.DataFrame,
    config: LabelConfig,
    min_abs_delta: float = 0.0,
    max_abs_delta: float = 1.0,
    max_entry_gross: float | None = None,
) -> UnitBacktestResult:
    """Build unit-sized backtest rows.

    Parameters
    ----------
    episode_labels  : one row per episode start (conservative bid/ask labels)
    quote_frame     : exit-date quote lookup, columns:
                      [exit_date, quote_key, exit_bid, exit_ask, exit_mid, exit_spread_pct]
    config          : LabelConfig (commission_per_contract, option_multiplier, max_label_spread_pct)
    min_abs_delta   : minimum abs(delta) for all selected legs (0.0 = no filter)
    max_abs_delta   : maximum abs(delta) for all selected legs (1.0 = no filter)
    max_entry_gross : if not None, invalidate rows whose entry_gross > this value
    """
    labels = episode_labels.copy()
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    labels["label_next_trading_date"] = pd.to_datetime(labels["label_next_trading_date"])
    for leg in ["A", "B", "C", "D"]:
        labels[f"feature_{leg}_expiration"] = pd.to_datetime(labels[f"feature_{leg}_expiration"])

    # Expand to one row per (episode, strategy)
    chunks = []
    for strategy in STRATEGIES:
        chunk = labels.copy()
        chunk["strategy"] = strategy
        chunk["selected_legs"] = ",".join(SELECTED_LEGS[strategy])
        chunks.append(chunk)
    rows = pd.concat(chunks, ignore_index=True)
    n = len(rows)

    rows["backtest_policy"] = "unit_sized_true_reversion_exit"
    rows["exit_date"]   = pd.to_datetime(rows["label_next_trading_date"])
    rows["exit_reason"] = np.where(
        rows["label_end_state"].eq("true_reversion"),
        "true_reversion_exit",
        rows["label_end_state"].astype(str),
    )
    rows["is_valid_backtest"]   = False
    rows["entry_gross_exposure"] = np.nan
    rows["price_pnl"]           = np.nan
    rows["transaction_cost"]    = np.nan
    rows["net_pnl"]             = np.nan
    rows["realized_return"]     = np.nan
    rows["actual_holding_days"] = np.nan

    is_call  = rows["feature_option_type"].eq("call").to_numpy()
    valid    = rows["label_end_state"].eq("true_reversion").to_numpy().copy()
    valid   &= rows["exit_date"].notna().to_numpy()

    gross           = np.zeros(n, dtype=float)
    price_pnl_arr   = np.zeros(n, dtype=float)
    txn_cost_arr    = np.zeros(n, dtype=float)
    selected_exp_min = np.full(n, np.datetime64("2262-04-11"), dtype="datetime64[ns]")

    quote_lookup = quote_frame.copy()
    quote_lookup["exit_date"] = pd.to_datetime(quote_lookup["exit_date"])

    for leg in ["A", "B", "C", "D"]:
        selected = rows["strategy"].map(lambda name, l=leg: l in SELECTED_LEGS[name]).to_numpy()
        qty  = np.ones(n, dtype=float)   # UNIT SIZING — always 1 contract
        side = np.where(is_call, float(_LEG_SIDE_CALL[leg]), float(_LEG_SIDE_PUT[leg]))

        entry_bid = rows[f"feature_{leg}_bid"].to_numpy(dtype=float)
        entry_ask = rows[f"feature_{leg}_ask"].to_numpy(dtype=float)
        entry_mid = rows[f"feature_{leg}_mid"].to_numpy(dtype=float)
        leg_exp   = pd.to_datetime(rows[f"feature_{leg}_expiration"]).to_numpy(dtype="datetime64[ns]")

        selected_exp_min = np.where(
            selected & (leg_exp < selected_exp_min), leg_exp, selected_exp_min
        )
        valid &= ~selected | _entry_valid(rows, leg, config.max_label_spread_pct)
        gross += np.where(selected, np.abs(entry_mid * qty * config.option_multiplier), 0.0)

        # Look up exit quotes
        exit_bid  = np.full(n, np.nan, dtype=float)
        exit_ask  = np.full(n, np.nan, dtype=float)
        exit_mid  = np.full(n, np.nan, dtype=float)
        exit_spr  = np.full(n, np.nan, dtype=float)
        key_mask  = selected & rows["exit_date"].notna().to_numpy()
        if key_mask.any():
            sym_col = f"feature_{leg}_option_symbol"
            cid_col = f"feature_{leg}_contract_id"
            sym = rows.loc[key_mask, sym_col] if sym_col in rows.columns else pd.Series(dtype=str)
            cid = rows.loc[key_mask, cid_col] if cid_col in rows.columns else pd.Series(dtype=str)
            quote_key = sym.where(sym.notna(), cid).astype(str).to_numpy()
            keys_df = pd.DataFrame({
                "_row":      np.flatnonzero(key_mask),
                "exit_date": rows.loc[key_mask, "exit_date"].to_numpy(),
                "quote_key": quote_key,
            })
            merged = keys_df.merge(quote_lookup, on=["exit_date", "quote_key"], how="left")
            idx = merged["_row"].to_numpy(dtype=int)
            exit_bid[idx] = merged["exit_bid"].to_numpy(dtype=float)
            exit_ask[idx] = merged["exit_ask"].to_numpy(dtype=float)
            exit_mid[idx] = merged["exit_mid"].to_numpy(dtype=float)
            exit_spr[idx] = merged["exit_spread_pct"].to_numpy(dtype=float)

        rows[f"{leg}_exit_bid"]        = exit_bid
        rows[f"{leg}_exit_ask"]        = exit_ask
        rows[f"{leg}_exit_mid"]        = exit_mid
        rows[f"{leg}_exit_spread_pct"] = exit_spr

        leg_exit_ok = _exit_valid(exit_bid, exit_ask, exit_mid, exit_spr, config.max_label_spread_pct)
        valid &= ~selected | leg_exit_ok

        long_pnl  = (exit_bid - entry_ask) * qty * config.option_multiplier
        short_pnl = (entry_bid - exit_ask) * qty * config.option_multiplier
        leg_pnl   = np.where(side == 1.0, long_pnl, short_pnl)
        leg_fees  = 2.0 * config.commission_per_contract * qty     # qty=1 always

        price_pnl_arr += np.where(selected, leg_pnl,  0.0)
        txn_cost_arr  += np.where(selected, leg_fees, 0.0)

    # Apply pre-expiration guard
    exit_ts      = pd.to_datetime(rows["exit_date"]).to_numpy(dtype="datetime64[ns]")
    pre_exp_exit = exit_ts < selected_exp_min
    valid       &= pre_exp_exit
    rows.loc[rows["label_end_state"].eq("true_reversion") & ~pre_exp_exit,
             "exit_reason"] = "exit_on_or_after_selected_expiration"

    # Apply delta filter (per-strategy: only selected legs checked)
    delta_ok = np.ones(n, dtype=bool)
    if min_abs_delta > 0.0 or max_abs_delta < 1.0:
        for strategy, legs in SELECTED_LEGS.items():
            strat_mask = rows["strategy"].eq(strategy).to_numpy()
            leg_ok = _delta_filter_mask(rows, legs, min_abs_delta, max_abs_delta)
            delta_ok = np.where(strat_mask, leg_ok, delta_ok)
    valid &= delta_ok
    rows.loc[~delta_ok & rows["label_end_state"].eq("true_reversion"), "exit_reason"] = "delta_filter_rejected"

    # Apply position size cap
    gross_valid = np.isfinite(gross) & (gross > 0)
    if max_entry_gross is not None:
        gross_valid &= gross <= max_entry_gross
        rows.loc[np.isfinite(gross) & (gross > 0) & (gross > max_entry_gross)
                 & rows["label_end_state"].eq("true_reversion"),
                 "exit_reason"] = "position_too_large"

    valid &= gross_valid
    rows.loc[~gross_valid & rows["label_end_state"].eq("true_reversion")
             & delta_ok & pre_exp_exit, "exit_reason"] = "bad_entry_gross"

    invalid_quote = (
        rows["label_end_state"].eq("true_reversion").to_numpy()
        & gross_valid & pre_exp_exit & delta_ok & ~valid
    )
    rows.loc[invalid_quote, "exit_reason"] = "invalid_entry_or_exit_quote"

    # Write results
    rows.loc[gross_valid, "entry_gross_exposure"] = gross[gross_valid]
    rows.loc[valid, "is_valid_backtest"]   = True
    rows.loc[valid, "price_pnl"]           = price_pnl_arr[valid]
    rows.loc[valid, "transaction_cost"]    = txn_cost_arr[valid]
    rows.loc[valid, "net_pnl"]             = price_pnl_arr[valid] - txn_cost_arr[valid]
    rows.loc[valid, "realized_return"]     = (
        rows.loc[valid, "net_pnl"] / rows.loc[valid, "entry_gross_exposure"]
    )
    rows.loc[valid, "actual_holding_days"] = (
        pd.to_datetime(rows.loc[valid, "exit_date"])
        - pd.to_datetime(rows.loc[valid, "signal_date"])
    ).dt.days

    keep = [
        "episode_id", "start_violation_id", "signal_date", "split_by_signal_date",
        "strategy", "selected_legs", "backtest_policy",
        "label_end_state", "label_end_state_detail",
        "duration_sessions", "episode_row_count",
        "feature_option_type", "feature_normalized_severity",
        "feature_avg_spread_pct", "feature_max_spread_pct",
        "feature_min_volume", "feature_total_volume",
        "feature_min_open_interest", "feature_total_open_interest",
        "exit_date", "exit_reason",
        "is_valid_backtest", "entry_gross_exposure",
        "price_pnl", "transaction_cost", "net_pnl",
        "realized_return", "actual_holding_days",
    ]
    for leg in ["A", "B", "C", "D"]:
        keep.extend([
            f"feature_{leg}_contract_id", f"feature_{leg}_option_symbol",
            f"feature_{leg}_expiration", f"feature_{leg}_strike",
            f"feature_{leg}_bid", f"feature_{leg}_ask", f"feature_{leg}_mid",
            f"{leg}_exit_bid", f"{leg}_exit_ask", f"{leg}_exit_mid", f"{leg}_exit_spread_pct",
        ])
    rows = rows[[c for c in keep if c in rows.columns]].copy()

    counts      = rows["exit_reason"].value_counts(dropna=False)
    by_strategy = (
        rows.groupby("strategy", dropna=False)
        .agg(
            candidate_rows=("episode_id", "size"),
            valid_rows=("is_valid_backtest", "sum"),
            mean_return=("realized_return", "mean"),
            median_return=("realized_return", "median"),
            net_pnl=("net_pnl", "sum"),
            transaction_cost=("transaction_cost", "sum"),
        )
        .reset_index()
    )
    summary = {
        "sizing": "unit_contract_per_leg",
        "min_abs_delta": min_abs_delta,
        "max_abs_delta": max_abs_delta,
        "max_entry_gross": max_entry_gross,
        "strategy_rows": int(len(rows)),
        "unique_episodes": int(rows["episode_id"].nunique()),
        "strategies": STRATEGIES,
        "valid_backtest_rows": int(rows["is_valid_backtest"].sum()),
        "valid_backtest_share": float(rows["is_valid_backtest"].mean()) if len(rows) else 0.0,
        "exit_reason_counts": {str(k): int(v) for k, v in counts.items()},
        "net_pnl_valid": float(rows.loc[rows["is_valid_backtest"], "net_pnl"].sum()),
        "transaction_cost_valid": float(rows.loc[rows["is_valid_backtest"], "transaction_cost"].sum()),
        "mean_realized_return_valid": float(rows.loc[rows["is_valid_backtest"], "realized_return"].mean()),
        "median_realized_return_valid": float(rows.loc[rows["is_valid_backtest"], "realized_return"].median()),
        "by_strategy": {
            str(r.strategy): {
                "candidate_rows":  int(r.candidate_rows),
                "valid_rows":      int(r.valid_rows),
                "mean_return":     None if pd.isna(r.mean_return)   else float(r.mean_return),
                "median_return":   None if pd.isna(r.median_return) else float(r.median_return),
                "net_pnl":         float(r.net_pnl),
                "transaction_cost": float(r.transaction_cost),
            }
            for r in by_strategy.itertuples(index=False)
        },
        "policy_note": (
            "Unit-sized backtest: 1 contract per selected leg (not product-replication qty). "
            "Exit at true-reversion next-state. Commission $0.65/side."
        ),
    }
    return UnitBacktestResult(rows=rows, summary=summary)
