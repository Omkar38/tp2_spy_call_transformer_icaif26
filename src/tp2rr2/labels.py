from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LabelConfig
from .strategies import STRATEGIES, strategy_legs


def _field(row, name: str):
    if isinstance(row, pd.Series):
        return row[name]
    return getattr(row, name)


def _valid_quote(row, max_spread_pct: float) -> bool:
    if row is None:
        return False
    return (
        pd.notna(_field(row, "bid"))
        and pd.notna(_field(row, "ask"))
        and _field(row, "bid") > 0
        and _field(row, "ask") > _field(row, "bid")
        and _field(row, "mid") > 0
        and _field(row, "spread_pct") <= max_spread_pct
    )


def _quote_key(row) -> str:
    symbol = getattr(row, "option_symbol", None)
    if symbol is not None and pd.notna(symbol):
        return str(symbol)
    return str(row.contractID)


def _build_quote_lookup(options: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], pd.Series]:
    return {
        (row.date, _quote_key(row)): row
        for row in options.itertuples(index=False)
    }


def _exit_date(
    entry_date: pd.Timestamp,
    selected_expirations: list[pd.Timestamp],
    trading_days: list[pd.Timestamp],
    horizon: int,
) -> tuple[pd.Timestamp | None, str]:
    future_days = [d for d in trading_days if d > entry_date]
    if not future_days:
        return None, "no_future_day"
    target = future_days[horizon - 1] if len(future_days) >= horizon else future_days[-1]
    earliest_exp = min(selected_expirations)
    valid_before_exp = [d for d in future_days if d < earliest_exp]
    if not valid_before_exp:
        return None, "no_pre_expiration_quote_day"
    if target < earliest_exp:
        return target, f"{horizon}d"
    return valid_before_exp[-1], "pre_expiration"


def _entry_gross(row: pd.Series, legs, multiplier: float) -> float:
    gross = 0.0
    for leg in legs:
        gross += abs(float(row[f"{leg.leg}_mid"]) * leg.quantity * multiplier)
    return gross


def label_one_strategy(
    row: pd.Series,
    quote_lookup: dict[tuple[pd.Timestamp, str], pd.Series],
    trading_days: list[pd.Timestamp],
    strategy: str,
    horizon: int,
    config: LabelConfig,
) -> dict:
    legs = strategy_legs(row, strategy)
    exit_dt, exit_reason = _exit_date(
        entry_date=row["date"],
        selected_expirations=[leg.expiration for leg in legs],
        trading_days=trading_days,
        horizon=horizon,
    )
    base = {
        "violation_id": row["violation_id"],
        "strategy": strategy,
        "label_horizon": horizon,
        "exit_date": exit_dt,
        "exit_reason": exit_reason,
        "is_valid_label": False,
        "net_pnl": np.nan,
        "entry_gross_exposure": np.nan,
        "realized_return": np.nan,
        "actual_holding_days": np.nan,
    }
    if exit_dt is None:
        return base

    gross = _entry_gross(row, legs, config.option_multiplier)
    if not np.isfinite(gross) or gross <= 0:
        base["exit_reason"] = "bad_entry_gross"
        return base

    net_pnl = 0.0
    for leg in legs:
        entry_bid = float(row[f"{leg.leg}_bid"])
        entry_ask = float(row[f"{leg.leg}_ask"])
        entry_mid = float(row[f"{leg.leg}_mid"])
        entry_spread_pct = float(row[f"{leg.leg}_spread_pct"])
        if not (entry_bid > 0 and entry_ask > entry_bid and entry_mid > 0 and entry_spread_pct <= config.max_label_spread_pct):
            base["exit_reason"] = "invalid_entry_quote"
            return base

        quote_key = leg.option_symbol if leg.option_symbol else leg.contract_id
        exit_quote = quote_lookup.get((exit_dt, quote_key))
        if exit_quote is None or not _valid_quote(exit_quote, config.max_label_spread_pct):
            base["exit_reason"] = "invalid_exit_quote"
            return base

        qty = leg.quantity
        if leg.side == 1:
            price_pnl = (float(_field(exit_quote, "bid")) - entry_ask) * qty * config.option_multiplier
        else:
            price_pnl = (entry_bid - float(_field(exit_quote, "ask"))) * qty * config.option_multiplier
        fees = 2.0 * config.commission_per_contract * abs(qty)
        net_pnl += price_pnl - fees

    base.update(
        {
            "is_valid_label": True,
            "net_pnl": net_pnl,
            "entry_gross_exposure": gross,
            "realized_return": net_pnl / gross,
            "actual_holding_days": int((exit_dt - row["date"]).days),
        }
    )
    return base


def build_labels(
    features: pd.DataFrame,
    options: pd.DataFrame,
    config: LabelConfig,
    horizons: list[int] | None = None,
    quote_lookup: dict[tuple[pd.Timestamp, str], pd.Series] | None = None,
    trading_days: list[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    horizons = horizons or [config.horizon_trading_days, config.secondary_horizon_trading_days]
    quote_lookup = quote_lookup or _build_quote_lookup(options)
    trading_days = trading_days or sorted(options["date"].drop_duplicates())
    records = []
    for row in features.itertuples(index=False):
        series = pd.Series(row._asdict())
        for strategy in STRATEGIES:
            for horizon in horizons:
                records.append(label_one_strategy(series, quote_lookup, trading_days, strategy, horizon, config))
    return pd.DataFrame(records)


def prepare_quote_frame(options: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "contractID", "bid", "ask", "mid", "spread_pct"]
    if "option_symbol" in options.columns:
        columns.insert(2, "option_symbol")
    out = options[columns].copy()
    out["quote_key"] = out["option_symbol"].astype(str) if "option_symbol" in out.columns else out["contractID"].astype(str)
    return out


def build_strategy_training_rows(
    features: pd.DataFrame,
    quote_frame: pd.DataFrame,
    trading_days: list[pd.Timestamp],
    config: LabelConfig,
) -> pd.DataFrame:
    rows = pd.concat([features.assign(strategy=strategy) for strategy in STRATEGIES], ignore_index=True)
    n = len(rows)
    rows["label_horizon"] = config.horizon_trading_days
    rows["exit_date"] = pd.NaT
    rows["exit_reason"] = "missing_quote_skip"
    rows["is_valid_label"] = False
    rows["net_pnl"] = np.nan
    rows["entry_gross_exposure"] = np.nan
    rows["realized_return"] = np.nan
    rows["actual_holding_days"] = np.nan
    if rows.empty:
        return rows

    days = pd.DatetimeIndex(trading_days).sort_values()
    day_values = days.values
    entry_values = pd.to_datetime(rows["date"]).values
    future_idx = np.searchsorted(day_values, entry_values, side="right")
    target_idx = np.minimum(future_idx + config.horizon_trading_days - 1, len(day_values) - 1)
    has_future = future_idx < len(day_values)
    target_dates = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    target_dates[has_future] = day_values[target_idx[has_future]]

    t1_exp = pd.to_datetime(rows["T1_expiration"]).values
    t2_exp = pd.to_datetime(rows["T2_expiration"]).values
    earliest_exp = np.where(rows["strategy"].eq("T2").to_numpy(), t2_exp, t1_exp)
    last_before_exp_idx = np.searchsorted(day_values, earliest_exp, side="left") - 1
    has_pre_exp = has_future & (last_before_exp_idx >= future_idx)
    target_before_exp = has_future & (target_dates < earliest_exp)
    exit_values = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    exit_values[target_before_exp] = target_dates[target_before_exp]
    pre_exp_mask = has_pre_exp & ~target_before_exp
    exit_values[pre_exp_mask] = day_values[last_before_exp_idx[pre_exp_mask]]

    rows["exit_date"] = pd.to_datetime(exit_values)
    rows.loc[target_before_exp, "exit_reason"] = f"{config.horizon_trading_days}d"
    rows.loc[pre_exp_mask, "exit_reason"] = "pre_expiration"
    rows.loc[~has_future, "exit_reason"] = "no_future_day"
    rows.loc[has_future & ~has_pre_exp & ~target_before_exp, "exit_reason"] = "no_pre_expiration_quote_day"

    is_call = rows["option_type"].eq("call")
    leg_specs = {
        "A": {
            "selected": rows["strategy"].isin(["full4", "T1", "K1"]),
            "qty": rows["B_mid"].astype(float),
            "side": np.where(is_call, 1, -1),
        },
        "B": {
            "selected": rows["strategy"].isin(["full4", "T2", "K2"]),
            "qty": rows["A_mid"].astype(float),
            "side": np.where(is_call, 1, -1),
        },
        "C": {
            "selected": rows["strategy"].isin(["full4", "T2", "K1"]),
            "qty": rows["D_mid"].astype(float),
            "side": np.where(is_call, -1, 1),
        },
        "D": {
            "selected": rows["strategy"].isin(["full4", "T1", "K2"]),
            "qty": rows["C_mid"].astype(float),
            "side": np.where(is_call, -1, 1),
        },
    }

    valid = pd.Series(has_pre_exp | target_before_exp, index=rows.index)
    gross = np.zeros(n, dtype=float)
    net_pnl = np.zeros(n, dtype=float)

    quote_lookup = quote_frame.rename(
        columns={
            "date": "exit_date",
            "bid": "exit_bid",
            "ask": "exit_ask",
            "mid": "exit_mid",
            "spread_pct": "exit_spread_pct",
        }
    )

    for leg, spec in leg_specs.items():
        selected = spec["selected"].to_numpy()
        qty = spec["qty"].to_numpy(dtype=float)
        side = spec["side"].astype(float)
        entry_bid = rows[f"{leg}_bid"].to_numpy(dtype=float)
        entry_ask = rows[f"{leg}_ask"].to_numpy(dtype=float)
        entry_mid = rows[f"{leg}_mid"].to_numpy(dtype=float)
        entry_spread_pct = rows[f"{leg}_spread_pct"].to_numpy(dtype=float)
        entry_valid = (
            (entry_bid > 0)
            & (entry_ask > entry_bid)
            & (entry_mid > 0)
            & (entry_spread_pct <= config.max_label_spread_pct)
        )
        valid &= ~selected | entry_valid
        gross += np.where(selected, np.abs(entry_mid * qty * config.option_multiplier), 0.0)

        key_mask = selected & pd.notna(rows["exit_date"]).to_numpy()
        exit_bid = np.full(n, np.nan, dtype=float)
        exit_ask = np.full(n, np.nan, dtype=float)
        exit_mid = np.full(n, np.nan, dtype=float)
        exit_spread_pct = np.full(n, np.nan, dtype=float)
        if key_mask.any():
            keys = pd.DataFrame(
                {
                    "_row": np.flatnonzero(key_mask),
                    "exit_date": rows.loc[key_mask, "exit_date"].to_numpy(),
                    "quote_key": (
                        rows.loc[key_mask, f"{leg}_option_symbol"].astype(str).to_numpy()
                        if f"{leg}_option_symbol" in rows.columns
                        else rows.loc[key_mask, f"{leg}_contract_id"].astype(str).to_numpy()
                    ),
                }
            )
            merged = keys.merge(quote_lookup, on=["exit_date", "quote_key"], how="left")
            row_idx = merged["_row"].to_numpy(dtype=int)
            exit_bid[row_idx] = merged["exit_bid"].to_numpy(dtype=float)
            exit_ask[row_idx] = merged["exit_ask"].to_numpy(dtype=float)
            exit_mid[row_idx] = merged["exit_mid"].to_numpy(dtype=float)
            exit_spread_pct[row_idx] = merged["exit_spread_pct"].to_numpy(dtype=float)

        exit_valid = (
            (exit_bid > 0)
            & (exit_ask > exit_bid)
            & (exit_mid > 0)
            & (exit_spread_pct <= config.max_label_spread_pct)
        )
        valid &= ~selected | exit_valid
        long_pnl = (exit_bid - entry_ask) * qty * config.option_multiplier
        short_pnl = (entry_bid - exit_ask) * qty * config.option_multiplier
        price_pnl = np.where(side == 1, long_pnl, short_pnl)
        fees = 2.0 * config.commission_per_contract * np.abs(qty)
        net_pnl += np.where(selected, price_pnl - fees, 0.0)

    valid &= np.isfinite(gross) & (gross > 0)
    gross_valid = np.isfinite(gross) & (gross > 0)
    rows.loc[gross_valid, "entry_gross_exposure"] = gross[gross_valid]
    rows.loc[valid, "is_valid_label"] = True
    rows.loc[valid, "net_pnl"] = net_pnl[valid]
    rows.loc[valid, "realized_return"] = net_pnl[valid] / gross[valid]
    rows.loc[valid, "actual_holding_days"] = (
        pd.to_datetime(rows.loc[valid, "exit_date"]) - pd.to_datetime(rows.loc[valid, "date"])
    ).dt.days
    rows.loc[valid & ~np.isfinite(rows["realized_return"].astype(float)), "is_valid_label"] = False
    rows.loc[~valid & rows["exit_date"].notna(), "exit_reason"] = "invalid_quote"
    return rows
