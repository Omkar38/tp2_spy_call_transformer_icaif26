from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LabelConfig
from .strategies import STRATEGIES


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    mid: float
    spread_pct: float


def _quote_key(row: pd.Series, leg: str) -> str:
    symbol_col = f"{leg}_option_symbol"
    if symbol_col in row and pd.notna(row[symbol_col]):
        return str(row[symbol_col])
    return str(row[f"{leg}_contract_id"])


def _build_quote_lookup(quote_frame: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], Quote]:
    out: dict[tuple[pd.Timestamp, str], Quote] = {}
    for row in quote_frame.itertuples(index=False):
        out[(row.date, str(row.quote_key))] = Quote(
            bid=float(row.bid),
            ask=float(row.ask),
            mid=float(row.mid),
            spread_pct=float(row.spread_pct),
        )
    return out


def _valid_quote(quote: Quote | None, max_spread_pct: float) -> bool:
    return (
        quote is not None
        and math.isfinite(quote.bid)
        and math.isfinite(quote.ask)
        and math.isfinite(quote.mid)
        and math.isfinite(quote.spread_pct)
        and quote.bid > 0
        and quote.ask > quote.bid
        and quote.mid > 0
        and quote.spread_pct <= max_spread_pct
    )


def _strategy_selected_legs(strategy: str) -> list[str]:
    if strategy == "full4":
        return ["A", "B", "C", "D"]
    if strategy == "T1":
        return ["A", "D"]
    if strategy == "T2":
        return ["B", "C"]
    if strategy == "K1":
        return ["A", "C"]
    if strategy == "K2":
        return ["B", "D"]
    raise ValueError(f"Unknown strategy: {strategy}")


def _leg_qty_side(row: pd.Series, leg: str) -> tuple[float, int]:
    is_call = row["option_type"] == "call"
    if leg == "A":
        return float(row["B_mid"]), 1 if is_call else -1
    if leg == "B":
        return float(row["A_mid"]), 1 if is_call else -1
    if leg == "C":
        return float(row["D_mid"]), -1 if is_call else 1
    if leg == "D":
        return float(row["C_mid"]), -1 if is_call else 1
    raise ValueError(f"Unknown leg: {leg}")


def _selected_expiration(row: pd.Series, leg: str) -> pd.Timestamp:
    return pd.Timestamp(row[f"{leg}_expiration"])


def _entry_valid(row: pd.Series, selected_legs: list[str], max_spread_pct: float) -> bool:
    for leg in selected_legs:
        bid = float(row[f"{leg}_bid"])
        ask = float(row[f"{leg}_ask"])
        mid = float(row[f"{leg}_mid"])
        spread_pct = float(row[f"{leg}_spread_pct"])
        if not (bid > 0 and ask > bid and mid > 0 and spread_pct <= max_spread_pct):
            return False
    return True


def _exit_valid(
    exit_date: pd.Timestamp,
    selected_legs: list[str],
    keys: dict[str, str],
    quote_lookup: dict[tuple[pd.Timestamp, str], Quote],
    max_spread_pct: float,
) -> bool:
    return all(_valid_quote(quote_lookup.get((exit_date, keys[leg])), max_spread_pct) for leg in selected_legs)


def _rectangle_quotes_valid(
    date: pd.Timestamp,
    keys: dict[str, str],
    quote_lookup: dict[tuple[pd.Timestamp, str], Quote],
    max_spread_pct: float,
) -> tuple[bool, dict[str, Quote]]:
    quotes = {leg: quote_lookup.get((date, keys[leg])) for leg in ["A", "B", "C", "D"]}
    valid = all(_valid_quote(q, max_spread_pct) for q in quotes.values())
    return valid, quotes  # type: ignore[return-value]


def _rectangle_normalized_severity(quotes: dict[str, Quote]) -> float:
    lhs = quotes["A"].mid * quotes["B"].mid
    rhs = quotes["C"].mid * quotes["D"].mid
    gross = lhs + rhs
    if gross <= 0:
        return np.nan
    return (rhs - lhs) / gross


def _find_reversion_exit(
    row: pd.Series,
    trading_days: pd.DatetimeIndex,
    keys: dict[str, str],
    selected_legs: list[str],
    quote_lookup: dict[tuple[pd.Timestamp, str], Quote],
    config: LabelConfig,
    reversion_threshold: float,
    max_holding_days: int | None,
) -> tuple[pd.Timestamp | None, str, float | None]:
    entry_date = pd.Timestamp(row["date"])
    earliest_rectangle_exp = min(pd.Timestamp(row["A_expiration"]), pd.Timestamp(row["D_expiration"]))
    selected_exp = min(_selected_expiration(row, leg) for leg in selected_legs)
    last_allowed_exp = selected_exp
    future_days = trading_days[trading_days > entry_date]
    if max_holding_days is not None:
        future_days = future_days[:max_holding_days]
    future_days = future_days[future_days < last_allowed_exp]
    if len(future_days) == 0:
        return None, "no_pre_expiration_quote_day", None

    last_valid_selected_exit: pd.Timestamp | None = None
    last_valid_selected_sev: float | None = None
    for day in future_days:
        if _exit_valid(day, selected_legs, keys, quote_lookup, config.max_label_spread_pct):
            last_valid_selected_exit = pd.Timestamp(day)

        if day >= earliest_rectangle_exp:
            continue
        rectangle_valid, rectangle_quotes = _rectangle_quotes_valid(day, keys, quote_lookup, config.max_label_spread_pct)
        if not rectangle_valid:
            continue
        severity = _rectangle_normalized_severity(rectangle_quotes)
        if math.isfinite(severity):
            last_valid_selected_sev = float(severity)
        if math.isfinite(severity) and severity <= reversion_threshold:
            if _exit_valid(day, selected_legs, keys, quote_lookup, config.max_label_spread_pct):
                return pd.Timestamp(day), "tp2_reversion", float(severity)

    if last_valid_selected_exit is not None:
        return last_valid_selected_exit, "latest_valid_pre_expiration", last_valid_selected_sev
    return None, "invalid_exit_quote", last_valid_selected_sev


def _pnl_for_strategy(
    row: pd.Series,
    exit_date: pd.Timestamp,
    selected_legs: list[str],
    keys: dict[str, str],
    quote_lookup: dict[tuple[pd.Timestamp, str], Quote],
    config: LabelConfig,
) -> tuple[float, float, float]:
    gross = 0.0
    net_pnl = 0.0
    for leg in selected_legs:
        qty, side = _leg_qty_side(row, leg)
        entry_bid = float(row[f"{leg}_bid"])
        entry_ask = float(row[f"{leg}_ask"])
        entry_mid = float(row[f"{leg}_mid"])
        exit_quote = quote_lookup[(exit_date, keys[leg])]
        gross += abs(entry_mid * qty * config.option_multiplier)
        if side == 1:
            price_pnl = (exit_quote.bid - entry_ask) * qty * config.option_multiplier
        else:
            price_pnl = (entry_bid - exit_quote.ask) * qty * config.option_multiplier
        fees = 2.0 * config.commission_per_contract * abs(qty)
        net_pnl += price_pnl - fees
    return gross, net_pnl, net_pnl / gross if gross > 0 else np.nan


def build_flexible_exit_strategy_training_rows(
    features: pd.DataFrame,
    quote_frame: pd.DataFrame,
    trading_days: list[pd.Timestamp],
    config: LabelConfig,
    reversion_threshold: float = 0.0001,
    max_holding_days: int | None = None,
) -> pd.DataFrame:
    quote_lookup = _build_quote_lookup(quote_frame)
    days = pd.DatetimeIndex(pd.to_datetime(trading_days)).sort_values()
    records = []
    for row_tuple in features.itertuples(index=False):
        row = pd.Series(row_tuple._asdict())
        keys = {leg: _quote_key(row, leg) for leg in ["A", "B", "C", "D"]}
        for strategy in STRATEGIES:
            selected_legs = _strategy_selected_legs(strategy)
            base = row.to_dict()
            base.update(
                {
                    "strategy": strategy,
                    "label_horizon": "flex_reversion",
                    "exit_date": pd.NaT,
                    "exit_reason": "invalid_entry_quote",
                    "is_valid_label": False,
                    "net_pnl": np.nan,
                    "entry_gross_exposure": np.nan,
                    "realized_return": np.nan,
                    "actual_holding_days": np.nan,
                    "exit_normalized_severity": np.nan,
                }
            )
            if not _entry_valid(row, selected_legs, config.max_label_spread_pct):
                records.append(base)
                continue
            exit_date, exit_reason, exit_severity = _find_reversion_exit(
                row=row,
                trading_days=days,
                keys=keys,
                selected_legs=selected_legs,
                quote_lookup=quote_lookup,
                config=config,
                reversion_threshold=reversion_threshold,
                max_holding_days=max_holding_days,
            )
            base["exit_reason"] = exit_reason
            if exit_severity is not None:
                base["exit_normalized_severity"] = exit_severity
            if exit_date is None:
                records.append(base)
                continue
            gross, net_pnl, realized_return = _pnl_for_strategy(row, exit_date, selected_legs, keys, quote_lookup, config)
            if not (math.isfinite(gross) and gross > 0 and math.isfinite(realized_return)):
                base["exit_reason"] = "bad_entry_gross"
                records.append(base)
                continue
            base.update(
                {
                    "exit_date": exit_date,
                    "is_valid_label": True,
                    "net_pnl": net_pnl,
                    "entry_gross_exposure": gross,
                    "realized_return": realized_return,
                    "actual_holding_days": int((exit_date - pd.Timestamp(row["date"])).days),
                }
            )
            records.append(base)
    return pd.DataFrame.from_records(records)
