from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ARCHIVE2_COLUMNS = [
    "quote_date",
    "contract_id",
    "option_symbol",
    "expiration_date",
    "strike",
    "option_type",
    "option_multiplier",
    "bid",
    "ask",
    "mid",
    "bid_size",
    "ask_size",
    "quote_time",
    "volume",
    "open_interest",
    "spy_close",
    "next_ex_div_date",
    "next_div_amount",
    "implied_vol",
    "delta",
    "theta",
    "vega",
    "forward",
    "risk_free_rate",
]


@dataclass(frozen=True)
class Archive2CleanConfig:
    min_dte: int = 1
    max_dte: int = 90
    calls_only: bool = True
    otm_calls_only: bool = True
    use_last_available_quote: bool = False
    min_quote_hour: int = 15
    min_quote_minute: int = 55
    max_spread_pct: float = 0.50
    require_bid_ask: bool = True
    require_positive_bid: bool = True
    require_positive_size: bool = False
    require_positive_open_interest: bool = False
    require_positive_volume: bool = False
    exclude_rate_fallback: bool = False
    require_call_price_bounds: bool = True
    call_lower_bound_tolerance: float = 1.0
    call_upper_bound_multiplier: float = 1.05
    call_upper_bound_tolerance: float = 1.0


def archive2_year_path(root: Path, year: int) -> Path:
    return root / f"year={year}" / "data.parquet"


def load_archive2_year(root: Path, year: int) -> pd.DataFrame:
    path = archive2_year_path(root, year)
    if not path.exists():
        raise FileNotFoundError(f"Missing Archive-2 year file: {path}")
    return pd.read_parquet(path, columns=ARCHIVE2_COLUMNS)


def deduplicate_archive2(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse producer duplicates to one row per quote_date/contract_id.

    Archive-2 raw files can contain repeated quote rows after merging unaggregated
    daily-volume records. Quote fields are duplicated, while volume can differ.
    We keep the last non-null quote/static value and sum duplicate volume rows.
    """
    if raw.empty:
        return raw.copy()

    df = raw.copy()
    df["quote_date"] = pd.to_datetime(df["quote_date"]).dt.normalize()
    df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.normalize()
    df["next_ex_div_date"] = pd.to_datetime(df["next_ex_div_date"], errors="coerce").dt.normalize()
    df["quote_time"] = pd.to_datetime(df["quote_time"], errors="coerce")

    sort_cols = ["quote_date", "contract_id", "quote_time"]
    df = df.sort_values(sort_cols, na_position="first")
    grouped = df.groupby(["quote_date", "contract_id"], sort=False)

    agg = {
        "option_symbol": "last",
        "expiration_date": "last",
        "strike": "last",
        "option_type": "last",
        "option_multiplier": "last",
        "bid": "last",
        "ask": "last",
        "mid": "last",
        "bid_size": "last",
        "ask_size": "last",
        "quote_time": "last",
        "volume": "sum",
        "open_interest": "last",
        "spy_close": "last",
        "next_ex_div_date": "last",
        "next_div_amount": "last",
        "implied_vol": "last",
        "delta": "last",
        "theta": "last",
        "vega": "last",
        "forward": "last",
        "risk_free_rate": "last",
    }
    return grouped.agg(agg).reset_index()


def normalize_for_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["quote_date"]).dt.normalize()
    out["expiration"] = pd.to_datetime(out["expiration_date"]).dt.normalize()
    out["contractID"] = out["contract_id"].astype(str)
    out["symbol"] = "SPY"
    out["type"] = out["option_type"].map({"C": "call", "P": "put"}).fillna(out["option_type"].astype(str).str.lower())
    out["underlying_price"] = out["spy_close"]
    out["underlying_source"] = "raw_close"
    out["implied_volatility"] = out["implied_vol"]
    out["rho"] = np.nan
    out["dte"] = (out["expiration"] - out["date"]).dt.days
    out["spread"] = out["ask"] - out["bid"]
    out["mid"] = out["mid"].fillna((out["bid"] + out["ask"]) / 2.0)
    out["spread_pct"] = out["spread"] / out["mid"]
    out["moneyness"] = out["strike"] / out["underlying_price"]
    return out


def _fresh_quote_mask(df: pd.DataFrame, config: Archive2CleanConfig) -> pd.Series:
    qt = pd.to_datetime(df["quote_time"], errors="coerce")
    if getattr(qt.dt, "tz", None) is not None:
        local = qt.dt.tz_convert("America/New_York")
    else:
        local = qt
    minutes = local.dt.hour * 60 + local.dt.minute + local.dt.second / 60.0
    threshold = config.min_quote_hour * 60 + config.min_quote_minute
    return minutes >= threshold


def _call_price_bounds_mask(df: pd.DataFrame, config: Archive2CleanConfig) -> pd.Series:
    """Flag call quotes that satisfy broad static no-arbitrage price bounds.

    For a nonnegative-dividend call, price should not be far below intrinsic
    value and should not exceed the underlying by a large margin. We keep the
    tolerance deliberately loose because this is a raw-data sanity filter, not a
    pricing model.
    """
    if not config.require_call_price_bounds:
        return pd.Series(True, index=df.index)

    is_call = df["type"].eq("call")
    underlying = pd.to_numeric(df["underlying_price"], errors="coerce")
    strike = pd.to_numeric(df["strike"], errors="coerce")
    mid = pd.to_numeric(df["mid"], errors="coerce")
    intrinsic = (underlying - strike).clip(lower=0)

    lower_ok = mid >= intrinsic - config.call_lower_bound_tolerance
    upper_ok = mid <= (
        underlying * config.call_upper_bound_multiplier
        + config.call_upper_bound_tolerance
    )
    valid_inputs = underlying.notna() & strike.notna() & mid.notna()
    return (~is_call) | (valid_inputs & lower_ok & upper_ok)


def _research_filter_mask(
    df: pd.DataFrame,
    config: Archive2CleanConfig,
    *,
    include_call_price_bounds: bool = True,
) -> pd.Series:
    mask = df["dte"].between(config.min_dte, config.max_dte)

    if config.calls_only:
        mask &= df["type"].eq("call")
    if config.otm_calls_only:
        mask &= df["strike"] > df["underlying_price"]
    if config.require_bid_ask:
        mask &= df["bid"].notna() & df["ask"].notna() & (df["ask"] > df["bid"]) & (df["mid"] > 0)
    if include_call_price_bounds:
        mask &= _call_price_bounds_mask(df, config)
    if config.require_positive_bid:
        mask &= df["bid"] > 0
    if config.max_spread_pct is not None:
        mask &= df["spread_pct"] <= config.max_spread_pct
    if config.require_positive_size:
        mask &= (df["bid_size"].fillna(0) > 0) & (df["ask_size"].fillna(0) > 0)
    if config.require_positive_open_interest:
        mask &= df["open_interest"].fillna(0) > 0
    if config.require_positive_volume:
        mask &= df["volume"].fillna(0) > 0
    if config.exclude_rate_fallback:
        mask &= df["risk_free_rate"] != 0.04

    if not config.use_last_available_quote:
        mask &= _fresh_quote_mask(df, config)
    return mask


def apply_research_filters(df: pd.DataFrame, config: Archive2CleanConfig) -> pd.DataFrame:
    out = df.copy()
    mask = _research_filter_mask(out, config, include_call_price_bounds=True)
    return _stabilize_output_dtypes(out.loc[mask].reset_index(drop=True))


def _stabilize_output_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    timestamp_cols = ["quote_date", "expiration_date", "next_ex_div_date", "date", "expiration"]
    for col in timestamp_cols:
        if col in out:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    string_cols = ["option_symbol", "option_type", "contractID", "symbol", "type", "underlying_source"]
    for col in string_cols:
        if col in out:
            out[col] = out[col].astype("string")

    int_cols = ["contract_id", "option_multiplier", "dte"]
    for col in int_cols:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("int64")

    float_cols = [
        "strike",
        "bid",
        "ask",
        "mid",
        "bid_size",
        "ask_size",
        "volume",
        "open_interest",
        "spy_close",
        "next_div_amount",
        "implied_vol",
        "delta",
        "theta",
        "vega",
        "forward",
        "risk_free_rate",
        "underlying_price",
        "implied_volatility",
        "rho",
        "spread",
        "spread_pct",
        "moneyness",
    ]
    for col in float_cols:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def clean_archive2_year(root: Path, year: int, config: Archive2CleanConfig) -> tuple[pd.DataFrame, dict]:
    raw = load_archive2_year(root, year)
    deduped = deduplicate_archive2(raw)
    normalized = normalize_for_pipeline(deduped)
    pre_price_bound_mask = _research_filter_mask(normalized, config, include_call_price_bounds=False)
    call_bounds_mask = _call_price_bounds_mask(normalized, config)
    cleaned = apply_research_filters(normalized, config)

    summary = {
        "year": year,
        "raw_rows": int(len(raw)),
        "deduped_rows": int(len(deduped)),
        "duplicate_rows_removed": int(len(raw) - len(deduped)),
        "clean_rows": int(len(cleaned)),
        "call_price_bound_fail_rows": int((pre_price_bound_mask & ~call_bounds_mask).sum()),
        "raw_sessions": int(pd.to_datetime(raw["quote_date"]).nunique()),
        "clean_sessions": int(cleaned["date"].nunique()) if not cleaned.empty else 0,
    }
    return cleaned, summary
