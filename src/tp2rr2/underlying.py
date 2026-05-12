from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_spy_close(path: Path) -> pd.DataFrame | None:
    """Load official raw SPY close from CSV if present.

    Expected columns are flexible: date plus one of close/Close/raw_close.
    Adjusted close is intentionally not used for moneyness.
    """
    if not path.exists():
        return None
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols:
        raise ValueError(f"{path} must contain a date column")
    close_col = None
    for candidate in ["raw_close", "close", "Close"]:
        if candidate.lower() in cols:
            close_col = cols[candidate.lower()]
            break
    if close_col is None:
        raise ValueError(f"{path} must contain raw_close or close")
    out = df[[cols["date"], close_col]].copy()
    out.columns = ["date", "underlying_price"]
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["underlying_price"] = pd.to_numeric(out["underlying_price"], errors="coerce")
    out = out.dropna()
    out["underlying_source"] = "raw_close"
    return out


def parity_proxy(options: pd.DataFrame) -> pd.DataFrame:
    """Build a conservative daily SPY proxy from near-ATM put-call parity candidates."""
    calls = options[(options["type"] == "call") & (options["dte"].between(7, 45))].copy()
    puts = options[(options["type"] == "put") & (options["dte"].between(7, 45))].copy()
    keys = ["date", "expiration", "strike"]
    pairs = calls.merge(
        puts,
        on=keys,
        suffixes=("_call", "_put"),
        how="inner",
    )
    if pairs.empty:
        return pd.DataFrame(columns=["date", "underlying_price", "underlying_source"])

    liquid = (
        (pairs["volume_call"].fillna(0) > 0)
        & (pairs["volume_put"].fillna(0) > 0)
        & (pairs["open_interest_call"].fillna(0) > 0)
        & (pairs["open_interest_put"].fillna(0) > 0)
        & (pairs["spread_pct_call"] <= 0.20)
        & (pairs["spread_pct_put"] <= 0.20)
    )
    pairs = pairs.loc[liquid].copy()
    if pairs.empty:
        return pd.DataFrame(columns=["date", "underlying_price", "underlying_source"])

    pairs["atm_score"] = (pairs["delta_call"] - 0.50).abs()
    pairs["s_proxy"] = pairs["mid_call"] - pairs["mid_put"] + pairs["strike"]
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna(subset=["s_proxy"])
    pairs = pairs[pairs["s_proxy"] > 0]
    if pairs.empty:
        return pd.DataFrame(columns=["date", "underlying_price", "underlying_source"])

    # Keep the near-ATM half of each day before taking a robust median.
    pairs["rank"] = pairs.groupby("date")["atm_score"].rank(method="first", pct=True)
    pairs = pairs[pairs["rank"] <= 0.50]
    out = pairs.groupby("date", as_index=False)["s_proxy"].median()
    out = out.rename(columns={"s_proxy": "underlying_price"})
    out["underlying_source"] = "parity_proxy"
    return out


def attach_underlying(options: pd.DataFrame, official_close: pd.DataFrame | None) -> pd.DataFrame:
    proxy = parity_proxy(options)
    if official_close is not None and not official_close.empty:
        underlying = official_close.copy()
        if not proxy.empty:
            missing_proxy = proxy[~proxy["date"].isin(underlying["date"])]
            underlying = pd.concat([underlying, missing_proxy], ignore_index=True)
    else:
        underlying = proxy

    out = options.merge(underlying, on="date", how="left")
    out["moneyness"] = out["strike"] / out["underlying_price"]
    return out
