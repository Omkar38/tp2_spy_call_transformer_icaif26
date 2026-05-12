from __future__ import annotations

import numpy as np
import pandas as pd


LEG_PREFIXES = ["A", "B", "C", "D"]


def build_features(violations: pd.DataFrame, options_with_underlying: pd.DataFrame) -> pd.DataFrame:
    if violations.empty:
        return violations.copy()
    daily_underlying = (
        options_with_underlying[["date", "underlying_price", "underlying_source"]]
        .drop_duplicates("date")
        .copy()
    )
    out = violations.merge(daily_underlying, on="date", how="left")
    out["moneyness_K1"] = out["K1"] / out["underlying_price"]
    out["moneyness_K2"] = out["K2"] / out["underlying_price"]
    out["distance_K1_ATM"] = (out["moneyness_K1"] - 1.0).abs()
    out["distance_K2_ATM"] = (out["moneyness_K2"] - 1.0).abs()

    spread_cols = [f"{p}_spread_pct" for p in LEG_PREFIXES]
    volume_cols = [f"{p}_volume" for p in LEG_PREFIXES]
    oi_cols = [f"{p}_open_interest" for p in LEG_PREFIXES]
    iv_cols = [f"{p}_implied_volatility" for p in LEG_PREFIXES]

    out["avg_spread_pct"] = out[spread_cols].mean(axis=1)
    out["max_spread_pct"] = out[spread_cols].max(axis=1)
    out["min_volume"] = out[volume_cols].min(axis=1)
    out["total_volume"] = out[volume_cols].sum(axis=1)
    out["min_open_interest"] = out[oi_cols].min(axis=1)
    out["total_open_interest"] = out[oi_cols].sum(axis=1)
    out["avg_iv"] = out[iv_cols].mean(axis=1)

    out["iv_slope_T1"] = out["D_implied_volatility"] - out["A_implied_volatility"]
    out["iv_slope_T2"] = out["B_implied_volatility"] - out["C_implied_volatility"]
    out["iv_term_K1"] = out["C_implied_volatility"] - out["A_implied_volatility"]
    out["iv_term_K2"] = out["B_implied_volatility"] - out["D_implied_volatility"]
    out["delta_gap_T1"] = out["D_delta"] - out["A_delta"]
    out["delta_gap_T2"] = out["B_delta"] - out["C_delta"]

    out = add_persistence_features(out)
    return out.replace([np.inf, -np.inf], np.nan)


def add_persistence_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["option_type", "K1", "K2", "T1_expiration", "T2_expiration", "date"]).copy()
    key_cols = ["option_type", "K1", "K2", "T1_expiration", "T2_expiration"]
    out["prev_date"] = out.groupby(key_cols)["date"].shift(1)
    out["prev_normalized_severity"] = out.groupby(key_cols)["normalized_severity"].shift(1)
    out["days_since_seen"] = (out["date"] - out["prev_date"]).dt.days
    out["seen_previous_trading_day"] = out["days_since_seen"].between(1, 4).fillna(False)
    out["normalized_severity_change_1d"] = out["normalized_severity"] - out["prev_normalized_severity"]
    streak = []
    last_key = None
    age = 0
    for row in out[key_cols + ["seen_previous_trading_day"]].itertuples(index=False, name=None):
        key = row[:-1]
        seen_prev = row[-1]
        if key != last_key or not seen_prev:
            age = 1
        else:
            age += 1
        streak.append(age)
        last_key = key
    out["violation_age_days"] = streak
    return out.drop(columns=["prev_date", "prev_normalized_severity", "days_since_seen"])
