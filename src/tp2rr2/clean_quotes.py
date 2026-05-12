from __future__ import annotations

import pandas as pd

from .config import CleanConfig


def clean_options(df: pd.DataFrame, config: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    mask = (
        (out["symbol"] == "SPY")
        & (out["type"].isin(["call", "put"]))
        & (out["dte"].between(config.min_dte, config.max_dte))
        & (out["bid"] > 0)
        & (out["ask"] > out["bid"])
        & (out["mid"] > 0)
        & (out["spread_pct"] > 0)
        & (out["spread_pct"] <= config.max_spread_pct)
        & (out["delta"].abs().between(config.min_abs_delta, config.max_abs_delta))
    )

    if config.require_positive_size_if_present:
        for col in ["bid_size", "ask_size"]:
            if col in out.columns:
                mask &= out[col].fillna(0) > 0

    cleaned = out.loc[mask].copy()
    cleaned = cleaned.sort_values(["date", "type", "expiration", "strike"]).reset_index(drop=True)
    return cleaned
