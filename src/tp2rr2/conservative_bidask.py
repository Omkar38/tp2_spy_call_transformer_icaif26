from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


REQUIRED_COLUMNS = [
    "episode_id",
    "signal_date",
    "feature_A_ask",
    "feature_B_ask",
    "feature_C_bid",
    "feature_D_bid",
    "feature_A_mid",
    "feature_B_mid",
    "feature_C_mid",
    "feature_D_mid",
]


@dataclass(frozen=True)
class ConservativeBidAskResult:
    rows: pd.DataFrame
    summary: dict


def add_conservative_bidask_columns(labels: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS).difference(labels.columns))
    if missing:
        raise ValueError(f"Episode-start labels are missing columns: {missing}")

    out = labels.copy()
    out["conservative_bidask_lhs"] = out["feature_A_ask"] * out["feature_B_ask"]
    out["conservative_bidask_rhs"] = out["feature_C_bid"] * out["feature_D_bid"]
    out["conservative_bidask_severity"] = out["conservative_bidask_rhs"] - out["conservative_bidask_lhs"]
    gross = out["conservative_bidask_lhs"] + out["conservative_bidask_rhs"]
    out["conservative_bidask_normalized_severity"] = out["conservative_bidask_severity"] / gross.where(gross > 0)
    out["is_conservative_bidask_tp2_violation"] = out["conservative_bidask_lhs"] < out["conservative_bidask_rhs"]

    out["mid_lhs"] = out["feature_A_mid"] * out["feature_B_mid"]
    out["mid_rhs"] = out["feature_C_mid"] * out["feature_D_mid"]
    out["bidask_to_mid_severity_ratio"] = (
        out["conservative_bidask_severity"] / (out["mid_rhs"] - out["mid_lhs"])
    )
    return out


def build_conservative_bidask_subset(labels: pd.DataFrame) -> ConservativeBidAskResult:
    enriched = add_conservative_bidask_columns(labels)
    subset = enriched[enriched["is_conservative_bidask_tp2_violation"]].copy()
    by_year = (
        enriched.assign(year=pd.to_datetime(enriched["signal_date"]).dt.year)
        .groupby("year", dropna=False)
        .agg(
            episode_starts=("episode_id", "size"),
            conservative_bidask_starts=("is_conservative_bidask_tp2_violation", "sum"),
            median_mid_normalized_severity=("feature_normalized_severity", "median"),
            median_bidask_normalized_severity=("conservative_bidask_normalized_severity", "median"),
        )
        .reset_index()
    )
    by_year["conservative_bidask_share"] = (
        by_year["conservative_bidask_starts"] / by_year["episode_starts"]
    )
    summary = {
        "episode_start_rows": int(len(enriched)),
        "conservative_bidask_rows": int(len(subset)),
        "conservative_bidask_share": float(len(subset) / len(enriched)) if len(enriched) else 0.0,
        "median_conservative_bidask_normalized_severity": float(
            subset["conservative_bidask_normalized_severity"].median()
        )
        if len(subset)
        else None,
        "mean_conservative_bidask_normalized_severity": float(
            subset["conservative_bidask_normalized_severity"].mean()
        )
        if len(subset)
        else None,
        "by_signal_year": {
            str(int(row.year)): {
                "episode_starts": int(row.episode_starts),
                "conservative_bidask_starts": int(row.conservative_bidask_starts),
                "conservative_bidask_share": float(row.conservative_bidask_share),
                "median_mid_normalized_severity": float(row.median_mid_normalized_severity),
                "median_bidask_normalized_severity": float(row.median_bidask_normalized_severity)
                if pd.notna(row.median_bidask_normalized_severity)
                else None,
            }
            for row in by_year.itertuples(index=False)
        },
        "filter_rule": "feature_A_ask * feature_B_ask < feature_C_bid * feature_D_bid",
    }
    return ConservativeBidAskResult(rows=subset, summary=summary)
