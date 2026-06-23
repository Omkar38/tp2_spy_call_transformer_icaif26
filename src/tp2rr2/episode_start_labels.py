from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


OBSERVABLE_STATES = {"true_reversion", "still_violation_or_reappeared"}
INVALID_STATES = {
    "left_leg_expired_or_missing",
    "right_leg_missing_or_expired",
    "liquidity_filter_failure",
}


@dataclass(frozen=True)
class EpisodeStartLabelResult:
    labels: pd.DataFrame
    summary: dict


def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == "date" or col.endswith("_date") or col.endswith("_expiration"):
            try:
                out[col] = pd.to_datetime(out[col])
            except (TypeError, ValueError):
                pass
    return out


def _rename_feature_columns(features: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for col in features.columns:
        if col == "violation_id":
            rename[col] = "start_violation_id"
        elif col == "date":
            rename[col] = "signal_date"
        else:
            rename[col] = f"feature_{col}"
    return features.rename(columns=rename)


def _label_quality(end_state: str) -> str:
    if end_state == "true_reversion":
        return "clean_positive_true_reversion"
    if end_state == "still_violation_or_reappeared":
        return "clean_negative_still_violation"
    if end_state == "sample_ended":
        return "censored_sample_end"
    if end_state in INVALID_STATES:
        return "invalid_missing_expired_or_liquidity"
    return "unresolved_other"


def _split_name(date: pd.Timestamp) -> str:
    if pd.isna(date):
        return "missing"
    if date <= pd.Timestamp("2022-12-31"):
        return "train_2013_2022"
    if date <= pd.Timestamp("2023-12-31"):
        return "validation_2023"
    return "test_2024_2026"


def build_episode_start_labels(
    detector_features: pd.DataFrame,
    episodes: pd.DataFrame,
    end_states: pd.DataFrame,
) -> EpisodeStartLabelResult:
    """Build one ML-label row per pair episode start.

    The output intentionally separates point-in-time detector features from
    future-known labels. Columns prefixed with ``feature_`` are the start-date
    detector row. Columns prefixed with ``label_`` are targets or future
    diagnostics and should not be used as model inputs.
    """

    detector_features = _coerce_dates(detector_features)
    episodes = _coerce_dates(episodes)
    end_states = _coerce_dates(end_states)

    required_episode_cols = {
        "episode_id",
        "representative_violation_id",
        "episode_start_date",
        "episode_end_date",
        "pair_key",
        "start_rectangle_key",
        "duration_sessions",
        "episode_row_count",
    }
    required_end_cols = {"episode_id", "end_state", "is_true_reversion", "is_observable_next_state"}
    required_feature_cols = {"violation_id", "date"}
    missing_episode = sorted(required_episode_cols.difference(episodes.columns))
    missing_end = sorted(required_end_cols.difference(end_states.columns))
    missing_features = sorted(required_feature_cols.difference(detector_features.columns))
    if missing_episode:
        raise ValueError(f"Episode table is missing columns: {missing_episode}")
    if missing_end:
        raise ValueError(f"End-state table is missing columns: {missing_end}")
    if missing_features:
        raise ValueError(f"Detector feature table is missing columns: {missing_features}")

    if episodes["episode_id"].duplicated().any():
        raise ValueError("Episode table contains duplicate episode_id values.")
    if end_states["episode_id"].duplicated().any():
        raise ValueError("End-state table contains duplicate episode_id values.")
    if detector_features["violation_id"].duplicated().any():
        raise ValueError("Detector feature table contains duplicate violation_id values.")

    episode_cols = [
        "episode_id",
        "representative_violation_id",
        "episode_start_date",
        "episode_end_date",
        "start_trading_day_index",
        "end_trading_day_index",
        "next_trading_day_index",
        "next_trading_date",
        "duration_sessions",
        "episode_row_count",
        "unique_violation_days",
        "unique_rectangle_count",
        "has_exact_rectangle_change",
        "start_year",
        "start_month",
        "end_year",
        "end_month",
        "start_normalized_severity",
        "max_normalized_severity",
        "mean_normalized_severity",
        "end_normalized_severity",
        "start_lhs",
        "start_rhs",
        "end_lhs",
        "end_rhs",
        "pair_key",
        "start_rectangle_key",
        "end_rectangle_key",
    ]
    episode_cols = [col for col in episode_cols if col in episodes.columns]
    starts = episodes[episode_cols].copy()
    starts = starts.rename(columns={"representative_violation_id": "start_violation_id"})

    feature_rows = detector_features[
        detector_features["violation_id"].isin(starts["start_violation_id"])
    ].copy()
    feature_rows = _rename_feature_columns(feature_rows)

    labels = starts.merge(feature_rows, on="start_violation_id", how="left", validate="one_to_one")
    missing_start_features = int(labels["signal_date"].isna().sum())
    if missing_start_features:
        raise ValueError(f"Missing detector start features for {missing_start_features:,} episodes.")

    end_cols = [
        "episode_id",
        "end_state",
        "end_state_detail",
        "is_true_reversion",
        "is_observable_next_state",
        "next_trading_date",
        "next_lhs",
        "next_rhs",
        "next_severity",
        "next_normalized_severity",
        "next_A_expiration",
        "next_A_strike",
        "next_B_expiration",
        "next_B_strike",
        "next_C_expiration",
        "next_C_strike",
        "next_D_expiration",
        "next_D_strike",
    ]
    end_cols = [col for col in end_cols if col in end_states.columns]
    end_label_rows = end_states[end_cols].copy()
    end_label_rows = end_label_rows.rename(
        columns={col: f"label_{col}" for col in end_label_rows.columns if col != "episode_id"}
    )
    labels = labels.merge(end_label_rows, on="episode_id", how="left", validate="one_to_one")

    labels["label_true_reversion_next_session"] = labels["label_end_state"].eq("true_reversion")
    labels["label_still_violation_next_session"] = labels["label_end_state"].eq(
        "still_violation_or_reappeared"
    )
    labels["label_observable_next_state"] = labels["label_end_state"].isin(OBSERVABLE_STATES)
    labels["label_invalid_or_unobservable"] = labels["label_end_state"].isin(INVALID_STATES)
    labels["label_sample_censored"] = labels["label_end_state"].eq("sample_ended")
    labels["label_quality_group"] = labels["label_end_state"].map(_label_quality)
    labels["ml_binary_reversion_target"] = np.where(
        labels["label_observable_next_state"],
        labels["label_true_reversion_next_session"].astype(int),
        np.nan,
    )
    labels["ml_sample_weight_default"] = np.where(labels["label_observable_next_state"], 1.0, 0.0)
    labels["split_by_signal_date"] = labels["signal_date"].map(_split_name)

    missing_end_state = int(labels["label_end_state"].isna().sum())
    duplicate_episode_ids = int(labels["episode_id"].duplicated().sum())
    signal_mismatch = int((labels["signal_date"] != labels["episode_start_date"]).sum())

    counts = labels["label_end_state"].value_counts(dropna=False)
    quality_counts = labels["label_quality_group"].value_counts(dropna=False)
    by_year = (
        labels.groupby(["start_year", "label_end_state"], dropna=False)
        .size()
        .rename("episode_count")
        .reset_index()
    )
    split_counts = labels["split_by_signal_date"].value_counts(dropna=False)

    summary = {
        "episode_start_label_rows": int(len(labels)),
        "unique_episode_ids": int(labels["episode_id"].nunique()),
        "duplicate_episode_ids": duplicate_episode_ids,
        "missing_end_state": missing_end_state,
        "missing_start_features": missing_start_features,
        "signal_date_episode_start_mismatches": signal_mismatch,
        "observable_next_state_rows": int(labels["label_observable_next_state"].sum()),
        "observable_next_state_share": float(labels["label_observable_next_state"].mean()) if len(labels) else 0.0,
        "true_reversion_rows": int(labels["label_true_reversion_next_session"].sum()),
        "true_reversion_share_all": float(labels["label_true_reversion_next_session"].mean()) if len(labels) else 0.0,
        "true_reversion_share_observable": float(
            labels.loc[labels["label_observable_next_state"], "label_true_reversion_next_session"].mean()
        )
        if labels["label_observable_next_state"].any()
        else 0.0,
        "label_end_state_counts": {str(k): int(v) for k, v in counts.items()},
        "label_quality_group_counts": {str(k): int(v) for k, v in quality_counts.items()},
        "split_by_signal_date_counts": {str(k): int(v) for k, v in split_counts.items()},
        "label_end_state_counts_by_start_year": {
            f"{int(row.start_year)}|{row.label_end_state}": int(row.episode_count)
            for row in by_year.itertuples(index=False)
        },
        "feature_column_prefix": "feature_",
        "label_column_prefix": "label_",
        "ml_feature_rule": "Use signal_date and feature_* point-in-time columns only; exclude label_* and future episode/end columns.",
    }
    return EpisodeStartLabelResult(labels=labels, summary=summary)
