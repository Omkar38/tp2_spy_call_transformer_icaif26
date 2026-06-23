from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


DATE_COLS = [
    "date",
    "A_expiration",
    "B_expiration",
    "C_expiration",
    "D_expiration",
    "T1_expiration",
    "T2_expiration",
]

EPISODE_INPUT_COLUMNS = [
    "date",
    "violation_id",
    "K1",
    "K2",
    "T1_expiration",
    "T2_expiration",
    "K1_forward_adjusted",
    "K2_forward_adjusted",
    "K1_forward_rounded",
    "K2_forward_rounded",
    "lhs",
    "rhs",
    "severity",
    "gross_product_exposure",
    "normalized_severity",
    "A_contract_id",
    "A_option_symbol",
    "A_expiration",
    "A_strike",
    "A_mid",
    "A_bid",
    "A_ask",
    "B_contract_id",
    "B_option_symbol",
    "B_expiration",
    "B_strike",
    "B_mid",
    "B_bid",
    "B_ask",
    "C_contract_id",
    "C_option_symbol",
    "C_expiration",
    "C_strike",
    "C_mid",
    "C_bid",
    "C_ask",
    "D_contract_id",
    "D_option_symbol",
    "D_expiration",
    "D_strike",
    "D_mid",
    "D_bid",
    "D_ask",
    "underlying_price",
    "avg_spread_pct",
    "max_spread_pct",
    "min_volume",
    "total_volume",
    "min_open_interest",
    "total_open_interest",
]


@dataclass(frozen=True)
class EpisodeBuildResult:
    members: pd.DataFrame
    episodes: pd.DataFrame
    summary: dict


def _format_key_part(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        values = pd.to_datetime(series)
        numeric = values.dt.year * 10000 + values.dt.month * 100 + values.dt.day
        return numeric.astype("Int64").astype("string").fillna("")
    if pd.api.types.is_float_dtype(series):
        values = series.to_numpy(dtype=float)
        formatted = np.empty(len(values), dtype=object)
        missing = np.isnan(values)
        formatted[missing] = ""
        formatted[~missing] = np.char.mod("%.6f", values[~missing])
        return pd.Series(formatted, index=series.index, dtype="string")
    return series.astype("string").fillna("")


def _join_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts = [_format_key_part(df[col]) for col in columns]
    out = parts[0]
    for part in parts[1:]:
        out = out.str.cat(part, sep="|")
    return out


def _prepare_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in DATE_COLS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col])
    return out


def add_episode_keys(violations: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "A_expiration",
        "A_strike",
        "B_expiration",
        "B_strike",
        "C_expiration",
        "C_strike",
        "D_expiration",
        "D_strike",
    }
    missing = sorted(required.difference(violations.columns))
    if missing:
        raise ValueError(f"Missing required episode key columns: {missing}")

    out = _prepare_dates(violations)
    out["pair_key"] = _join_key(out, ["A_expiration", "A_strike", "B_expiration", "B_strike"])
    out["rectangle_key"] = _join_key(
        out,
        [
            "A_expiration",
            "A_strike",
            "B_expiration",
            "B_strike",
            "C_expiration",
            "C_strike",
            "D_expiration",
            "D_strike",
        ],
    )
    return out


def build_trading_calendar(clean_options: pd.DataFrame) -> pd.DataFrame:
    if "date" not in clean_options.columns:
        raise ValueError("Clean options data must contain a date column.")
    dates = pd.Series(pd.to_datetime(clean_options["date"]).drop_duplicates()).sort_values().reset_index(drop=True)
    return pd.DataFrame({"date": dates, "trading_day_index": np.arange(len(dates), dtype=np.int64)})


def assign_pair_episodes(violations: pd.DataFrame, trading_calendar: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "A_expiration",
        "A_strike",
        "B_expiration",
        "B_strike",
        "C_expiration",
        "C_strike",
        "D_expiration",
        "D_strike",
    }
    missing = sorted(required.difference(violations.columns))
    if missing:
        raise ValueError(f"Missing required episode key columns: {missing}")

    keyed = _prepare_dates(violations)
    calendar = trading_calendar[["date", "trading_day_index"]].copy()
    calendar["date"] = pd.to_datetime(calendar["date"])

    pair_cols = ["A_expiration", "A_strike", "B_expiration", "B_strike"]
    label_cols = [
        "violation_id",
        "date",
        *pair_cols,
        "C_expiration",
        "C_strike",
        "D_expiration",
        "D_strike",
    ]
    labels = keyed[label_cols].merge(calendar, on="date", how="left")
    if labels["trading_day_index"].isna().any():
        bad_dates = sorted(labels.loc[labels["trading_day_index"].isna(), "date"].dt.strftime("%Y-%m-%d").unique())
        raise ValueError(f"Violation dates missing from trading calendar: {bad_dates[:10]}")

    labels = labels.sort_values(pair_cols + ["trading_day_index", "violation_id"]).reset_index(drop=True)
    same_pair_as_previous = pd.Series(True, index=labels.index)
    for col in pair_cols:
        same_pair_as_previous &= labels[col].eq(labels[col].shift(1))
    prev_idx = labels["trading_day_index"].shift(1)
    new_episode = (~same_pair_as_previous) | ((labels["trading_day_index"] - prev_idx) != 1)
    labels["episode_number"] = new_episode.cumsum().astype("int64")
    labels["episode_id"] = labels["episode_number"].map(lambda value: f"pair_ep_{value:09d}")

    episode_cols = ["violation_id", "trading_day_index", "episode_number", "episode_id"]
    if labels["violation_id"].duplicated().any():
        raise ValueError("Duplicate violation_id values found before episode label merge.")
    return keyed.merge(labels[episode_cols], on="violation_id", how="left")


def summarize_episodes(members: pd.DataFrame, trading_calendar: pd.DataFrame) -> pd.DataFrame:
    if members.empty:
        return pd.DataFrame()

    summary_cols = [
        "episode_number",
        "episode_id",
        "trading_day_index",
        "violation_id",
        "date",
        "normalized_severity",
        "lhs",
        "rhs",
        "A_contract_id",
        "A_option_symbol",
        "A_expiration",
        "A_strike",
        "B_contract_id",
        "B_option_symbol",
        "B_expiration",
        "B_strike",
        "C_contract_id",
        "C_option_symbol",
        "C_expiration",
        "C_strike",
        "D_contract_id",
        "D_option_symbol",
        "D_expiration",
        "D_strike",
    ]
    members = members[[col for col in summary_cols if col in members.columns]].sort_values(
        ["episode_number", "trading_day_index", "violation_id"]
    ).copy()
    calendar_max = int(trading_calendar["trading_day_index"].max())
    calendar_lookup = trading_calendar.set_index("trading_day_index")["date"].to_dict()

    episode_numbers = members["episode_number"].to_numpy()
    first_mask = np.r_[True, episode_numbers[1:] != episode_numbers[:-1]]
    first_pos = np.flatnonzero(first_mask)
    last_pos = np.r_[first_pos[1:] - 1, len(members) - 1]
    counts = last_pos - first_pos + 1

    severity = members["normalized_severity"].to_numpy(dtype=float)
    max_severity = np.maximum.reduceat(severity, first_pos)
    sum_severity = np.add.reduceat(severity, first_pos)

    first_rows = members.iloc[first_pos].set_index("episode_number")
    last_rows = members.iloc[last_pos].set_index("episode_number")

    episodes = pd.DataFrame(
        {
            "episode_id": first_rows["episode_id"].to_numpy(),
            "episode_start_date": first_rows["date"].to_numpy(),
            "episode_end_date": last_rows["date"].to_numpy(),
            "start_trading_day_index": first_rows["trading_day_index"].to_numpy(dtype=np.int64),
            "end_trading_day_index": last_rows["trading_day_index"].to_numpy(dtype=np.int64),
            "episode_row_count": counts.astype(np.int64),
            "unique_violation_days": counts.astype(np.int64),
            "start_normalized_severity": first_rows["normalized_severity"].to_numpy(dtype=float),
            "max_normalized_severity": max_severity,
            "mean_normalized_severity": sum_severity / counts,
            "end_normalized_severity": last_rows["normalized_severity"].to_numpy(dtype=float),
            "start_lhs": first_rows["lhs"].to_numpy(dtype=float),
            "start_rhs": first_rows["rhs"].to_numpy(dtype=float),
            "end_lhs": last_rows["lhs"].to_numpy(dtype=float),
            "end_rhs": last_rows["rhs"].to_numpy(dtype=float),
            "representative_violation_id": first_rows["violation_id"].to_numpy(),
        },
        index=first_rows.index,
    )
    episodes["median_normalized_severity"] = episodes["mean_normalized_severity"]

    episodes["unique_rectangle_count"] = 1
    multi_episode_numbers = episodes.index[episodes["episode_row_count"].gt(1)]
    if len(multi_episode_numbers):
        rect_cols = [
            "A_expiration",
            "A_strike",
            "B_expiration",
            "B_strike",
            "C_expiration",
            "C_strike",
            "D_expiration",
            "D_strike",
        ]
        rect_counts = (
            members.loc[members["episode_number"].isin(multi_episode_numbers), ["episode_number", *rect_cols]]
            .drop_duplicates()
            .groupby("episode_number")
            .size()
        )
        episodes.loc[rect_counts.index, "unique_rectangle_count"] = rect_counts.astype(int)

    episodes["duration_sessions"] = (
        episodes["end_trading_day_index"] - episodes["start_trading_day_index"] + 1
    ).astype("int64")
    episodes["has_exact_rectangle_change"] = episodes["unique_rectangle_count"] > 1
    episodes["start_year"] = episodes["episode_start_date"].dt.year
    episodes["start_month"] = episodes["episode_start_date"].dt.to_period("M").astype(str)
    episodes["end_year"] = episodes["episode_end_date"].dt.year
    episodes["end_month"] = episodes["episode_end_date"].dt.to_period("M").astype(str)
    episodes["next_trading_day_index"] = episodes["end_trading_day_index"] + 1
    episodes["next_trading_date"] = episodes["next_trading_day_index"].map(calendar_lookup)
    episodes["end_reason"] = np.where(
        episodes["end_trading_day_index"].eq(calendar_max),
        "sample_ended",
        "not_detected_next_session",
    )

    for leg in ["A", "B", "C", "D"]:
        for field in ["contract_id", "option_symbol", "expiration", "strike"]:
            col = f"{leg}_{field}"
            if col in first_rows.columns:
                episodes[f"start_{col}"] = first_rows[col]
            if col in last_rows.columns:
                episodes[f"end_{col}"] = last_rows[col]

    episodes["pair_key"] = _join_key(
        episodes,
        ["start_A_expiration", "start_A_strike", "start_B_expiration", "start_B_strike"],
    )
    episodes["start_rectangle_key"] = _join_key(
        episodes,
        [
            "start_A_expiration",
            "start_A_strike",
            "start_B_expiration",
            "start_B_strike",
            "start_C_expiration",
            "start_C_strike",
            "start_D_expiration",
            "start_D_strike",
        ],
    )
    episodes["end_rectangle_key"] = _join_key(
        episodes,
        [
            "end_A_expiration",
            "end_A_strike",
            "end_B_expiration",
            "end_B_strike",
            "end_C_expiration",
            "end_C_strike",
            "end_D_expiration",
            "end_D_strike",
        ],
    )

    return episodes.reset_index(drop=True)


def add_episode_context_to_members(members: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "episode_id",
        "episode_start_date",
        "episode_end_date",
        "duration_sessions",
        "episode_row_count",
        "unique_rectangle_count",
        "has_exact_rectangle_change",
        "end_reason",
    ]
    context = episodes[cols].copy()
    out = members.merge(context, on="episode_id", how="left")
    out["is_episode_start"] = out["date"].eq(out["episode_start_date"])
    out["is_episode_end"] = out["date"].eq(out["episode_end_date"])
    return out


def build_episode_summary(
    members: pd.DataFrame,
    episodes: pd.DataFrame,
    input_path: str,
    clean_calendar_rows: int,
) -> dict:
    if episodes.empty:
        return {
            "input": input_path,
            "daily_violation_rows": int(len(members)),
            "episode_rows": 0,
            "clean_calendar_sessions": int(clean_calendar_rows),
        }

    starts_by_year = episodes["start_year"].value_counts().sort_index()
    rows_by_year = members["date"].dt.year.value_counts().sort_index()
    single_day = int(episodes["duration_sessions"].eq(1).sum())
    repeated_pair_keys = int((episodes["pair_key"].value_counts() > 1).sum())
    exact_changes = int(episodes["has_exact_rectangle_change"].sum())
    split_overlaps = _split_overlap_summary(episodes)

    duration = episodes["duration_sessions"]
    return {
        "input": input_path,
        "daily_violation_rows": int(len(members)),
        "episode_rows": int(len(episodes)),
        "clean_calendar_sessions": int(clean_calendar_rows),
        "unique_pair_keys": int(episodes["pair_key"].nunique()),
        "repeated_pair_keys_with_multiple_episodes": repeated_pair_keys,
        "single_day_episodes": single_day,
        "single_day_episode_share": float(single_day / len(episodes)),
        "episodes_with_exact_rectangle_change": exact_changes,
        "episodes_with_exact_rectangle_change_share": float(exact_changes / len(episodes)),
        "duration_sessions": {
            "min": int(duration.min()),
            "p25": float(duration.quantile(0.25)),
            "median": float(duration.median()),
            "mean": float(duration.mean()),
            "p75": float(duration.quantile(0.75)),
            "p95": float(duration.quantile(0.95)),
            "max": int(duration.max()),
        },
        "daily_rows_by_year": {str(int(k)): int(v) for k, v in rows_by_year.items()},
        "episode_starts_by_year": {str(int(k)): int(v) for k, v in starts_by_year.items()},
        "end_reason_counts": {str(k): int(v) for k, v in episodes["end_reason"].value_counts().items()},
        "split_overlap_diagnostics": split_overlaps,
    }


def _split_overlap_summary(episodes: pd.DataFrame) -> dict:
    splits = {
        "train_2013_2022": (pd.Timestamp("2013-01-01"), pd.Timestamp("2022-12-31")),
        "validation_2023": (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")),
        "test_2024_2026": (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-12-31")),
    }
    out: dict[str, int] = {}
    for name, (start, end) in splits.items():
        overlaps = (episodes["episode_start_date"] <= end) & (episodes["episode_end_date"] >= start)
        starts_inside = (episodes["episode_start_date"] >= start) & (episodes["episode_start_date"] <= end)
        out[f"{name}_overlapping_episodes"] = int(overlaps.sum())
        out[f"{name}_episode_starts"] = int(starts_inside.sum())

    boundaries = {
        "crosses_2022_2023_boundary": pd.Timestamp("2023-01-01"),
        "crosses_2023_2024_boundary": pd.Timestamp("2024-01-01"),
    }
    for name, boundary in boundaries.items():
        out[name] = int(((episodes["episode_start_date"] < boundary) & (episodes["episode_end_date"] >= boundary)).sum())
    return out


def build_pair_episodes(
    violations: pd.DataFrame,
    clean_options: pd.DataFrame,
    input_path: str = "",
) -> EpisodeBuildResult:
    calendar = build_trading_calendar(clean_options)
    members = assign_pair_episodes(violations, calendar)
    episodes = summarize_episodes(members, calendar)
    members = add_episode_context_to_members(members, episodes)
    summary = build_episode_summary(members, episodes, input_path, len(calendar))
    return EpisodeBuildResult(members=members, episodes=episodes, summary=summary)
