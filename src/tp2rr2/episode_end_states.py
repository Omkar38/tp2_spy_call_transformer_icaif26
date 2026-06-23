from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .archive2_ingest import deduplicate_archive2, normalize_for_pipeline


EPISODE_COLUMNS = [
    "episode_id",
    "episode_start_date",
    "episode_end_date",
    "duration_sessions",
    "episode_row_count",
    "end_year",
    "end_trading_day_index",
    "next_trading_day_index",
    "next_trading_date",
    "end_A_expiration",
    "end_A_strike",
    "end_B_expiration",
    "end_B_strike",
    "end_C_expiration",
    "end_C_strike",
    "end_D_expiration",
    "end_D_strike",
    "end_lhs",
    "end_rhs",
    "end_normalized_severity",
    "pair_key",
]

CLEAN_COLUMNS = [
    "date",
    "expiration",
    "strike",
    "type",
    "contractID",
    "option_symbol",
    "bid",
    "ask",
    "mid",
    "spread_pct",
    "volume",
    "open_interest",
    "forward",
]

RAW_COLUMNS = [
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
class EndStateResult:
    end_states: pd.DataFrame
    summary: dict


def _round_up(strikes: np.ndarray, target: float) -> float | None:
    idx = int(np.searchsorted(strikes, target, side="left"))
    if idx >= len(strikes):
        return None
    return float(strikes[idx])


def _quality_ok(row, max_spread_pct: float) -> bool:
    bid = pd.to_numeric(getattr(row, "bid", np.nan), errors="coerce")
    ask = pd.to_numeric(getattr(row, "ask", np.nan), errors="coerce")
    mid = pd.to_numeric(getattr(row, "mid", np.nan), errors="coerce")
    spread_pct = pd.to_numeric(getattr(row, "spread_pct", np.nan), errors="coerce")
    return (
        pd.notna(bid)
        and pd.notna(ask)
        and pd.notna(mid)
        and bid > 0
        and ask > bid
        and mid > 0
        and pd.notna(spread_pct)
        and spread_pct <= max_spread_pct
    )


def _forward_for_chain(chain: pd.DataFrame) -> float:
    values = pd.to_numeric(chain["forward"], errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _option_map(day: pd.DataFrame) -> dict[tuple[pd.Timestamp, float], pd.Series]:
    if day.empty:
        return {}
    typed = day[day["type"].eq("call")].copy()
    typed["expiration"] = pd.to_datetime(typed["expiration"])
    typed["strike"] = pd.to_numeric(typed["strike"], errors="coerce").astype(float)
    typed = typed.dropna(subset=["expiration", "strike"])
    typed = typed.sort_values(["expiration", "strike", "contractID"]).drop_duplicates(
        ["expiration", "strike"], keep="last"
    )
    return {(pd.Timestamp(row.expiration), float(row.strike)): row for row in typed.itertuples(index=False)}


def _chain_map(day: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    if day.empty:
        return {}
    typed = day[day["type"].eq("call")].copy()
    typed["expiration"] = pd.to_datetime(typed["expiration"])
    typed["strike"] = pd.to_numeric(typed["strike"], errors="coerce").astype(float)
    typed = typed.dropna(subset=["expiration", "strike"])
    chains: dict[pd.Timestamp, pd.DataFrame] = {}
    for exp, chain in typed.groupby("expiration", sort=True):
        chains[pd.Timestamp(exp)] = chain.sort_values("strike").drop_duplicates("strike", keep="last").reset_index(drop=True)
    return chains


def _raw_presence_set(raw_day: pd.DataFrame) -> set[tuple[pd.Timestamp, float]]:
    if raw_day.empty:
        return set()
    typed = raw_day[raw_day["type"].eq("call")].copy()
    typed["expiration"] = pd.to_datetime(typed["expiration"])
    typed["strike"] = pd.to_numeric(typed["strike"], errors="coerce").astype(float)
    typed = typed.dropna(subset=["expiration", "strike"])
    return set(zip(typed["expiration"].map(pd.Timestamp), typed["strike"].astype(float)))


def _load_raw_normalized_year(archive_root: Path, year: int) -> pd.DataFrame:
    path = archive_root / f"year={year}" / "data.parquet"
    if not path.exists():
        return pd.DataFrame(columns=CLEAN_COLUMNS)
    raw = pd.read_parquet(path, columns=RAW_COLUMNS)
    deduped = deduplicate_archive2(raw)
    normalized = normalize_for_pipeline(deduped)
    return normalized[CLEAN_COLUMNS].copy()


def _group_by_date(df: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    if df.empty:
        return {}
    return {pd.Timestamp(date): day for date, day in df.groupby("date", sort=False)}


def _classify_one(
    episode: pd.Series,
    clean_options: dict[tuple[pd.Timestamp, float], pd.Series],
    clean_chains: dict[pd.Timestamp, pd.DataFrame],
    raw_presence: set[tuple[pd.Timestamp, float]],
    max_spread_pct: float,
    min_normalized_severity: float,
) -> dict:
    next_date = pd.Timestamp(episode["next_trading_date"]) if pd.notna(episode.get("next_trading_date")) else pd.NaT
    if pd.isna(next_date):
        return {
            "episode_id": episode["episode_id"],
            "next_trading_date": pd.NaT,
            "end_state": "sample_ended",
            "end_state_detail": "episode ends on final clean-data session",
            "next_lhs": np.nan,
            "next_rhs": np.nan,
            "next_normalized_severity": np.nan,
        }

    t1 = pd.Timestamp(episode["end_A_expiration"])
    t2 = pd.Timestamp(episode["end_B_expiration"])
    k1 = float(episode["end_A_strike"])
    k2 = float(episode["end_B_strike"])

    base = {
        "episode_id": episode["episode_id"],
        "next_trading_date": next_date,
        "next_A_expiration": t1,
        "next_A_strike": k1,
        "next_B_expiration": t2,
        "next_B_strike": k2,
        "next_lhs": np.nan,
        "next_rhs": np.nan,
        "next_severity": np.nan,
        "next_normalized_severity": np.nan,
        "next_C_expiration": pd.NaT,
        "next_C_strike": np.nan,
        "next_D_expiration": pd.NaT,
        "next_D_strike": np.nan,
    }

    if next_date >= t1 or next_date >= t2:
        return base | {
            "end_state": "left_leg_expired_or_missing",
            "end_state_detail": "A or B expiration reached before next-state check",
        }

    a_key = (t1, k1)
    b_key = (t2, k2)
    a = clean_options.get(a_key)
    b = clean_options.get(b_key)
    if a is None or b is None:
        raw_has_a = a_key in raw_presence
        raw_has_b = b_key in raw_presence
        if raw_has_a or raw_has_b:
            return base | {
                "end_state": "liquidity_filter_failure",
                "end_state_detail": "A/B present in raw next day but missing from clean quote-quality universe",
            }
        return base | {
            "end_state": "left_leg_expired_or_missing",
            "end_state_detail": "A or B not present in clean or raw next day",
        }

    c1 = clean_chains.get(t1)
    c2 = clean_chains.get(t2)
    if c1 is None or c2 is None or c1.empty or c2.empty:
        return base | {
            "end_state": "right_leg_missing_or_expired",
            "end_state_detail": "one required expiration chain is unavailable in clean data",
        }

    f1 = _forward_for_chain(c1)
    f2 = _forward_for_chain(c2)
    if not np.isfinite(f1) or not np.isfinite(f2) or f1 <= 0 or f2 <= 0:
        return base | {
            "end_state": "right_leg_missing_or_expired",
            "end_state_detail": "forward unavailable for required expiration chain",
        }

    k1_adjusted = k1 * f2 / f1
    k2_adjusted = k2 * f1 / f2
    k2_grid = c2["strike"].to_numpy(dtype=float)
    k1_grid = c1["strike"].to_numpy(dtype=float)
    c_strike = _round_up(k2_grid, k1_adjusted)
    d_strike = _round_up(k1_grid, k2_adjusted)
    base["next_C_expiration"] = t2
    base["next_D_expiration"] = t1
    base["next_C_strike"] = np.nan if c_strike is None else c_strike
    base["next_D_strike"] = np.nan if d_strike is None else d_strike

    if c_strike is None or d_strike is None:
        return base | {
            "end_state": "right_leg_missing_or_expired",
            "end_state_detail": "forward-adjusted rounded C/D strike out of available clean-chain bounds",
        }

    c_key = (t2, c_strike)
    d_key = (t1, d_strike)
    c = clean_options.get(c_key)
    d = clean_options.get(d_key)
    if c is None or d is None:
        raw_has_c = c_key in raw_presence
        raw_has_d = d_key in raw_presence
        if raw_has_c or raw_has_d:
            return base | {
                "end_state": "liquidity_filter_failure",
                "end_state_detail": "C/D present in raw next day but missing from clean quote-quality universe",
            }
        return base | {
            "end_state": "right_leg_missing_or_expired",
            "end_state_detail": "required rounded C or D leg unavailable in clean or raw next day",
        }

    legs = {"A": a, "B": b, "C": c, "D": d}
    bad_legs = [name for name, row in legs.items() if not _quality_ok(row, max_spread_pct)]
    if bad_legs:
        return base | {
            "end_state": "liquidity_filter_failure",
            "end_state_detail": "clean legs failed explicit quote-quality recheck: " + ",".join(bad_legs),
        }

    a_mid = float(a.mid)
    b_mid = float(b.mid)
    c_mid = float(c.mid)
    d_mid = float(d.mid)
    lhs = a_mid * b_mid
    rhs = c_mid * d_mid
    severity = rhs - lhs
    gross = lhs + rhs
    normalized = severity / gross if gross > 0 else np.nan
    state = "true_reversion" if lhs >= rhs else "still_violation_or_reappeared"
    if state == "still_violation_or_reappeared" and np.isfinite(normalized) and normalized < min_normalized_severity:
        detail = "lhs < rhs on next day, but below detector severity threshold"
    elif state == "still_violation_or_reappeared":
        detail = "lhs < rhs on next day with all legs observable"
    else:
        detail = "all legs observable and lhs >= rhs"
    return base | {
        "end_state": state,
        "end_state_detail": detail,
        "next_lhs": lhs,
        "next_rhs": rhs,
        "next_severity": severity,
        "next_normalized_severity": normalized,
    }


def classify_episode_end_states(
    episodes: pd.DataFrame,
    clean: pd.DataFrame,
    archive_root: Path,
    max_spread_pct: float = 0.50,
    min_normalized_severity: float = 0.0001,
) -> EndStateResult:
    episodes = episodes.copy()
    for col in [
        "episode_start_date",
        "episode_end_date",
        "next_trading_date",
        "end_A_expiration",
        "end_B_expiration",
    ]:
        if col in episodes:
            episodes[col] = pd.to_datetime(episodes[col])

    clean = clean.copy()
    clean["date"] = pd.to_datetime(clean["date"])
    clean["expiration"] = pd.to_datetime(clean["expiration"])
    clean["strike"] = pd.to_numeric(clean["strike"], errors="coerce").astype(float)

    rows: list[dict] = []
    sample_ended = episodes[episodes["next_trading_date"].isna()]
    for _, episode in sample_ended.iterrows():
        rows.append(_classify_one(episode, {}, {}, set(), max_spread_pct, min_normalized_severity))

    pending = episodes[episodes["next_trading_date"].notna()].copy()
    pending["next_year"] = pending["next_trading_date"].dt.year.astype(int)
    clean_by_date = {pd.Timestamp(date): day for date, day in clean.groupby("date", sort=False)}

    raw_year_cache: dict[int, dict[pd.Timestamp, pd.DataFrame]] = {}
    grouped_pending = list(pending.groupby("next_trading_date", sort=True))
    total_dates = len(grouped_pending)
    for date_idx, (next_date, group) in enumerate(grouped_pending, 1):
        next_date = pd.Timestamp(next_date)
        year = int(next_date.year)
        if year not in raw_year_cache:
            raw_year_cache[year] = _group_by_date(_load_raw_normalized_year(archive_root, year))
        raw_day = raw_year_cache[year].get(next_date, pd.DataFrame(columns=CLEAN_COLUMNS))
        clean_day = clean_by_date.get(next_date, pd.DataFrame(columns=CLEAN_COLUMNS))
        clean_options = _option_map(clean_day)
        clean_chains = _chain_map(clean_day)
        raw_presence = _raw_presence_set(raw_day)
        for episode in group.to_dict("records"):
            rows.append(
                _classify_one(
                    episode,
                    clean_options,
                    clean_chains,
                    raw_presence,
                    max_spread_pct=max_spread_pct,
                    min_normalized_severity=min_normalized_severity,
                )
            )
        if date_idx == total_dates or date_idx % 250 == 0:
            print(
                f"Classified next-state dates {date_idx:,}/{total_dates:,}; rows so far {len(rows):,}",
                flush=True,
            )

    end_states = pd.DataFrame(rows)
    out = episodes.merge(end_states, on="episode_id", how="left", suffixes=("", "_classified"))
    out["is_true_reversion"] = out["end_state"].eq("true_reversion")
    out["is_observable_next_state"] = out["end_state"].isin(["true_reversion", "still_violation_or_reappeared"])

    counts = out["end_state"].value_counts(dropna=False)
    by_year = out.groupby(["end_year", "end_state"], dropna=False).size().reset_index(name="episode_count")
    summary = {
        "episode_rows": int(len(out)),
        "unique_episode_ids": int(out["episode_id"].nunique()),
        "missing_end_state": int(out["end_state"].isna().sum()),
        "duplicate_episode_ids": int(out["episode_id"].duplicated().sum()),
        "end_state_counts": {str(k): int(v) for k, v in counts.items()},
        "true_reversion_share": float(out["is_true_reversion"].mean()) if len(out) else 0.0,
        "observable_next_state_share": float(out["is_observable_next_state"].mean()) if len(out) else 0.0,
        "max_spread_pct": float(max_spread_pct),
        "min_normalized_severity": float(min_normalized_severity),
    }
    summary["end_state_counts_by_year"] = {
        f"{int(row.end_year)}|{row.end_state}": int(row.episode_count)
        for row in by_year.itertuples(index=False)
    }
    return EndStateResult(end_states=out, summary=summary)
