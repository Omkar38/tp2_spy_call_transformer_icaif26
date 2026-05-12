from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import get_context
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DetectionConfig


LEG_NAMES = {
    "A": ("K1", "T1"),
    "B": ("K2", "T2"),
    "C": ("K1", "T2"),
    "D": ("K2", "T1"),
}


@dataclass(frozen=True)
class LegQuote:
    contract_id: str
    expiration: pd.Timestamp
    strike: float
    bid: float
    ask: float
    mid: float
    spread: float
    spread_pct: float
    volume: float
    open_interest: float
    bid_size: float | None
    ask_size: float | None
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def _select_strikes(exp_df: pd.DataFrame, config: DetectionConfig) -> pd.DataFrame:
    if len(exp_df) <= config.max_strikes_per_expiration:
        return exp_df
    if "underlying_price" in exp_df.columns and exp_df["underlying_price"].notna().any():
        center = exp_df["underlying_price"].dropna().iloc[0]
        return exp_df.assign(_dist=(exp_df["strike"] - center).abs()).nsmallest(
            config.max_strikes_per_expiration, "_dist"
        ).drop(columns=["_dist"])
    return exp_df.assign(_dist=(exp_df["delta"].abs() - 0.35).abs()).nsmallest(
        config.max_strikes_per_expiration, "_dist"
    ).drop(columns=["_dist"])


def _leg(row: pd.Series) -> LegQuote:
    return LegQuote(
        contract_id=str(row["contractID"]),
        expiration=row["expiration"],
        strike=float(row["strike"]),
        bid=float(row["bid"]),
        ask=float(row["ask"]),
        mid=float(row["mid"]),
        spread=float(row["spread"]),
        spread_pct=float(row["spread_pct"]),
        volume=float(row.get("volume", np.nan)),
        open_interest=float(row.get("open_interest", np.nan)),
        bid_size=float(row["bid_size"]) if "bid_size" in row and pd.notna(row["bid_size"]) else None,
        ask_size=float(row["ask_size"]) if "ask_size" in row and pd.notna(row["ask_size"]) else None,
        implied_volatility=float(row.get("implied_volatility", np.nan)),
        delta=float(row.get("delta", np.nan)),
        gamma=float(row.get("gamma", np.nan)),
        theta=float(row.get("theta", np.nan)),
        vega=float(row.get("vega", np.nan)),
        rho=float(row.get("rho", np.nan)),
    )


def _emit_record(
    date: pd.Timestamp,
    option_type: str,
    k1: float,
    k2: float,
    t1: pd.Timestamp,
    t2: pd.Timestamp,
    legs: dict[str, pd.Series],
    lhs: float,
    rhs: float,
) -> dict:
    is_call = option_type == "call"
    severity = rhs - lhs if is_call else lhs - rhs
    gross = lhs + rhs
    normalized = severity / gross if gross > 0 else np.nan
    record = {
        "date": date,
        "violation_type": "TP2" if is_call else "RR2",
        "option_type": option_type,
        "K1": k1,
        "K2": k2,
        "T1_expiration": t1,
        "T2_expiration": t2,
        "T1_DTE": int((t1 - date).days),
        "T2_DTE": int((t2 - date).days),
        "strike_gap": k2 - k1,
        "term_gap": int((t2 - t1).days),
        "lhs": lhs,
        "rhs": rhs,
        "severity": severity,
        "gross_product_exposure": gross,
        "normalized_severity": normalized,
    }
    for name, row in legs.items():
        leg = _leg(row)
        prefix = f"{name}_"
        for field, value in leg.__dict__.items():
            record[prefix + field] = value
    return record


def detect_day(day_options: pd.DataFrame, config: DetectionConfig) -> pd.DataFrame:
    if day_options.empty:
        return pd.DataFrame()
    date = day_options["date"].iloc[0]
    frames: list[pd.DataFrame] = []

    for option_type in ["call", "put"]:
        typed = day_options[day_options["type"] == option_type]
        expirations = sorted(typed["expiration"].drop_duplicates())
        if config.max_expirations_per_day:
            expirations = expirations[: config.max_expirations_per_day]
        chains = {}
        for exp in expirations:
            exp_df = typed[typed["expiration"] == exp].drop_duplicates("strike", keep="last")
            exp_df = _select_strikes(exp_df, config)
            chains[exp] = exp_df.set_index("strike", drop=False).sort_index()

        for i, t1 in enumerate(expirations):
            for t2 in expirations[i + 1 :]:
                c1 = chains[t1]
                c2 = chains[t2]
                common = c1.index.intersection(c2.index).sort_values()
                n_common = len(common)
                if n_common < 2:
                    continue

                left_idx, right_idx = np.triu_indices(n_common, k=1)
                k1 = common.to_numpy(dtype=float)[left_idx]
                k2 = common.to_numpy(dtype=float)[right_idx]

                c1_common = c1.loc[common]
                c2_common = c2.loc[common]
                a_mid = c1_common["mid"].to_numpy(dtype=float)[left_idx]
                b_mid = c2_common["mid"].to_numpy(dtype=float)[right_idx]
                c_mid = c2_common["mid"].to_numpy(dtype=float)[left_idx]
                d_mid = c1_common["mid"].to_numpy(dtype=float)[right_idx]
                lhs = a_mid * b_mid
                rhs = c_mid * d_mid
                severity = rhs - lhs if option_type == "call" else lhs - rhs
                gross = lhs + rhs
                normalized = np.divide(severity, gross, out=np.full_like(severity, np.nan), where=gross > 0)
                keep = (gross > 0) & (normalized >= config.min_normalized_severity)
                if not keep.any():
                    continue

                if option_type == "call":
                    keep &= lhs < rhs
                    violation_type = "TP2"
                else:
                    keep &= lhs > rhs
                    violation_type = "RR2"
                if not keep.any():
                    continue

                a_rows = c1_common.iloc[left_idx[keep]].reset_index(drop=True)
                b_rows = c2_common.iloc[right_idx[keep]].reset_index(drop=True)
                c_rows = c2_common.iloc[left_idx[keep]].reset_index(drop=True)
                d_rows = c1_common.iloc[right_idx[keep]].reset_index(drop=True)

                chunk = pd.DataFrame(
                    {
                        "date": date,
                        "violation_type": violation_type,
                        "option_type": option_type,
                        "K1": k1[keep],
                        "K2": k2[keep],
                        "T1_expiration": t1,
                        "T2_expiration": t2,
                        "T1_DTE": int((t1 - date).days),
                        "T2_DTE": int((t2 - date).days),
                        "strike_gap": k2[keep] - k1[keep],
                        "term_gap": int((t2 - t1).days),
                        "lhs": lhs[keep],
                        "rhs": rhs[keep],
                        "severity": severity[keep],
                        "gross_product_exposure": gross[keep],
                        "normalized_severity": normalized[keep],
                    }
                )
                for prefix, rows in {"A": a_rows, "B": b_rows, "C": c_rows, "D": d_rows}.items():
                    _append_leg_columns(chunk, prefix, rows)
                frames.append(chunk)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out["violation_id"] = (
            out["date"].dt.strftime("%Y%m%d")
            + "_"
            + out["option_type"]
            + "_"
            + out["K1"].astype(str)
            + "_"
            + out["K2"].astype(str)
            + "_"
            + out["T1_expiration"].dt.strftime("%Y%m%d")
            + "_"
            + out["T2_expiration"].dt.strftime("%Y%m%d")
        )
    return out


def _append_leg_columns(out: pd.DataFrame, prefix: str, rows: pd.DataFrame) -> None:
    leg_columns = {
        "contract_id": "contractID",
        "expiration": "expiration",
        "strike": "strike",
        "bid": "bid",
        "ask": "ask",
        "mid": "mid",
        "spread": "spread",
        "spread_pct": "spread_pct",
        "volume": "volume",
        "open_interest": "open_interest",
        "bid_size": "bid_size",
        "ask_size": "ask_size",
        "implied_volatility": "implied_volatility",
        "delta": "delta",
        "gamma": "gamma",
        "theta": "theta",
        "vega": "vega",
        "rho": "rho",
    }
    for output_name, source_name in leg_columns.items():
        out[f"{prefix}_{output_name}"] = rows[source_name].to_numpy() if source_name in rows else np.nan


def _detect_day_worker(args: tuple[pd.DataFrame, DetectionConfig]) -> pd.DataFrame:
    day, config = args
    return detect_day(day, config)


def _iter_days(options: pd.DataFrame) -> Iterable[pd.DataFrame]:
    for _, day in options.groupby("date", sort=True):
        yield day


def detect_violations(options: pd.DataFrame, config: DetectionConfig, progress: bool = False) -> pd.DataFrame:
    frames = []
    days = list(_iter_days(options))
    total = len(days)
    if config.n_jobs and config.n_jobs > 1 and total > 1:
        ctx = get_context("fork")
        with ctx.Pool(processes=config.n_jobs) as pool:
            for idx, found in enumerate(pool.imap_unordered(_detect_day_worker, [(day, config) for day in days], chunksize=1), 1):
                if not found.empty:
                    frames.append(found)
                if progress and (idx == total or idx % 25 == 0):
                    rows = sum(len(frame) for frame in frames)
                    print(f"Detected {rows:,} rows through {idx:,}/{total:,} days", flush=True)
    else:
        for idx, day in enumerate(days, 1):
            found = detect_day(day, config)
            if not found.empty:
                frames.append(found)
            if progress and (idx == total or idx % 25 == 0):
                rows = sum(len(frame) for frame in frames)
                print(f"Detected {rows:,} rows through {idx:,}/{total:,} days", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
