from __future__ import annotations

from multiprocessing import get_context
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DetectionConfig
from .detect_violations import _append_leg_columns, _select_strikes


def _forward_for_chain(chain: pd.DataFrame) -> float:
    if "forward" not in chain.columns:
        return np.nan
    values = pd.to_numeric(chain["forward"], errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _round_up_indices(strikes: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.searchsorted(strikes, targets, side="left")


def detect_forward_tp2_day(day_options: pd.DataFrame, config: DetectionConfig) -> pd.DataFrame:
    """Detect forward-adjusted TP2 call violations for one date.

    Uses the Glasserman-Li-Pirjol call condition with forward-adjusted strikes:

        C(K1,T1) C(K2,T2) >= C(K1 F2/F1,T2) C(K2 F1/F2,T1)

    The adjusted strikes are rounded up to the nearest available strike in the
    relevant expiration, matching the conservative empirical convention in the
    motivating paper.
    """
    if day_options.empty:
        return pd.DataFrame()

    date = pd.Timestamp(day_options["date"].iloc[0])
    typed = day_options[day_options["type"] == "call"].copy()
    expirations = sorted(typed["expiration"].drop_duplicates())
    if config.max_expirations_per_day:
        expirations = expirations[: config.max_expirations_per_day]

    chains: dict[pd.Timestamp, pd.DataFrame] = {}
    forwards: dict[pd.Timestamp, float] = {}
    for exp in expirations:
        exp_df = typed[typed["expiration"] == exp].drop_duplicates("strike", keep="last")
        exp_df = _select_strikes(exp_df, config).sort_values("strike").reset_index(drop=True)
        if len(exp_df) >= 2:
            chains[exp] = exp_df
            forwards[exp] = _forward_for_chain(exp_df)

    frames: list[pd.DataFrame] = []
    for i, t1 in enumerate(expirations):
        if t1 not in chains:
            continue
        c1 = chains[t1]
        f1 = forwards.get(t1, np.nan)
        if not np.isfinite(f1) or f1 <= 0:
            continue
        k1_grid = c1["strike"].to_numpy(dtype=float)

        for t2 in expirations[i + 1 :]:
            if t2 not in chains:
                continue
            c2 = chains[t2]
            f2 = forwards.get(t2, np.nan)
            if not np.isfinite(f2) or f2 <= 0:
                continue
            k2_grid = c2["strike"].to_numpy(dtype=float)

            left_idx, right_idx = np.meshgrid(
                np.arange(len(k1_grid)), np.arange(len(k2_grid)), indexing="ij"
            )
            left_idx = left_idx.ravel()
            right_idx = right_idx.ravel()
            k1 = k1_grid[left_idx]
            k2 = k2_grid[right_idx]

            forward_order = (k1 / f1) < (k2 / f2)
            if not forward_order.any():
                continue
            left_idx = left_idx[forward_order]
            right_idx = right_idx[forward_order]
            k1 = k1[forward_order]
            k2 = k2[forward_order]

            k1_adjusted = k1 * f2 / f1
            k2_adjusted = k2 * f1 / f2
            c_idx = _round_up_indices(k2_grid, k1_adjusted)
            d_idx = _round_up_indices(k1_grid, k2_adjusted)
            in_bounds = (c_idx < len(k2_grid)) & (d_idx < len(k1_grid))
            if not in_bounds.any():
                continue

            left_idx = left_idx[in_bounds]
            right_idx = right_idx[in_bounds]
            c_idx = c_idx[in_bounds]
            d_idx = d_idx[in_bounds]
            k1 = k1[in_bounds]
            k2 = k2[in_bounds]
            k1_adjusted = k1_adjusted[in_bounds]
            k2_adjusted = k2_adjusted[in_bounds]

            a_mid = c1["mid"].to_numpy(dtype=float)[left_idx]
            b_mid = c2["mid"].to_numpy(dtype=float)[right_idx]
            c_mid = c2["mid"].to_numpy(dtype=float)[c_idx]
            d_mid = c1["mid"].to_numpy(dtype=float)[d_idx]
            lhs = a_mid * b_mid
            rhs = c_mid * d_mid
            severity = rhs - lhs
            gross = lhs + rhs
            normalized = np.divide(severity, gross, out=np.full_like(severity, np.nan), where=gross > 0)
            keep = (gross > 0) & (lhs < rhs) & (normalized >= config.min_normalized_severity)
            if not keep.any():
                continue

            a_rows = c1.iloc[left_idx[keep]].reset_index(drop=True)
            b_rows = c2.iloc[right_idx[keep]].reset_index(drop=True)
            c_rows = c2.iloc[c_idx[keep]].reset_index(drop=True)
            d_rows = c1.iloc[d_idx[keep]].reset_index(drop=True)
            chunk = pd.DataFrame(
                {
                    "date": date,
                    "detection_method": "forward_adjusted_round_up",
                    "violation_type": "TP2",
                    "option_type": "call",
                    "K1": k1[keep],
                    "K2": k2[keep],
                    "T1_expiration": t1,
                    "T2_expiration": t2,
                    "T1_DTE": int((pd.Timestamp(t1) - date).days),
                    "T2_DTE": int((pd.Timestamp(t2) - date).days),
                    "strike_gap": k2[keep] - k1[keep],
                    "term_gap": int((pd.Timestamp(t2) - pd.Timestamp(t1)).days),
                    "F_T1": f1,
                    "F_T2": f2,
                    "K1_forward_adjusted": k1_adjusted[keep],
                    "K2_forward_adjusted": k2_adjusted[keep],
                    "K1_forward_rounded": c_rows["strike"].to_numpy(dtype=float),
                    "K2_forward_rounded": d_rows["strike"].to_numpy(dtype=float),
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
            + "_fwd_call_"
            + out["K1"].astype(str)
            + "_"
            + out["K2"].astype(str)
            + "_"
            + out["T1_expiration"].dt.strftime("%Y%m%d")
            + "_"
            + out["T2_expiration"].dt.strftime("%Y%m%d")
            + "_"
            + out["K1_forward_rounded"].astype(str)
            + "_"
            + out["K2_forward_rounded"].astype(str)
        )
    return out


def _iter_days(options: pd.DataFrame) -> Iterable[pd.DataFrame]:
    for _, day in options.groupby("date", sort=True):
        yield day


def _detect_day_worker(args: tuple[pd.DataFrame, DetectionConfig]) -> pd.DataFrame:
    day, config = args
    return detect_forward_tp2_day(day, config)


def detect_forward_tp2(options: pd.DataFrame, config: DetectionConfig, progress: bool = False) -> pd.DataFrame:
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
                    print(f"Detected {rows:,} forward-adjusted rows through {idx:,}/{total:,} days", flush=True)
    else:
        for idx, day in enumerate(days, 1):
            found = detect_forward_tp2_day(day, config)
            if not found.empty:
                frames.append(found)
            if progress and (idx == total or idx % 25 == 0):
                rows = sum(len(frame) for frame in frames)
                print(f"Detected {rows:,} forward-adjusted rows through {idx:,}/{total:,} days", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
