"""
ml/features.py — Feature extraction from SQLite for DS-CAT training.

Two feature streams per violation:

  Temporal stream (seq_len × d_temporal):
    For the N snapshot hours BEFORE detection, capture:
    - Market regime (spot return, realized vol, IV level/skew/slope)
    - Surface health (n_instruments, n_violations, avg/max violation)
    - The specific quad's prices and TP2 ratio
    - Cyclical time encoding (hour-of-day, day-of-week)

  Surface stream (n_instruments × d_surface):
    Full option surface at detection time:
    - Moneyness, TTM, IV, spread, normalized price, OI, volume
    - Binary flag: is this instrument a leg of the detected violation?

Labels:
    Binary — fast reversion (<4h) = 1, else 0.
    Source: reversions table, revert_speed field.
"""

import math
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

log = logging.getLogger("ml.features")


# ── Feature dimensions (change here, everything adjusts) ──────
# Temporal stream
TEMPORAL_FEATURES = [
    # Market regime (5)
    "spot_log_return",       # log(spot_t / spot_t-1)
    "realized_vol",          # annualized realized vol from recent returns
    "iv_level",              # avg mark_iv across surface
    "iv_skew",               # OTM iv - ATM iv
    "term_slope",            # far iv - near iv
    # Surface health (4)
    "n_clean",               # number of clean instruments
    "n_violations",          # violations detected this hour
    "avg_viol_pct",          # avg violation_pct this hour
    "max_viol_pct",          # max violation_pct this hour
    # Quad-specific (6)
    "quad_c11",              # C(K1,T1) / spot  (normalized)
    "quad_c21",              # C(K2,T1) / spot
    "quad_c12",              # C(K1,T2) / spot
    "quad_c22",              # C(K2,T2) / spot
    "quad_tp2_ratio",        # RHS / LHS  (>1 = violation)
    "quad_avg_spread",       # avg spread_pct across 4 legs
    # Time encoding (4) — cyclical to capture periodicity
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
]
D_TEMPORAL = len(TEMPORAL_FEATURES)  # 19

# Surface stream
SURFACE_FEATURES = [
    "moneyness",             # strike / spot
    "ttm",                   # time to expiry in days (log-scaled)
    "mark_iv",               # implied volatility
    "spread_pct",            # bid-ask spread / mid
    "norm_price",            # mid_price / spot
    "log_oi",                # log(1 + open_interest)
    "log_volume",            # log(1 + volume_24h)
    "is_violation_leg",      # 1 if this is one of the 4 legs
]
D_SURFACE = len(SURFACE_FEATURES)  # 8

# Defaults
DEFAULT_SEQ_LEN      = 60   # hours of lookback
DEFAULT_MAX_SURFACE  = 400  # max instruments (pad/truncate to this)


# ── Label extraction ──────────────────────────────────────────

def get_labeled_violations(conn: sqlite3.Connection) -> list[dict]:
    """
    Get all violations that have a definitive reversion label.

    A violation is 'labeled' once:
      - It has reverted (revert_speed in fast/medium/slow), OR
      - 48+ hours have passed without reversion (revert_speed = 'never')

    Returns list of dicts with violation columns + binary label.
    """
    rows = conn.execute("""
        SELECT
            v.id,
            v.snapshot_hour,
            v.T1, v.T2, v.K1, v.K2,
            v.C_K1_T1, v.C_K2_T1, v.C_K1_T2, v.C_K2_T2,
            v.LHS, v.RHS, v.violation_pct,
            v.maturity_gap, v.strike_gap_pct,
            v.moneyness_K1, v.moneyness_K2,
            v.avg_spread_pct, v.est_total_cost_pct, v.net_edge,
            v.btc_spot,
            v.btc_return_1h, v.btc_return_24h,
            v.realized_vol_24h,
            v.iv_level_surface, v.iv_skew,
            v.term_structure_slope,
            v.hour_of_day, v.day_of_week,
            v.profitability,
            r.revert_speed
        FROM violations v
        JOIN reversions r ON r.violation_id = v.id
        WHERE r.revert_speed IN ('fast', 'medium', 'slow', 'never')
        GROUP BY v.id
        HAVING MIN(
            CASE r.revert_speed
                WHEN 'fast'   THEN 1
                WHEN 'medium' THEN 2
                WHEN 'slow'   THEN 3
                WHEN 'never'  THEN 4
            END
        )
        ORDER BY v.snapshot_hour ASC
    """).fetchall()

    violations = []
    for r in rows:
        speed = r["revert_speed"]
        # Binary label: fast reversion (<4h) is the profitable class
        label = 1 if speed == "fast" else 0

        violations.append({
            "id":              r["id"],
            "snapshot_hour":   r["snapshot_hour"],
            "T1":              r["T1"],
            "T2":              r["T2"],
            "K1":              r["K1"],
            "K2":              r["K2"],
            "C_K1_T1":         r["C_K1_T1"],
            "C_K2_T1":         r["C_K2_T1"],
            "C_K1_T2":         r["C_K1_T2"],
            "C_K2_T2":         r["C_K2_T2"],
            "btc_spot":        r["btc_spot"],
            "violation_pct":   r["violation_pct"],
            "net_edge":        r["net_edge"],
            "avg_spread_pct":  r["avg_spread_pct"],
            "revert_speed":    speed,
            "label":           label,
        })

    n_pos = sum(v["label"] for v in violations)
    n_neg = len(violations) - n_pos
    log.info(
        f"Labeled violations: {len(violations)} total | "
        f"{n_pos} fast ({100*n_pos/max(len(violations),1):.1f}%) | "
        f"{n_neg} slow/never"
    )
    return violations


# ── Temporal feature extraction ───────────────────────────────

def _get_snapshot_hours(conn: sqlite3.Connection) -> list[str]:
    """All distinct snapshot hours, chronologically sorted."""
    rows = conn.execute("""
        SELECT DISTINCT snapshot_hour FROM snapshots
        WHERE is_clean = 1
        ORDER BY snapshot_hour ASC
    """).fetchall()
    return [r["snapshot_hour"] for r in rows]


def _build_hourly_regime_cache(conn: sqlite3.Connection) -> dict:
    """
    Pre-compute regime features for every snapshot hour.
    Returns: {snapshot_hour → dict of feature values}

    This runs ONCE and caches — avoids N×seq_len SQL queries.
    """
    hours = _get_snapshot_hours(conn)
    cache = {}

    # ── Batch: spot prices per hour ──
    spot_rows = conn.execute("""
        SELECT snapshot_hour, AVG(underlying_price) as avg_spot, COUNT(*) as n_clean
        FROM snapshots WHERE is_clean = 1
        GROUP BY snapshot_hour ORDER BY snapshot_hour
    """).fetchall()
    spot_map = {r["snapshot_hour"]: (r["avg_spot"], r["n_clean"]) for r in spot_rows}

    # ── Batch: IV level per hour ──
    iv_rows = conn.execute("""
        SELECT snapshot_hour, AVG(mark_iv) as iv
        FROM snapshots WHERE is_clean = 1 AND mark_iv IS NOT NULL
        GROUP BY snapshot_hour
    """).fetchall()
    iv_map = {r["snapshot_hour"]: r["iv"] for r in iv_rows}

    # ── Batch: violation stats per hour ──
    viol_rows = conn.execute("""
        SELECT snapshot_hour,
               COUNT(*) as n,
               AVG(violation_pct) as avg_pct,
               MAX(violation_pct) as max_pct
        FROM violations
        GROUP BY snapshot_hour
    """).fetchall()
    viol_map = {r["snapshot_hour"]: (r["n"], r["avg_pct"], r["max_pct"]) for r in viol_rows}

    # ── Build cache ──
    prev_spot = None
    for hour in hours:
        spot, n_clean = spot_map.get(hour, (None, 0))

        # Log return
        if spot and prev_spot and prev_spot > 0:
            spot_lr = math.log(spot / prev_spot)
        else:
            spot_lr = 0.0

        iv_level = iv_map.get(hour)
        n_viols, avg_viol, max_viol = viol_map.get(hour, (0, 0.0, 0.0))

        dt = datetime.fromisoformat(hour)

        cache[hour] = {
            "spot":          spot,
            "spot_log_return": spot_lr,
            "iv_level":      iv_level or 0.0,
            "n_clean":       n_clean,
            "n_violations":  n_viols,
            "avg_viol_pct":  avg_viol or 0.0,
            "max_viol_pct":  max_viol or 0.0,
            "hour_of_day":   dt.hour,
            "day_of_week":   dt.weekday(),
        }
        prev_spot = spot

    # ── Second pass: rolling realized vol + skew + term slope ──
    # (These need the feature computation from processor/features.py,
    #  but we approximate with cached data to avoid N queries.)
    # For skew and term slope, pull from violation table where available.
    viol_regime = conn.execute("""
        SELECT DISTINCT snapshot_hour, iv_skew, term_structure_slope, realized_vol_24h
        FROM violations
    """).fetchall()
    for r in viol_regime:
        h = r["snapshot_hour"]
        if h in cache:
            cache[h]["iv_skew"]    = r["iv_skew"] or 0.0
            cache[h]["term_slope"] = r["term_structure_slope"] or 0.0
            cache[h]["realized_vol"] = r["realized_vol_24h"] or 0.0

    # Fill missing skew/slope/vol with 0
    for h in cache:
        cache[h].setdefault("iv_skew", 0.0)
        cache[h].setdefault("term_slope", 0.0)
        cache[h].setdefault("realized_vol", 0.0)

    log.info(f"Regime cache built: {len(cache)} hours")
    return cache


def build_temporal_sequence(
    conn: sqlite3.Connection,
    violation: dict,
    regime_cache: dict,
    all_hours: list[str],
    seq_len: int = DEFAULT_SEQ_LEN,
) -> np.ndarray:
    """
    Build the temporal feature tensor for one violation.

    Returns: np.ndarray of shape (seq_len, D_TEMPORAL)
             Left-padded with zeros if fewer than seq_len hours available.
    """
    detection_hour = violation["snapshot_hour"]
    spot = violation["btc_spot"] or 1.0  # normalize prices by spot

    # Find the index of detection_hour in the sorted list
    try:
        det_idx = all_hours.index(detection_hour)
    except ValueError:
        # Detection hour not in clean snapshots — use zeros
        return np.zeros((seq_len, D_TEMPORAL), dtype=np.float32)

    # Slice the last seq_len hours up to and including detection
    start_idx = max(0, det_idx - seq_len + 1)
    window_hours = all_hours[start_idx : det_idx + 1]

    # ── Look up quad prices across these hours ──
    # Check if the same (T1,T2,K1,K2) quad existed in earlier snapshots
    t1, t2 = violation["T1"], violation["T2"]
    k1, k2 = violation["K1"], violation["K2"]

    # Query: get the 4 leg mid prices for each hour in the window
    placeholders = ",".join("?" * len(window_hours))
    quad_rows = conn.execute(f"""
        SELECT snapshot_hour, expiry, strike, mid_price, spread_pct
        FROM snapshots
        WHERE snapshot_hour IN ({placeholders})
          AND is_clean = 1
          AND option_type = 'C'
          AND expiry IN (?, ?)
          AND strike IN (?, ?)
        ORDER BY snapshot_hour
    """, window_hours + [t1, t2, k1, k2]).fetchall()

    # Organize: hour → {(expiry, strike) → (mid, spread)}
    quad_by_hour = {}
    for r in quad_rows:
        h = r["snapshot_hour"]
        if h not in quad_by_hour:
            quad_by_hour[h] = {}
        quad_by_hour[h][(r["expiry"], r["strike"])] = (
            r["mid_price"], r["spread_pct"]
        )

    # ── Build tensor ──
    tensor = np.zeros((seq_len, D_TEMPORAL), dtype=np.float32)
    offset = seq_len - len(window_hours)  # left-pad index

    for i, hour in enumerate(window_hours):
        row_idx = offset + i
        regime = regime_cache.get(hour, {})

        # Market regime features
        tensor[row_idx, 0] = regime.get("spot_log_return", 0.0)
        tensor[row_idx, 1] = regime.get("realized_vol", 0.0)
        tensor[row_idx, 2] = regime.get("iv_level", 0.0)
        tensor[row_idx, 3] = regime.get("iv_skew", 0.0)
        tensor[row_idx, 4] = regime.get("term_slope", 0.0)

        # Surface health
        tensor[row_idx, 5] = regime.get("n_clean", 0) / 500.0   # normalize
        tensor[row_idx, 6] = regime.get("n_violations", 0) / 50.0
        tensor[row_idx, 7] = regime.get("avg_viol_pct", 0.0)
        tensor[row_idx, 8] = regime.get("max_viol_pct", 0.0)

        # Quad-specific
        qh = quad_by_hour.get(hour, {})
        c11 = qh.get((t1, k1), (0, 0))[0] or 0
        c21 = qh.get((t1, k2), (0, 0))[0] or 0
        c12 = qh.get((t2, k1), (0, 0))[0] or 0
        c22 = qh.get((t2, k2), (0, 0))[0] or 0

        tensor[row_idx, 9]  = c11 / spot
        tensor[row_idx, 10] = c21 / spot
        tensor[row_idx, 11] = c12 / spot
        tensor[row_idx, 12] = c22 / spot

        # TP2 ratio: RHS/LHS (>1 means violation)
        lhs = c11 * c22
        rhs = c12 * c21
        tensor[row_idx, 13] = (rhs / lhs) if lhs > 0 else 1.0

        # Avg spread of the 4 legs
        spreads = [qh.get((t1, k1), (0, 0))[1] or 0,
                   qh.get((t1, k2), (0, 0))[1] or 0,
                   qh.get((t2, k1), (0, 0))[1] or 0,
                   qh.get((t2, k2), (0, 0))[1] or 0]
        tensor[row_idx, 14] = sum(spreads) / 4.0

        # Cyclical time encoding
        hod = regime.get("hour_of_day", 0)
        dow = regime.get("day_of_week", 0)
        tensor[row_idx, 15] = math.sin(2 * math.pi * hod / 24)
        tensor[row_idx, 16] = math.cos(2 * math.pi * hod / 24)
        tensor[row_idx, 17] = math.sin(2 * math.pi * dow / 7)
        tensor[row_idx, 18] = math.cos(2 * math.pi * dow / 7)

    return tensor


# ── Surface feature extraction ─────────────────────────────────

def build_surface_snapshot(
    conn: sqlite3.Connection,
    violation: dict,
    max_instruments: int = DEFAULT_MAX_SURFACE,
) -> tuple[np.ndarray, int]:
    """
    Build the surface feature tensor for one violation.

    Returns:
        tensor: np.ndarray of shape (max_instruments, D_SURFACE)
                Zero-padded if fewer instruments available.
        n_actual: number of real instruments (before padding)
    """
    hour = violation["snapshot_hour"]
    spot = violation["btc_spot"] or 1.0
    t1, t2 = violation["T1"], violation["T2"]
    k1, k2 = violation["K1"], violation["K2"]

    # Violation leg instrument identifiers
    violation_legs = {(t1, k1), (t1, k2), (t2, k1), (t2, k2)}

    rows = conn.execute("""
        SELECT expiry, strike, mid_price, mark_iv, spread_pct,
               moneyness, time_to_expiry_d, open_interest, volume_24h
        FROM snapshots
        WHERE snapshot_hour = ?
          AND is_clean = 1
          AND option_type = 'C'
          AND mid_price > 0
        ORDER BY expiry, strike
    """, (hour,)).fetchall()

    n_actual = min(len(rows), max_instruments)
    tensor = np.zeros((max_instruments, D_SURFACE), dtype=np.float32)

    for i, r in enumerate(rows[:max_instruments]):
        tensor[i, 0] = r["moneyness"] or 0.0
        tensor[i, 1] = math.log1p(max(r["time_to_expiry_d"] or 0, 0))  # log(1+TTM)
        tensor[i, 2] = (r["mark_iv"] or 0.0) / 100.0                   # scale to [0, ~5]
        tensor[i, 3] = r["spread_pct"] or 0.0
        tensor[i, 4] = (r["mid_price"] or 0.0) / spot                  # normalize by spot
        tensor[i, 5] = math.log1p(r["open_interest"] or 0)
        tensor[i, 6] = math.log1p(r["volume_24h"] or 0)
        tensor[i, 7] = 1.0 if (r["expiry"], r["strike"]) in violation_legs else 0.0

    return tensor, n_actual


# ── Full dataset construction ──────────────────────────────────

def build_dataset(
    conn: sqlite3.Connection,
    seq_len: int = DEFAULT_SEQ_LEN,
    max_instruments: int = DEFAULT_MAX_SURFACE,
) -> dict:
    """
    Build the complete dataset from the DB.

    Returns dict with numpy arrays ready for TP2Dataset:
        temporal:     (N, seq_len, D_TEMPORAL)
        surface:      (N, max_instruments, D_SURFACE)
        surface_mask: (N, max_instruments) — 1 for real instruments, 0 for padding
        labels:       (N,)
        metadata:     list of dicts with violation info for analysis
    """
    violations = get_labeled_violations(conn)
    if not violations:
        log.warning("No labeled violations found — run more reversion checks")
        return None

    N = len(violations)
    log.info(f"Building dataset: {N} samples, seq_len={seq_len}, max_instr={max_instruments}")

    # Pre-compute caches
    regime_cache = _build_hourly_regime_cache(conn)
    all_hours = _get_snapshot_hours(conn)

    temporal     = np.zeros((N, seq_len, D_TEMPORAL), dtype=np.float32)
    surface      = np.zeros((N, max_instruments, D_SURFACE), dtype=np.float32)
    surface_mask = np.zeros((N, max_instruments), dtype=np.float32)
    labels       = np.zeros(N, dtype=np.int64)

    for i, v in enumerate(violations):
        if i % 100 == 0:
            log.info(f"  Processing sample {i}/{N}...")

        temporal[i] = build_temporal_sequence(
            conn, v, regime_cache, all_hours, seq_len
        )
        surf, n_actual = build_surface_snapshot(conn, v, max_instruments)
        surface[i] = surf
        surface_mask[i, :n_actual] = 1.0
        labels[i] = v["label"]

    pos_rate = labels.sum() / max(len(labels), 1) * 100
    log.info(
        f"Dataset built: {N} samples | "
        f"temporal {temporal.shape} | surface {surface.shape} | "
        f"positive rate {pos_rate:.2f}%"
    )

    return {
        "temporal":     temporal,
        "surface":      surface,
        "surface_mask": surface_mask,
        "labels":       labels,
        "metadata":     violations,   # for analysis / split logic
    }
