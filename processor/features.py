"""
processor/features.py — Feature engineering pipeline.

Takes clean snapshots and computes regime features:
- BTC returns (1h, 24h)
- Realized volatility
- IV surface level and skew
- Term structure slope
- Time-of-day / day-of-week

These become the ML features in Phase III.
"""

import logging
import sqlite3
import math
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

log = logging.getLogger("features")


def compute_regime_features(conn: sqlite3.Connection, snapshot_hour: str) -> dict:
    """
    Compute market regime features for a given snapshot hour.
    Returns dict of feature_name → value.
    """
    features = {}
    snap_dt = datetime.fromisoformat(snapshot_hour)

    # ── BTC spot at this hour ─────────────────────────────────
    row = conn.execute("""
        SELECT AVG(underlying_price) as spot
        FROM snapshots
        WHERE snapshot_hour = ? AND is_clean = 1
    """, (snapshot_hour,)).fetchone()
    spot_now = row["spot"] if row else None
    features["btc_spot"] = spot_now

    # ── BTC spot 1 hour ago ───────────────────────────────────
    prev_hours = conn.execute("""
        SELECT snapshot_hour, AVG(underlying_price) as spot
        FROM snapshots
        WHERE snapshot_hour < ? AND is_clean = 1
        GROUP BY snapshot_hour
        ORDER BY snapshot_hour DESC
        LIMIT 24
    """, (snapshot_hour,)).fetchall()

    if len(prev_hours) >= 1 and spot_now:
        spot_1h_ago = prev_hours[0]["spot"]
        if spot_1h_ago and spot_1h_ago > 0:
            features["btc_return_1h"] = round(math.log(spot_now / spot_1h_ago), 6)
        else:
            features["btc_return_1h"] = None
    else:
        features["btc_return_1h"] = None

    # ── BTC return 24h ─────────────────────────────────────────
    if len(prev_hours) >= 24 and spot_now:
        spot_24h_ago = prev_hours[23]["spot"]
        if spot_24h_ago and spot_24h_ago > 0:
            features["btc_return_24h"] = round(math.log(spot_now / spot_24h_ago), 6)
        else:
            features["btc_return_24h"] = None
    else:
        features["btc_return_24h"] = None

    # ── Realized vol (24h) ────────────────────────────────────
    if len(prev_hours) >= 4:
        spots = [r["spot"] for r in prev_hours if r["spot"] and r["spot"] > 0]
        if len(spots) >= 3:
            returns = [math.log(spots[i] / spots[i+1]) for i in range(len(spots)-1)]
            mean_r  = sum(returns) / len(returns)
            var_r   = sum((r - mean_r)**2 for r in returns) / len(returns)
            rvol    = math.sqrt(var_r) * math.sqrt(24) * 100  # annualized-ish, in %
            features["realized_vol_24h"] = round(rvol, 4)
        else:
            features["realized_vol_24h"] = None
    else:
        features["realized_vol_24h"] = None

    # ── IV surface level (average across all clean options) ──
    row = conn.execute("""
        SELECT AVG(mark_iv) as avg_iv, COUNT(*) as n
        FROM snapshots
        WHERE snapshot_hour = ? AND is_clean = 1 AND mark_iv IS NOT NULL
    """, (snapshot_hour,)).fetchone()
    features["iv_level_surface"] = round(row["avg_iv"], 4) if row["avg_iv"] else None

    # ── IV skew (OTM iv - ATM iv, for calls) ─────────────────
    # ATM = moneyness between 0.95 and 1.05
    # OTM = moneyness between 1.10 and 1.30
    row_atm = conn.execute("""
        SELECT AVG(mark_iv) as iv
        FROM snapshots
        WHERE snapshot_hour = ? AND is_clean = 1
          AND option_type = 'C'
          AND moneyness BETWEEN 0.95 AND 1.05
          AND mark_iv IS NOT NULL
    """, (snapshot_hour,)).fetchone()

    row_otm = conn.execute("""
        SELECT AVG(mark_iv) as iv
        FROM snapshots
        WHERE snapshot_hour = ? AND is_clean = 1
          AND option_type = 'C'
          AND moneyness BETWEEN 1.10 AND 1.30
          AND mark_iv IS NOT NULL
    """, (snapshot_hour,)).fetchone()

    if row_atm["iv"] and row_otm["iv"]:
        features["iv_skew"] = round(row_otm["iv"] - row_atm["iv"], 4)
    else:
        features["iv_skew"] = None

    # ── Term structure slope (far IV - near IV) ───────────────
    # Near = expiries 1-6, Far = expiries 9+
    near_expiries = [k for k, v in cfg.EXPIRY_ORDER.items() if 1 <= v <= 6]
    far_expiries  = [k for k, v in cfg.EXPIRY_ORDER.items() if v >= 9]

    near_ph = ",".join("?" * len(near_expiries))
    far_ph  = ",".join("?" * len(far_expiries))

    if near_expiries and far_expiries:
        near_iv = conn.execute(f"""
            SELECT AVG(mark_iv) as iv FROM snapshots
            WHERE snapshot_hour = ? AND is_clean = 1
              AND expiry IN ({near_ph}) AND mark_iv IS NOT NULL
        """, [snapshot_hour] + near_expiries).fetchone()

        far_iv = conn.execute(f"""
            SELECT AVG(mark_iv) as iv FROM snapshots
            WHERE snapshot_hour = ? AND is_clean = 1
              AND expiry IN ({far_ph}) AND mark_iv IS NOT NULL
        """, [snapshot_hour] + far_expiries).fetchone()

        if near_iv["iv"] and far_iv["iv"]:
            features["term_structure_slope"] = round(far_iv["iv"] - near_iv["iv"], 4)
        else:
            features["term_structure_slope"] = None
    else:
        features["term_structure_slope"] = None

    # ── Time features ─────────────────────────────────────────
    features["hour_of_day"] = snap_dt.hour
    features["day_of_week"] = snap_dt.weekday()   # 0=Mon, 6=Sun

    return features
