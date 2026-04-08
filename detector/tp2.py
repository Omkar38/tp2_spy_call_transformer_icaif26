"""
detector/tp2.py — Core TP2 violation detection engine.

For each hourly snapshot:
1. Load clean options from DB
2. For all valid (K1<K2, T1<T2) pairs (calendar-ordered)
3. Check C(K1,T1)*C(K2,T2) >= C(K1,T2)*C(K2,T1)
4. Save violations with full feature set to DB

Key fix baked in: T1 must be NEARER than T2 in calendar time.
This was the bug that caused 302/308 fake violations before.
"""

import logging
import sqlite3
from itertools import combinations
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from processor.features import compute_regime_features

log = logging.getLogger("tp2")


def detect_violations(conn: sqlite3.Connection, snapshot_hour: str, currency: str = None) -> dict:
    """
    Run TP2 detection for one snapshot hour.
    Optional currency filter for V2 multi-currency support.
    Returns summary dict with counts.
    """
    # Load clean calls for this hour, grouped by expiry
    # V2: optionally filter by currency
    if currency:
        rows = conn.execute("""
            SELECT expiry, strike, mid_price, spread_pct,
                   moneyness, mark_iv, underlying_price
            FROM snapshots
            WHERE snapshot_hour = ?
              AND currency = ?
              AND is_clean = 1
              AND option_type = 'C'
              AND mid_price > 0
            ORDER BY expiry, strike
        """, (snapshot_hour, currency)).fetchall()
    else:
        rows = conn.execute("""
            SELECT expiry, strike, mid_price, spread_pct,
                   moneyness, mark_iv, underlying_price
            FROM snapshots
            WHERE snapshot_hour = ?
              AND is_clean = 1
              AND option_type = 'C'
              AND mid_price > 0
            ORDER BY expiry, strike
        """, (snapshot_hour,)).fetchall()

    if not rows:
        log.warning(f"No clean data for {snapshot_hour}")
        return {"snapshot_hour": snapshot_hour, "pairs_checked": 0, "violations": 0}

    # Build surface dict: expiry → {strike → row}
    surface = {}
    for row in rows:
        exp = row["expiry"]
        if exp not in surface:
            surface[exp] = {}
        surface[exp][row["strike"]] = row

    # Get spot price for this hour
    spot = rows[0]["underlying_price"] if rows else None

    # Compute regime features once for this snapshot
    regime = compute_regime_features(conn, snapshot_hour)

    # Sort expiries by calendar order — dynamic parsing, works for any date range
    expiries = sorted(
        [e for e in surface.keys() if cfg.parse_expiry_to_date(e) is not None],
        key=cfg.expiry_sort_key
    )

    snap_dt      = datetime.fromisoformat(snapshot_hour)
    total_pairs  = 0
    violations   = []

    for t1, t2 in combinations(expiries, 2):
        # ── CRITICAL: enforce T1 is nearer than T2 ─────────────
        # Uses dynamic date parsing — works for any historical period
        t1_date = cfg.parse_expiry_to_date(t1)
        t2_date = cfg.parse_expiry_to_date(t2)
        if t1_date is None or t2_date is None or t1_date >= t2_date:
            continue

        e1 = surface[t1]
        e2 = surface[t2]
        common_strikes = sorted(set(e1.keys()) & set(e2.keys()))

        if len(common_strikes) < 2:
            continue

        mat_gap = (t2_date - t1_date).days

        for i in range(len(common_strikes) - 1):
            k1, k2 = common_strikes[i], common_strikes[i + 1]

            r11 = e1[k1]; r21 = e1[k2]
            r12 = e2[k1]; r22 = e2[k2]

            c11 = r11["mid_price"]   # C(K1, T1)
            c21 = r21["mid_price"]   # C(K2, T1)
            c12 = r12["mid_price"]   # C(K1, T2)
            c22 = r22["mid_price"]   # C(K2, T2)

            if any(v is None or v <= 0 for v in [c11, c21, c12, c22]):
                continue

            lhs = c11 * c22
            rhs = c12 * c21
            total_pairs += 1

            if rhs <= lhs:
                continue   # TP2 holds for this quad

            viol_pct = (rhs - lhs) / lhs * 100

            if viol_pct < cfg.MIN_VIOLATION_PCT:
                continue   # numerical noise

            # ── Spread cost estimate ─────────────────────────────
            spreads = [
                r11["spread_pct"], r21["spread_pct"],
                r12["spread_pct"], r22["spread_pct"]
            ]
            valid_spreads   = [s for s in spreads if s is not None]
            avg_spread_pct  = sum(valid_spreads) / len(valid_spreads) * 100 if valid_spreads else None
            total_cost_pct  = avg_spread_pct * 4 if avg_spread_pct else None
            net_edge        = viol_pct - total_cost_pct if total_cost_pct else None

            # ── Phantom violation gates ──────────────────────────
            # Gate 1: any single leg spread > 20% → untradeable phantom
            if valid_spreads and max(valid_spreads) > 0.20:
                continue

            # Gate 2: any leg mid < MIN_MID_PRICE → near-zero, illiquid
            mids = [c11, c21, c12, c22]
            if min(mids) < cfg.MIN_MID_PRICE:
                continue

            # Gate 3: executable profit must be positive
            # Long: buy K1T1 at ask, buy K2T2 at ask; sell K1T2 at bid, sell K2T1 at bid
            ask_k1t1 = c11 * (1 + (r11["spread_pct"] or 0) / 2)
            ask_k2t2 = c22 * (1 + (r22["spread_pct"] or 0) / 2)
            bid_k1t2 = c12 * (1 - (r12["spread_pct"] or 0) / 2)
            bid_k2t1 = c21 * (1 - (r21["spread_pct"] or 0) / 2)
            executable = (bid_k1t2 * bid_k2t1) - (ask_k1t1 * ask_k2t2)
            if executable <= 0:
                continue

            # ── Profitability label ──────────────────────────────
            if net_edge is not None:
                if net_edge > 0:
                    label = "TRADEABLE"
                elif net_edge > -5:
                    label = "marginal"
                else:
                    label = "noise_likely"
            else:
                label = "unclassified"

            violations.append((
                snapshot_hour,
                currency or "BTC",
                t1, t2, k1, k2,
                round(c11, 8), round(c21, 8),
                round(c12, 8), round(c22, 8),
                round(lhs, 10), round(rhs, 10),
                round(viol_pct, 4),
                mat_gap,
                round(k2 - k1, 2),
                round((k2 - k1) / spot, 6) if spot else None,
                round(k1 / spot, 6) if spot else None,
                round(k2 / spot, 6) if spot else None,
                round(avg_spread_pct, 4) if avg_spread_pct else None,
                round(total_cost_pct, 4) if total_cost_pct else None,
                round(net_edge, 4) if net_edge else None,
                spot,
                regime.get("btc_return_1h"),
                regime.get("btc_return_24h"),
                regime.get("realized_vol_24h"),
                regime.get("iv_level_surface"),
                regime.get("iv_skew"),
                regime.get("term_structure_slope"),
                regime.get("hour_of_day"),
                regime.get("day_of_week"),
                label,
            ))

    # Save violations
    # Detect V2 schema (has currency column) vs V1
    has_currency_col = False
    try:
        col_info = conn.execute("PRAGMA table_info(violations)").fetchall()
        has_currency_col = any(c["name"] == "currency" for c in col_info)
    except Exception:
        pass

    if violations:
        if has_currency_col:
            conn.executemany("""
                INSERT OR REPLACE INTO violations (
                    snapshot_hour, currency, T1, T2, K1, K2,
                    C_K1_T1, C_K2_T1, C_K1_T2, C_K2_T2,
                    LHS, RHS, violation_pct,
                    maturity_gap, strike_gap, strike_gap_pct,
                    moneyness_K1, moneyness_K2,
                    avg_spread_pct, est_total_cost_pct, net_edge,
                    btc_spot,
                    btc_return_1h, btc_return_24h, realized_vol_24h,
                    iv_level_surface, iv_skew, term_structure_slope,
                    hour_of_day, day_of_week,
                    profitability
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, violations)
        else:
            # V1 DB — strip the currency field (index 1) from each row
            v1_rows = [row[:1] + row[2:] for row in violations]
            conn.executemany("""
                INSERT OR REPLACE INTO violations (
                    snapshot_hour, T1, T2, K1, K2,
                    C_K1_T1, C_K2_T1, C_K1_T2, C_K2_T2,
                    LHS, RHS, violation_pct,
                    maturity_gap, strike_gap, strike_gap_pct,
                    moneyness_K1, moneyness_K2,
                    avg_spread_pct, est_total_cost_pct, net_edge,
                    btc_spot,
                    btc_return_1h, btc_return_24h, realized_vol_24h,
                    iv_level_surface, iv_skew, term_structure_slope,
                    hour_of_day, day_of_week,
                    profitability
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, v1_rows)
        conn.commit()

    viol_rate = len(violations) / total_pairs * 100 if total_pairs > 0 else 0

    log.info(
        f"TP2 [{snapshot_hour}]: "
        f"{total_pairs} pairs checked | "
        f"{len(violations)} violations ({viol_rate:.2f}%) | "
        f"tradeable: {sum(1 for v in violations if v[-1]=='TRADEABLE')}"
    )

    return {
        "snapshot_hour": snapshot_hour,
        "pairs_checked": total_pairs,
        "violations":    len(violations),
        "violation_rate": round(viol_rate, 4),
        "tradeable":     sum(1 for v in violations if v[-1] == "TRADEABLE"),
    }


def check_reversions(conn: sqlite3.Connection, current_hour: str):
    """
    For all open violations, check if they've reverted in the current snapshot.
    A violation is reverted when the same (T1,T2,K1,K2) pair no longer violates TP2.

    This is the core of Phase II — building the reversion labels.
    """
    # Get all violations not yet fully labeled
    open_viols = conn.execute("""
        SELECT v.id, v.snapshot_hour, v.T1, v.T2, v.K1, v.K2, v.violation_pct,
               v.snapshot_hour as detected_at
        FROM violations v
        WHERE NOT EXISTS (
            SELECT 1 FROM reversions r
            WHERE r.violation_id = v.id
              AND r.reverted = 1
        )
        ORDER BY v.snapshot_hour DESC
        LIMIT 500
    """).fetchall()

    if not open_viols:
        return

    # Load current clean surface
    rows = conn.execute("""
        SELECT expiry, strike, mid_price FROM snapshots
        WHERE snapshot_hour = ? AND is_clean = 1 AND option_type = 'C' AND mid_price > 0
    """, (current_hour,)).fetchall()

    if not rows:
        return

    surface_now = {}
    for r in rows:
        surface_now.setdefault(r["expiry"], {})[r["strike"]] = r["mid_price"]

    current_dt = datetime.fromisoformat(current_hour)
    reversion_rows = []

    for v in open_viols:
        t1, t2 = v["T1"], v["T2"]
        k1, k2 = v["K1"], v["K2"]

        # Check if this pair still violates
        try:
            c11 = surface_now.get(t1, {}).get(k1)
            c21 = surface_now.get(t1, {}).get(k2)
            c12 = surface_now.get(t2, {}).get(k1)
            c22 = surface_now.get(t2, {}).get(k2)

            if any(x is None or x <= 0 for x in [c11, c21, c12, c22]):
                current_viol = None   # can't measure — option may have expired
                reverted     = 0
            else:
                lhs = c11 * c22
                rhs = c12 * c21
                if rhs > lhs:
                    current_viol = round((rhs - lhs) / lhs * 100, 4)
                    reverted     = 0
                else:
                    current_viol = 0.0
                    reverted     = 1

        except Exception:
            current_viol = None
            reverted     = 0

        detected_dt   = datetime.fromisoformat(v["detected_at"])
        hours_elapsed = (current_dt - detected_dt).total_seconds() / 3600

        if reverted:
            if hours_elapsed <= cfg.REVERSION_SUPERFAST_H:
                speed = "superfast"
            elif hours_elapsed <= cfg.REVERSION_FAST_H:
                speed = "fast"
            elif hours_elapsed <= cfg.REVERSION_MEDIUM_H:
                speed = "medium"
            else:
                speed = "slow"
        else:
            speed = "not_yet" if hours_elapsed < cfg.REVERSION_NEVER_H else "never"

        reversion_rows.append((
            v["id"], current_hour,
            round(hours_elapsed, 2),
            current_viol, reverted, speed
        ))

    if reversion_rows:
        conn.executemany("""
            INSERT OR REPLACE INTO reversions
              (violation_id, checked_hour, hours_elapsed, current_viol, reverted, revert_speed)
            VALUES (?,?,?,?,?,?)
        """, reversion_rows)
        conn.commit()
        reverted_count = sum(1 for r in reversion_rows if r[4] == 1)
        log.info(f"Reversion check [{current_hour}]: {reverted_count}/{len(reversion_rows)} reverted")
