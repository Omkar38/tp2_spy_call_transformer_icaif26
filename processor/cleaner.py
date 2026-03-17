"""
processor/cleaner.py — Data cleaning pipeline.

Takes raw snapshots from DB and applies all filters.
Fills the derived columns (spread, moneyness, etc.)
Marks rows is_clean=1 when they pass all checks.

Every filter has an explicit reason tied back to the project.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

log = logging.getLogger("cleaner")

# Expiry string → approximate days to expiry (for time_to_expiry filter)
# We parse this from the expiry string directly
import re

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}


def parse_expiry_date(expiry_str: str) -> datetime | None:
    """
    '27MAR26' → datetime(2026, 3, 27, 8, 0, tzinfo=UTC)
    Deribit options expire at 08:00 UTC on expiry date.
    """
    try:
        m = re.match(r"(\d{1,2})([A-Z]{3})(\d{2})", expiry_str)
        if not m:
            return None
        day, mon, yr = int(m.group(1)), MONTH_MAP[m.group(2)], 2000 + int(m.group(3))
        return datetime(yr, mon, day, 8, 0, tzinfo=timezone.utc)
    except Exception:
        return None


def run_cleaning_pass(conn: sqlite3.Connection, snapshot_hour: str = None) -> dict:
    """
    Run the full cleaning pipeline on snapshots.
    If snapshot_hour given, only clean that hour's data.
    Returns dict of counts at each filter step.

    Filter order matters — we go from cheapest to most expensive check.
    """
    where = f"WHERE snapshot_hour = '{snapshot_hour}'" if snapshot_hour else "WHERE is_clean = 0"

    # Load unprocessed rows
    rows = conn.execute(f"""
        SELECT id, instrument_name, expiry, strike, option_type,
               bid_price, ask_price, mark_price, mark_iv,
               underlying_price, open_interest, snapshot_hour
        FROM snapshots {where}
    """).fetchall()

    if not rows:
        log.info("No new snapshots to clean")
        return {}

    counts = {"total": len(rows)}
    log.info(f"Cleaning {len(rows)} snapshot rows...")

    kept_ids   = []
    update_rows = []   # (spread, spread_pct, moneyness, tte, is_clean, id)

    for row in rows:
        bid   = row["bid_price"]
        ask   = row["ask_price"]
        mid   = ((bid or 0) + (ask or 0)) / 2 if bid and ask else None
        spot  = row["underlying_price"]
        strike = row["strike"]
        opt   = row["option_type"]
        expiry = row["expiry"]
        oi    = row["open_interest"]
        iv    = row["mark_iv"]

        # ── Filter 1: calls only (for now) ──────────────────────
        if cfg.CALLS_ONLY and opt != "C":
            continue

        # ── Filter 2: valid bid ──────────────────────────────────
        if bid is None or bid <= cfg.MIN_BID:
            continue

        # ── Filter 3: valid ask > bid ────────────────────────────
        if ask is None or ask <= bid:
            continue

        # ── Filter 4: mid is computable ──────────────────────────
        if mid is None or mid <= 0:
            continue

        # ── Filter 5: open interest ──────────────────────────────
        if oi is None or oi < cfg.MIN_OPEN_INTEREST:
            continue

        # ── Filter 6: spread ratio ───────────────────────────────
        spread     = ask - bid
        spread_pct = spread / mid
        if spread_pct > cfg.MAX_SPREAD_PCT:
            continue

        # ── Filter 7: moneyness range ────────────────────────────
        if spot is None or spot <= 0:
            continue
        moneyness = strike / spot
        if moneyness < cfg.MIN_MONEYNESS or moneyness > cfg.MAX_MONEYNESS:
            continue

        # ── Filter 8: time to expiry ─────────────────────────────
        expiry_dt = parse_expiry_date(expiry)
        snap_dt   = datetime.fromisoformat(row["snapshot_hour"])
        if expiry_dt and snap_dt:
            tte_days = (expiry_dt - snap_dt).total_seconds() / 86400
            if tte_days < cfg.MIN_DAYS_TO_EXPIRY:
                continue
        else:
            tte_days = None

        # ── Filter 9: IV sanity ───────────────────────────────────
        if iv is not None and (iv < cfg.MIN_IV_PCT or iv > cfg.MAX_IV_PCT):
            continue

        # ── Passed all filters ───────────────────────────────────
        kept_ids.append(row["id"])
        update_rows.append((
            round(spread, 8),
            round(spread_pct, 6),
            round(moneyness, 6),
            round(tte_days, 4) if tte_days is not None else None,
            1,                  # is_clean
            row["id"]
        ))

    # Batch update
    if update_rows:
        conn.executemany("""
            UPDATE snapshots
            SET spread=?, spread_pct=?, moneyness=?, time_to_expiry_d=?, is_clean=?
            WHERE id=?
        """, update_rows)
        conn.commit()

    counts["kept"]    = len(kept_ids)
    counts["removed"] = counts["total"] - counts["kept"]
    counts["pct_kept"] = round(counts["kept"] / counts["total"] * 100, 1) if counts["total"] else 0

    log.info(f"  Clean: {counts['kept']}/{counts['total']} ({counts['pct_kept']}%) passed filters")
    return counts
