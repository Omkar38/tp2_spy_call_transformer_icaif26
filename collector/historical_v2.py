"""
collector/historical_v2.py — V2 snapshot collection with all fields.

Changes from V1:
  - Multi-currency: loops over DERIBIT_CURRENCIES (BTC, ETH, SOL)
  - Collects BOTH calls and puts (no CALLS_ONLY filter at collection time)
  - Captures new fields: interest_rate, last, volume_usd, high, low, price_change
  - Saves to V2 DB schema (tp2_crypto_v2.db)
  - After TP2 detection, fetches Greeks for violation-leg instruments

V1 historical.py is untouched — V1 DB stays as-is.
"""

import time
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from collector.deribit import (
    get_surface_snapshot_multi, get_index_price,
    get_greeks_batch,
)

log = logging.getLogger("historical_v2")


def _parse_instrument(name: str) -> tuple:
    """BTC-27MAR26-98000-C → ('BTC', '27MAR26', 98000.0, 'C')"""
    try:
        parts = name.split("-")
        return parts[0], parts[1], float(parts[2]), parts[3]
    except Exception:
        return None, None, None, None


def fetch_and_save_snapshot_v2(db_conn) -> dict:
    """
    Fetch surface snapshots for ALL currencies and save to V2 DB.
    Returns: {currency: n_rows_saved}
    """
    all_surfaces = get_surface_snapshot_multi()

    now = datetime.now(timezone.utc)
    interval = cfg.SNAPSHOT_INTERVAL_MINUTES
    floored_min = (now.minute // interval) * interval
    now_h = now.replace(minute=floored_min, second=0, microsecond=0)
    hour_str = now_h.isoformat()

    results = {}

    for currency, raw in all_surfaces.items():
        if not raw:
            log.warning(f"Empty snapshot for {currency}")
            results[currency] = 0
            continue

        spot = get_index_price(currency)

        rows = []
        for item in raw:
            name = item.get("instrument_name", "")
            curr, expiry, strike, opt_type = _parse_instrument(name)
            if expiry is None:
                continue

            bid = item.get("bid_price")
            ask = item.get("ask_price")
            mid = ((bid or 0) + (ask or 0)) / 2 if bid and ask else None

            rows.append((
                hour_str,
                name,
                currency,
                expiry,
                strike,
                opt_type,
                bid,
                ask,
                mid,
                item.get("mark_price"),
                item.get("mark_iv"),
                item.get("underlying_price") or spot,
                item.get("open_interest"),
                item.get("volume"),
                item.get("quote_currency"),
                # V2 new fields
                item.get("interest_rate"),
                item.get("last"),
                item.get("volume_usd"),
                item.get("high"),
                item.get("low"),
                item.get("price_change"),
            ))

        _insert_snapshots_v2(db_conn, rows)
        results[currency] = len(rows)
        log.info(f"Snapshot saved [{currency}]: {len(rows)} instruments @ {hour_str}")

    total = sum(results.values())
    log.info(f"V2 snapshot total: {total} instruments across {len(results)} currencies")
    return results


def fetch_and_save_greeks(db_conn, instrument_names: list, currency: str = "BTC") -> int:
    """
    Fetch Greeks for specific instruments and save to V2 greeks table.
    Called after TP2 detection for violation-leg instruments.
    """
    if not instrument_names:
        return 0

    now = datetime.now(timezone.utc)
    interval = cfg.SNAPSHOT_INTERVAL_MINUTES
    floored_min = (now.minute // interval) * interval
    hour_str = now.replace(minute=floored_min, second=0, microsecond=0).isoformat()

    greeks_data = get_greeks_batch(instrument_names)

    rows = []
    for name, g in greeks_data.items():
        rows.append((
            hour_str,
            name,
            currency,
            g.get("delta"),
            g.get("gamma"),
            g.get("vega"),
            g.get("theta"),
            g.get("rho"),
            g.get("underlying_price"),
            g.get("mark_price"),
        ))

    if rows:
        db_conn.executemany("""
            INSERT OR REPLACE INTO greeks (
                snapshot_hour, instrument_name, currency,
                delta, gamma, vega, theta, rho,
                underlying_price, mark_price
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """, rows)
        db_conn.commit()
        log.info(f"Greeks saved: {len(rows)} instruments @ {hour_str}")

    return len(rows)


# ── DB insert helpers ──────────────────────────────────────────

def _insert_snapshots_v2(conn, rows: list):
    if not rows:
        return
    conn.executemany("""
        INSERT OR REPLACE INTO snapshots (
            snapshot_hour, instrument_name, currency,
            expiry, strike, option_type,
            bid_price, ask_price, mid_price, mark_price, mark_iv,
            underlying_price, open_interest, volume_24h, quote_currency,
            interest_rate, last_price, volume_usd, high_24h, low_24h, price_change_24h
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
