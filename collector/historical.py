"""
collector/historical.py — Historical data collection strategy.

Deribit public API reality check:
  ✓ get_book_summary_by_currency   → full surface NOW (works great)
  ✓ get_last_trades_by_currency    → last ~1000 trades (recent only)
  ✗ get_last_trades_by_currency_and_time → 400 for dates > ~7 days without auth

Strategy for 30-day historical data:
  1. Recent trades  (last 24h):  get_last_trades_by_currency        ← FREE
  2. Past 7 days:                get_last_trades_by_currency_and_time ← FREE (with fixes)
  3. 7-30 days ago:              NOT available on public API
                                 → Use Tardis.dev free tier (1st of each month)
                                 → Or collect going forward hourly

For Phase I presentation: option 1+2 is enough.
For Phase III ML: need Tardis or forward-collected data.
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
    get_trades_page, get_recent_trades,
    get_surface_snapshot, get_index_price
)

log = logging.getLogger("historical")


def _parse_instrument(name: str) -> tuple:
    """BTC-27MAR26-98000-C → ('27MAR26', 98000.0, 'C')"""
    try:
        parts = name.split("-")
        return parts[1], float(parts[2]), parts[3]
    except Exception:
        return None, None, None


def fetch_full_trade_history(db_conn, days_back: int = cfg.HISTORY_DAYS):
    """
    Fetch BTC option trade history.
    
    Public API limits mean we split into two strategies:
      - Last 7 days: use timed endpoint (with fixed params)
      - Older: log a warning and recommend Tardis
    """
    now_utc   = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=days_back)
    cutoff    = now_utc - timedelta(days=7)  # public API safe window

    log.info(f"Historical fetch: {start_utc.date()} → {now_utc.date()} ({days_back} days)")
    log.info(f"Public API safe window: last 7 days ({cutoff.date()} → {now_utc.date()})")

    if days_back > 7:
        log.warning(
            f"Requested {days_back} days but Deribit public API only provides ~7 days "
            f"of trade history without authentication. "
            f"Fetching last 7 days only. "
            f"For full {days_back}-day history, use Tardis.dev (see README)."
        )

    total_saved = 0

    # ── Strategy A: Timed endpoint for last 7 days ─────────────
    fetch_start = max(start_utc, cutoff)
    chunk_start = fetch_start

    while chunk_start < now_utc:
        chunk_end = min(chunk_start + timedelta(days=cfg.TRADES_CHUNK_DAYS), now_utc)
        start_ms  = int(chunk_start.timestamp() * 1000)
        end_ms    = int(chunk_end.timestamp()   * 1000)

        log.info(f"  Chunk: {chunk_start.strftime('%Y-%m-%d %H:%M')} → {chunk_end.strftime('%Y-%m-%d %H:%M')}")
        chunk_saved = _fetch_timed_chunk(db_conn, start_ms, end_ms)
        total_saved += chunk_saved
        log.info(f"  Saved: {chunk_saved} trades | Total: {total_saved}")

        chunk_start = chunk_end
        time.sleep(0.3)

    # ── Strategy B: Bulk recent trades (catches any gaps) ──────
    log.info("Fetching latest 1000 trades as safety net...")
    recent = get_recent_trades(count=1000)
    if recent:
        rows = _build_trade_rows(recent)
        _insert_trades(db_conn, rows)
        total_saved += len(rows)
        log.info(f"  Recent trades saved: {len(rows)}")

    log.info(f"Historical fetch complete. Total: {total_saved} trades")
    
    if days_back > 7:
        log.info(
            "\n" + "="*60 +
            "\n  For full 30-90 day history, use Tardis.dev:"
            "\n  1. Sign up at https://tardis.dev (free tier available)"
            "\n  2. Add TARDIS_API_KEY to your .env file"
            "\n  3. Run: python main.py --mode tardis --days 30"
            "\n" + "="*60
        )

    return total_saved


def _fetch_timed_chunk(db_conn, start_ms: int, end_ms: int) -> int:
    """Paginate through one time window using the timed trades endpoint."""
    saved     = 0
    batch_end = end_ms
    page      = 0

    while True:
        result = get_trades_page(start_ms, batch_end)
        trades = result.get("trades", [])

        if not trades:
            break

        page += 1
        rows = _build_trade_rows(trades)
        _insert_trades(db_conn, rows)
        saved += len(rows)

        if not result.get("has_more", False):
            break

        oldest_ts = min(t["timestamp"] for t in trades)
        if oldest_ts <= start_ms:
            break
        batch_end = oldest_ts - 1
        time.sleep(0.15)

    return saved


def _build_trade_rows(trades: list) -> list:
    """Convert raw Deribit trade dicts to DB row tuples."""
    rows = []
    for t in trades:
        name = t.get("instrument_name", "")
        expiry, strike, opt_type = _parse_instrument(name)
        if expiry is None:
            continue
        rows.append((
            t.get("timestamp"),
            name, expiry, strike, opt_type,
            t.get("price"),
            t.get("amount"),
            t.get("iv"),
            t.get("mark_price"),
            t.get("index_price"),
            t.get("direction"),
            t.get("trade_id"),
        ))
    return rows


def fetch_and_save_snapshot(db_conn) -> int:
    """
    Fetch current surface snapshot and save to DB.
    Called every hour by the live scheduler.
    This is the PRIMARY data collection method.
    """
    raw  = get_surface_snapshot()
    if not raw:
        log.warning("Empty snapshot returned")
        return 0

    spot  = get_index_price()
    now = datetime.now(timezone.utc)
    interval = cfg.SNAPSHOT_INTERVAL_MINUTES
    floored_min = (now.minute // interval) * interval
    now_h = now.replace(minute=floored_min, second=0, microsecond=0)

    rows = []
    for item in raw:
        name = item.get("instrument_name", "")
        expiry, strike, opt_type = _parse_instrument(name)
        if expiry is None:
            continue
        bid = item.get("bid_price")
        ask = item.get("ask_price")
        mid = ((bid or 0) + (ask or 0)) / 2 if bid and ask else None

        rows.append((
            now_h.isoformat(),
            name, expiry, strike, opt_type,
            bid, ask, mid,
            item.get("mark_price"),
            item.get("mark_iv"),
            item.get("underlying_price") or spot,
            item.get("open_interest"),
            item.get("volume"),
            item.get("quote_currency"),
        ))

    _insert_snapshots(db_conn, rows)
    log.info(f"Snapshot saved: {len(rows)} instruments @ {now_h.isoformat()}")
    return len(rows)


# ── DB insert helpers ──────────────────────────────────────────

def _insert_trades(conn, rows: list):
    if not rows:
        return
    conn.executemany("""
        INSERT OR IGNORE INTO trades
          (timestamp_ms, instrument_name, expiry, strike, option_type,
           price, amount, iv, mark_price, index_price, direction, trade_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()


def _insert_snapshots(conn, rows: list):
    if not rows:
        return
    conn.executemany("""
        INSERT OR REPLACE INTO snapshots
          (snapshot_hour, instrument_name, expiry, strike, option_type,
           bid_price, ask_price, mid_price, mark_price, mark_iv,
           underlying_price, open_interest, volume_24h, quote_currency)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
