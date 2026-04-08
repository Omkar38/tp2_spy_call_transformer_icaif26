"""
collector/spx_spot.py — SPX (^GSPC) daily closing price loader.

Databento ohlcv-1d provides option prices but NOT the underlying index level.
This module fetches ^GSPC daily closes from yfinance and backfills the
`underlying_price` column in the equity snapshots table.

Must run BEFORE the cleaner — the cleaner needs spot to compute moneyness.

Usage:
    python -m collector.spx_spot            # auto-detect date range from DB
    python -m collector.spx_spot --dry-run  # just print what would be updated

Dependencies:
    pip install yfinance
"""

import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

log = logging.getLogger("spx_spot")


def fetch_spx_prices(start: str, end: str) -> dict:
    """
    Fetch ^GSPC daily closing prices from yfinance.

    Args:
        start: "YYYY-MM-DD" (inclusive)
        end:   "YYYY-MM-DD" (exclusive — yfinance convention)

    Returns:
        dict mapping "YYYY-MM-DD" → float closing price
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    log.info(f"Fetching ^GSPC from yfinance: {start} → {end}")
    ticker = yf.Ticker("^GSPC")
    hist = ticker.history(start=start, end=end, auto_adjust=True)

    prices = {}
    for dt, row in hist.iterrows():
        date_str = dt.strftime("%Y-%m-%d")
        prices[date_str] = round(float(row["Close"]), 2)

    log.info(f"Fetched {len(prices)} SPX closing prices")
    return prices


def update_underlying_prices(conn: sqlite3.Connection, prices: dict) -> int:
    """
    Backfill underlying_price in snapshots table for all rows on each date.

    Equity snapshots use snapshot_hour = "YYYY-MM-DDT21:00:00+00:00" (≈16:00 ET).
    We match on the date prefix.

    Returns: number of rows updated.
    """
    total_updated = 0

    for date_str, price in sorted(prices.items()):
        # Match any snapshot_hour starting with this date
        result = conn.execute("""
            UPDATE snapshots
            SET underlying_price = ?
            WHERE snapshot_hour LIKE ?
              AND underlying_price IS NULL
        """, (price, date_str + "T%"))

        n = result.rowcount
        if n > 0:
            log.debug(f"  {date_str}: {n} rows → SPX={price}")
        total_updated += n

    conn.commit()
    log.info(f"Backfilled underlying_price for {total_updated} snapshot rows")
    return total_updated


def run(db_path: str = None, dry_run: bool = False) -> int:
    """
    End-to-end: find date range in DB, fetch prices, update rows.

    Returns: number of rows updated (0 on dry_run).
    """
    if db_path is None:
        db_path = str(cfg.DB_PATH)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Find the date range that needs updating
    row = conn.execute("""
        SELECT MIN(snapshot_hour) as min_h, MAX(snapshot_hour) as max_h
        FROM snapshots
        WHERE underlying_price IS NULL
    """).fetchone()

    if row["min_h"] is None:
        log.info("All rows already have underlying_price — nothing to do")
        conn.close()
        return 0

    start = row["min_h"][:10]                                          # "YYYY-MM-DD"
    max_dt = datetime.fromisoformat(row["max_h"]).date()
    end = (max_dt + timedelta(days=2)).strftime("%Y-%m-%d")            # +2 for yfinance exclusivity

    log.info(f"DB needs prices from {start} to {end}")

    prices = fetch_spx_prices(start, end)

    if dry_run:
        log.info(f"[DRY RUN] Would update {len(prices)} dates, sample: {list(prices.items())[:3]}")
        conn.close()
        return 0

    n = update_underlying_prices(conn, prices)
    conn.close()
    return n


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format=cfg.LOG_FORMAT)

    parser = argparse.ArgumentParser(description="Backfill SPX spot prices into equity DB")
    parser.add_argument("--db", default=None, help="Path to equity DB (default: cfg.DB_PATH)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without writing")
    args = parser.parse_args()

    n = run(db_path=args.db, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"Updated {n} rows with SPX underlying prices")
