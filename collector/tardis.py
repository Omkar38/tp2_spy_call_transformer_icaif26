"""
collector/tardis.py — Historical BTC options data from Tardis.dev

Downloads Deribit options_chain CSV files (one per day) and backfills
the snapshots table with properly timestamped surface snapshots, then
runs the full cleaning + TP2 detection pipeline on each new hour.

Endpoint:
    https://datasets.tardis.dev/v1/deribit/options_chain/{year}/{month}/{day}/OPTIONS.csv.gz

Authentication:
    Authorization: Bearer <TARDIS_API_KEY>

Free tier:
    Without an API key, only the 1st of each month is accessible.
    Add TARDIS_API_KEY to .env for full historical access (~$50-200/month).

Usage:
    python main.py --mode tardis --days 90
"""

import io
import os
import re
import gzip
import logging
import sqlite3
import time
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from processor.cleaner import run_cleaning_pass
from detector.tp2 import detect_violations, check_reversions

log = logging.getLogger("tardis")

TARDIS_BASE = "https://datasets.tardis.dev/v1"
EXCHANGE    = "deribit"
DATA_TYPE   = "options_chain"
SYMBOL      = "OPTIONS"

_MONTH_NUM = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_MONTH_STR = {v: k for k, v in _MONTH_NUM.items()}


# ── API key ────────────────────────────────────────────────────

def _get_api_key() -> Optional[str]:
    """Read TARDIS_API_KEY from environment or .env file."""
    # Try environment first
    key = os.environ.get("TARDIS_API_KEY", "")
    if key and key != "your_tardis_key_here":
        return key

    # Try .env file
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("TARDIS_API_KEY="):
                val = line.split("=", 1)[1].strip()
                if val and val != "your_tardis_key_here":
                    return val
    return None


# ── Download ───────────────────────────────────────────────────

def _stream_day(target_date: date, api_key: Optional[str],
                interval_minutes: int, conn: sqlite3.Connection):
    """
    Stream one day's CSV.gz from Tardis and process in chunks.
    Never loads more than CHUNK_ROWS rows into RAM at once (~50MB peak).
    Inserts snapshot rows directly into DB as each chunk is processed.

    Returns number of snapshot rows inserted, or -1 on error.
    """
    CHUNK_ROWS = 200_000  # ~50MB per chunk in RAM

    url = (
        f"{TARDIS_BASE}/{EXCHANGE}/{DATA_TYPE}/"
        f"{target_date.year}/{target_date.month:02d}/{target_date.day:02d}/{SYMBOL}.csv.gz"
    )
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    log.info(f"Streaming {target_date} ...")

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=180, stream=True)

            if resp.status_code == 401:
                log.error("Tardis: 401 Unauthorized — check TARDIS_API_KEY in .env")
                return -1
            if resp.status_code == 404:
                log.warning(f"Tardis: no data for {target_date} (404)")
                return 0
            if resp.status_code == 403:
                log.warning(
                    f"Tardis: 403 Forbidden for {target_date}. "
                    "Free tier = 1st of month only. Add API key for full access."
                )
                return 0
            resp.raise_for_status()

            # Stream gzip → decompress on the fly → chunked CSV parse
            # Peak RAM: ~100MB regardless of file size (no full-file load)
            gz_stream = gzip.GzipFile(fileobj=resp.raw)
            total_inserted = 0
            rolling: dict = {}   # (snap_hour, symbol) → compact tuple
            current_max_hour = ""

            reader = pd.read_csv(
                gz_stream,
                chunksize=CHUNK_ROWS,
                low_memory=False,
                on_bad_lines="skip",
            )

            for chunk_idx, chunk in enumerate(reader):
                chunk_max = _process_chunk(chunk, interval_minutes, rolling)

                # Flush hours we've definitely finished (older than chunk max)
                if chunk_max and chunk_max > current_max_hour:
                    flushed = _flush_completed(rolling, chunk_max)
                    if flushed:
                        _insert_snapshots(conn, flushed)
                        total_inserted += len(flushed)
                    current_max_hour = chunk_max

                if chunk_idx % 50 == 0:
                    log.info(f"  {target_date}: chunk {chunk_idx} | "
                             f"{total_inserted:,} rows inserted | "
                             f"{len(rolling)} buffered")

            # Flush remainder
            final_rows = _flush_all(rolling)
            if final_rows:
                _insert_snapshots(conn, final_rows)
                total_inserted += len(final_rows)

            log.info(f"  {target_date}: done — {total_inserted:,} snapshot rows inserted")
            return total_inserted

        except Exception as e:
            log.warning(f"  Attempt {attempt + 1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))

    log.error(f"  All retries failed for {target_date}")
    return -1


# ── Field mapping ──────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    try:
        f = float(val)
        return None if f != f else f  # NaN → None
    except (TypeError, ValueError):
        return None


def _fix_mark_iv(iv: Optional[float]) -> Optional[float]:
    """
    Tardis mirrors Deribit's WebSocket ticker where mark_iv is decimal
    (0.65 = 65%). Our schema stores it as percentage. Multiply if < 5.
    BTC IV is rarely below 20%, so anything < 5 is almost certainly decimal.
    """
    if iv is None:
        return None
    return round(iv * 100, 4) if iv < 5.0 else round(iv, 4)


def _process_chunk(chunk: pd.DataFrame, interval_minutes: int,
                   rolling: dict) -> int:
    """
    Process one CSV chunk using vectorized groupby — fast, low memory.

    Strategy:
    - groupby(snap_hour, symbol).last() on the chunk (vectorized, <0.1s)
    - Update rolling dict with results (only stores last seen per key)
    - Flush completed hours (any hour older than the current chunk's max)
      directly to the caller via the returned count

    rolling dict stores compact tuples, not full pandas Series.
    Peak memory: ~100MB for a full day (288 intervals × 1000 instruments).
    """
    # BTC options only — vectorized string filter
    mask = chunk["symbol"].str.startswith("BTC-", na=False)
    chunk = chunk.loc[mask].copy()
    if chunk.empty:
        return 0

    # Parse timestamps vectorized
    chunk["ts"] = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce")
    chunk.dropna(subset=["ts"], inplace=True)
    if chunk.empty:
        return 0

    chunk["snapshot_hour"] = (
        chunk["ts"].dt.floor(f"{interval_minutes}min")
                   .dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    )

    # Detect the volume column name (varies across Tardis schema versions)
    vol_col = "volume_24h" if "volume_24h" in chunk.columns else "volume"

    # Vectorized groupby — get last tick per (snap_hour, symbol)
    # This is O(n log n) but fully vectorized: ~10ms for 200k rows
    keep_cols = [
        "snapshot_hour", "symbol",
        "best_bid_price", "best_ask_price",
        "mark_price", "mark_iv",
        "underlying_price", "open_interest", vol_col,
    ]
    available = [c for c in keep_cols if c in chunk.columns]
    chunk_small = chunk[available].copy()

    chunk_last = (
        chunk_small.sort_values("snapshot_hour")
                   .groupby(["snapshot_hour", "symbol"], sort=False)
                   .last()
    )

    # Update rolling dict with compact tuples (not Series — saves memory)
    for (snap_hour, symbol), row in chunk_last.iterrows():
        rolling[(snap_hour, symbol)] = (
            snap_hour, symbol,
            _safe_float(row.get("best_bid_price")),
            _safe_float(row.get("best_ask_price")),
            _safe_float(row.get("mark_price")),
            _safe_float(row.get("mark_iv")),
            _safe_float(row.get("underlying_price")),
            _safe_float(row.get("open_interest")),
            _safe_float(row.get(vol_col if vol_col in row.index else "volume")),
        )

    # Return the current max snap_hour so the caller can flush old hours
    return chunk_last.index.get_level_values("snapshot_hour").max()


def _flush_completed(rolling: dict, up_to_hour: str) -> list:
    """Emit and remove all rolling entries with snap_hour < up_to_hour."""
    rows = []
    done = [k for k in rolling if k[0] < up_to_hour]
    for k in done:
        r = _compact_to_db_tuple(rolling.pop(k))
        if r:
            rows.append(r)
    return rows


def _flush_all(rolling: dict) -> list:
    """Emit all remaining rolling entries at end of file."""
    rows = []
    for entry in rolling.values():
        r = _compact_to_db_tuple(entry)
        if r:
            rows.append(r)
    rolling.clear()
    return rows


def _compact_to_db_tuple(entry: tuple) -> Optional[tuple]:
    """Convert compact rolling entry → snapshots DB insert tuple."""
    snap_hour, symbol, bid, ask, mark_price, mark_iv, underlying, oi, volume = entry

    parts = symbol.split("-")
    if len(parts) != 4:
        return None
    expiry_str = parts[1]
    try:
        strike   = float(parts[2])
        opt_type = parts[3]
    except (ValueError, IndexError):
        return None

    mid     = ((bid + ask) / 2) if (bid and ask and bid > 0 and ask > 0) else None
    mark_iv = _fix_mark_iv(mark_iv)

    return (
        snap_hour,
        symbol, expiry_str, strike, opt_type,
        bid, ask, mid,
        mark_price, mark_iv,
        underlying, oi, volume,
        "USD",
    )


# ── DB helpers ─────────────────────────────────────────────────

def _insert_snapshots(conn: sqlite3.Connection, rows: list):
    if not rows:
        return
    conn.executemany("""
        INSERT OR IGNORE INTO snapshots
          (snapshot_hour, instrument_name, expiry, strike, option_type,
           bid_price, ask_price, mid_price, mark_price, mark_iv,
           underlying_price, open_interest, volume_24h, quote_currency)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()


def _update_expiry_order(conn: sqlite3.Connection):
    """
    Scan all distinct expiry strings in the DB and patch cfg.EXPIRY_ORDER
    so the TP2 detector can order historical expiries correctly.
    This is needed because config.py only lists current/future expiries.
    """
    rows = conn.execute(
        "SELECT DISTINCT expiry FROM snapshots WHERE expiry IS NOT NULL ORDER BY expiry"
    ).fetchall()
    expiries = [r[0] for r in rows]

    def _to_date(s):
        m = re.match(r"(\d{1,2})([A-Z]{3})(\d{2})", str(s))
        if not m:
            return None
        try:
            return date(2000 + int(m.group(3)), _MONTH_NUM[m.group(2)], int(m.group(1)))
        except Exception:
            return None

    dated = [(e, _to_date(e)) for e in expiries]
    dated = sorted([(e, d) for e, d in dated if d is not None], key=lambda x: x[1])

    for i, (expiry, _) in enumerate(dated, start=1):
        cfg.EXPIRY_ORDER[expiry] = i

    log.info(f"EXPIRY_ORDER patched: {len(cfg.EXPIRY_ORDER)} expiries registered")


# ── Main entry point ───────────────────────────────────────────

def backfill(conn: sqlite3.Connection,
             days: int = 90,
             run_detection: bool = True) -> int:
    """
    Backfill historical snapshots from Tardis.dev.

    Args:
        conn:           SQLite connection
        days:           How many calendar days back to fetch
        run_detection:  Run cleaning + TP2 detection after inserting

    Returns:
        Number of new snapshot hours inserted.

    Free tier note:
        Without an API key only the 1st of each month is available.
        Add TARDIS_API_KEY=<your_key> to .env for full day-by-day access.
    """
    api_key = _get_api_key()
    today   = datetime.now(timezone.utc).date()

    if not api_key:
        log.warning(
            "No TARDIS_API_KEY found — using free tier.\n"
            "Free tier: 1st of each month only.\n"
            "Add TARDIS_API_KEY=<your_key> to .env for full access."
        )

    # Build list of dates to fetch
    dates_to_fetch = []
    for i in range(1, days + 1):
        target = today - timedelta(days=i)
        if not api_key and target.day != 1:
            continue  # free tier: skip non-first-of-month days
        dates_to_fetch.append(target)

    if not dates_to_fetch:
        log.warning("No dates to fetch (free tier with no 1st-of-month days in range?)")
        return 0

    log.info(
        f"Tardis backfill: {len(dates_to_fetch)} days | "
        f"{'paid' if api_key else 'FREE TIER — 1st of month only'}"
    )

    all_new_hours: set = set()

    for target_date in sorted(dates_to_fetch):
        # Query existing snapshot hours before streaming so we can diff after
        before = set(
            r[0] for r in conn.execute(
                "SELECT DISTINCT snapshot_hour FROM snapshots"
            ).fetchall()
        )

        n_inserted = _stream_day(target_date, api_key, cfg.SNAPSHOT_INTERVAL_MINUTES, conn)
        if n_inserted <= 0:
            continue

        after = set(
            r[0] for r in conn.execute(
                "SELECT DISTINCT snapshot_hour FROM snapshots"
            ).fetchall()
        )
        new_hours = after - before
        all_new_hours.update(new_hours)
        log.info(f"  {target_date}: {len(new_hours)} new snapshot hours added")
        time.sleep(0.5)

    if not all_new_hours:
        log.warning("Tardis backfill: no new data inserted")
        return 0

    log.info(f"All inserts done. {len(all_new_hours)} new snapshot hours total.")

    if run_detection:
        _update_expiry_order(conn)

        sorted_hours = sorted(all_new_hours)
        log.info(f"Running pipeline on {len(sorted_hours)} snapshot hours...")

        total_violations = 0
        for i, hour in enumerate(sorted_hours):
            run_cleaning_pass(conn, hour)
            result = detect_violations(conn, hour)
            check_reversions(conn, hour)
            total_violations += result.get("violations", 0)

            if (i + 1) % 100 == 0 or (i + 1) == len(sorted_hours):
                log.info(
                    f"  Progress: {i+1}/{len(sorted_hours)} hours | "
                    f"{total_violations} violations so far"
                )

        log.info(
            f"Tardis backfill complete: "
            f"{len(sorted_hours)} hours processed | "
            f"{total_violations} total violations detected"
        )

    return len(all_new_hours)
