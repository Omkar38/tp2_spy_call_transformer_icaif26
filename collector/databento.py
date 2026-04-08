"""
collector/databento.py — Databento OPRA collector for SPX options.

Fetches historical S&P 500 (SPX) options data from Databento's OPRA feed
and normalizes it into the same snapshot schema used by the Deribit collector.

This enables running the identical TP2 detection pipeline on equity data
for paper replication (Glasserman, Li & Pirjol 2025).

Setup:
    pip install databento
    export DATABENTO_API_KEY="db-..."

Usage:
    from collector.databento import fetch_spx_snapshots, estimate_cost
    estimate_cost("2023-04-01", "2023-04-02")   # check cost FIRST
    fetch_spx_snapshots(conn, start="2023-04-01", end="2023-06-30")

Databento data:
    - Dataset: OPRA.PILLAR (US equity options, all 17 exchanges)
    - Schema: mbp-1 (top-of-book bid/ask) — expensive! Always estimate first
    - SPX options use OCC symbology: "SPX   240315C04000000"
    - SPXW = weekly expirations, SPX = standard monthly
    - Coverage: March 28, 2023 onward
    - Cost: $125 free signup credit, usage-based per GB

Paper context:
    - Glasserman et al. use 2000–2022 SPX EOD data (likely OptionMetrics)
    - Databento starts March 2023, so we replicate methodology on 2023+ data
    - Professor directive: use same calendar dates for SPX and BTC comparison

IMPORTANT: OPRA data is extremely high-volume. A single day of SPX mbp-1
can be multiple GB. ALWAYS call estimate_cost() before fetching.
Consider using ohlcv-1d schema for EOD snapshots to minimize cost.
"""

import logging
import re
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

log = logging.getLogger("databento_collector")


# ── OCC symbology parser ──────────────────────────────────────
# SPX option symbols follow OCC/OSI format (21 characters total):
#   "SPX   240315C04000000"
#    ^^^^^^ ^^^^^^ ^ ^^^^^^^^
#    root   YYMMDD C/P strike (× 1000, zero-padded to 8 digits)
#    (6ch)  (6ch)  (1) (8ch)
#
# Root is space-padded to 6 chars: "SPX   " or "SPXW  "
# Strike 04000000 / 1000 = $4000.000

_OCC_RE = re.compile(
    r"^(?P<root>[A-Z]{1,6})\s*"
    r"(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"(?P<cp>[CP])"
    r"(?P<strike>\d{8})$"
)

_MONTH_ABBR = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}


def parse_occ_symbol(symbol: str) -> Optional[dict]:
    """
    Parse OCC option symbol into components.

    'SPX   240315C04000000' → {
        'root': 'SPX', 'expiry': '15MAR24', 'strike': 4000.0,
        'option_type': 'C', 'expiry_date': datetime(2024,3,15)
    }

    Also handles SPXW (weekly) symbols — normalized to same format.
    """
    symbol = symbol.strip()
    m = _OCC_RE.match(symbol)
    if not m:
        return None

    yy = int(m.group("yy"))
    mm = int(m.group("mm"))
    dd = int(m.group("dd"))
    strike = int(m.group("strike")) / 1000.0
    root = m.group("root").rstrip()

    # Build Deribit-style expiry string: "15MAR24"
    expiry_str = f"{dd:02d}{_MONTH_ABBR[mm]}{yy:02d}"

    return {
        "root": root,
        "expiry": expiry_str,
        "strike": strike,
        "option_type": m.group("cp"),
        "expiry_date": datetime(2000 + yy, mm, dd, tzinfo=timezone.utc),
    }


# ── Databento client helpers ──────────────────────────────────

def _check_databento_available():
    """Check if databento package is installed and API key is set."""
    try:
        import databento  # noqa: F401
    except ImportError:
        raise ImportError(
            "databento package not installed. Run: pip install databento"
        )
    if not cfg.DATABENTO_API_KEY:
        raise ValueError(
            "DATABENTO_API_KEY not set. "
            "Sign up at https://databento.com and export DATABENTO_API_KEY=db-..."
        )


def _get_client():
    """Get an authenticated Databento Historical client."""
    _check_databento_available()
    import databento as db
    return db.Historical(cfg.DATABENTO_API_KEY)


# ── Definitions fetcher ───────────────────────────────────────
# The mbp-1 schema gives bid/ask but NOT strike/expiry metadata.
# We need the 'definition' schema to get instrument details,
# then join on instrument_id.

def fetch_definitions(start: str, end: str) -> dict:
    """
    Fetch instrument definitions (strike, expiry, type) for SPX options.

    Returns dict mapping instrument_id → {expiry, strike, option_type, ...}
    """
    client = _get_client()

    log.info(f"Fetching SPX option definitions: {start} → {end}")

    data = client.timeseries.get_range(
        dataset=cfg.DATABENTO_DATASET,
        symbols="SPX.OPT",
        schema="definition",
        start=start,
        end=end,
        stype_in="parent",
    )

    definitions = {}
    for record in data:
        iid = record.instrument_id

        # Use raw_symbol to parse, or use the structured fields
        raw_sym = record.raw_symbol if hasattr(record, "raw_symbol") else None
        parsed = parse_occ_symbol(raw_sym) if raw_sym else None

        if parsed is None:
            # Fallback: use structured fields from definition record
            if hasattr(record, "strike_price") and hasattr(record, "expiration"):
                strike = record.strike_price / 1e9 if record.strike_price else None
                exp_dt = record.expiration if hasattr(record, "expiration") else None
                opt_type = getattr(record, "instrument_class", None)

                if strike and exp_dt and opt_type:
                    # Build expiry string from expiration datetime
                    if hasattr(exp_dt, "month"):
                        expiry_str = f"{exp_dt.day:02d}{_MONTH_ABBR[exp_dt.month]}{exp_dt.year % 100:02d}"
                    else:
                        continue

                    root = getattr(record, "underlying", "SPX")
                    definitions[iid] = {
                        "root": root,
                        "expiry": expiry_str,
                        "strike": strike,
                        "option_type": opt_type,
                        "raw_symbol": raw_sym or "",
                    }
            continue

        definitions[iid] = parsed
        definitions[iid]["raw_symbol"] = raw_sym

    log.info(f"Loaded {len(definitions)} instrument definitions")
    return definitions


# ── Main fetch function ───────────────────────────────────────

def fetch_spx_snapshots(
    conn: sqlite3.Connection,
    start: str,
    end: str,
    use_ohlcv: bool = True,
) -> int:
    """
    Fetch SPX options data from Databento and save to snapshots table.

    Args:
        conn: DB connection (should be equity DB via TP2_ASSET_CLASS=equity)
        start: Start date "YYYY-MM-DD"
        end: End date "YYYY-MM-DD"
        use_ohlcv: If True, use ohlcv-1d (cheap, EOD only). If False, use
                   mbp-1 (expensive, tick-level). Paper uses EOD → True.

    Returns:
        Number of snapshot rows saved.

    The data is normalized into the same schema as Deribit snapshots:
        instrument_name, expiry, strike, option_type, bid_price, ask_price,
        mid_price, underlying_price, open_interest, etc.
    """
    client = _get_client()

    log.info(f"Fetching SPX options from Databento: {start} → {end}")
    log.info(f"Schema: {'ohlcv-1d (EOD, cheap)' if use_ohlcv else 'mbp-1 (tick, expensive)'}")

    # Step 1: Get instrument definitions (strike, expiry, type)
    definitions = fetch_definitions(start, end)
    if not definitions:
        log.error("No definitions returned — cannot map instrument IDs to strikes/expiries")
        return 0

    # Step 2: Fetch price data
    if use_ohlcv:
        # ohlcv-1d: one bar per instrument per day — cheapest option
        # close price ≈ EOD price (what the paper uses)
        data = client.timeseries.get_range(
            dataset=cfg.DATABENTO_DATASET,
            symbols="SPX.OPT",
            schema="ohlcv-1d",
            start=start,
            end=end,
            stype_in="parent",
        )
    else:
        # mbp-1: every quote update — very expensive for OPRA
        # Only use for intraday analysis or if ohlcv doesn't have bid/ask
        data = client.timeseries.get_range(
            dataset=cfg.DATABENTO_DATASET,
            symbols="SPX.OPT",
            schema="mbp-1",
            start=start,
            end=end,
            stype_in="parent",
        )

    total_saved = 0
    daily_snapshots = {}  # date_str → {instrument_name → record_dict}

    for record in data:
        iid = record.instrument_id

        # Look up instrument metadata
        defn = definitions.get(iid)
        if defn is None:
            continue

        # Only keep SPX and SPXW (skip other roots)
        root = defn.get("root", "")
        if root not in ("SPX", "SPXW"):
            continue

        # Get timestamp
        ts_ns = record.ts_event
        dt_utc = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
        date_str = dt_utc.strftime("%Y-%m-%d")

        if date_str not in daily_snapshots:
            daily_snapshots[date_str] = {}

        # Build instrument name in Deribit-style format for pipeline compat
        inst_name = (
            f"SPX-{defn['expiry']}-{int(defn['strike'])}-{defn['option_type']}"
        )

        if use_ohlcv:
            # ohlcv-1d fields: open, high, low, close, volume
            close = record.close / 1e9 if hasattr(record, "close") and record.close else None
            volume = record.volume if hasattr(record, "volume") else None

            if close is None or close <= 0:
                continue

            # For ohlcv we don't have bid/ask — use close as mid
            # The paper also uses mid prices for core analysis
            daily_snapshots[date_str][inst_name] = {
                "instrument_name": inst_name,
                "expiry": defn["expiry"],
                "strike": defn["strike"],
                "option_type": defn["option_type"],
                "bid_price": close,       # approximation: close ≈ mid
                "ask_price": close,       # no spread info in ohlcv
                "mid_price": close,
                "mark_price": close,
                "mark_iv": None,
                "underlying_price": None,  # filled from index data
                "open_interest": None,     # available via statistics schema
                "volume_24h": volume,
                "quote_currency": "USD",
            }
        else:
            # mbp-1 fields: bid_px_00, ask_px_00, bid_sz_00, ask_sz_00
            bid = record.bid_px_00 / 1e9 if hasattr(record, "bid_px_00") and record.bid_px_00 else None
            ask = record.ask_px_00 / 1e9 if hasattr(record, "ask_px_00") and record.ask_px_00 else None

            if bid is None or ask is None or bid <= 0:
                continue

            mid = (bid + ask) / 2.0

            # For mbp-1, keep the LAST record per instrument per day (≈ EOD)
            daily_snapshots[date_str][inst_name] = {
                "instrument_name": inst_name,
                "expiry": defn["expiry"],
                "strike": defn["strike"],
                "option_type": defn["option_type"],
                "bid_price": bid,
                "ask_price": ask,
                "mid_price": mid,
                "mark_price": mid,
                "mark_iv": None,
                "underlying_price": None,
                "open_interest": None,
                "volume_24h": None,
                "quote_currency": "USD",
            }

    # Step 3: Save each day as a snapshot_hour
    # Use 21:00 UTC (≈ 16:00 ET market close) as the snapshot timestamp
    for date_str, instruments in sorted(daily_snapshots.items()):
        snapshot_hour = f"{date_str}T21:00:00+00:00"

        rows = []
        for inst_name, d in instruments.items():
            rows.append((
                snapshot_hour,
                d["instrument_name"],
                d["expiry"],
                d["strike"],
                d["option_type"],
                d["bid_price"],
                d["ask_price"],
                d["mid_price"],
                d["mark_price"],
                d["mark_iv"],
                d["underlying_price"],
                d["open_interest"],
                d["volume_24h"],
                d["quote_currency"],
            ))

        if rows:
            conn.executemany("""
                INSERT OR REPLACE INTO snapshots
                  (snapshot_hour, instrument_name, expiry, strike, option_type,
                   bid_price, ask_price, mid_price, mark_price, mark_iv,
                   underlying_price, open_interest, volume_24h, quote_currency)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, rows)
            conn.commit()
            total_saved += len(rows)
            log.info(f"  {date_str}: saved {len(rows)} instruments")

    log.info(
        f"Databento fetch complete. "
        f"Total: {total_saved} snapshot rows across {len(daily_snapshots)} trading days."
    )
    return total_saved


# ── Cost estimation ───────────────────────────────────────────

def estimate_cost(start: str, end: str, schema: str = "ohlcv-1d") -> dict:
    """
    Estimate the Databento API cost for a given date range
    WITHOUT actually fetching data.

    ALWAYS call this before fetch_spx_snapshots() to avoid surprise charges.

    Args:
        start: Start date "YYYY-MM-DD"
        end: End date "YYYY-MM-DD"
        schema: "ohlcv-1d" (cheap, EOD) or "mbp-1" (expensive, tick)

    Returns:
        dict with estimated cost.
    """
    client = _get_client()

    cost = client.metadata.get_cost(
        dataset=cfg.DATABENTO_DATASET,
        symbols="SPX.OPT",
        schema=schema,
        start=start,
        end=end,
        stype_in="parent",
    )

    result = {
        "start": start,
        "end": end,
        "schema": schema,
        "estimated_cost_usd": cost,
        "free_credit_remaining": max(0, 125.0 - cost),
    }

    log.info(f"Estimated cost for {start} → {end} ({schema}): ${cost:.2f}")
    if cost > 125.0:
        log.warning(f"Cost ${cost:.2f} exceeds $125 free credit!")

    return result


def list_available_dates() -> dict:
    """Check what date range is available for OPRA data on Databento."""
    client = _get_client()
    info = client.metadata.get_dataset_range(dataset=cfg.DATABENTO_DATASET)
    log.info(f"OPRA data available: {info}")
    return info


# ── CLI entry point ───────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format=cfg.LOG_FORMAT,
    )

    from storage.db import get_connection, init_db

    # Rolling window: train on last ROLLING_WINDOW_MONTHS months
    # Predict window: next ROLLING_PREDICT_DAYS days (not fetched — future data)
    today = datetime.now(tz=timezone.utc).date()
    train_start = today - timedelta(days=24 * 30)  # 24 months history for DS-CAT
    train_end   = today - timedelta(days=2)         # OPRA has ~2-day publication lag

    start_str = train_start.strftime("%Y-%m-%d")
    end_str   = train_end.strftime("%Y-%m-%d")
    predict_end = (today + timedelta(days=cfg.ROLLING_PREDICT_DAYS)).strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  Databento SPX Options Collector")
    print(f"  Train window : {start_str} → {end_str}  (24 months)")
    print(f"  Predict window: {end_str} → {predict_end} ({cfg.ROLLING_PREDICT_DAYS} days ahead)")
    print(f"  DB : {cfg.DB_PATH}")
    print(f"{'='*60}\n")

    # Step 1: Init equity DB
    conn = get_connection(cfg.DB_PATH)
    init_db(conn)

    # Step 2: Check what's already in the DB — only fetch the missing gap
    row = conn.execute(
        "SELECT MIN(snapshot_hour), MAX(snapshot_hour) FROM snapshots"
    ).fetchone()
    existing_min, existing_max = row[0], row[1]

    if existing_min:
        existing_start = existing_min[:10]
        print(f">>> DB already has data: {existing_start} → {existing_max[:10]}")
        if existing_start <= start_str:
            print("    No gap to fill — DB already covers the full window.")
            conn.close()
            sys.exit(0)
        end_str = (
            datetime.strptime(existing_start, "%Y-%m-%d").date() - timedelta(days=1)
        ).strftime("%Y-%m-%d")
        print(f">>> Fetching missing gap only: {start_str} → {end_str}\n")
    else:
        print(f">>> DB is empty — fetching full window: {start_str} → {end_str}\n")

    # Step 3: Estimate cost BEFORE fetching
    print(">>> Estimating cost (ohlcv-1d schema)...")
    cost_info = estimate_cost(start_str, end_str, schema="ohlcv-1d")
    print(f"    Estimated cost : ${cost_info['estimated_cost_usd']:.4f}")
    print(f"    Free credit remaining after pull: ${cost_info['free_credit_remaining']:.2f}\n")

    if cost_info["estimated_cost_usd"] > 125.0:
        print("ERROR: Estimated cost exceeds $125 free credit. Aborting.")
        sys.exit(1)

    # Step 4: Fetch in 3-month chunks to avoid gateway timeouts
    # Databento times out when fetching definitions for long date ranges in one shot
    CHUNK_DAYS = 90
    fetch_start = datetime.strptime(start_str, "%Y-%m-%d").date()
    fetch_end   = datetime.strptime(end_str,   "%Y-%m-%d").date()

    total = 0
    chunk_s = fetch_start
    while chunk_s < fetch_end:
        chunk_e = min(chunk_s + timedelta(days=CHUNK_DAYS), fetch_end)
        cs = chunk_s.strftime("%Y-%m-%d")
        ce = chunk_e.strftime("%Y-%m-%d")
        print(f">>> Chunk: {cs} → {ce} ...")
        total += fetch_spx_snapshots(conn, start=cs, end=ce, use_ohlcv=True)
        chunk_s = chunk_e + timedelta(days=1)

    print(f"\nDone. Saved {total} snapshot rows to {cfg.DB_PATH}")
    print(f"Next step: run the TP2 detector on this data.")
    conn.close()
