"""
collector/live_v2.py — V2 live collection scheduler.

Runs on interval. Each cycle:
1. Fetch surface snapshot for ALL currencies (BTC, ETH, SOL)
2. Save to V2 DB (with all new fields)
3. Run cleaning pass (calls + puts)
4. Run TP2 detection per currency
5. Fetch Greeks for violation-leg instruments
6. Check reversions on open violations
7. Log summary

Usage:
    from collector.live_v2 import start_v2_scheduler
    start_v2_scheduler()   # non-blocking, runs in background thread
"""

import time
import logging
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from storage.db_v2 import get_v2_connection, init_v2_db, get_v2_stats
from collector.historical_v2 import fetch_and_save_snapshot_v2, fetch_and_save_greeks
from processor.cleaner import run_cleaning_pass
from detector.tp2 import detect_violations, check_reversions

log = logging.getLogger("scheduler_v2")

_stop_event = threading.Event()


def _get_violation_leg_instruments(conn, hour_str: str, currency: str) -> list:
    """Extract unique instrument names from violations detected this hour."""
    rows = conn.execute("""
        SELECT DISTINCT T1, T2, K1, K2
        FROM violations
        WHERE snapshot_hour = ? AND currency = ?
    """, (hour_str, currency)).fetchall()

    instruments = set()
    for r in rows:
        # Reconstruct instrument names from violation quad
        for expiry in [r["T1"], r["T2"]]:
            for strike in [r["K1"], r["K2"]]:
                strike_str = f"{int(strike)}" if strike == int(strike) else f"{strike}"
                # We need both C and P legs — but violations are currently calls only
                instruments.add(f"{currency}-{expiry}-{strike_str}-C")
    return list(instruments)


def run_one_cycle_v2(conn) -> dict:
    """
    Single V2 collection cycle.
    Returns summary dict for logging/monitoring.
    """
    now = datetime.now(timezone.utc)
    interval = cfg.SNAPSHOT_INTERVAL_MINUTES
    floored_min = (now.minute // interval) * interval
    hour_str = now.replace(minute=floored_min, second=0, microsecond=0).isoformat()

    log.info(f"{'='*60}")
    log.info(f"V2 Cycle start: {hour_str}")

    summary = {"hour": hour_str}

    # Step 1: Fetch multi-currency snapshot
    snap_results = fetch_and_save_snapshot_v2(conn)
    summary["snapshots"] = snap_results
    total_saved = sum(snap_results.values())

    if total_saved == 0:
        log.warning("All snapshots returned 0 rows — skipping rest of cycle")
        return summary

    # Step 2: Clean (runs on all new data for this hour)
    clean_counts = run_cleaning_pass(conn, hour_str)
    summary["clean_kept"] = clean_counts.get("kept", 0)
    summary["clean_removed"] = clean_counts.get("removed", 0)

    # Step 3 & 4: TP2 detection + Greeks per currency
    summary["violations"] = {}
    summary["greeks"] = {}

    for currency in cfg.DERIBIT_CURRENCIES:
        # Check if we have data for this currency this hour
        count = conn.execute("""
            SELECT COUNT(*) as n FROM snapshots
            WHERE snapshot_hour = ? AND currency = ? AND is_clean = 1
        """, (hour_str, currency)).fetchone()["n"]

        if count == 0:
            log.info(f"  No clean data for {currency} — skipping detection")
            continue

        # TP2 detection (runs on calls, same detector logic)
        viol_summary = detect_violations(conn, hour_str, currency=currency)
        summary["violations"][currency] = viol_summary

        # Fetch Greeks for violation legs
        if viol_summary.get("violations", 0) > 0:
            leg_instruments = _get_violation_leg_instruments(conn, hour_str, currency)
            n_greeks = fetch_and_save_greeks(conn, leg_instruments, currency)
            summary["greeks"][currency] = n_greeks

    # Step 5: Reversion check on all open violations
    check_reversions(conn, hour_str)

    # Step 6: Log DB health
    stats = get_v2_stats(conn)
    log.info(
        f"V2 DB → snapshots:{stats['snapshots']:,} "
        f"violations:{stats['violations']:,} "
        f"reversions:{stats['reversions']:,} "
        f"greeks:{stats['greeks']:,}"
    )
    if stats.get("by_currency"):
        for curr, n in stats["by_currency"].items():
            log.info(f"  {curr}: {n:,} snapshots")

    log.info(f"V2 Cycle complete: {hour_str}")
    log.info(f"{'='*60}")

    return summary


def _scheduler_loop(interval_minutes: int):
    """Internal loop that runs cycles on a fixed interval."""
    conn = get_v2_connection()
    init_v2_db(conn)
    log.info(f"V2 Scheduler started — interval: {interval_minutes} min")

    while not _stop_event.is_set():
        try:
            run_one_cycle_v2(conn)
        except Exception as e:
            log.error(f"V2 Cycle failed: {e}", exc_info=True)

        # Sleep until next interval, checking stop_event every 10s
        next_run = datetime.now(timezone.utc) + timedelta(minutes=interval_minutes)
        log.info(f"Next run at: {next_run.strftime('%H:%M UTC')}")

        while datetime.now(timezone.utc) < next_run:
            if _stop_event.is_set():
                break
            time.sleep(10)

    conn.close()
    log.info("V2 Scheduler stopped cleanly")


def start_v2_scheduler(interval_minutes: int = cfg.SNAPSHOT_INTERVAL_MINUTES) -> threading.Thread:
    """
    Start the V2 scheduler in a background thread.
    Returns the thread so you can join() it if needed.
    """
    _stop_event.clear()
    t = threading.Thread(
        target=_scheduler_loop,
        args=(interval_minutes,),
        daemon=True,
        name="tp2-v2-scheduler"
    )
    t.start()
    log.info(f"V2 Scheduler thread started (daemon=True, interval={interval_minutes}min)")
    return t


def stop_v2_scheduler():
    """Gracefully stop the V2 scheduler."""
    _stop_event.set()
    log.info("Stop signal sent to V2 scheduler")
