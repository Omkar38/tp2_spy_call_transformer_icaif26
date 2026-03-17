"""
collector/live.py — Hourly live collection scheduler.

Runs in a background thread. Every hour:
1. Fetch surface snapshot from Deribit
2. Save to DB
3. Run cleaning pass
4. Run TP2 detection
5. Check reversions on open violations
6. Log summary

Usage:
    from collector.live import start_scheduler
    start_scheduler()   # non-blocking, runs in background thread
"""

import time
import logging
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from storage.db import get_connection, get_stats, maybe_backup
from collector.historical import fetch_and_save_snapshot
from processor.cleaner import run_cleaning_pass
from detector.tp2 import detect_violations, check_reversions

log = logging.getLogger("scheduler")

_stop_event = threading.Event()


def run_one_cycle(conn) -> dict:
    """
    Single collection cycle. Called every hour.
    Returns summary dict for logging/monitoring.
    """
    now = datetime.now(timezone.utc)
    interval = cfg.SNAPSHOT_INTERVAL_MINUTES
    floored_min = (now.minute // interval) * interval
    hour_str = now.replace(minute=floored_min, second=0, microsecond=0).isoformat()

    log.info(f"{'='*50}")
    log.info(f"Cycle start: {hour_str}")

    summary = {"hour": hour_str}

    # Step 1: Fetch snapshot
    n_saved = fetch_and_save_snapshot(conn)
    summary["snapshot_rows"] = n_saved

    if n_saved == 0:
        log.warning("Snapshot fetch returned 0 rows — skipping rest of cycle")
        return summary

    # Step 2: Clean
    clean_counts = run_cleaning_pass(conn, hour_str)
    summary["clean_kept"]    = clean_counts.get("kept", 0)
    summary["clean_removed"] = clean_counts.get("removed", 0)

    # Step 3: TP2 detection
    viol_summary = detect_violations(conn, hour_str)
    summary["pairs_checked"] = viol_summary["pairs_checked"]
    summary["violations"]    = viol_summary["violations"]
    summary["viol_rate"]     = viol_summary["violation_rate"]
    summary["tradeable"]     = viol_summary["tradeable"]

    # Step 4: Reversion check (on violations from previous hours)
    check_reversions(conn, hour_str)

    # Step 5: Log DB health
    stats = get_stats(conn)
    log.info(
        f"DB state → snapshots:{stats['snapshots']:,} "
        f"violations:{stats['violations']:,} "
        f"reversions:{stats['reversions']:,}"
    )
    # Step 6: Backup DB to SSD if it exceeded 1GB threshold
    maybe_backup(conn)

    log.info(f"Cycle complete: {hour_str}")
    log.info(f"{'='*50}")

    return summary


def _scheduler_loop(interval_minutes: int):
    """Internal loop that runs cycles on a fixed interval."""
    conn = get_connection()
    log.info(f"Scheduler started — interval: {interval_minutes} min")

    while not _stop_event.is_set():
        try:
            run_one_cycle(conn)
        except Exception as e:
            log.error(f"Cycle failed: {e}", exc_info=True)

        # Sleep until next interval, checking stop_event every 10s
        next_run = datetime.now(timezone.utc) + timedelta(minutes=interval_minutes)
        log.info(f"Next run at: {next_run.strftime('%H:%M UTC')}")

        while datetime.now(timezone.utc) < next_run:
            if _stop_event.is_set():
                break
            time.sleep(10)

    conn.close()
    log.info("Scheduler stopped cleanly")


def start_scheduler(interval_minutes: int = cfg.SNAPSHOT_INTERVAL_MINUTES) -> threading.Thread:
    """
    Start the hourly scheduler in a background thread.
    Returns the thread so you can join() it if needed.

    Usage in Jupyter:
        from collector.live import start_scheduler
        t = start_scheduler()
        # continues running every hour in background
    """
    _stop_event.clear()
    t = threading.Thread(
        target=_scheduler_loop,
        args=(interval_minutes,),
        daemon=True,
        name="tp2-scheduler"
    )
    t.start()
    log.info(f"Scheduler thread started (daemon=True, interval={interval_minutes}min)")
    return t


def stop_scheduler():
    """Gracefully stop the scheduler."""
    _stop_event.set()
    log.info("Stop signal sent to scheduler")
