"""
main.py — Single entry point for the entire TP2 project.

USAGE:

  # 1. First-time setup + backfill 30 days of history, then start live
  python main.py --mode full

  # 2. Just backfill historical data (run once)
  python main.py --mode history --days 30

  # 3. Just start live hourly collection
  python main.py --mode live

  # 4. Run one snapshot right now (good for testing)
  python main.py --mode once

  # 5. Backfill 90 days from Tardis.dev (requires API key in .env)
  python main.py --mode tardis --days 90

  # 6. Check DB health
  python main.py --mode status

In Jupyter notebook, import and call directly:
  from main import setup, run_once, start_live
  setup()          # init DB
  run_once()       # immediate snapshot + detection
  start_live()     # start hourly background thread
"""

import logging
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Setup logging before anything else ──────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import config as cfg

logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL),
    format=cfg.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg.LOG_FILE),
    ]
)
log = logging.getLogger("main")

# ── Imports ──────────────────────────────────────────────────────
from storage.db import get_connection, init_db, get_stats, maybe_restore
from collector.historical import fetch_full_trade_history, fetch_and_save_snapshot
from collector.live import start_scheduler, run_one_cycle
from collector.tardis import backfill as tardis_backfill
from processor.cleaner import run_cleaning_pass


def setup() -> object:
    """Initialise DB and return connection. Always call this first."""
    log.info("Setting up TP2 project...")
    maybe_restore()
    conn = get_connection()
    init_db(conn)
    stats = get_stats(conn)
    log.info(f"DB ready: {stats}")
    return conn


def run_history(conn, days: int = cfg.HISTORY_DAYS):
    """
    Backfill historical trades from Deribit.
    Run once. Takes 5-15 minutes for 30 days.
    """
    log.info(f"Starting historical backfill: {days} days...")
    total = fetch_full_trade_history(conn, days_back=days)
    log.info(f"Historical backfill complete: {total:,} trades saved")
    return total


def run_once(conn) -> dict:
    """
    Fetch one snapshot right now + clean + detect violations.
    Good for testing. Returns summary.
    """
    log.info("Running single cycle...")
    return run_one_cycle(conn)


def start_live(conn=None, interval: int = cfg.SNAPSHOT_INTERVAL_MINUTES):
    """
    Start the hourly live collection scheduler.
    Runs in background thread — non-blocking.
    """
    log.info(f"Starting live scheduler (interval={interval}min)...")
    t = start_scheduler(interval_minutes=interval)
    return t


def show_status(conn):
    """Print current DB state."""
    stats = get_stats(conn)
    print("\n" + "="*50)
    print("  TP2 PROJECT — DATABASE STATUS")
    print("="*50)
    print(f"  Snapshots:    {stats.get('snapshots', 0):>8,}")
    print(f"  Trades:       {stats.get('trades', 0):>8,}")
    print(f"  Violations:   {stats.get('violations', 0):>8,}")
    print(f"  Reversions:   {stats.get('reversions', 0):>8,}")
    r = stats.get("snapshot_range", (None, None))
    if r[0]:
        print(f"  Date range:   {r[0][:16]}  →  {r[1][:16]}")
    if "avg_violation_pct" in stats:
        print(f"  Avg violation: {stats['avg_violation_pct']:.3f}%")
        print(f"  Max violation: {stats['max_violation_pct']:.3f}%")
    print("="*50 + "\n")


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMS 520 TP2 Options Project")
    parser.add_argument("--mode", choices=["full", "history", "live", "once", "status", "tardis"],
                        default="once", help="What to run")
    parser.add_argument("--days", type=int, default=cfg.HISTORY_DAYS,
                        help="Days of history to fetch (default: 30)")
    parser.add_argument("--interval", type=int, default=cfg.SNAPSHOT_INTERVAL_MINUTES,
                        help="Live collection interval in minutes (default: 60)")
    args = parser.parse_args()

    conn = setup()

    if args.mode == "status":
        show_status(conn)

    elif args.mode == "once":
        summary = run_once(conn)
        show_status(conn)
        print(f"Cycle result: {summary}")

    elif args.mode == "history":
        run_history(conn, days=args.days)
        show_status(conn)

    elif args.mode == "live":
        show_status(conn)
        t = start_live(conn, interval=args.interval)
        log.info("Live scheduler running. Press Ctrl+C to stop.")
        try:
            t.join()
        except KeyboardInterrupt:
            from collector.live import stop_scheduler
            stop_scheduler()
            log.info("Stopped.")

    elif args.mode == "tardis":
        log.info(f"Tardis backfill: {args.days} days")
        n = tardis_backfill(conn, days=args.days)
        log.info(f"Tardis backfill done: {n} snapshot hours inserted")
        show_status(conn)

    elif args.mode == "full":
        # Full setup: backfill history THEN start live collection
        log.info("FULL MODE: backfill history + start live collection")
        run_history(conn, days=args.days)
        show_status(conn)
        t = start_live(conn, interval=args.interval)
        log.info("Live scheduler running. Press Ctrl+C to stop.")
        try:
            t.join()
        except KeyboardInterrupt:
            from collector.live import stop_scheduler
            stop_scheduler()
            log.info("Stopped.")
