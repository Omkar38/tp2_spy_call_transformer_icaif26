"""
notebook.py — Clean helper for Jupyter notebook usage.

Instead of running CLI commands, import from here in your notebook cells.

Example notebook workflow:
─────────────────────────────────────────────────────────
Cell 1: Setup
    from notebook import *
    conn = setup()

Cell 2: Backfill history (run once)
    backfill(days=30)   # ~10 minutes

Cell 3: Fetch current snapshot + detect violations
    summary = snapshot_now()
    print(summary)

Cell 4: Start hourly collection in background
    start_live()        # non-blocking

Cell 5: Check what's in the DB
    status()

Cell 6: Load violations as DataFrame
    df_viols = get_violations_df()
    df_viols.head(20)

Cell 7: Load reversion data
    df_rev = get_reversions_df()
    df_rev.head()

Cell 8: Stop scheduler when done
    stop_live()
─────────────────────────────────────────────────────────
"""

import logging
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from main import setup as _setup, run_once, run_history, start_live as _start_live, show_status
from storage.db import get_connection, get_stats
from collector.live import stop_scheduler

log = logging.getLogger("notebook")

# Global connection (re-used across cells)
_conn = None


def setup():
    """Initialize everything. Run this first in Cell 1."""
    global _conn
    from storage.db import init_db
    _conn = get_connection()
    init_db(_conn)
    show_status(_conn)
    print("✓ Ready. Run backfill() next if this is your first time.")
    return _conn


def backfill(days: int = 30):
    """
    Fetch historical trade data. Run ONCE at project start.
    Takes 5-15 minutes for 30 days.
    """
    global _conn
    if _conn is None: _conn = setup()
    print(f"Starting {days}-day backfill... (this takes ~10 minutes, let it run)")
    total = run_history(_conn, days=days)
    print(f"✓ Backfill done: {total:,} trades saved")
    show_status(_conn)


def snapshot_now():
    """Fetch one snapshot right now. Returns summary dict."""
    global _conn
    if _conn is None: _conn = setup()
    summary = run_once(_conn)
    print(f"✓ Snapshot: {summary['snapshot_rows']} rows | "
          f"{summary.get('violations',0)} violations | "
          f"rate: {summary.get('viol_rate',0):.2f}%")
    return summary


def start_live(interval_minutes: int = 60):
    """Start hourly background collection. Non-blocking."""
    global _conn
    if _conn is None: _conn = setup()
    t = _start_live(_conn, interval=interval_minutes)
    print(f"✓ Live scheduler started (every {interval_minutes} min, background thread)")
    print("  Run stop_live() to stop it.")
    return t


def stop_live():
    """Stop the background scheduler."""
    stop_scheduler()
    print("✓ Scheduler stopped")


def status():
    """Print DB health summary."""
    global _conn
    if _conn is None: _conn = setup()
    show_status(_conn)


def get_violations_df(min_viol_pct: float = 0.0) -> pd.DataFrame:
    """
    Load all violations as a pandas DataFrame.
    Filter with min_viol_pct to see only significant ones.
    """
    global _conn
    if _conn is None: _conn = setup()
    df = pd.read_sql_query(f"""
        SELECT * FROM violations
        WHERE violation_pct >= {min_viol_pct}
        ORDER BY snapshot_hour DESC, violation_pct DESC
    """, _conn)
    print(f"Loaded {len(df)} violations (min_viol_pct={min_viol_pct}%)")
    return df


def get_reversions_df() -> pd.DataFrame:
    """
    Load violations with their reversion labels joined.
    This is your Phase II dataset.
    """
    global _conn
    if _conn is None: _conn = setup()
    df = pd.read_sql_query("""
        SELECT
            v.snapshot_hour,
            v.T1, v.T2, v.K1, v.K2,
            v.violation_pct,
            v.avg_spread_pct,
            v.est_total_cost_pct,
            v.net_edge,
            v.moneyness_K1,
            v.maturity_gap,
            v.btc_spot,
            v.btc_return_1h,
            v.realized_vol_24h,
            v.iv_level_surface,
            v.iv_skew,
            v.hour_of_day,
            v.profitability,
            r.hours_elapsed,
            r.current_viol,
            r.reverted,
            r.revert_speed
        FROM violations v
        LEFT JOIN reversions r ON r.violation_id = v.id
        ORDER BY v.snapshot_hour DESC
    """, _conn)
    n_with_reversion = df["reverted"].notna().sum()
    print(f"Loaded {len(df)} violation-reversion pairs ({n_with_reversion} with reversion data)")
    return df


def get_snapshots_df(snapshot_hour: str = None, clean_only: bool = True) -> pd.DataFrame:
    """Load raw or clean surface snapshots."""
    global _conn
    if _conn is None: _conn = setup()
    where = []
    if clean_only:
        where.append("is_clean = 1")
    if snapshot_hour:
        where.append(f"snapshot_hour = '{snapshot_hour}'")
    where_str = "WHERE " + " AND ".join(where) if where else ""
    df = pd.read_sql_query(f"SELECT * FROM snapshots {where_str} ORDER BY snapshot_hour, expiry, strike", _conn)
    print(f"Loaded {len(df)} snapshot rows")
    return df


def violation_summary() -> pd.DataFrame:
    """Quick summary table — violations by expiry pair."""
    global _conn
    if _conn is None: _conn = setup()
    df = pd.read_sql_query("""
        SELECT
            T1 || ' → ' || T2 as expiry_pair,
            COUNT(*) as count,
            ROUND(AVG(violation_pct), 2) as avg_viol_pct,
            ROUND(MAX(violation_pct), 2) as max_viol_pct,
            ROUND(AVG(avg_spread_pct), 2) as avg_spread_pct,
            SUM(CASE WHEN profitability = 'TRADEABLE' THEN 1 ELSE 0 END) as tradeable_count,
            SUM(CASE WHEN profitability = 'noise_likely' THEN 1 ELSE 0 END) as noise_count
        FROM violations
        GROUP BY T1, T2
        ORDER BY count DESC
    """, _conn)
    return df
