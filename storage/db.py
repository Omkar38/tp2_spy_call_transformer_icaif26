"""
storage/db.py — SQLite database layer.

SQLite is perfect here:
- Zero setup, single file, works everywhere
- Fast enough for 100k rows
- Easy to inspect with DB Browser for SQLite
- One file you can share with your team

Schema:
  snapshots   → surface snapshots (one row per instrument per snapshot)
  trades      → executed trades (Deribit or Databento)
  violations  → detected TP2 violations with features
  reversions  → reversion labels (filled in as time passes)

Multi-asset: config.DB_PATH points to asset-specific DB (tp2_crypto.db / tp2_equity.db).
The schema is identical — the asset class context comes from which DB file you open.
"""

import sqlite3
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

log = logging.getLogger("db")


def get_connection(db_path=None) -> sqlite3.Connection:
    """Get a DB connection with WAL mode for concurrent reads.

    Args:
        db_path: Override DB path. Defaults to config.DB_PATH (asset-class-specific).
    """
    path = db_path or cfg.DB_PATH
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row   # lets you access columns by name
    return conn


def init_db(conn: sqlite3.Connection):
    """Create all tables if they don't exist."""
    log.info(f"Initialising database at {cfg.DB_PATH}")

    conn.executescript("""
    -- ── Raw hourly surface snapshots ───────────────────────────
    CREATE TABLE IF NOT EXISTS snapshots (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_hour     TEXT    NOT NULL,      -- ISO8601, floor to hour
        instrument_name   TEXT    NOT NULL,
        expiry            TEXT    NOT NULL,
        strike            REAL    NOT NULL,
        option_type       TEXT    NOT NULL,      -- 'C' or 'P'
        bid_price         REAL,
        ask_price         REAL,
        mid_price         REAL,
        mark_price        REAL,
        mark_iv           REAL,
        underlying_price  REAL,
        open_interest     REAL,
        volume_24h        REAL,
        quote_currency    TEXT,
        -- derived columns filled by processor
        spread            REAL,
        spread_pct        REAL,
        moneyness         REAL,
        time_to_expiry_d  REAL,
        is_clean          INTEGER DEFAULT 0,     -- 1 after cleaning pass
        UNIQUE(snapshot_hour, instrument_name)
    );

    -- ── Historical executed trades ──────────────────────────────
    CREATE TABLE IF NOT EXISTS trades (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_ms    INTEGER NOT NULL,
        instrument_name TEXT    NOT NULL,
        expiry          TEXT,
        strike          REAL,
        option_type     TEXT,
        price           REAL,
        amount          REAL,
        iv              REAL,
        mark_price      REAL,
        index_price     REAL,
        direction       TEXT,
        trade_id        TEXT    UNIQUE
    );

    -- ── Detected TP2 violations ─────────────────────────────────
    CREATE TABLE IF NOT EXISTS violations (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_hour        TEXT    NOT NULL,
        T1                   TEXT    NOT NULL,
        T2                   TEXT    NOT NULL,
        K1                   REAL    NOT NULL,
        K2                   REAL    NOT NULL,
        -- The four option prices
        C_K1_T1              REAL    NOT NULL,
        C_K2_T1              REAL    NOT NULL,
        C_K1_T2              REAL    NOT NULL,
        C_K2_T2              REAL    NOT NULL,
        -- TP2 inequality
        LHS                  REAL,
        RHS                  REAL,
        violation_pct        REAL,
        -- Surface context
        maturity_gap         INTEGER,
        strike_gap           REAL,
        strike_gap_pct       REAL,
        moneyness_K1         REAL,
        moneyness_K2         REAL,
        -- Liquidity context
        avg_spread_pct       REAL,
        est_total_cost_pct   REAL,
        net_edge             REAL,
        -- Market regime at detection time
        btc_spot             REAL,
        btc_return_1h        REAL,
        btc_return_24h       REAL,
        realized_vol_24h     REAL,
        iv_level_surface     REAL,
        iv_skew              REAL,
        term_structure_slope REAL,
        hour_of_day          INTEGER,
        day_of_week          INTEGER,
        -- Status
        profitability        TEXT DEFAULT 'unclassified',
        UNIQUE(snapshot_hour, T1, T2, K1, K2)
    );

    -- ── Reversion labels (filled as time passes) ───────────────
    CREATE TABLE IF NOT EXISTS reversions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        violation_id    INTEGER REFERENCES violations(id),
        checked_hour    TEXT    NOT NULL,       -- when we re-checked
        hours_elapsed   REAL    NOT NULL,
        current_viol    REAL,                   -- violation at check time (0 = reverted)
        reverted        INTEGER DEFAULT 0,      -- 1 if violation_pct dropped to <0.5%
        revert_speed    TEXT,                   -- 'fast'(<4h) 'medium'(4-24h) 'slow'(>24h) 'never'
        UNIQUE(violation_id, checked_hour)
    );

    -- ── Indices for fast queries ────────────────────────────────
    CREATE INDEX IF NOT EXISTS idx_snap_hour  ON snapshots(snapshot_hour);
    CREATE INDEX IF NOT EXISTS idx_snap_instr ON snapshots(instrument_name);
    CREATE INDEX IF NOT EXISTS idx_snap_clean ON snapshots(is_clean);
    CREATE INDEX IF NOT EXISTS idx_trade_ts   ON trades(timestamp_ms);
    CREATE INDEX IF NOT EXISTS idx_trade_instr ON trades(instrument_name);
    CREATE INDEX IF NOT EXISTS idx_viol_hour  ON violations(snapshot_hour);
    CREATE INDEX IF NOT EXISTS idx_viol_pct   ON violations(violation_pct);
    CREATE INDEX IF NOT EXISTS idx_rev_vid    ON reversions(violation_id);
    """)

    conn.commit()
    log.info("Database initialised")


def maybe_restore():
    """
    On startup: if the SSD backup DB has more data than the local DB,
    restore from backup so the local DB is up to date.

    Safe to call every time — only copies when the backup is ahead.
    Returns True if a restore was performed.
    """
    if not cfg.SSD_AVAILABLE:
        return False

    backup_path = cfg.SSD_BACKUP_DB
    if not backup_path.exists():
        log.info("No SSD backup found — skipping restore.")
        return False

    # Compare snapshot counts
    def _count(path):
        try:
            c = sqlite3.connect(path)
            n = c.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
            c.close()
            return n
        except Exception:
            return 0

    local_n  = _count(cfg.DB_PATH)
    backup_n = _count(backup_path)

    if backup_n <= local_n:
        log.info(
            f"Local DB up to date ({local_n:,} snapshots). "
            f"SSD backup has {backup_n:,}. No restore needed."
        )
        return False

    log.info(
        f"SSD backup is ahead: {backup_n:,} snapshots vs local {local_n:,}. "
        f"Restoring from {backup_path} ..."
    )
    try:
        src  = sqlite3.connect(backup_path)
        dest = sqlite3.connect(cfg.DB_PATH)
        src.backup(dest)
        dest.close()
        src.close()
        log.info("Restore complete. Local DB is now in sync with SSD backup.")
        return True
    except Exception as e:
        log.error(f"Restore failed: {e}")
        return False


def maybe_backup(conn: sqlite3.Connection):
    """
    Copy the DB to the external SSD if:
      1. SSD is plugged in
      2. DB has grown past DB_BACKUP_THRESHOLD_GB

    Uses SQLite's built-in backup API (safe while DB is live).
    Called automatically at the end of each collection cycle.
    """
    if not cfg.SSD_AVAILABLE:
        return

    db_size_gb = cfg.DB_PATH.stat().st_size / 1e9
    if db_size_gb < cfg.DB_BACKUP_THRESHOLD_GB:
        return

    log.info(
        f"DB is {db_size_gb:.2f} GB — backing up to {cfg.SSD_BACKUP_DB} ..."
    )
    try:
        dest = sqlite3.connect(cfg.SSD_BACKUP_DB)
        conn.backup(dest)
        dest.close()
        log.info("DB backup complete.")
    except Exception as e:
        log.error(f"DB backup failed: {e}")


def get_stats(conn: sqlite3.Connection) -> dict:
    """Quick health check — how much data do we have?"""
    stats = {}
    for table in ["snapshots", "trades", "violations", "reversions"]:
        row = conn.execute(f"SELECT COUNT(*) as n FROM {table}").fetchone()
        stats[table] = row["n"]

    # Date range of snapshots
    row = conn.execute("""
        SELECT MIN(snapshot_hour) as earliest, MAX(snapshot_hour) as latest
        FROM snapshots
    """).fetchone()
    stats["snapshot_range"] = (row["earliest"], row["latest"])

    # Violation rate
    if stats["violations"] > 0:
        row = conn.execute("""
            SELECT AVG(violation_pct) as avg_v, MAX(violation_pct) as max_v
            FROM violations
        """).fetchone()
        stats["avg_violation_pct"] = round(row["avg_v"], 3)
        stats["max_violation_pct"] = round(row["max_v"], 3)

    return stats
