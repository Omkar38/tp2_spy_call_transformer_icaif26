"""
storage/db_v2.py — V2 database schema with additional fields.

New vs v1:
  snapshots: +currency, +interest_rate, +last_price, +volume_usd, +high_24h, +low_24h, +price_change_24h
  greeks:    NEW table — per-instrument Greeks (delta, gamma, vega, theta, rho)
  violations, reversions, trades: unchanged schema

V2 DB lives at data/db/tp2_crypto_v2.db — old DB is untouched.
After validation, use merge_v1_into_v2() to bring old data forward.
"""

import sqlite3
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

log = logging.getLogger("db_v2")

# V2 DB path — separate from v1
V2_DB_PATH = cfg.DB_DIR / f"tp2_{cfg.ASSET_CLASS}_v2.db"


def get_v2_connection(db_path=None) -> sqlite3.Connection:
    """Get a V2 DB connection with WAL mode."""
    path = db_path or V2_DB_PATH
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_v2_db(conn: sqlite3.Connection):
    """Create V2 tables. Safe to call repeatedly."""
    log.info(f"Initialising V2 database at {V2_DB_PATH}")

    conn.executescript("""
    -- ── V2 surface snapshots (superset of V1) ────────────────────
    CREATE TABLE IF NOT EXISTS snapshots (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_hour     TEXT    NOT NULL,
        instrument_name   TEXT    NOT NULL,
        currency          TEXT    NOT NULL DEFAULT 'BTC',
        expiry            TEXT    NOT NULL,
        strike            REAL    NOT NULL,
        option_type       TEXT    NOT NULL,
        bid_price         REAL,
        ask_price         REAL,
        mid_price         REAL,
        mark_price        REAL,
        mark_iv           REAL,
        underlying_price  REAL,
        open_interest     REAL,
        volume_24h        REAL,
        quote_currency    TEXT,
        -- NEW in V2
        interest_rate     REAL,
        last_price        REAL,
        volume_usd        REAL,
        high_24h          REAL,
        low_24h           REAL,
        price_change_24h  REAL,
        -- derived columns filled by processor
        spread            REAL,
        spread_pct        REAL,
        moneyness         REAL,
        time_to_expiry_d  REAL,
        is_clean          INTEGER DEFAULT 0,
        UNIQUE(snapshot_hour, instrument_name)
    );

    -- ── Greeks (NEW in V2) ────────────────────────────────────────
    -- Fetched per-instrument via get_ticker, stored separately
    -- because not every snapshot instrument gets Greeks (rate-limited)
    CREATE TABLE IF NOT EXISTS greeks (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_hour   TEXT    NOT NULL,
        instrument_name TEXT    NOT NULL,
        currency        TEXT    NOT NULL DEFAULT 'BTC',
        delta           REAL,
        gamma           REAL,
        vega            REAL,
        theta           REAL,
        rho             REAL,
        underlying_price REAL,
        mark_price      REAL,
        UNIQUE(snapshot_hour, instrument_name)
    );

    -- ── Trades (same as V1) ──────────────────────────────────────
    CREATE TABLE IF NOT EXISTS trades (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_ms    INTEGER NOT NULL,
        instrument_name TEXT    NOT NULL,
        currency        TEXT    NOT NULL DEFAULT 'BTC',
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

    -- ── Violations (same as V1 + currency) ───────────────────────
    CREATE TABLE IF NOT EXISTS violations (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_hour        TEXT    NOT NULL,
        currency             TEXT    NOT NULL DEFAULT 'BTC',
        T1                   TEXT    NOT NULL,
        T2                   TEXT    NOT NULL,
        K1                   REAL    NOT NULL,
        K2                   REAL    NOT NULL,
        C_K1_T1              REAL    NOT NULL,
        C_K2_T1              REAL    NOT NULL,
        C_K1_T2              REAL    NOT NULL,
        C_K2_T2              REAL    NOT NULL,
        LHS                  REAL,
        RHS                  REAL,
        violation_pct        REAL,
        maturity_gap         INTEGER,
        strike_gap           REAL,
        strike_gap_pct       REAL,
        moneyness_K1         REAL,
        moneyness_K2         REAL,
        avg_spread_pct       REAL,
        est_total_cost_pct   REAL,
        net_edge             REAL,
        btc_spot             REAL,
        btc_return_1h        REAL,
        btc_return_24h       REAL,
        realized_vol_24h     REAL,
        iv_level_surface     REAL,
        iv_skew              REAL,
        term_structure_slope REAL,
        hour_of_day          INTEGER,
        day_of_week          INTEGER,
        profitability        TEXT DEFAULT 'unclassified',
        UNIQUE(snapshot_hour, currency, T1, T2, K1, K2)
    );

    -- ── Reversions (same as V1) ──────────────────────────────────
    CREATE TABLE IF NOT EXISTS reversions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        violation_id    INTEGER REFERENCES violations(id),
        checked_hour    TEXT    NOT NULL,
        hours_elapsed   REAL    NOT NULL,
        current_viol    REAL,
        reverted        INTEGER DEFAULT 0,
        revert_speed    TEXT,
        UNIQUE(violation_id, checked_hour)
    );

    -- ── Indices ──────────────────────────────────────────────────
    CREATE INDEX IF NOT EXISTS idx_v2_snap_hour     ON snapshots(snapshot_hour);
    CREATE INDEX IF NOT EXISTS idx_v2_snap_instr    ON snapshots(instrument_name);
    CREATE INDEX IF NOT EXISTS idx_v2_snap_clean    ON snapshots(is_clean);
    CREATE INDEX IF NOT EXISTS idx_v2_snap_currency ON snapshots(currency);
    CREATE INDEX IF NOT EXISTS idx_v2_greeks_hour   ON greeks(snapshot_hour);
    CREATE INDEX IF NOT EXISTS idx_v2_greeks_instr  ON greeks(instrument_name);
    CREATE INDEX IF NOT EXISTS idx_v2_trade_ts      ON trades(timestamp_ms);
    CREATE INDEX IF NOT EXISTS idx_v2_trade_instr   ON trades(instrument_name);
    CREATE INDEX IF NOT EXISTS idx_v2_viol_hour     ON violations(snapshot_hour);
    CREATE INDEX IF NOT EXISTS idx_v2_viol_pct      ON violations(violation_pct);
    CREATE INDEX IF NOT EXISTS idx_v2_viol_currency ON violations(currency);
    CREATE INDEX IF NOT EXISTS idx_v2_rev_vid       ON reversions(violation_id);
    """)

    conn.commit()
    log.info("V2 database initialised")


def merge_v1_into_v2(v1_path=None, v2_path=None):
    """
    Migrate all data from V1 DB into V2 DB.
    V1 rows get currency='BTC' and NULL for new columns.
    Safe to run multiple times — uses INSERT OR IGNORE.

    Call this AFTER both DBs exist and v2 is initialised.
    """
    v1 = v1_path or cfg.DB_PATH
    v2 = v2_path or V2_DB_PATH

    if not Path(v1).exists():
        log.error(f"V1 DB not found at {v1}")
        return 0

    log.info(f"Merging V1 ({v1}) → V2 ({v2})")

    conn_v2 = get_v2_connection(v2)
    init_v2_db(conn_v2)

    # Attach V1 as 'old'
    conn_v2.execute(f"ATTACH DATABASE ? AS old", (str(v1),))

    # ── Migrate snapshots ──
    conn_v2.execute("""
        INSERT OR IGNORE INTO snapshots (
            snapshot_hour, instrument_name, currency,
            expiry, strike, option_type,
            bid_price, ask_price, mid_price, mark_price, mark_iv,
            underlying_price, open_interest, volume_24h, quote_currency,
            spread, spread_pct, moneyness, time_to_expiry_d, is_clean
        )
        SELECT
            snapshot_hour, instrument_name, 'BTC',
            expiry, strike, option_type,
            bid_price, ask_price, mid_price, mark_price, mark_iv,
            underlying_price, open_interest, volume_24h, quote_currency,
            spread, spread_pct, moneyness, time_to_expiry_d, is_clean
        FROM old.snapshots
    """)
    snap_count = conn_v2.execute("SELECT changes()").fetchone()[0]
    log.info(f"  Migrated {snap_count:,} snapshots")

    # ── Migrate trades ──
    conn_v2.execute("""
        INSERT OR IGNORE INTO trades (
            timestamp_ms, instrument_name, currency,
            expiry, strike, option_type,
            price, amount, iv, mark_price, index_price,
            direction, trade_id
        )
        SELECT
            timestamp_ms, instrument_name, 'BTC',
            expiry, strike, option_type,
            price, amount, iv, mark_price, index_price,
            direction, trade_id
        FROM old.trades
    """)
    trade_count = conn_v2.execute("SELECT changes()").fetchone()[0]
    log.info(f"  Migrated {trade_count:,} trades")

    # ── Migrate violations ──
    # We need to remap IDs for reversions, so track old→new mapping
    old_violations = conn_v2.execute("""
        SELECT id, snapshot_hour, T1, T2, K1, K2,
               C_K1_T1, C_K2_T1, C_K1_T2, C_K2_T2,
               LHS, RHS, violation_pct,
               maturity_gap, strike_gap, strike_gap_pct,
               moneyness_K1, moneyness_K2,
               avg_spread_pct, est_total_cost_pct, net_edge,
               btc_spot, btc_return_1h, btc_return_24h,
               realized_vol_24h, iv_level_surface, iv_skew,
               term_structure_slope, hour_of_day, day_of_week,
               profitability
        FROM old.violations
    """).fetchall()

    id_map = {}  # old_id → new_id
    for v in old_violations:
        old_id = v["id"]
        conn_v2.execute("""
            INSERT OR IGNORE INTO violations (
                snapshot_hour, currency, T1, T2, K1, K2,
                C_K1_T1, C_K2_T1, C_K1_T2, C_K2_T2,
                LHS, RHS, violation_pct,
                maturity_gap, strike_gap, strike_gap_pct,
                moneyness_K1, moneyness_K2,
                avg_spread_pct, est_total_cost_pct, net_edge,
                btc_spot, btc_return_1h, btc_return_24h,
                realized_vol_24h, iv_level_surface, iv_skew,
                term_structure_slope, hour_of_day, day_of_week,
                profitability
            ) VALUES (
                ?, 'BTC', ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?
            )
        """, (
            v["snapshot_hour"], v["T1"], v["T2"], v["K1"], v["K2"],
            v["C_K1_T1"], v["C_K2_T1"], v["C_K1_T2"], v["C_K2_T2"],
            v["LHS"], v["RHS"], v["violation_pct"],
            v["maturity_gap"], v["strike_gap"], v["strike_gap_pct"],
            v["moneyness_K1"], v["moneyness_K2"],
            v["avg_spread_pct"], v["est_total_cost_pct"], v["net_edge"],
            v["btc_spot"], v["btc_return_1h"], v["btc_return_24h"],
            v["realized_vol_24h"], v["iv_level_surface"], v["iv_skew"],
            v["term_structure_slope"], v["hour_of_day"], v["day_of_week"],
            v["profitability"],
        ))
        # Get the new ID
        new_row = conn_v2.execute("""
            SELECT id FROM violations
            WHERE snapshot_hour = ? AND currency = 'BTC'
              AND T1 = ? AND T2 = ? AND K1 = ? AND K2 = ?
        """, (v["snapshot_hour"], v["T1"], v["T2"], v["K1"], v["K2"])).fetchone()
        if new_row:
            id_map[old_id] = new_row["id"]

    log.info(f"  Migrated {len(id_map):,} violations")

    # ── Migrate reversions (with remapped violation_id) ──
    old_reversions = conn_v2.execute("""
        SELECT violation_id, checked_hour, hours_elapsed,
               current_viol, reverted, revert_speed
        FROM old.reversions
    """).fetchall()

    rev_count = 0
    for r in old_reversions:
        new_vid = id_map.get(r["violation_id"])
        if new_vid is None:
            continue
        conn_v2.execute("""
            INSERT OR IGNORE INTO reversions (
                violation_id, checked_hour, hours_elapsed,
                current_viol, reverted, revert_speed
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (new_vid, r["checked_hour"], r["hours_elapsed"],
              r["current_viol"], r["reverted"], r["revert_speed"]))
        rev_count += 1

    log.info(f"  Migrated {rev_count:,} reversions")

    # Commit BEFORE detach to avoid lock errors on large merges
    conn_v2.commit()

    try:
        conn_v2.execute("DETACH DATABASE old")
    except Exception as e:
        log.warning(f"DETACH failed (non-critical): {e}")

    conn_v2.close()

    log.info("V1 → V2 migration complete")
    return snap_count


def get_v2_stats(conn: sqlite3.Connection) -> dict:
    """Quick health check for V2 DB."""
    stats = {}
    for table in ["snapshots", "trades", "violations", "reversions", "greeks"]:
        row = conn.execute(f"SELECT COUNT(*) as n FROM {table}").fetchone()
        stats[table] = row["n"]

    # Currency breakdown
    cur_rows = conn.execute("""
        SELECT currency, COUNT(*) as n FROM snapshots GROUP BY currency
    """).fetchall()
    stats["by_currency"] = {r["currency"]: r["n"] for r in cur_rows}

    # Option type breakdown
    type_rows = conn.execute("""
        SELECT option_type, COUNT(*) as n FROM snapshots GROUP BY option_type
    """).fetchall()
    stats["by_type"] = {r["option_type"]: r["n"] for r in type_rows}

    # Date range
    row = conn.execute("""
        SELECT MIN(snapshot_hour) as earliest, MAX(snapshot_hour) as latest
        FROM snapshots
    """).fetchone()
    stats["snapshot_range"] = (row["earliest"], row["latest"])

    return stats
