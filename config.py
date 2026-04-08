"""
config.py — Single source of truth for the entire project.
Change settings here only. Everything else imports from here.

Supports two asset classes:
  - "crypto"  → Deribit (BTC, ETH, SOL options)
  - "equity"  → Databento OPRA (SPX options)

Set ASSET_CLASS below to switch the pipeline.
"""

import os
from pathlib import Path

# Load .env file (keeps secrets out of git)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv not installed — fall back to environment variables

# ── Asset class toggle ─────────────────────────────────────────
# "crypto" = Deribit (BTC/ETH/SOL), "equity" = Databento (SPX)
ASSET_CLASS = os.environ.get("TP2_ASSET_CLASS", "crypto")

# ── Paths ──────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
LOG_DIR = ROOT / "logs"

# Separate DBs per asset class to keep data clean
DB_DIR  = ROOT / "data" / "db"
DB_PATH = DB_DIR / f"tp2_{ASSET_CLASS}.db"

# V2 DB path — enhanced schema with Greeks, multi-currency, full fields
DB_V2_PATH = DB_DIR / f"tp2_{ASSET_CLASS}_v2.db"

# External SSD (1TB, /Volumes/Omkar) — used for large raw data downloads
# and DB backups. Project works fine without it.
_SSD          = Path("/Volumes/Omkar/tp2_data")
SSD_AVAILABLE = _SSD.exists()
SSD_RAW_DIR   = _SSD / "raw"        # Tardis CSV downloads go here
SSD_BACKUP_DB = _SSD / "db" / f"tp2_{ASSET_CLASS}_backup.db"

# Auto-backup DB to SSD when it exceeds this size
DB_BACKUP_THRESHOLD_GB = 1.0

# Local fallback dirs (used when SSD is absent)
RAW_DIR   = SSD_RAW_DIR if SSD_AVAILABLE else ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

for d in [DB_PATH.parent, RAW_DIR, PROCESSED, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if SSD_AVAILABLE:
    (SSD_BACKUP_DB.parent).mkdir(parents=True, exist_ok=True)

# ── Deribit API (crypto only) ─────────────────────────────────
DERIBIT_BASE   = "https://www.deribit.com/api/v2/public"
DERIBIT_CURRENCIES = ["BTC", "ETH", "SOL"]   # professor requested multi-currency
CURRENCY       = "BTC"          # primary currency for live collection
KIND           = "option"
REQUEST_TIMEOUT = 30            # seconds
MAX_RETRIES    = 3
RETRY_DELAY    = 2              # seconds between retries

# ── Databento API (equity only) ────────────────────────────────
# Sign up at https://databento.com ($125 free credit)
# Set DATABENTO_API_KEY in environment or .env file
DATABENTO_API_KEY = os.environ.get("DATABENTO_API_KEY", "")
DATABENTO_DATASET = "OPRA.PILLAR"      # US equity options
SPX_ROOT_SYMBOL   = "SPX"              # S&P 500 index options
DATABENTO_SCHEMA  = "mbp-1"            # top-of-book bid/ask

# ── Collection schedule ────────────────────────────────────────
SNAPSHOT_INTERVAL_MINUTES = 1   # change to 30 for denser data

# ── Historical data ────────────────────────────────────────────
HISTORY_DAYS        = 7        # how far back to pull trades
TRADES_PAGE_SIZE    = 1000     # max per Deribit page
TRADES_CHUNK_DAYS   = 2         # fetch 3 days at a time to avoid timeout

# ── Rolling window (for ML training) ──────────────────────────
# Professor directive: shorter window = higher accuracy
ROLLING_WINDOW_MONTHS  = 3     # train on this many months of data
ROLLING_STEP_DAYS      = 15    # slide forward by 15 days (predict window)
ROLLING_PREDICT_DAYS   = 15    # predict on this many days ahead
CRYPTO_FINETUNE_MONTHS = 1     # shorter window for crypto transfer learning

# ── Data cleaning thresholds ───────────────────────────────────
MIN_BID             = 0.0       # bid must be > this
MAX_SPREAD_PCT      = 0.25      # spread/mid must be < this (25%) — was 0.40, tightened to cut phantoms
MIN_MID_PRICE       = 0.0005    # mid must be >= this (≈$42 at $85K spot) — Filter 10
MIN_OPEN_INTEREST   = 1         # OI must be >= this
MIN_MONEYNESS       = 0.50      # strike/spot must be >= this
MAX_MONEYNESS       = 3.00      # strike/spot must be <= this
MIN_DAYS_TO_EXPIRY  = 1         # skip options expiring today
MAX_IV_PCT          = 500.0     # mark_iv sanity cap
MIN_IV_PCT          = 0.5       # mark_iv sanity floor
CALLS_ONLY          = True      # V1 collector: calls only. V2 collects both, filters in detection.
CALLS_ONLY_V2       = False     # V2 collector: collect both calls and puts

# ── TP2 detection ──────────────────────────────────────────────
# Dynamic expiry ordering: parse any expiry string to a date for sorting.
# No more hardcoded dict — works for any historical period or asset class.
import re as _re
from datetime import datetime as _dt, timezone as _tz

_MONTH_MAP = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
              "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

def parse_expiry_to_date(expiry_str: str):
    """Parse '27MAR26' → datetime for calendar ordering. Works for any year."""
    m = _re.match(r"(\d{1,2})([A-Z]{3})(\d{2})", expiry_str)
    if not m:
        return None
    try:
        return _dt(2000 + int(m.group(3)), _MONTH_MAP[m.group(2)],
                   int(m.group(1)), 8, 0, tzinfo=_tz.utc)
    except (ValueError, KeyError):
        return None

def expiry_sort_key(expiry_str: str):
    """Sort key for any expiry string. Returns (year, month, day) tuple."""
    d = parse_expiry_to_date(expiry_str)
    if d is None:
        return (9999, 12, 31)
    return (d.year, d.month, d.day)

# Legacy hardcoded dict — kept for backwards compat with existing DB data.
# New code should use expiry_sort_key() instead of EXPIRY_ORDER lookups.
EXPIRY_ORDER = {
    "11MAR26":  1,  "12MAR26":  2,  "13MAR26":  3,  "14MAR26":  4,
    "20MAR26":  5,  "21MAR26":  6,  "27MAR26":  7,  "28MAR26":  8,
    "03APR26":  9,  "10APR26": 10,  "24APR26": 11,  "30APR26": 12,
    "29MAY26": 13,  "26JUN26": 14,  "25SEP26": 15,  "25DEC26": 16,
    "26DEC26": 17,  "27MAR27": 18,  "25JUN27": 19,  "31DEC27": 20,
}

MIN_VIOLATION_PCT   = 0.10      # ignore sub-0.1% violations (numerical noise)

# ── Reversion analysis ─────────────────────────────────────────
# Equity (daily EOD): snapshot every 24h, so thresholds scale to trading days
# Crypto (hourly):    snapshot every ~1h, so thresholds scale to hours
if ASSET_CLASS == "equity":
    REVERSION_WINDOWS_H   = [24, 48, 72, 120, 240, 480]  # 1,2,3,5,10,20 trading days
    REVERSION_SUPERFAST_H = 24.0    # 1 trading day
    REVERSION_FAST_H      = 48.0    # 2 trading days
    REVERSION_MEDIUM_H    = 120.0   # 5 trading days
    REVERSION_NEVER_H     = 480.0   # 20 trading days without reversion = "never"
else:
    REVERSION_WINDOWS_H   = [1, 2, 4, 8, 24, 48]
    REVERSION_SUPERFAST_H = 0.5     # < 30 minutes → genuine liquid arb
    REVERSION_FAST_H      = 2.0     # 30 min – 2h  → real but execution-constrained
    REVERSION_MEDIUM_H    = 8.0     # 2h – 8h      → regime-locked / structural
    REVERSION_NEVER_H     = 48.0    # > 48h without reversion = never
# > MEDIUM_H = slow,  > NEVER_H without reversion = never

# ── Logging ───────────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE   = LOG_DIR / "tp2.log"