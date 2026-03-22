"""
config.py — Single source of truth for the entire project.
Change settings here only. Everything else imports from here.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
LOG_DIR = ROOT / "logs"

# DB always lives locally (fast NVMe, always available)
# Backed up to SSD automatically once it exceeds DB_BACKUP_THRESHOLD_GB
DB_PATH = ROOT / "data" / "db" / "tp2.db"

# External SSD (1TB, /Volumes/Omkar) — used for large raw data downloads
# and DB backups. Project works fine without it.
_SSD          = Path("/Volumes/Omkar/tp2_data")
SSD_AVAILABLE = _SSD.exists()
SSD_RAW_DIR   = _SSD / "raw"        # Tardis CSV downloads go here
SSD_BACKUP_DB = _SSD / "db" / "tp2_backup.db"

# Auto-backup DB to SSD when it exceeds this size
DB_BACKUP_THRESHOLD_GB = 1.0

# Local fallback dirs (used when SSD is absent)
RAW_DIR   = SSD_RAW_DIR if SSD_AVAILABLE else ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

for d in [DB_PATH.parent, RAW_DIR, PROCESSED, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if SSD_AVAILABLE:
    (SSD_BACKUP_DB.parent).mkdir(parents=True, exist_ok=True)

# ── Deribit API ────────────────────────────────────────────────
DERIBIT_BASE   = "https://www.deribit.com/api/v2/public"
CURRENCY       = "BTC"          # BTC or ETH
KIND           = "option"
REQUEST_TIMEOUT = 30            # seconds
MAX_RETRIES    = 3
RETRY_DELAY    = 2              # seconds between retries

# ── Collection schedule ────────────────────────────────────────
# Recommendation: 1 hour for academic project.
# 5 min would hit rate limits and generate 288 snapshots/day.
# 1 hour = 24 snapshots/day = clean and manageable.
SNAPSHOT_INTERVAL_MINUTES = 1   # change to 30 for denser data

# ── Historical data ────────────────────────────────────────────
# Deribit trades endpoint can go back ~90 days depending on instrument.
# We fetch in chunks to avoid timeouts.
HISTORY_DAYS        = 7        # how far back to pull trades
TRADES_PAGE_SIZE    = 1000     # max per Deribit page
TRADES_CHUNK_DAYS   = 2         # fetch 3 days at a time to avoid timeout

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
CALLS_ONLY          = True      # start with calls, add puts in Phase II

# ── TP2 detection ──────────────────────────────────────────────
# Expiry calendar order for BTC options (extend as new expiries appear)
EXPIRY_ORDER = {
    "11MAR26":  1,  "12MAR26":  2,  "13MAR26":  3,  "14MAR26":  4,
    "20MAR26":  5,  "21MAR26":  6,  "27MAR26":  7,  "28MAR26":  8,
    "03APR26":  9,  "10APR26": 10,  "24APR26": 11,  "30APR26": 12,
    "29MAY26": 13,  "26JUN26": 14,  "25SEP26": 15,  "25DEC26": 16,
    "26DEC26": 17,  "27MAR27": 18,  "25JUN27": 19,  "31DEC27": 20,
}

MIN_VIOLATION_PCT   = 0.10      # ignore sub-0.1% violations (numerical noise)

# ── Reversion analysis ─────────────────────────────────────────
REVERSION_WINDOWS_H = [1, 2, 4, 8, 24, 48]   # hours to check reversion at

# ── Logging ───────────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE   = LOG_DIR / "tp2.log"