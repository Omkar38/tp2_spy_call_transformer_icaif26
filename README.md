# AMS 520 — TP2 Options Project
**BTC Options TP2 Violation Detection, Reversion Analysis & ML Ranking**

---

## Project Structure
```
tp2_project/
├── config.py              ← ALL settings live here
├── main.py                ← CLI entry point
├── notebook.py            ← Jupyter-friendly API
│
├── collector/
│   ├── deribit.py         ← Deribit API wrapper (all network calls)
│   ├── historical.py      ← Backfill 30-day trade history
│   └── live.py            ← Hourly scheduler (background thread)
│
├── processor/
│   ├── cleaner.py         ← 8-step data cleaning pipeline
│   └── features.py        ← Regime features (BTC vol, IV skew, etc.)
│
├── detector/
│   └── tp2.py             ← TP2 violation detection + reversion labeling
│
├── storage/
│   └── db.py              ← SQLite schema + all DB operations
│
└── data/
    └── db/tp2.db          ← Single database file (shareable with team)
```

---

## Quickstart (Terminal)

```bash
# 1. First time — backfill 30 days of history + start live collection
python main.py --mode full --days 30

# 2. Just fetch one snapshot right now (good for testing)
python main.py --mode once

# 3. Check database health
python main.py --mode status

# 4. Start live collection only (if history already done)
python main.py --mode live

# 5. Change collection interval to 30 minutes
python main.py --mode live --interval 30
```

---

## Quickstart (Jupyter Notebook)

```python
# Cell 1 — Setup (run once per session)
from notebook import *
conn = setup()

# Cell 2 — Backfill history (run ONCE, takes ~10 min)
backfill(days=30)

# Cell 3 — Fetch snapshot right now + detect violations
summary = snapshot_now()

# Cell 4 — Start hourly background collection
start_live(interval_minutes=60)

# Cell 5 — Check database
status()

# Cell 6 — Load violations as DataFrame
df_viols = get_violations_df()
df_viols.head(20)

# Cell 7 — Load reversion data (Phase II)
df_rev = get_reversions_df()

# Cell 8 — Summary table by expiry pair
violation_summary()

# Cell 9 — Stop scheduler when done
stop_live()
```

---

## What Each Phase Needs

| Phase | What you need | How long |
|-------|--------------|----------|
| Phase I | 1 snapshot + violations detected + cleaning stats | Done now |
| Phase II | 7-30 days of hourly snapshots + reversion labels | 7+ days |
| Phase III | 90 days + ML features + train/test split | 90 days |

---

## Data Flow

```
Deribit API
    │
    ▼
collector/historical.py  ──── backfills 30 days of trades
collector/live.py        ──── fetches snapshot every hour
    │
    ▼
storage/db.py            ──── saves to SQLite (snapshots + trades tables)
    │
    ▼
processor/cleaner.py     ──── applies 8 filters, marks is_clean=1
processor/features.py    ──── computes BTC return, IV skew, realized vol
    │
    ▼
detector/tp2.py          ──── detects violations + checks reversions
    │
    ▼
storage/db.py            ──── saves to violations + reversions tables
    │
    ▼
notebook.py              ──── load as DataFrames for analysis + ML
```

---

## Key Settings (config.py)

| Setting | Default | Change to |
|---------|---------|-----------|
| HISTORY_DAYS | 30 | 90 for full ML dataset |
| SNAPSHOT_INTERVAL_MINUTES | 60 | 30 for denser data |
| MAX_SPREAD_PCT | 0.40 | 0.30 to be stricter |
| CALLS_ONLY | True | False to add puts (Phase II) |
| MIN_VIOLATION_PCT | 0.10 | 0.5 to remove tiny violations |

---

## Requirements
- Python 3.10+
- `requests` — pip install requests
- `pandas` — pip install pandas
- `sqlite3` — built into Python (no install needed)

No other dependencies. SQLite works out of the box.
