# EOD SPY TP2/RR2 ML Strategy Selection

This repo implements an end-of-day SPY ETF option research pipeline inspired by TP2/RR2 option-price violations.

Important scope warning: the paper in `Literature/5392317.pdf` studies SPX European index options. The available archive here is SPY ETF options, which are American-style and dividend/friction affected. Results should be interpreted as SPY EOD relative-value research, not pure SPX arbitrage replication.

## Data

Input archive:

```text
archive/spy_options_data_19.json
archive/spy_options_data_20.json
archive/spy_options_data_21.json
archive/spy_options_data_22.json
archive/spy_options_data_23.json
archive/spy_options_data_24.json
```

Optional official raw SPY close file:

```text
data/raw/spy_daily.csv
```

Expected columns are `date` and `close` or `raw_close`. Raw close is used for moneyness. If unavailable for a date, the pipeline falls back to a conservative put-call parity proxy.

## Pipeline

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python scripts/run_ingest.py
PYTHONPATH=src python scripts/run_detect.py
PYTHONPATH=src python scripts/run_labels.py
PYTHONPATH=src python scripts/run_train.py
PYTHONPATH=src python scripts/run_backtest.py
```

For a small smoke test, pass a single year to ingestion:

```bash
PYTHONPATH=src python scripts/run_ingest.py --years 2024 --output data/processed/options_clean_2024.parquet
```

## Model Design

Each detected violation expands into five strategy rows:

```text
full4
T1
T2
K1
K2
```

One LightGBM regressor predicts after-cost return with `strategy` as a categorical feature. The backtest scores all five rows and chooses at most one strategy per violation after liquidity/cost gates.

## Splits

```text
Train: 2019-2022
Validation: 2023
Test: 2024
```

Thresholds and portfolio limits must be tuned only on validation, never on test.
