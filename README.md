# ML-Based Trading of TP2-Inspired Violations in SPY Call Options

This repository contains the current Archive-2 research pipeline for detecting
forward-adjusted TP2-inspired violations in SPY ETF call options. The project is
intended for an ICAIF 2026-style empirical finance/ML research workflow.

The project uses the Glasserman-Li-Pirjol TP2 call inequality as theoretical
motivation, but it is not a direct replication of their SPX European-option
study. Our data is SPY ETF options, which are American-style and dividend/friction
affected. Current results should be interpreted as SPY EOD relative-value
research.

## Current Scope

- Underlying: SPY ETF options.
- Option side: calls only.
- Sample: professor-provided Archive-2 data, 2013-04-01 to 2026-06-04.
- Clean quote convention: latest available quote per contract-day.
- Filters: DTE 1-90, valid bid/ask, positive bid, positive mid, spread percentage
  no greater than 50%.
- Detector: forward-adjusted call TP2 inequality with rounded adjusted strikes.
- Current stage: EDA plus full detector output. ML episode tracking and final
  trading backtest are next steps.

## Repository Data Policy

Large raw and derived parquet files are intentionally not pushed to GitHub
because normal GitHub blobs are limited to 100 MB and Git LFS is not installed
in this workspace.

Tracked in Git:

- Source code.
- Archive metadata and sample files.
- Summary JSON files.
- EDA CSV tables and figures.
- Current PDF reports.

Local-only:

- `Archive-2/spy_options_daily/year=*/data.parquet`
- `data/processed/*.parquet`
- `data/features/**/*.parquet`

The active local files used for the latest EDA run are:

```text
data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet
data/features/archive2_lastquote_forward/tp2_call_forward_features_lastquote_2013_2026.parquet
```

## Key Results from Current EDA

- Active clean rows: 4,912,891 SPY call contract-day quotes.
- Trading sessions: 3,256.
- Full TP2 feature rows: 2,430,220 daily four-contract violation rectangles.
- Detector validation:
  - zero rows with `lhs >= rhs`
  - zero rows with invalid expiration ordering
  - zero duplicate `violation_id` values

Important interpretation: one violation row is one four-contract TP2 rectangle
on one quote date, not one option contract and not one executed trade.

## Main Reports

```text
reports/raw_archive2_eda/archive2_raw_data_eda_report.pdf
reports/eda_professor_presentation_script.pdf
reports/archive2_dataset_ingestion_report.pdf
```

The professor presentation script is the shortest meeting-facing summary. The
raw EDA report contains the detailed data diagnostics, tables, and charts.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run commands with:

```bash
PYTHONPATH=src python3 <script>
```

## Rebuild the Active Clean Dataset

The current active dataset uses all SPY calls, all moneyness, DTE 1-90, and the
latest available raw quote per contract-day.

```bash
PYTHONPATH=src python3 scripts/run_ingest_archive2.py \
  --archive2-root Archive-2/spy_options_daily \
  --years 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 \
  --include-atm-itm \
  --use-last-available-quote \
  --output data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet \
  --summary-output data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026_summary.json
```

## Run Full TP2 Detection

```bash
PYTHONPATH=src python3 scripts/run_detect_archive2_lastquote_by_year.py \
  --output-dir data/features/archive2_lastquote_forward \
  --combined-output data/features/archive2_lastquote_forward/tp2_call_forward_features_lastquote_2013_2026.parquet \
  --summary-output data/features/archive2_lastquote_forward/tp2_call_forward_summary_lastquote_2013_2026.json \
  --n-jobs 1 \
  --progress
```

This writes one parquet file per year plus a combined local parquet file and a
tracked summary JSON.

## Regenerate EDA Tables and Reports

```bash
PYTHONPATH=src python3 scripts/generate_lastquote_tp2_violation_eda.py
PYTHONPATH=src python3 scripts/generate_archive2_raw_eda_report.py
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory reports/raw_archive2_eda \
  reports/raw_archive2_eda/archive2_raw_data_eda_report.tex
```

Professor-facing script PDF:

```bash
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory reports \
  reports/eda_professor_presentation_script.tex
```

## Current Pipeline Files

Core Archive-2 modules:

```text
src/tp2rr2/archive2_ingest.py
src/tp2rr2/detect_forward_tp2.py
src/tp2rr2/features.py
```

Current runner/report scripts:

```text
scripts/run_ingest_archive2.py
scripts/run_detect_archive2_lastquote_by_year.py
scripts/generate_lastquote_tp2_violation_eda.py
scripts/generate_archive2_raw_eda_report.py
```

Future ML/backtest work should build on episode-level violations rather than
raw daily violation rows.

## Next Research Step

The next structural step is episode tracking:

1. Build a stable rectangle key from the four option legs.
2. Group repeated same-rectangle daily violations into one episode.
3. Label episode starts by reversion/trading outcome.
4. Train ML on rolling windows using episode-level observations.
5. Backtest selected trades with transaction costs and liquidity constraints.

