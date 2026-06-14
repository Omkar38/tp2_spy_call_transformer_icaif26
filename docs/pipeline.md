# Current Archive-2 TP2 Pipeline

This document describes the current project pipeline after the Archive-2
dataset reset. It supersedes the older 2019-2024 JSON, put/call, five-strategy
pipeline.

## 1. Research Scope

- Underlying: SPY ETF options.
- Option type: calls only.
- Style caveat: SPY options are American-style and dividend/friction affected.
- Theory base: forward-adjusted call TP2 inequality inspired by
  Glasserman-Li-Pirjol.
- Sample: professor-provided Archive-2 data, 2013-04-01 to 2026-06-04.
- Current stage: EDA and full detector output; final ML trading pipeline comes
  after episode tracking.

This is SPY EOD relative-value research, not a pure SPX European arbitrage
replication.

## 2. Raw Data

Raw input lives locally under:

```text
Archive-2/spy_options_daily/year=*/data.parquet
```

The raw files are not committed to GitHub because several yearly parquet files
exceed GitHub's normal 100 MB file limit. The repository tracks metadata,
summaries, scripts, EDA tables, and reports.

One raw row is an option quote observation. The raw files can contain multiple
observations for the same `quote_date, contract_id`, so ingestion must collapse
raw observations to one quote per contract-day.

## 3. Active Clean Dataset

Active local clean file:

```text
data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet
```

Cleaning rules:

- SPY calls only.
- All moneyness: ITM, ATM, and OTM calls are included.
- DTE 1-90.
- Latest available quote per contract-day.
- Valid bid/ask.
- Positive bid and positive mid.
- Spread percentage no greater than 50%.
- One row per `date, contractID`.

The active clean dataset has 4,912,891 SPY call contract-day rows across 3,256
sessions.

Rebuild command:

```bash
PYTHONPATH=src python3 scripts/run_ingest_archive2.py
```

The script defaults now point to the active full-sample all-call lastquote
dataset. Use `--otm-only` only for a narrower diagnostic universe.

## 4. Forward-Adjusted TP2 Detection

For calls with `T1 < T2` and `K1/F_T1 < K2/F_T2`, the no-violation condition is:

```text
C(K1,T1) C(K2,T2)
>=
C(K1 F_T2/F_T1, T2) C(K2 F_T1/F_T2, T1)
```

The detector defines:

```text
lhs = A_mid * B_mid
rhs = C_mid * D_mid
```

where:

```text
A = C(K1, T1)
B = C(K2, T2)
C = C(K1 adjusted to T2, T2)
D = C(K2 adjusted to T1, T1)
```

The call violation condition is:

```text
lhs < rhs
```

Adjusted strikes are rounded up to the nearest available strike in the target
expiration chain. The default detector threshold is:

```text
normalized_severity >= 0.0001
```

Full detector command:

```bash
PYTHONPATH=src python3 scripts/run_detect_archive2_lastquote_by_year.py --progress
```

Output:

```text
data/features/archive2_lastquote_forward/tp2_call_forward_features_lastquote_YYYY.parquet
data/features/archive2_lastquote_forward/tp2_call_forward_features_lastquote_2013_2026.parquet
data/features/archive2_lastquote_forward/tp2_call_forward_summary_lastquote_2013_2026.json
```

The combined detector output has 2,430,220 daily TP2 violation rows.

## 5. What One Violation Row Means

One clean row is one individual option quote for one contract on one date.

One violation row is one four-contract TP2 rectangle on one date. It is not one
option contract, not one selected ML trade, and not one executed trade.

Example interpretation:

```text
date = 2024-03-15
A = one call at K1,T1
B = one call at K2,T2
C = adjusted K1 call at T2
D = adjusted K2 call at T1
```

If `A_mid * B_mid < C_mid * D_mid`, the detector writes one violation row.

For 2013:

```text
99,129 clean option rows
17,847,362 raw K1/K2 strike pairs across expiration pairs
8,470,629 after forward-order condition
6,886,206 valid four-leg TP2 rectangles
73,619 TP2 violation rows
```

## 6. EDA Reports

Current report outputs:

```text
reports/raw_archive2_eda/archive2_raw_data_eda_report.pdf
reports/eda_professor_presentation_script.pdf
reports/archive2_dataset_ingestion_report.pdf
```

Main EDA tables:

```text
reports/raw_archive2_eda/tables/lastquote_tp2_violation_counts_by_year.csv
reports/raw_archive2_eda/tables/lastquote_tp2_violation_counts_by_month.csv
reports/raw_archive2_eda/tables/lastquote_forward_coverage_by_year.csv
reports/raw_archive2_eda/tables/clean_dataset_version_comparison.csv
```

Regenerate TP2 violation EDA:

```bash
PYTHONPATH=src python3 scripts/generate_lastquote_tp2_violation_eda.py
```

Regenerate raw EDA report source:

```bash
PYTHONPATH=src python3 scripts/generate_archive2_raw_eda_report.py
```

Compile PDF:

```bash
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory reports/raw_archive2_eda \
  reports/raw_archive2_eda/archive2_raw_data_eda_report.tex
```

## 7. Why Episode Tracking Is Next

The current detector output is daily rectangle-level. The same economic
rectangle can violate on adjacent days. Counting each day independently can
inflate the sample size and can leak repeated observations across rolling
train/test splits.

The next pipeline stage should:

1. Build a stable rectangle key from the four option legs.
2. Group consecutive trading-day appearances of the same rectangle into one
   episode.
3. Define episode start date, end date, duration, max severity, and reversion
   outcome.
4. Label episode starts for ML.
5. Use rolling-window ML and backtesting on episode-level observations.

This makes the final research design more defensible because the model learns
from economic events rather than repeated daily rows of the same event.
