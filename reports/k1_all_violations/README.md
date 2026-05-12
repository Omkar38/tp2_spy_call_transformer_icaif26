# K1 All 2024 Violations All 2024 Violations, No ML

This folder is separate from the main pipeline outputs.

Policy:

- Test period: 2024 only
- Strategy: `K1`
- Candidate rule: include every 2024 `K1` violation row
- Trade rule: trade every valid/labeled row; invalid rows are reported as skipped
- Gross exposure: fixed at `$1,000` per trade
- Input: `data/labels/strategy_training_rows.parquet`

| Metric | Value |
|---|---:|
| 2024 candidates | 1,905,494 |
| Valid/traded rows | 521,974 |
| Invalid/skipped rows | 1,383,520 |
| Trades | 521,974 |
| Net PnL | $-12,812,180.81 |
| Average return | -2.45% |
| Median return | -2.00% |
| Win rate | 30.59% |
| Profit factor | 0.7460 |
| Max drawdown | $-24,889,120.01 |
| Daily Sharpe-like ratio | -1.0198 |
| Average actual holding days | 5.33 |
