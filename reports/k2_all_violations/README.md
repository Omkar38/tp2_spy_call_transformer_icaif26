# K2 All 2024 Violations All 2024 Violations, No ML

This folder is separate from the main pipeline outputs.

Policy:

- Test period: 2024 only
- Strategy: `K2`
- Candidate rule: include every 2024 `K2` violation row
- Trade rule: trade every valid/labeled row; invalid rows are reported as skipped
- Gross exposure: fixed at `$1,000` per trade
- Input: `data/labels/strategy_training_rows.parquet`

| Metric | Value |
|---|---:|
| 2024 candidates | 1,905,494 |
| Valid/traded rows | 797,946 |
| Invalid/skipped rows | 1,107,548 |
| Trades | 797,946 |
| Net PnL | $-136,087,327.99 |
| Average return | -17.05% |
| Median return | -11.88% |
| Win rate | 4.02% |
| Profit factor | 0.1940 |
| Max drawdown | $-146,814,250.64 |
| Daily Sharpe-like ratio | -6.8605 |
| Average actual holding days | 5.28 |
