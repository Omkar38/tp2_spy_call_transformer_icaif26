# T2 All 2024 Violations All 2024 Violations, No ML

This folder is separate from the main pipeline outputs.

Policy:

- Test period: 2024 only
- Strategy: `T2`
- Candidate rule: include every 2024 `T2` violation row
- Trade rule: trade every valid/labeled row; invalid rows are reported as skipped
- Gross exposure: fixed at `$1,000` per trade
- Input: `data/labels/strategy_training_rows.parquet`

| Metric | Value |
|---|---:|
| 2024 candidates | 1,905,494 |
| Valid/traded rows | 1,284,673 |
| Invalid/skipped rows | 620,821 |
| Trades | 1,284,673 |
| Net PnL | $-129,768,003.44 |
| Average return | -10.10% |
| Median return | -3.54% |
| Win rate | 19.07% |
| Profit factor | 0.0900 |
| Max drawdown | $-129,543,379.14 |
| Daily Sharpe-like ratio | -7.0865 |
| Average actual holding days | 6.92 |
