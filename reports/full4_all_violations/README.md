# Full4 All 2024 Violations All 2024 Violations, No ML

This folder is separate from the main pipeline outputs.

Policy:

- Test period: 2024 only
- Strategy: `full4`
- Candidate rule: include every 2024 `full4` violation row
- Trade rule: trade every valid/labeled row; invalid rows are reported as skipped
- Gross exposure: fixed at `$1,000` per trade
- Input: `data/labels/strategy_training_rows.parquet`

| Metric | Value |
|---|---:|
| 2024 candidates | 1,905,494 |
| Valid/traded rows | 493,843 |
| Invalid/skipped rows | 1,411,651 |
| Trades | 493,843 |
| Net PnL | $-27,886,611.76 |
| Average return | -5.65% |
| Median return | -4.20% |
| Win rate | 4.58% |
| Profit factor | 0.2335 |
| Max drawdown | $-27,703,938.04 |
| Daily Sharpe-like ratio | -9.1928 |
| Average actual holding days | 5.35 |
