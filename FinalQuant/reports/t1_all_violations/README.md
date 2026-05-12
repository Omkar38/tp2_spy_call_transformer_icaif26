# T1 All 2024 Violations, No ML

This folder is separate from the main pipeline outputs. It does not modify the
existing model, backtest code, reports, or selected-trade artifacts.

Policy:

- Test period: 2024 only
- Strategy: `T1`
- Candidate rule: include every 2024 `T1` violation row
- Trade rule: trade every valid/labeled 2024 `T1` row; invalid rows are reported as skipped because no valid entry/exit PnL can be computed
- ML selection: none
- Gross exposure: fixed at `$1,000` per trade
- Input: `data/labels/strategy_training_rows.parquet`

Results:

| Metric | T1 all 2024 violations |
|---|---:|
| 2024 T1 violation candidates | 1,905,494 |
| Valid/traded 2024 T1 rows | 499,801 |
| Invalid/skipped 2024 T1 rows | 1,405,693 |
| Trades | 499,801 |
| Net PnL | $3,591,186.72 |
| Average return | 0.72% |
| Median return | -3.63% |
| Win rate | 28.95% |
| Profit factor | 1.0899 |
| Max drawdown | -$7,242,614.43 |
| Daily Sharpe-like ratio | 0.3043 |
| Average actual holding days | 5.35 |

Benchmark from the current ML-selected fixed-gross backtest:

| Metric | ML selector |
|---|---:|
| Trades | 6,132 |
| Net PnL | $524,095.82 |
| Average return | 8.55% |
| Median return | -1.51% |
| Win rate | 45.58% |
| Profit factor | 1.6530 |
| Max drawdown | -$134,245.33 |
| Daily Sharpe-like ratio | 2.31 |

The T1-all policy earns more total dollars only because it takes far more trades.
Its average trade quality, win rate, profit factor, and drawdown are much worse
than the ML-selected fixed-gross benchmark.
