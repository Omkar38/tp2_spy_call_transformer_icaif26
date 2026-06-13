# Final Pipeline

## 1. Universe

- SPY ETF EOD options from 2019-2024.
- DTE 1-60.
- 0DTE excluded.
- Two-strike TP2/RR2 approximation.
- This is EOD relative-value research, not pure SPX arbitrage.

## 2. Moneyness

Use official raw SPY close first from `data/raw/spy_daily.csv`.

If raw close is unavailable for a date, use a put-call parity backup:

```text
S_proxy = call_mid - put_mid + strike
```

The proxy is built only from liquid, short-dated, near-ATM call/put pairs and is marked with `underlying_source = parity_proxy`.

## 3. Detection

For calls:

```text
A = C(K1,T1)
B = C(K2,T2)
C = C(K1,T2)
D = C(K2,T1)

violation if A*B < C*D
```

For puts:

```text
violation if A*B > C*D
```

Severity is normalized by `A*B + C*D`.

## 4. Strategies

The strategy engine first determines cheap side and rich side.

Calls:

```text
cheap = A*B
rich = C*D
```

Puts:

```text
cheap = C*D
rich = A*B
```

Full4 product-replication quantities:

If cheap side is `A*B`:

```text
long A qty B_mid
long B qty A_mid
short C qty D_mid
short D qty C_mid
```

If cheap side is `C*D`:

```text
long C qty D_mid
long D qty C_mid
short A qty B_mid
short B qty A_mid
```

Subset strategies inherit those directions and quantities:

```text
T1: A and D
T2: B and C
K1: A and C
K2: B and D
```

Exposure and return are normalized per selected strategy using entry-time gross exposure only.

## 5. Labels

Main target is 5 trading day after-cost return.

Short-DTE fix:

```text
exit_date = min(5th future trading day, last available EOD quote date before earliest selected-leg expiration)
```

If no valid pre-expiration exit quote exists, the strategy label is skipped.

Execution:

- Long entry at ask, exit at bid.
- Short entry at bid, exit at ask.
- Commission is `$0.65 * abs(quantity)` per contract per leg per side.
- Multiplier is 100.

## 6. ML Backtest

One gradient-boosted model scores all five strategy rows for each violation. The default is LightGBM; `scripts/run_train.py --model-type catboost_native` trains the CatBoost variant with native categorical handling and the same target.

Backtest rules:

- Choose at most one strategy per violation.
- Apply liquidity/cost gates before accepting trade.
- Apply daily and contract-overlap limits.
- Tune thresholds on 2023 validation only.
- Evaluate once on 2024 test.
