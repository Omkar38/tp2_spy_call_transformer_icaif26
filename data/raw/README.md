# Raw Supplemental Data

Place official raw SPY daily close data here when available:

```text
data/raw/spy_daily.csv
```

Required:

```text
date, close
```

or:

```text
date, raw_close
```

Use raw close, not adjusted close, for moneyness. If this file is absent or a date is missing, the pipeline falls back to a conservative put-call parity proxy and records `underlying_source = parity_proxy`.
