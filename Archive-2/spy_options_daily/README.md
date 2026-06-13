# SPY Options Daily EOD Panel (2013–present)

**One row per (trading session, listed SPY option contract)**: end-of-day consolidated NBBO
quote, daily volume, open interest, underlier close and dividend schedule, plus
model-derived implied volatility, greeks, forward and risk-free rate.

- **Underlier:** SPY (SPDR S&P 500 ETF Trust)
- **Universe:** the *entire* listed SPY chain each session — every strike × every expiration
  (0DTE/dailies, weeklies, monthlies, quarterlies, LEAPS). No moneyness or tenor filter.
- **Range:** 2013-04-01 → present (earliest OPRA history available from the vendor)
- **Grain / primary key:** (`quote_date`, `contract_id`); (`quote_date`, `option_symbol`) is an
  equivalent alternate key. No duplicates (maintenance dedups on the primary key, keep-last).
- **Size:** ~55–60M rows, ~1.8 GB Parquet (zstd-free, pyarrow defaults), 14 year partitions.

---

## 1. Layout & loading

```
spy_options_daily/
  year=YYYY/data.parquet   # hive-style partition per calendar year (2013 … 2026)
  README.md                # this file
  run_manifest.json        # config + package versions of the producing run
  skipped_days.txt         # sessions known missing but recoverable (pending backfill); may be absent when clean
  known_gaps.txt           # sessions the vendor itself has no data for (permanent), one ISO date per line
```

```python
import pandas as pd
df = pd.read_parquet("spy_options_daily")            # all years; hive partitioning adds a 'year' column
df24 = pd.read_parquet("spy_options_daily/year=2024/data.parquet")   # single year (no 'year' column inside files)
```

```sql
-- duckdb
SELECT * FROM read_parquet('spy_options_daily/year=*/data.parquet', hive_partitioning=true);
```

```python
import polars as pl
lf = pl.scan_parquet("spy_options_daily/year=*/data.parquet")
```

The `year` column is **not stored inside the files**; it materializes from the directory name
when reading with hive partitioning.

---

## 2. Column dictionary

| column | logical type | source | description |
|---|---|---|---|
| `quote_date` | date | — | Trading session (US/Eastern calendar date). |
| `contract_id` | int | Databento | OPRA `instrument_id` for the contract (stable per contract). |
| `option_symbol` | str (21 chars) | Databento | OSI symbol: `ROOT(6, space-padded) + YYMMDD + C/P + strike×1000 (8 digits)`, e.g. `SPY   270331C00735000`. |
| `expiration_date` | date | parsed from OSI | Contract expiration. |
| `strike` | float, USD | parsed from OSI | Strike price (OSI integer / 1000). |
| `option_type` | `'C'`/`'P'` | parsed from OSI | Call / put. |
| `option_multiplier` | int | constant | Always **100** (standard equity-option deliverable; SPY has never split). Set explicitly because the OPRA definition field carries an INT32_MAX null sentinel. |
| `bid`, `ask` | float, USD/share | Databento `cbbo-1m` | Consolidated (NBBO across ~17 OPRA venues) closing quote: the last 1-minute BBO bar with label ≤ 16:00:00 ET within the [15:58, 16:00] ET window. Premium per contract = price × 100. One side may be NaN (one-sided market); rows missing **both** sides are dropped. |
| `mid` | float, USD/share | computed | `(bid + ask)/2`; NaN if either side missing. |
| `bid_size`, `ask_size` | float (int-valued) | Databento `cbbo-1m` | Contracts at the NBBO on each side. |
| `quote_time` | timestamp (tz-naive, **UTC instant**) | Databento `cbbo-1m` | Label of the closing bar used. Bars are labeled at minute *end*: the bar covering 15:59–16:00 ET is labeled 16:00 ET = 20:00 UTC (EDT) / 21:00 UTC (EST). |
| `volume` | float (int-valued) | Databento `ohlcv-1d` | Contracts traded that session, all venues. **0 = no trades** (filled; absence of a daily bar means the contract did not print). |
| `open_interest` | float, NaN-able | Databento `statistics` | OPRA open-interest statistic keyed to the session via the feed's reference date. OPRA disseminates OI each morning reflecting positions after the prior settlement; if you need a strict as-of-close convention, shift by one session to taste. NaN = not reported. |
| `spy_close` | float, USD | Yahoo Finance | **Raw, unadjusted** SPY daily close (`auto_adjust=False`). Stooq is the automatic fallback source. |
| `next_ex_div_date` | date, NaT-able | Yahoo Finance | Next SPY ex-dividend date **on or after** `quote_date`, from the realized SPDR distribution series. ⚠ Look-ahead caveat below. |
| `next_div_amount` | float, USD/share | Yahoo Finance | Cash amount of that next distribution. |
| `forward` | float, USD | computed | Parity-implied forward `F` for this (`quote_date`, `expiration_date`). |
| `risk_free_rate` | float (cont. comp.) | computed | `r = −ln(D)/T` from the parity-implied discount factor `D`. Rows where parity could not be fit carry the fallback **exactly 0.04** (filter on equality to exclude them). |
| `implied_vol` | float, NaN-able | computed | Black–Scholes IV inverted from `mid`. NaN when `mid ≤` model intrinsic, `T ≤ 0` (expiry-day at the close), either quote side missing, or the solver leaves [1e-4, 5]. |
| `delta` | float | computed | BS spot delta `cp·e^{−qT}·N(cp·d1)` at `implied_vol`. |
| `theta` | float, USD/day | computed | BS theta **per calendar day** (annual/365), includes the dividend-yield term. |
| `vega` | float, USD per 1.00 vol | computed | BS vega per unit (1.00 = 100 vol points) change in σ. Divide by 100 for per-vol-point. |

Physical dtypes may surface as float64 where NaN is possible (sizes, OI, volume); check `df.dtypes`.

---

## 3. Sources

| What | Source | Detail |
|---|---|---|
| All option market data (quotes, sizes, volume, OI, symbols/strikes/expiries) | **Databento, dataset `OPRA.PILLAR`** | Schemas: `cbbo-1m` (consolidated BBO, the only NBBO schema spanning 2013→present), `ohlcv-1d`, `statistics`. Chain selected via parent symbology `SPY.OPT` (whole chain, no filter). Strike/expiry/type are decoded from the OSI symbol on each record — no separate reference fetch. |
| SPY close, dividends, trading calendar | **Yahoo Finance** (`yfinance`, `auto_adjust=False`); Stooq fallback for the close | The session calendar driving the per-day requests is the SPY daily-bar index. Databento carries no corporate actions, hence the external source. |
| `mid`, `multiplier`, IV, greeks, `forward`, `risk_free_rate` | **Computed** (see §4) | Model outputs, not market data. Recompute under your own conventions if needed. |

---

## 4. Methodology (precise definitions)

**EOD quote selection.** For each session, the producer requests `cbbo-1m` for
[16:00−2min, 16:00:00] ET (window configurable) and keeps, per `contract_id`, the **last** bar
labeled ≤ 16:00:00 ET. Consequence: a contract appears on a given day **only if its NBBO
updated inside that window** — deeply illiquid far-OTM/long-dated lines can be absent on
quiet days. This is a quote (NBBO) panel, **not** settlement prices and not trade prices.

**Forward & rate (per `quote_date` × `expiration_date`).** Cross-strike OLS of
`(C_mid − P_mid)` on `K` over strikes with both legs quoted:
put-call parity gives slope `= −D` and intercept `= D·F`, hence discount factor `D`,
`F = intercept/D`, `r = −ln(D)/T`, implied carry `q = r − ln(F/S)/T` with `S = spy_close`.
Day count **ACT/365**: `T = (expiration_date − quote_date)/365`. Fit validity requires ≥2
paired strikes, `0 < D ≤ 1`, `F > 0`; otherwise the row falls back to `r = 0.04` and
`q = 4·next_div_amount/S` (0 if unknown), `F = S·e^{(r−q)T}`.

**IV & greeks.** Vectorized Newton on the Black–Scholes price (init σ=0.30, tol 1e-8,
bounds [1e-4, 5]) against `mid`, using `(S, K, T, r, q)` as above; greeks evaluated in closed
form at the solved σ. Conventions: spot delta with continuous dividend yield; theta per
calendar day; vega per 1.00 of σ.

---

## 5. Coverage snapshot

Verified counts as of **2026-06-04** (run `python ../validate_and_backfill.py` for live numbers;
`year=2026` was added 2026-06-05 and backfill of storm-skipped days may still be in flight):

| year | rows | sessions | distinct contracts |
|---|---|---|---|
| 2013 (from Apr 1) | 1,090,018 | 189 | 9,650 |
| 2014 | 1,741,109 | 249 | 14,376 |
| 2015 | 2,083,775 | 250 | 16,900 |
| 2016 | 2,406,752 | 251 | 21,617 |
| 2017 | 2,441,126 | 249 | 23,641 |
| 2018 | 3,332,563 | 248 | 36,893 |
| 2019 | 3,836,239 | 249 | 38,817 |
| 2020 | 6,143,825 | 251 | 50,014 |
| 2021 | 6,107,956 | 250 | 52,455 |
| 2022 | 7,062,256 | 251 | 60,966 |
| 2023 | 5,746,810 | 248 | 37,850 |
| 2024 | 6,044,559 | 236* | 33,971 |
| 2025 | 7,119,099 | 232* | 27,032 |
| 2026 (Jan–Jun) | present | ~100 | — |

\* below the NYSE calendar pending backfill — see `skipped_days.txt`. Per-day chain size grows
from ~2,700 contracts (2013) to ~14,600 (2026, 0DTE era).

**Session accounting:** a calendar session can be absent for exactly three reasons:
1. **Permanent vendor gap** — listed in `known_gaps.txt` (currently one: **2021-06-18**, which
   Databento's dataset-condition endpoint flags `missing`; no OPRA data exists for it).
2. **Pending backfill** — listed in `skipped_days.txt` (transient gateway failures during the
   bulk pull; recoverable, vendor condition `available`).
3. **No closing-window records** — a handful of half-days/odd sessions where the feed had no
   bars in the final minutes (e.g. some Christmas Eves, post-Thanksgiving Fridays).

---

## 6. Caveats for research use

- **Quotes, not settlements.** Marks derived from `mid` inherit NBBO microstructure at the close.
- **`next_ex_div_date`/`next_div_amount` contain look-ahead.** They come from the *realized*
  distribution series, so on dates before a dividend's declaration they embed information that
  was not yet public. For strict point-in-time backtests, reconstruct the dividend expectation
  yourself; for pricing/parity work the fields are fine.
- **Fallback-rate rows.** `risk_free_rate == 0.04` exactly ⇒ parity fit unavailable for that
  (date, expiry); its `forward`, IV and greeks use the fallback carry. Mostly thin LEAPS/odd expiries.
- **IV coverage.** NaN IV concentrates in deep-ITM (mid below intrinsic), one-sided quotes, and
  expiry-day (`T=0`) rows. This is by construction, not data loss.
- **OI timing.** See column note; decide your as-of convention and shift if needed.
- **Survivorship/adjustments.** Not applicable: SPY options only, no splits, standard 100
  multiplier throughout; no adjusted (non-standard) contracts in the chain.
- **Timezones.** `quote_date` is an ET session date; `quote_time` is a UTC instant.

---

## 7. Provenance & maintenance

Produced by `download_spy_options.py` (same folder as the dataset's parent, `TP2/`), which is
deterministic given its config; `run_manifest.json` records the exact config and package
versions of the producing run. Key behaviors: per-year Parquet writes with resume
(existing `year=YYYY` partitions are skipped; `--force` rebuilds), per-day request isolation
(a failed session is recorded, never aborts a year), capped-jitter retries.

Companion tool `validate_and_backfill.py`:

```bash
python validate_and_backfill.py                  # coverage report
python validate_and_backfill.py --audit          # rebuild skipped_days/known_gaps from ground truth:
                                                 #   (SPY calendar ∩ vendor 'available') − sessions present
python validate_and_backfill.py --audit --backfill --max-workers 1   # re-pull exactly the missing sessions
```

`--audit` is authoritative; `skipped_days.txt` is otherwise an append-log and can drift.
Backfill merges into existing partitions with primary-key dedup (keep-last) and never creates
a year partition from scratch (absent years require the full downloader).

Operational notes (vendor): requires a Databento OPRA subscription; under the producing
account the daily/reference schemas were free for the full history, the most recent ~12 months
of `cbbo-1m` were free, and deep-history closing windows were metered (~$60 total for
2013→2025). The historical gateway is load-sensitive — keep request concurrency ≤3 for large
pulls and expect occasional 504 waves; the tooling retries and records rather than crashes.

*Document generated 2026-06-05.*
