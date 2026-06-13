#!/usr/bin/env python3
"""
download_spy_options.py
=======================

Pull the *full available daily history* of SPY (ETF) options from Databento
``OPRA.PILLAR`` and assemble a per-day option-quote panel matching the fields in
``Data Description.docx`` — extended to the entire available history
(2013-04-01 -> present) rather than only Jan-2024+.

Field -> source map
-------------------
    quote_date, bid, ask, bid_size, ask_size   OPRA.PILLAR / cbbo-1m  (EOD NBBO @ 16:00 ET)
    volume                                      OPRA.PILLAR / ohlcv-1d
    open_interest                               OPRA.PILLAR / statistics (StatType.OPEN_INTEREST)
    option_symbol, expiration, strike, right    parsed from the OPRA OSI symbol on each record
    option_multiplier                           100 (SPY equity options; never split -> no adjusted contracts)
    spy_close                                    yfinance SPY 'Close' (auto_adjust=False -> raw, unadjusted)
    spy ex-div date / amount                     yfinance SPY 'Dividends' (= SPDR distribution history)
    implied_vol, delta, theta, vega,            COMPUTED. Per (quote_date, expiration) the discount
    forward, risk_free_rate                      factor D and forward F are implied from put-call parity
                                                 (C - P) = D * (F - K) by a cross-strike least-squares fit;
                                                 r = -ln(D)/T, q = r - ln(F/S)/T. IV is inverted from
                                                 Black-Scholes on the mid; delta/theta/vega are closed form.
                                                 Self-contained: no external rate feed required.

Why cbbo-1m
-----------
It is the only *consolidated* NBBO schema on OPRA.PILLAR that spans the full
history (cmbp-1 starts 2023-03-28, cbbo-1s 2025-02-20). We pull only a short
window around 16:00 ET each session and keep the last bar <= the close, so the
volume is a tiny fraction of the intraday tape. With the account's OPRA
subscription, streaming these schemas is included (metadata cost estimate = $0).

Usage
-----
    export DATABENTO_API_KEY=db-...                  # or pass --api-key
    python download_spy_options.py --out ./spy_options_daily
    python download_spy_options.py --dry-run         # print ranges/record counts/cost, download nothing

Requirements
------------
    databento>=0.34  pandas>=2.0  numpy>=1.24  scipy>=1.10  pyarrow>=12  yfinance>=0.2

NOTE: This script is intentionally NOT executed by its author. Review the config
and run a --dry-run first.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import io
import json
import logging
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Third-party clients are imported lazily inside the functions that need them so
# that --help / --dry-run argument parsing works without every dependency present.

LOG = logging.getLogger("spy_options")
ET = ZoneInfo("America/New_York")
UTC = timezone.utc


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    """Immutable run configuration. Everything reproducibility-relevant lives here."""

    api_key: str
    out_dir: Path
    start: date = date(2013, 4, 1)          # earliest OPRA.PILLAR history for these schemas
    end: date = field(default_factory=lambda: datetime.now(ET).date())

    dataset: str = "OPRA.PILLAR"
    parent_symbol: str = "SPY.OPT"          # parent symbology -> every SPY option

    # End-of-day quote window (US/Eastern). We request [close - window, close] and
    # keep the last cbbo-1m bar with ts_event <= close for each instrument.
    market_close: dtime = dtime(16, 0)      # SPY equity options close 16:00 ET
    # Closing window pulled per session. 2 min captures the 16:00 bar plus one fallback bar
    # cheaply; widen (e.g. 5) for better coverage of illiquid contracts at higher cost.
    eod_window_min: int = 2

    option_multiplier: int = 100            # SPY never split; OPRA definition's contract_multiplier is the
                                            # INT32_MAX null sentinel (verified live), so we set 100 explicitly

    # Equity (SPY underlier) source: 'yfinance' covers the full 2013->present span
    # with raw close + dividends in one call. 'databento' is offered for the period
    # Databento equities actually cover (typically 2018+).
    equity_source: str = "yfinance"
    equity_dataset: str = "XNAS.BASIC"      # used only when equity_source == 'databento'

    day_count: float = 365.0                # ACT/365 for T
    r_fallback: float = 0.04                # used only when parity cannot be solved for a (date, expiry)
    iv_lower: float = 1e-4
    iv_upper: float = 5.0
    iv_init: float = 0.30
    iv_max_iter: int = 100
    iv_tol: float = 1e-8

    seed: int = 0
    max_retries: int = 5
    retry_backoff_s: float = 2.0
    max_workers: int = 8                    # concurrent per-session EOD quote requests
    log_level: str = "INFO"

    def years(self) -> list[int]:
        return list(range(self.start.year, self.end.year + 1))


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def retry(max_retries: int, backoff_s: float) -> Callable:
    """Exponential-backoff retry decorator for transient API/network errors."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - intentional broad transient catch
                    last = exc
                    # Capped exponential backoff with jitter so concurrent workers don't
                    # retry in lockstep and re-overload the gateway (thundering herd).
                    wait = min(backoff_s * (2 ** (attempt - 1)), 30.0) * (0.7 + 0.6 * random.random())
                    LOG.warning("%s attempt %d/%d failed: %s -- retrying in %.1fs",
                                fn.__name__, attempt, max_retries, exc, wait)
                    time.sleep(wait)
            assert last is not None
            raise last

        return wrapper

    return deco


def month_starts(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Yield [month_start, next_month_start) half-open windows covering [start, end]."""
    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
        yield max(cur, start), min(nxt, end + timedelta(days=1))
        cur = nxt


def iso_utc(d: date, t: dtime) -> str:
    """Combine an Eastern wall-clock (date, time) and render as a UTC RFC3339 string."""
    return datetime.combine(d, t, ET).astimezone(UTC).isoformat()


# --------------------------------------------------------------------------- #
# Databento client + raw pulls
# --------------------------------------------------------------------------- #
def make_client(cfg: Config):
    import databento as db

    return db.Historical(key=cfg.api_key)


def _to_df(store, *, price_type: str = "float") -> pd.DataFrame:
    """DBNStore -> tidy DataFrame with ts_event as a column and mapped raw symbols."""
    df = store.to_df(price_type=price_type, pretty_ts=True, map_symbols=True)
    if df.index.name is not None:
        df = df.reset_index()
    return df


@retry(5, 2.0)
def _get_range(client, **kwargs):
    return client.timeseries.get_range(**kwargs)


def parse_osi(symbol: pd.Series) -> pd.DataFrame:
    """
    Vectorized parse of OPRA OSI symbols -> (expiration, right, strike).

    OSI layout (right-anchored, robust to root padding):
        <root> <YYMMDD> <C|P> <strike*1000, 8 digits>

    Parameters
    ----------
    symbol : Series[str]  shape (N,)

    Returns
    -------
    DataFrame[expiration: date, option_type: {'C','P'}, strike: float]  shape (N, 3)
    """
    s = symbol.astype("string").str.replace(" ", "", regex=False)
    strike = s.str[-8:].astype("int64") / 1000.0
    right = s.str[-9]
    expiration = pd.to_datetime(s.str[-15:-9], format="%y%m%d").dt.date
    return pd.DataFrame({"expiration": expiration, "option_type": right, "strike": strike})


def _fetch_day_quotes(cfg: Config, d: date, win_start: dtime, win_end: dtime) -> pd.DataFrame:
    """One session's EOD NBBO, reduced to the last bar (<= 16:00 ET) per contract.

    Uses its own client so it is safe to call from a worker thread."""
    store = _get_range(
        make_client(cfg), dataset=cfg.dataset, symbols=[cfg.parent_symbol], stype_in="parent",
        schema="cbbo-1m", start=iso_utc(d, win_start), end=iso_utc(d, win_end),
    )
    df = _to_df(store)
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("ts_event").groupby("instrument_id", as_index=False).tail(1)
    return pd.DataFrame({
        "quote_date": d,
        "instrument_id": df["instrument_id"].to_numpy(),
        "option_symbol": df["symbol"].to_numpy(),
        "quote_time": df["ts_event"].dt.tz_convert(ET).to_numpy(),
        "bid": df["bid_px_00"].to_numpy(),
        "ask": df["ask_px_00"].to_numpy(),
        "bid_size": df["bid_sz_00"].to_numpy(),
        "ask_size": df["ask_sz_00"].to_numpy(),
    })


def fetch_eod_quotes(client, cfg: Config, trading_days: Sequence[date]) -> pd.DataFrame:
    """
    EOD consolidated NBBO from cbbo-1m, fetched concurrently (one request per session).

    Each session keeps the last bar (ts_event <= 16:00 ET) per instrument_id. Per-day
    failures are isolated: once the retry budget is exhausted the session is logged,
    appended to <out_dir>/skipped_days.txt, and skipped -- one bad day never aborts the year.

    Returns
    -------
    DataFrame  shape (~n_instruments_per_day * n_days, 8)
        columns: quote_date, instrument_id, option_symbol, quote_time, bid, ask, bid_size, ask_size
    """
    win_start = (datetime.combine(date.min, cfg.market_close, ET)
                 - timedelta(minutes=cfg.eod_window_min)).time()
    # end is exclusive in the API; +1s guarantees the 16:00:00-labelled bar is included.
    win_end = (datetime.combine(date.min, cfg.market_close, ET) + timedelta(seconds=1)).time()

    frames: list[pd.DataFrame] = []
    failed: list[date] = []
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futs = {pool.submit(_fetch_day_quotes, cfg, d, win_start, win_end): d for d in trading_days}
        for fut in as_completed(futs):
            d = futs[fut]
            try:
                out = fut.result()
            except Exception as exc:  # noqa: BLE001 - isolate one session, keep the year alive
                LOG.warning("cbbo-1m %s: FAILED after retries (%s) -- skipping", d, exc)
                failed.append(d)
                continue
            if out.empty:
                LOG.info("cbbo-1m %s: no records", d)
            else:
                frames.append(out)
                LOG.info("cbbo-1m %s: %d contracts", d, len(out))

    if failed:
        with (cfg.out_dir / "skipped_days.txt").open("a") as fh:
            fh.write("".join(f"{d}\n" for d in sorted(failed)))
        LOG.warning("cbbo-1m: skipped %d session(s) this year; recorded in skipped_days.txt", len(failed))

    cols = ["quote_date", "instrument_id", "option_symbol", "quote_time", "bid", "ask", "bid_size", "ask_size"]
    if not frames:
        return pd.DataFrame(columns=cols)
    quotes = pd.concat(frames, ignore_index=True)
    # Databento maps undefined prices to NaN with price_type='float'; drop two-sided-missing rows.
    quotes = quotes[~(quotes["bid"].isna() & quotes["ask"].isna())].reset_index(drop=True)
    return quotes


def fetch_daily_volume(client, cfg: Config, lo: date, hi: date) -> pd.DataFrame:
    """ohlcv-1d daily volume per contract over [lo, hi). shape (M, 3): quote_date, instrument_id, volume."""
    store = _get_range(
        client, dataset=cfg.dataset, symbols=[cfg.parent_symbol], stype_in="parent",
        schema="ohlcv-1d", start=lo.isoformat(), end=hi.isoformat(),
    )
    df = _to_df(store)
    if df.empty:
        return pd.DataFrame(columns=["quote_date", "instrument_id", "volume"])
    df["quote_date"] = df["ts_event"].dt.tz_convert(ET).dt.date
    return df[["quote_date", "instrument_id", "volume"]].copy()


def fetch_open_interest(client, cfg: Config, lo: date, hi: date) -> pd.DataFrame:
    """
    Open interest from the statistics schema over [lo, hi).

    Keeps only StatType.OPEN_INTEREST and keys it by the session it refers to
    (ts_ref when present, else ts_event date).

    Returns DataFrame shape (M, 3): quote_date, instrument_id, open_interest.
    """
    import databento as db

    store = _get_range(
        client, dataset=cfg.dataset, symbols=[cfg.parent_symbol], stype_in="parent",
        schema="statistics", start=lo.isoformat(), end=hi.isoformat(),
    )
    df = _to_df(store)
    if df.empty:
        return pd.DataFrame(columns=["quote_date", "instrument_id", "open_interest"])

    oi_code = int(db.StatType.OPEN_INTEREST)
    df = df[df["stat_type"].astype("int64") == oi_code].copy()
    if df.empty:
        return pd.DataFrame(columns=["quote_date", "instrument_id", "open_interest"])

    ref = df["ts_ref"] if "ts_ref" in df.columns else df["ts_event"]
    ref = pd.to_datetime(ref, utc=True, errors="coerce")
    ref = ref.fillna(df["ts_event"])
    df["quote_date"] = ref.dt.tz_convert(ET).dt.date
    df = df.rename(columns={"quantity": "open_interest"})
    # One OI per (date, contract); keep the last reported.
    df = (df.sort_values("ts_event")
            .groupby(["quote_date", "instrument_id"], as_index=False)["open_interest"].last())
    return df


# --------------------------------------------------------------------------- #
# SPY underlier: raw close + dividends
# --------------------------------------------------------------------------- #
def _spy_yfinance(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SPY raw close + SPDR dividend series from yfinance (auto_adjust=False)."""
    import yfinance as yf

    hist = yf.Ticker("SPY").history(start=cfg.start.isoformat(),
                                    end=(cfg.end + timedelta(days=1)).isoformat(),
                                    interval="1d", auto_adjust=False, actions=True).reset_index()
    if hist.empty:
        raise RuntimeError("yfinance returned no rows (rate-limited?)")
    hist["quote_date"] = pd.to_datetime(hist["Date"]).dt.date
    close = hist[["quote_date", "Close"]].rename(columns={"Close": "spy_close"})
    divs = (hist.loc[hist["Dividends"] > 0, ["quote_date", "Dividends"]]
                .rename(columns={"quote_date": "ex_div_date", "Dividends": "dividend_amount"})
                .reset_index(drop=True))
    return close, divs


def _spy_close_stooq(cfg: Config) -> pd.DataFrame:
    """Raw (unadjusted) SPY daily close from Stooq — fallback when yfinance is unavailable."""
    url = f"https://stooq.com/q/d/l/?s=spy.us&i=d&d1={cfg.start:%Y%m%d}&d2={cfg.end:%Y%m%d}"
    raw = urllib.request.urlopen(url, timeout=30).read().decode()
    df = pd.read_csv(io.StringIO(raw))
    df["quote_date"] = pd.to_datetime(df["Date"]).dt.date
    return df[["quote_date", "Close"]].rename(columns={"Close": "spy_close"})


def _spy_close_databento(cfg: Config) -> pd.DataFrame:
    """SPY daily close from a Databento equity dataset (covers the years Databento equities span)."""
    client = make_client(cfg)
    store = _get_range(client, dataset=cfg.equity_dataset, symbols=["SPY"], stype_in="raw_symbol",
                       schema="ohlcv-1d", start=cfg.start.isoformat(), end=cfg.end.isoformat())
    df = _to_df(store)
    df["quote_date"] = df["ts_event"].dt.tz_convert(ET).dt.date
    return df[["quote_date", "close"]].rename(columns={"close": "spy_close"})


def fetch_spy_close_dividends(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    SPY raw (unadjusted) daily close and the dividend/ex-date series.

    Close follows cfg.equity_source ('yfinance' default, 'databento' optional) with a Stooq
    fallback if yfinance is unavailable. Dividends/ex-dates are the SPDR distribution series
    from yfinance (Databento carries no corporate actions); if unavailable, the dividend
    columns are left empty with a warning.

    Returns
    -------
    close : DataFrame[quote_date, spy_close]                        shape (D, 2)
    divs  : DataFrame[ex_div_date, dividend_amount] (ex-dates only)  shape (K, 2)
    """
    empty_divs = pd.DataFrame(columns=["ex_div_date", "dividend_amount"])

    if cfg.equity_source == "databento":
        close = _spy_close_databento(cfg)
        try:
            _, divs = _spy_yfinance(cfg)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("dividends unavailable (%s); leaving dividend columns empty", exc)
            divs = empty_divs
        return close, divs

    if cfg.equity_source == "yfinance":
        try:
            return _spy_yfinance(cfg)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("yfinance unavailable (%s); using Stooq for SPY close, dividends empty", exc)
            return _spy_close_stooq(cfg), empty_divs

    raise ValueError(f"unknown equity_source: {cfg.equity_source!r}")


def attach_dividends(df: pd.DataFrame, divs: pd.DataFrame) -> pd.DataFrame:
    """
    Attach, per row, the *next* SPY ex-dividend on/after quote_date (the cash flow
    relevant to pricing toward an expiry) as columns next_ex_div_date, next_div_amount.
    """
    if divs.empty:
        df["next_ex_div_date"] = pd.NaT
        df["next_div_amount"] = np.nan
        return df
    d = divs.sort_values("ex_div_date").reset_index(drop=True)
    q = pd.DataFrame({"quote_date": pd.to_datetime(df["quote_date"].unique())}).sort_values("quote_date")
    merged = pd.merge_asof(
        q, d.assign(ex=pd.to_datetime(d["ex_div_date"])).sort_values("ex"),
        left_on="quote_date", right_on="ex", direction="forward",
    )
    lut = merged.set_index(merged["quote_date"].dt.date)
    df["next_ex_div_date"] = df["quote_date"].map(lut["ex_div_date"])
    df["next_div_amount"] = df["quote_date"].map(lut["dividend_amount"])
    return df


# --------------------------------------------------------------------------- #
# Forward / rate from put-call parity, then Black-Scholes IV & greeks
# --------------------------------------------------------------------------- #
def imply_forward_rate(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Per (quote_date, expiration), fit (C_mid - P_mid) = a + b*K across strikes.

    Put-call parity:  C - P = D * (F - K)  =>  slope b = -D,  intercept a = D*F.
    Hence the discount factor D = -b, forward F = a/D, rate r = -ln(D)/T, and the
    continuous dividend yield q = r - ln(F/S)/T.

    Adds columns: D, forward, risk_free_rate, q  (merged back onto every row).
    Groups with < 2 paired strikes or a degenerate fit fall back to r_fallback and
    F = S * exp((r - q_proxy) * T) with q_proxy from the local dividend yield (0 if none).
    """
    mid = (df["bid"].to_numpy() + df["ask"].to_numpy()) / 2.0
    work = df.assign(mid=mid)
    calls = work.loc[work["option_type"] == "C", ["quote_date", "expiration", "strike", "mid"]]
    puts = work.loc[work["option_type"] == "P", ["quote_date", "expiration", "strike", "mid"]]
    cp = calls.merge(puts, on=["quote_date", "expiration", "strike"], suffixes=("_c", "_p")).dropna()
    cp["y"] = cp["mid_c"] - cp["mid_p"]
    cp["x"] = cp["strike"]
    cp["xy"] = cp["x"] * cp["y"]
    cp["xx"] = cp["x"] * cp["x"]

    agg = (cp.groupby(["quote_date", "expiration"])
             .agg(n=("x", "size"), Sx=("x", "sum"), Sy=("y", "sum"),
                  Sxx=("xx", "sum"), Sxy=("xy", "sum"))
             .reset_index())
    den = agg["n"] * agg["Sxx"] - agg["Sx"] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (agg["n"] * agg["Sxy"] - agg["Sx"] * agg["Sy"]) / den
        intercept = (agg["Sy"] - slope * agg["Sx"]) / agg["n"]
        D = -slope
        F = intercept / D
    valid = (agg["n"] >= 2) & np.isfinite(D) & np.isfinite(F) & (D > 0) & (D <= 1.0 + 1e-6) & (F > 0)
    agg["D"] = np.where(valid, D.clip(upper=1.0), np.nan)
    agg["forward"] = np.where(valid, F, np.nan)

    out = df.merge(agg[["quote_date", "expiration", "D", "forward"]],
                   on=["quote_date", "expiration"], how="left")

    T = _tenor(out, cfg)
    S = out["spy_close"].to_numpy()
    # Fallbacks where parity could not be solved.
    need = ~np.isfinite(out["D"].to_numpy())
    D_arr = out["D"].to_numpy().copy()
    F_arr = out["forward"].to_numpy().copy()
    q_proxy = _dividend_yield_proxy(out)
    with np.errstate(divide="ignore", invalid="ignore"):
        D_arr[need] = np.exp(-cfg.r_fallback * T[need])
        F_arr[need] = S[need] * np.exp((cfg.r_fallback - q_proxy[need]) * T[need])
        r = -np.log(D_arr) / T
        q = r - np.log(F_arr / S) / T
    out["D"] = D_arr
    out["forward"] = F_arr
    out["risk_free_rate"] = r
    out["q"] = q
    out["T"] = T
    return out


def _tenor(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    exp = pd.to_datetime(df["expiration"])
    qd = pd.to_datetime(df["quote_date"])
    days = (exp - qd).dt.days.to_numpy().astype("float64")
    return np.maximum(days, 0.0) / cfg.day_count


def _dividend_yield_proxy(df: pd.DataFrame) -> np.ndarray:
    """Crude trailing dividend yield = 4 * next_div_amount / spy_close (0 if unknown)."""
    col = df["next_div_amount"] if "next_div_amount" in df.columns else pd.Series(np.nan, index=df.index)
    nd = pd.to_numeric(col, errors="coerce").to_numpy()
    s = df["spy_close"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(np.isfinite(nd) & (s > 0), 4.0 * nd / s, 0.0)
    return np.nan_to_num(y, nan=0.0)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF. Uses scipy when available (fast, fully vectorized) and
    falls back to a math.erf implementation so the module has no hard scipy dependency."""
    try:
        from scipy.special import ndtr

        return ndtr(x)
    except Exception:
        import math

        erf = np.vectorize(math.erf, otypes=[np.float64])
        return 0.5 * (1.0 + erf(np.asarray(x, dtype=np.float64) / np.sqrt(2.0)))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _bs_price(S, K, T, r, q, sigma, cp):
    """Black-Scholes price; cp = +1 call, -1 put. All array-broadcastable."""
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_t = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol_t
        d2 = d1 - vol_t
    return cp * (S * np.exp(-q * T) * _norm_cdf(cp * d1)
                 - K * np.exp(-r * T) * _norm_cdf(cp * d2))


def implied_vol_and_greeks(out: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Vectorized Newton inversion of Black-Scholes for IV, then closed-form greeks.

    Conventions: vega is per 1.00 change in vol (divide by 100 for per-1%-vol);
    theta is per calendar day (dPrice/dt). IV is NaN where the mid is below intrinsic,
    T<=0, or the iteration leaves the [iv_lower, iv_upper] band.
    """
    S = out["spy_close"].to_numpy(dtype="float64")
    K = out["strike"].to_numpy(dtype="float64")
    T = out["T"].to_numpy(dtype="float64")
    r = out["risk_free_rate"].to_numpy(dtype="float64")
    q = out["q"].to_numpy(dtype="float64")
    cp = np.where(out["option_type"].to_numpy() == "C", 1.0, -1.0)
    price = ((out["bid"].to_numpy(dtype="float64") + out["ask"].to_numpy(dtype="float64")) / 2.0)

    intrinsic = np.maximum(cp * (S * np.exp(-q * T) - K * np.exp(-r * T)), 0.0)
    solvable = np.isfinite(price) & (T > 0) & (S > 0) & (K > 0) & (price > intrinsic + 1e-8)

    sigma = np.full_like(price, cfg.iv_init)
    for _ in range(cfg.iv_max_iter):
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_t = sigma * np.sqrt(T)
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol_t
            vega = S * np.exp(-q * T) * _norm_pdf(d1) * np.sqrt(T)
            diff = _bs_price(S, K, T, r, q, sigma, cp) - price
            step = np.where(vega > 1e-12, diff / vega, 0.0)
        sigma = np.clip(sigma - step, cfg.iv_lower, cfg.iv_upper)
        if np.nanmax(np.abs(np.where(solvable, step, 0.0))) < cfg.iv_tol:
            break

    sigma = np.where(solvable & (sigma > cfg.iv_lower) & (sigma < cfg.iv_upper), sigma, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        vol_t = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol_t
        d2 = d1 - vol_t
        delta = cp * np.exp(-q * T) * _norm_cdf(cp * d1)
        vega = S * np.exp(-q * T) * _norm_pdf(d1) * np.sqrt(T)             # per 1.00 vol
        theta = ((-(S * np.exp(-q * T) * _norm_pdf(d1) * sigma) / (2.0 * np.sqrt(T))
                  - cp * r * K * np.exp(-r * T) * _norm_cdf(cp * d2)
                  + cp * q * S * np.exp(-q * T) * _norm_cdf(cp * d1)) / cfg.day_count)  # per day

    out = out.copy()
    out["implied_vol"] = sigma
    out["delta"] = delta
    out["vega"] = vega
    out["theta"] = theta
    return out


# --------------------------------------------------------------------------- #
# Assembly + output
# --------------------------------------------------------------------------- #
OUTPUT_COLUMNS: list[str] = [
    "quote_date", "contract_id", "option_symbol", "expiration_date", "strike",
    "option_type", "option_multiplier", "bid", "ask", "mid", "bid_size", "ask_size",
    "quote_time", "volume", "open_interest", "spy_close",
    "next_ex_div_date", "next_div_amount",
    "implied_vol", "delta", "theta", "vega",
    "forward", "risk_free_rate",
]


def _safe_fetch(fn, client, cfg: Config, lo: date, hi: date, what: str) -> pd.DataFrame:
    """Run a monthly fetch; on persistent failure log and return empty so the year survives."""
    try:
        return fn(client, cfg, lo, hi)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("%s: FAILED after retries (%s) -- skipping chunk", what, exc)
        return pd.DataFrame()


def assemble_year(client, cfg: Config, year: int, trading_days: Sequence[date],
                  close: pd.DataFrame, divs: pd.DataFrame) -> pd.DataFrame:
    """Build the full daily panel for a single calendar year."""
    lo = max(cfg.start, date(year, 1, 1))
    hi = min(cfg.end + timedelta(days=1), date(year + 1, 1, 1))
    days = [d for d in trading_days if lo <= d < hi]
    if not days:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    LOG.info("=== year %d: %d trading days ===", year, len(days))
    quotes = fetch_eod_quotes(client, cfg, days)
    if quotes.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    months = list(month_starts(lo, hi - timedelta(days=1)))
    vol = pd.concat([_safe_fetch(fetch_daily_volume, client, cfg, a, b, f"ohlcv-1d {a:%Y-%m}") for a, b in months]
                    + [pd.DataFrame(columns=["quote_date", "instrument_id", "volume"])], ignore_index=True)
    oi = pd.concat([_safe_fetch(fetch_open_interest, client, cfg, a, b, f"statistics {a:%Y-%m}") for a, b in months]
                   + [pd.DataFrame(columns=["quote_date", "instrument_id", "open_interest"])], ignore_index=True)

    terms = parse_osi(quotes["option_symbol"])
    df = pd.concat([quotes.reset_index(drop=True), terms], axis=1)
    df = df.merge(vol, on=["quote_date", "instrument_id"], how="left")
    df = df.merge(oi, on=["quote_date", "instrument_id"], how="left")
    df = df.merge(close, on="quote_date", how="left")
    df = attach_dividends(df, divs)

    df = imply_forward_rate(df, cfg)
    df = implied_vol_and_greeks(df, cfg)

    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["option_multiplier"] = cfg.option_multiplier
    df["volume"] = df["volume"].fillna(0)          # no ohlcv-1d bar => the contract did not trade that day
    df = df.rename(columns={"instrument_id": "contract_id", "expiration": "expiration_date"})
    df["year"] = year
    return df.reindex(columns=OUTPUT_COLUMNS + ["year"])


def write_year(df: pd.DataFrame, cfg: Config, year: int) -> Path:
    part = cfg.out_dir / f"year={year}"
    part.mkdir(parents=True, exist_ok=True)
    path = part / "data.parquet"
    df.drop(columns=["year"]).to_parquet(path, engine="pyarrow", index=False)
    LOG.info("wrote %s  (%d rows)", path, len(df))
    return path


# --------------------------------------------------------------------------- #
# Dry-run metadata
# --------------------------------------------------------------------------- #
def dry_run(client, cfg: Config) -> None:
    rng = client.metadata.get_dataset_range(dataset=cfg.dataset)
    print(json.dumps({"dataset_range": rng}, indent=2, default=str))
    for schema in ("cbbo-1m", "ohlcv-1d", "statistics"):
        cost = client.metadata.get_cost(
            dataset=cfg.dataset, symbols=[cfg.parent_symbol], stype_in="parent",
            schema=schema, start=cfg.start.isoformat(), end=cfg.end.isoformat(),
            mode="historical-streaming")
        count = client.metadata.get_record_count(
            dataset=cfg.dataset, symbols=[cfg.parent_symbol], stype_in="parent",
            schema=schema, start=cfg.start.isoformat(), end=cfg.end.isoformat())
        print(f"{schema:11}  full-range cost=${cost:<12}  records={count:,}")
    print("\nNote: cbbo-1m above is the FULL intraday tape; the actual pull is only the "
          "EOD window per day (~1/390 of this).")


# --------------------------------------------------------------------------- #
# Trading calendar
# --------------------------------------------------------------------------- #
def trading_calendar(close: pd.DataFrame, cfg: Config) -> list[date]:
    """Trading days = the SPY daily-bar dates within [start, end] (NYSE sessions)."""
    days = sorted(d for d in close["quote_date"].tolist() if cfg.start <= d <= cfg.end)
    return days


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def write_manifest(cfg: Config, written: list[Path], n_rows: int) -> None:
    import importlib.metadata as ilm

    def ver(pkg: str) -> str:
        try:
            return ilm.version(pkg)
        except Exception:
            return "unknown"

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {k: str(v) for k, v in dataclasses.asdict(cfg).items() if k != "api_key"},
        "rows_total": n_rows,
        "partitions": [str(p) for p in written],
        "versions": {p: ver(p) for p in ("databento", "pandas", "numpy", "scipy", "pyarrow", "yfinance")},
        "python": sys.version,
    }
    (cfg.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    LOG.info("wrote run_manifest.json")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def build_config(args: argparse.Namespace) -> Config:
    api_key = args.api_key or os.environ.get("DATABENTO_API_KEY", "")
    if not api_key:
        sys.exit("ERROR: set DATABENTO_API_KEY or pass --api-key")
    kw: dict = {"api_key": api_key, "out_dir": Path(args.out).expanduser().resolve()}
    if args.start:
        kw["start"] = date.fromisoformat(args.start)
    if args.end:
        kw["end"] = date.fromisoformat(args.end)
    if args.equity_source:
        kw["equity_source"] = args.equity_source
    if args.window_min is not None:
        kw["eod_window_min"] = args.window_min
    if args.r_fallback is not None:
        kw["r_fallback"] = args.r_fallback
    if args.max_workers is not None:
        kw["max_workers"] = args.max_workers
    return Config(**kw)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download full-history SPY options from Databento OPRA.PILLAR.")
    p.add_argument("--out", default="./spy_options_daily", help="output root for the partitioned Parquet dataset")
    p.add_argument("--api-key", default=None, help="Databento API key (else $DATABENTO_API_KEY)")
    p.add_argument("--start", default=None, help="YYYY-MM-DD (default 2013-04-01)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (default today, ET)")
    p.add_argument("--equity-source", choices=["yfinance", "databento"], default=None)
    p.add_argument("--window-min", type=int, default=None, help="EOD cbbo-1m window in minutes (default 2)")
    p.add_argument("--max-workers", type=int, default=None, help="concurrent EOD quote requests (default 8)")
    p.add_argument("--r-fallback", type=float, default=None, help="fallback risk-free rate when parity fails")
    p.add_argument("--dry-run", action="store_true", help="print ranges/record counts/cost and exit")
    p.add_argument("--year", type=int, default=None, help="restrict to a single calendar year (debugging)")
    p.add_argument("--force", action="store_true", help="rebuild years whose Parquet partition already exists")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    logging.basicConfig(level=getattr(logging, cfg.log_level),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    np.random.seed(cfg.seed)

    client = make_client(cfg)

    if args.dry_run:
        dry_run(client, cfg)
        return 0

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Fetching SPY close + dividends via %s", cfg.equity_source)
    close, divs = fetch_spy_close_dividends(cfg)
    calendar = trading_calendar(close, cfg)
    LOG.info("Trading calendar: %d sessions %s -> %s", len(calendar), calendar[0], calendar[-1])

    years = [args.year] if args.year else cfg.years()
    written: list[Path] = []
    total = 0
    for year in years:
        part = cfg.out_dir / f"year={year}" / "data.parquet"
        if part.exists() and not args.force:
            LOG.info("year %d already written (%s) -> skipping (use --force to rebuild)", year, part)
            written.append(part)
            continue
        df = assemble_year(client, cfg, year, calendar, close, divs)
        if df.empty:
            continue
        written.append(write_year(df, cfg, year))
        total += len(df)

    write_manifest(cfg, written, total)
    LOG.info("DONE: %d rows across %d partitions -> %s", total, len(written), cfg.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
