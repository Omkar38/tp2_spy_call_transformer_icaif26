#!/usr/bin/env python3
"""
Offline unit tests for download_spy_options.py — pure functions only, NO API calls.

Run either:
    pytest -q test_spy_options.py
    python test_spy_options.py        # standalone runner, prints PASS/FAIL
"""
from __future__ import annotations

from datetime import date, time as dtime
from math import exp, log
from pathlib import Path

import numpy as np
import pandas as pd

import download_spy_options as m

CFG = m.Config(api_key="TEST", out_dir=Path("."))


# --------------------------------------------------------------------------- #
# OSI symbol parsing
# --------------------------------------------------------------------------- #
def test_parse_osi() -> None:
    syms = pd.Series([
        "SPY   240119C00450000",   # space-padded root, call, $450.000
        "SPY   251219P00500000",   # put, $500.000
        "SPY   240119C00450500",   # fractional strike $450.500
        "SPYW  240621P00123450",   # 4-char root, $123.450
        "SPY240119C01000000",      # no padding, $1000.000
    ])
    r = m.parse_osi(syms)
    assert r.loc[0, "option_type"] == "C" and r.loc[0, "expiration"] == date(2024, 1, 19)
    assert abs(r.loc[0, "strike"] - 450.0) < 1e-9
    assert r.loc[1, "option_type"] == "P" and abs(r.loc[1, "strike"] - 500.0) < 1e-9
    assert abs(r.loc[2, "strike"] - 450.5) < 1e-9
    assert r.loc[3, "option_type"] == "P" and abs(r.loc[3, "strike"] - 123.45) < 1e-9
    assert r.loc[3, "expiration"] == date(2024, 6, 21)
    assert r.loc[4, "option_type"] == "C" and abs(r.loc[4, "strike"] - 1000.0) < 1e-9


# --------------------------------------------------------------------------- #
# EOD window timezone handling (DST)
# --------------------------------------------------------------------------- #
def test_iso_utc_dst() -> None:
    # July -> EDT (UTC-4): 16:00 ET == 20:00 UTC
    assert m.iso_utc(date(2024, 7, 1), dtime(16, 0)).startswith("2024-07-01T20:00:00")
    # January -> EST (UTC-5): 16:00 ET == 21:00 UTC
    assert m.iso_utc(date(2024, 1, 2), dtime(16, 0)).startswith("2024-01-02T21:00:00")


# --------------------------------------------------------------------------- #
# Black-Scholes put-call parity at the price level
# --------------------------------------------------------------------------- #
def test_bs_put_call_parity() -> None:
    S, K, T, r, q, sig = 100.0, 105.0, 0.5, 0.03, 0.01, 0.2
    c = float(m._bs_price(S, K, T, r, q, sig, +1.0))
    p = float(m._bs_price(S, K, T, r, q, sig, -1.0))
    parity = S * exp(-q * T) - K * exp(-r * T)
    assert abs((c - p) - parity) < 1e-9


# --------------------------------------------------------------------------- #
# IV round-trip + finite-difference greeks via implied_vol_and_greeks
# --------------------------------------------------------------------------- #
def _iv_frame(S, r, q, T, sig, strikes) -> pd.DataFrame:
    rows = []
    for K in strikes:
        for cp, typ in ((+1.0, "C"), (-1.0, "P")):
            px = float(m._bs_price(S, K, T, r, q, sig, cp))
            rows.append(dict(spy_close=S, strike=K, T=T, risk_free_rate=r, q=q,
                             option_type=typ, bid=px, ask=px))
    return pd.DataFrame(rows)


def test_iv_roundtrip() -> None:
    S, r, q, T, sig = 450.0, 0.04, 0.015, 0.25, 0.18
    strikes = np.array([400.0, 425.0, 450.0, 475.0, 500.0])
    out = m.implied_vol_and_greeks(_iv_frame(S, r, q, T, sig, strikes), CFG)
    iv = out["implied_vol"].to_numpy()
    assert np.isfinite(iv).all()
    assert np.nanmax(np.abs(iv - sig)) < 1e-4


def test_greeks_finite_difference() -> None:
    S, r, q, T, sig = 450.0, 0.04, 0.015, 0.25, 0.18
    strikes = np.array([450.0])
    out = m.implied_vol_and_greeks(_iv_frame(S, r, q, T, sig, strikes), CFG)
    row = out[(out.option_type == "C")].iloc[0]

    cp = +1.0
    V = lambda S=S, K=450.0, T=T, r=r, q=q, s=sig: float(m._bs_price(S, K, T, r, q, s, cp))
    hS, hs, hT = 1e-2, 1e-4, 1e-5
    delta_fd = (V(S=S + hS) - V(S=S - hS)) / (2 * hS)
    vega_fd = (V(s=sig + hs) - V(s=sig - hs)) / (2 * hs)
    theta_fd = -((V(T=T + hT) - V(T=T - hT)) / (2 * hT)) / 365.0  # per calendar day

    assert 0.0 < row["delta"] < 1.0
    assert row["vega"] > 0.0
    assert abs(row["delta"] - delta_fd) < 1e-4
    assert abs(row["vega"] - vega_fd) / vega_fd < 1e-3
    assert abs(row["theta"] - theta_fd) < 1e-3


# --------------------------------------------------------------------------- #
# Put-call parity recovery of discount factor / forward / rate
# --------------------------------------------------------------------------- #
def test_imply_forward_rate() -> None:
    qd, exp_d = date(2025, 6, 2), date(2025, 8, 31)          # 90 calendar days
    T = (exp_d - qd).days / CFG.day_count
    D_true = exp(-0.043 * T)
    F_true = 455.0
    S = 450.0
    strikes = np.array([440.0, 445.0, 450.0, 455.0, 460.0, 465.0])

    rows = []
    for K in strikes:
        p_mid = 10.0                                          # arbitrary; only C-P is identified
        c_mid = p_mid + D_true * (F_true - K)                # parity: C - P = D (F - K)
        rows.append(dict(quote_date=qd, expiration=exp_d, strike=K, option_type="C",
                         bid=c_mid, ask=c_mid, spy_close=S))
        rows.append(dict(quote_date=qd, expiration=exp_d, strike=K, option_type="P",
                         bid=p_mid, ask=p_mid, spy_close=S))
    out = m.imply_forward_rate(pd.DataFrame(rows), CFG)

    assert abs(out["D"].iloc[0] - D_true) < 1e-9
    assert abs(out["forward"].iloc[0] - F_true) < 1e-6
    assert abs(out["risk_free_rate"].iloc[0] - (-log(D_true) / T)) < 1e-9


# --------------------------------------------------------------------------- #
# Next ex-dividend attachment
# --------------------------------------------------------------------------- #
def test_attach_dividends() -> None:
    divs = pd.DataFrame({
        "ex_div_date": [date(2025, 3, 21), date(2025, 6, 20), date(2025, 9, 19)],
        "dividend_amount": [1.50, 1.60, 1.70],
    })
    df = pd.DataFrame({
        "quote_date": [date(2025, 3, 20), date(2025, 3, 21), date(2025, 4, 1), date(2025, 9, 19)],
        "spy_close": [500.0, 500.0, 500.0, 500.0],
    })
    out = m.attach_dividends(df, divs)
    assert out["next_ex_div_date"].tolist() == [
        date(2025, 3, 21), date(2025, 3, 21), date(2025, 6, 20), date(2025, 9, 19)]
    assert out["next_div_amount"].tolist() == [1.50, 1.50, 1.60, 1.70]


# --------------------------------------------------------------------------- #
def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
