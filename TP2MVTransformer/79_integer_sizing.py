"""brief-18: integer-contract sizing sensitivity for the cash-budgeted backtest.
Reuses script 42's committed trade sets + trading calendar. Two sizings:
  fractional : per_trade dollars deployed continuously (script 42) -- GATE vs Table 3 CAGRs.
  integer    : contracts q = max(1, round(per_trade_dollars / entry_gross_exposure)); a trade is
               skipped if even 1 contract exceeds the day's headroom. P&L = q*exposure*realized_return.
Shows whether whole-contract sizing (no fractional contracts) changes the cash-budgeted result.
Ratio-tolerance across legs is noted (needs per-leg contract counts). Artifact + CSV.
"""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent; ACTION = OUT/"tables"/"action_list"
s = importlib.util.spec_from_file_location("cap42c", OUT/"42_capital_scaled_portfolio_backtest.py")
cap = importlib.util.module_from_spec(s); s.loader.exec_module(cap)
CAP0, FRAC, MAXX = cap.STARTING_CAPITAL, cap.DAILY_DEPLOY_FRACTION, cap.MAX_GROSS_EXPOSURE_MULTIPLE

def backtest(trades, calendar, integer):
    trades = trades.copy(); trades["signal_date"] = pd.to_datetime(trades["signal_date"]); trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    entries = trades.groupby("signal_date")
    all_dates = pd.DatetimeIndex(sorted(set(calendar) | set(trades["exit_date"])))
    exits = {}; cash = CAP0; outstanding = 0.0; rows = []
    for date in all_dates:
        prof = 0.0
        for pos in exits.pop(date, []):
            prof += pos["size"] * pos["ret"]; outstanding -= pos["size"]
        if date in entries.groups:
            td = trades.loc[entries.groups[date]]; n = len(td)
            avail = min(FRAC * cash, max(0.0, MAXX * cash - outstanding))
            if avail > 0 and n > 0:
                per = avail / n
                for _, r in td.iterrows():
                    ex = r["entry_gross_exposure"]
                    if ex <= 0: continue
                    if integer:
                        q = max(1, int(round(per / ex))); size = q * ex
                        if outstanding + size > MAXX * cash: continue     # 1 contract doesn't fit -> skip
                    else:
                        size = per
                    exits.setdefault(r["exit_date"], []).append({"size": size, "ret": r["realized_return"]})
                    outstanding += size
        cash += prof; rows.append({"date": date, "cash": cash})
    dfc = pd.DataFrame(rows).set_index("date"); ny = (dfc.index.max() - dfc.index.min()).days / 365.25
    cagr = (dfc["cash"].iloc[-1] / CAP0) ** (1 / ny) - 1 if ny > 0 else float("nan")
    dd = (dfc["cash"] / dfc["cash"].cummax() - 1).min()
    return {"ending_capital": float(dfc["cash"].iloc[-1]), "cagr": float(cagr), "max_drawdown": float(dd)}

labels = cap.load_labels(); tcal = cap.load_trading_calendar()
GATE = {"LightGBM": 0.146, "MLP": 0.172, "Transformer": 0.006}
result = {}; csv_rows = []
print(f"{'model':14} {'fractional CAGR':>16} {'(Table3 gate)':>14} {'integer CAGR':>14} {'int ending':>14}")
for name in ["LightGBM", "MLP", "Transformer"]:
    tr = cap.build_trade_set(name, labels)
    frac = backtest(tr, tcal, integer=False); intg = backtest(tr, tcal, integer=True)
    ok = abs(frac["cagr"] - GATE[name]) < 0.005
    assert ok, f"{name} fractional CAGR gate FAILED {frac['cagr']:.3f} vs {GATE[name]}"
    result[name] = {"fractional": frac, "integer": intg}
    csv_rows.append({"model": name, "frac_cagr": frac["cagr"], "int_cagr": intg["cagr"],
                     "frac_ending": frac["ending_capital"], "int_ending": intg["ending_capital"]})
    print(f"{name:14} {frac['cagr']*100:>15.1f}% {'('+str(GATE[name]*100)+'%)':>14} {intg['cagr']*100:>13.1f}% ${intg['ending_capital']:>13,.0f}")

json.dump({"models": result, "starting_capital": CAP0, "daily_deploy_fraction": FRAC,
           "note": "brief-18. Fractional gate reproduces Table 3 CAGRs (LightGBM 14.6/MLP 17.2/Transformer 0.6%). "
                   "Integer = whole contracts, q=max(1,round(dollars/entry_gross_exposure)), skip if 1 contract "
                   "exceeds headroom. Per-leg ratio tolerance (1/2.5/5%) needs per-leg contract counts -- noted."},
          open(ACTION/"integer_sizing.json", "w"), indent=2, default=float)
pd.DataFrame(csv_rows).to_csv(ACTION/"integer_sizing.csv", index=False)
print("\nwrote", ACTION/"integer_sizing.json", "+ .csv")
