"""brief-18 + brief-19 on the NEW models (per user directive): cash-budgeted backtest, integer-contract
sizing, and margin-aware capital, using the new-model selected trades. LightGBM stays old-149.
New-model trade sets = new selected trades (episode_id, strategy) joined to the labels for
exit_date / entry_gross_exposure / realized_return. No Table-3 gate for the new models (their
cash-budgeted CAGR is a new baseline); LightGBM(old) still reproduces Table 3 as a sanity anchor.
Reuses script 42's backtest. Reg-T margin as in brief-19. Artifact + CSV.
"""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent; ROOT = OUT.parent; ACTION = OUT/"tables"/"action_list"
NESTED = ROOT/"FeatureDataDocs_2026-08-04"/"nested"
s = importlib.util.spec_from_file_location("cap42n", OUT/"42_capital_scaled_portfolio_backtest.py")
cap = importlib.util.module_from_spec(s); s.loader.exec_module(cap)
CAP0, FRAC, MAXX, MULT = cap.STARTING_CAPITAL, cap.DAILY_DEPLOY_FRACTION, cap.MAX_GROSS_EXPOSURE_MULTIPLE, 100.0

labels_full = pd.read_parquet(ROOT/"data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet",
    columns=["episode_id", "strategy", "signal_date", "exit_date", "exit_policy", "is_valid_backtest",
             "entry_gross_exposure", "realized_return", "feature_A_mid", "feature_B_mid", "feature_C_mid",
             "feature_D_mid", "feature_C_strike", "feature_D_strike"])
lab = labels_full[(labels_full.exit_policy == "exit_selected_leg_expiration_payoff") & labels_full.is_valid_backtest & labels_full.strategy.isin(["T1", "K2"])].copy()
lab["signal_date"] = pd.to_datetime(lab.signal_date)
u = pd.read_parquet(ROOT/"data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet", columns=["date", "spy_close"]).drop_duplicates("date")
u["date"] = pd.to_datetime(u["date"]); lab = lab.merge(u, left_on="signal_date", right_on="date", how="left")
U = lab["spy_close"].to_numpy(float)
def regt(strike, mid): return (np.maximum(0.20*U - np.maximum(0.0, strike-U), 0.10*U) + mid) * MULT
lab["margin"] = (regt(lab.feature_C_strike.to_numpy(float), lab.feature_C_mid.to_numpy(float))
                 + regt(lab.feature_D_strike.to_numpy(float), lab.feature_D_mid.to_numpy(float))
                 + (lab.feature_A_mid.to_numpy(float) + lab.feature_B_mid.to_numpy(float)) * MULT)
LAB = lab[["episode_id", "strategy", "signal_date", "exit_date", "entry_gross_exposure", "realized_return", "margin"]]

def newmodel_set(f):
    d = pd.read_parquet(NESTED/f)
    if "selected" in d.columns: d = d[d["selected"].astype(bool)]
    picks = d[["episode_id", "strategy"]].drop_duplicates()
    return picks.merge(LAB, on=["episode_id", "strategy"], how="inner")

MODELS = {"LightGBM(149,old)": ("__cap__", None), "MLP-5(new)": (None, "trades_mlp5.parquet"),
          "Transformer-plain-5(new)": (None, "trades_transformer_plain.parquet"),
          "Transformer-ALiBi-5(new)": (None, "trades_transformer_alibi.parquet"),
          "Logistic(new)": (None, "trades_logistic.parquet")}

def integer_backtest(tr):
    tr = tr.copy(); tr["exit_date"] = pd.to_datetime(tr["exit_date"]); ent = tr.groupby("signal_date")
    alld = pd.DatetimeIndex(sorted(set(cap.load_trading_calendar()) | set(tr["exit_date"])))
    exits = {}; cash = CAP0; outст = 0.0; rows = []
    for date in alld:
        prof = 0.0
        for pos in exits.pop(date, []): prof += pos["s"]*pos["r"]; outст -= pos["s"]
        if date in ent.groups:
            td = tr.loc[ent.groups[date]]; n = len(td); avail = min(FRAC*cash, max(0.0, MAXX*cash - outст))
            if avail > 0 and n > 0:
                per = avail/n
                for _, r in td.iterrows():
                    ex = r["entry_gross_exposure"]
                    if ex <= 0: continue
                    q = max(1, int(round(per/ex))); size = q*ex
                    if outст + size > MAXX*cash: continue
                    exits.setdefault(r["exit_date"], []).append({"s": size, "r": r["realized_return"]}); outст += size
        cash += prof; rows.append(cash)
    ny = (alld.max()-alld.min()).days/365.25
    return (rows[-1]/CAP0)**(1/ny)-1 if ny > 0 else float("nan")

result = {}; csv_rows = []
print(f"{'model':26} {'premium CAGR':>13} {'integer CAGR':>13} {'margin CAGR':>12}")
for name, (tag, f) in MODELS.items():
    tr = cap.build_trade_set("LightGBM", cap.load_labels()).merge(LAB[["episode_id", "strategy", "margin"]], on=["episode_id", "strategy"], how="left") if tag == "__cap__" else newmodel_set(f)
    prem = cap.run_capital_scaled_backtest(tr, cap.load_trading_calendar())["cagr"]
    integ = integer_backtest(tr)
    trm = tr.copy(); trm = trm[trm["margin"].notna() & (trm["margin"] > 0)]; trm["realized_return"] = trm["realized_return"]*(trm["entry_gross_exposure"]/trm["margin"])
    marg = cap.run_capital_scaled_backtest(trm, cap.load_trading_calendar())["cagr"]
    result[name] = {"premium_cagr": float(prem), "integer_cagr": float(integ), "margin_cagr": float(marg), "n_trades": int(len(tr))}
    csv_rows.append({"model": name, **result[name]})
    print(f"{name:26} {prem*100:>12.1f}% {integ*100:>12.1f}% {marg*100:>11.1f}%")

json.dump({"models": result, "note": "brief-18 + brief-19 on NEW models (LightGBM=old 149). New-model trade sets = "
           "new selected trades joined to labels. No Table-3 gate for new models (new baseline); LightGBM(old) "
           "anchors Table 3. integer = whole contracts (min 1); margin = Reg-T conservative (effective_return = "
           "realized_return * premium/margin). Reuses script 42."},
          open(ACTION/"newmodel_cash_budgeted.json", "w"), indent=2, default=float)
pd.DataFrame(csv_rows).to_csv(ACTION/"newmodel_cash_budgeted.csv", index=False)
print("\nwrote", ACTION/"newmodel_cash_budgeted.json", "+ .csv")
