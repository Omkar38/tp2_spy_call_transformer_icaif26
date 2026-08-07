"""New-model equity-curves figure (per user directive): the principal figure on the NEW-model set
{Unconditional, LightGBM-149(old), MLP-5(new), plain-5(new), ALiBi-5(new), logistic(new)}.
Panel (a): flat-notional cumulative P&L from the new-model selected trades. Panel (b): cash-budgeted
equity from the new-model trade sets (LightGBM old, MLP-5/plain new) via script 42. Kept as a SEPARATE
file (equity_curves_newmodel.pdf); the committed equity_curves.pdf still matches the committed Table 2/3.
"""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.size": 7, "axes.linewidth": 0.6})

OUT = Path(__file__).resolve().parent; ROOT = OUT.parent; DATA = OUT/"data"; NESTED = ROOT/"FeatureDataDocs_2026-08-04"/"nested"
FIGS = ROOT/"AICIF"/"figures"
s = importlib.util.spec_from_file_location("cap42e", OUT/"42_capital_scaled_portfolio_backtest.py"); cap = importlib.util.module_from_spec(s); s.loader.exec_module(cap)
COL = {"Unconditional": "black", "LightGBM": "#E69F00", "MLP-5": "#0072B2", "Transformer-plain-5": "#56B4E9", "Transformer-ALiBi-5": "#009E73", "Logistic": "#CC79A7"}

def load(p, col):
    d = pd.read_parquet(p); d["signal_date"] = pd.to_datetime(d["signal_date"]); return d, col
# panel (a): new-model selected trades
SER = {"LightGBM": (DATA/"lgbm_final_t1k2_selected_trades.parquet", "norm_pnl"),
       "MLP-5": (NESTED/"trades_mlp5.parquet", "target_norm_pnl"),
       "Transformer-plain-5": (NESTED/"trades_transformer_plain.parquet", "target_norm_pnl"),
       "Transformer-ALiBi-5": (NESTED/"trades_transformer_alibi.parquet", "target_norm_pnl"),
       "Logistic": (NESTED/"trades_logistic.parquet", "target_norm_pnl")}
pop = pd.read_parquet(DATA/"lgbm_final_test_predictions.parquet"); pop["signal_date"] = pd.to_datetime(pop["signal_date"])
if "is_valid" in pop.columns: pop = pop[pop.is_valid.astype(bool)]
pop = pop[pop.strategy.isin(["T1", "K2"])]
daily = {"Unconditional": pop.groupby(pop.signal_date.dt.normalize())["norm_pnl"].sum()}
for name, (p, c) in SER.items():
    d, col = load(p, c)
    if "selected" in d.columns: d = d[d["selected"].astype(bool)]
    daily[name] = d.groupby(d.signal_date.dt.normalize())[col].sum()
cal = pd.DatetimeIndex(sorted(set().union(*[set(x.index) for x in daily.values()])))
cum = pd.DataFrame({k: v.reindex(cal, fill_value=0).cumsum() for k, v in daily.items()}, index=cal)
print("panel (a) terminals (new models):", {k: f"${cum[k].iloc[-1]:,.0f}" for k in cum.columns})

# panel (b): new-model cash-budgeted daily equity
labels = cap.load_labels(); tcal = cap.load_trading_calendar()
def nmset(f):
    d = pd.read_parquet(NESTED/f); picks = d[["episode_id", "strategy"]].drop_duplicates()
    return picks.merge(labels, on=["episode_id", "strategy"], how="inner")
cash = {}
cash["LightGBM"] = cap.run_capital_scaled_backtest(cap.build_trade_set("LightGBM", labels), tcal)["daily_curve"]
cash["MLP-5"] = cap.run_capital_scaled_backtest(nmset("trades_mlp5.parquet"), tcal)["daily_curve"]
cash["Transformer-plain-5"] = cap.run_capital_scaled_backtest(nmset("trades_transformer_plain.parquet"), tcal)["daily_curve"]

fig, (axA, axB) = plt.subplots(2, 1, figsize=(3.5, 5.0), sharex=True)
for name in ["Unconditional", "LightGBM", "MLP-5", "Transformer-plain-5", "Transformer-ALiBi-5", "Logistic"]:
    st = dict(color=COL[name], lw=1.1); st.update(ls="--") if name == "Unconditional" else None
    axA.plot(cum.index, cum[name]/1e6, label=name.replace("Transformer-", "TF-"), **st)
axA.axhline(0, color="grey", lw=0.4); axA.set_ylabel("Cumulative P&L ($M), flat-notional")
axA.set_title("(a) Flat-notional cumulative P&L (new models)", fontsize=7, loc="left")
axA.grid(axis="y", lw=0.3, color="0.85"); axA.legend(frameon=False, fontsize=5, labelcolor="0.25", ncol=2)
for name in ["LightGBM", "MLP-5", "Transformer-plain-5"]:
    dc = cash[name].copy(); dc["date"] = pd.to_datetime(dc["date"])
    axB.plot(dc["date"], dc["cash"]/1e6, color=COL[name], lw=1.1, label=name.replace("Transformer-", "TF-"))
axB.axhline(1.0, color="grey", lw=0.4, ls=":"); axB.set_ylabel("Account equity ($M), cash-budgeted")
axB.set_title("(b) Cash-budgeted equity ($1M, 1%/day cap)", fontsize=7, loc="left")
axB.grid(axis="y", lw=0.3, color="0.85"); axB.legend(frameon=False, fontsize=5.5, labelcolor="0.25")
for ax in (axA, axB):
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
fig.tight_layout(pad=0.4); fig.savefig(FIGS/"equity_curves_newmodel.pdf", bbox_inches="tight")
cum.to_parquet(OUT/"tables"/"action_list"/"daily_pnl_series_newmodel.parquet")
print("wrote", FIGS/"equity_curves_newmodel.pdf", "+ daily_pnl_series_newmodel.parquet")
