"""brief-25: SPY decomposition ladder -- how the SPX->SPY reversal accrues, rung by rung.
mid (SPX external) -> bid/ask -> +commissions -> full model -> no exposed shorts -> R1/R2 only.
Exact rungs use committed label columns (price_pnl = bid/ask pre-commission; net_pnl = +commissions;
gate: unconditional net = -$12.52M). Model rung = each model's committed net; ex-div rung reuses
brief-16. mid (SPX 95-100% win) is external; R1/R2-only needs the pending episode regime labels.
metrics() daily convention. Artifact + CSV.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent; ROOT = OUT.parent; ACTION = OUT/"tables"/"action_list"
LABELS = ROOT/"data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet"
NORM = 1000.0; ANN = np.sqrt(252)

d = pd.read_parquet(LABELS, columns=["episode_id", "strategy", "signal_date", "exit_policy", "is_valid_backtest",
                                     "price_pnl", "transaction_cost", "net_pnl", "entry_gross_exposure", "realized_return"])
d = d[(d.exit_policy == "exit_selected_leg_expiration_payoff") & d.is_valid_backtest & d.strategy.isin(["T1", "K2"])].copy()
d["signal_date"] = pd.to_datetime(d["signal_date"]); d = d[d.signal_date.dt.year >= 2021]
d["bidask_norm"] = d.price_pnl / d.entry_gross_exposure * NORM        # bid/ask, pre-commission
d["net_norm"] = d.net_pnl / d.entry_gross_exposure * NORM             # +commissions (= target_norm_pnl)

def dsharpe(df, col):
    daily = df.groupby("signal_date")[col].sum()
    return float(daily.mean() / daily.std() * ANN) if daily.std() > 0 else float("nan")

ladder = []
def rung(name, df, col, note=""):
    r = {"rung": name, "n": int(len(df)), "pnl": float(df[col].sum()),
         "win_rate": float((df[col] > 0).mean()), "sharpe": dsharpe(df, col), "note": note}
    ladder.append(r)
    print(f"  {name:34} N={r['n']:>7,} PnL=${r['pnl']:>13,.0f} win={r['win_rate']*100:>5.1f}% Sh={r['sharpe']:>6.2f}")
    return r

print("=== SPY decomposition ladder (unconditional, valid T1/K2 2021-26) ===")
ladder.append({"rung": "0. SPX at mid (external)", "note": "SPX four-contract structure wins 95-100% at mid (Glasserman); not recomputed here"})
print("  0. SPX at mid (external): 95-100% win on SPX (reference)")
rung("1. SPY bid/ask (pre-commission)", d, "bidask_norm", "conservative execution, no commissions")
gate = rung("2. SPY +commissions (unconditional net)", d, "net_norm", "GATE: paper unconditional -$12.52M")
assert abs(gate["pnl"] + 12521429) < 5000, "unconditional net gate FAILED"
comm = d["bidask_norm"].sum() - d["net_norm"].sum()
print(f"     -> commission drag: ${comm:,.0f} (turns +${d['bidask_norm'].sum():,.0f} into ${d['net_norm'].sum():,.0f})")

# rung 3 (full model) + rung 4 (no exposed shorts): from committed selected trades + brief-16
exdiv = json.load(open(ACTION/"newmodel_reruns.json"))["exdiv_newmodel"]  # NEW-model rungs
print("\n  3. full-model net / 4. no-exposed-shorts (per model, from Table 2 + brief-16):")
model_rungs = {}
for m, v in exdiv.items():
    model_rungs[m] = {"full_model_pnl": v["full"]["pnl"], "full_model_sharpe": v["full"]["sharpe"],
                      "no_exdiv_pnl": v["excl_exdiv"]["pnl"], "no_exdiv_sharpe": v["excl_exdiv"]["sharpe"]}
    print(f"     {m:22} full=${v['full']['pnl']:>11,.0f} (Sh {v['full']['sharpe']:.2f}) | no-exdiv=${v['excl_exdiv']['pnl']:>11,.0f} (Sh {v['excl_exdiv']['sharpe']:.2f})")

out = {"unconditional_ladder": ladder, "commission_drag": float(comm), "model_rungs": model_rungs,
       "rung5_R1R2_only_unconditional_net": json.load(open(ACTION/"regime_stratified_performance.json"))["unconditional_R1R2_only"],
       "note": "brief-25. Rungs 1-2 exact from committed label P&L (net gate -$12.52M reproduced). "
               "Headline: the structure is +$61M at bid/ask; commissions (-$73.5M at $0.65/contract/side) drive "
               "the unconditional loss; model selection then recovers positive P&L; survives ex-div exclusion. "
               "mid rung is the external SPX reference; R1/R2 rung pending regime labels."}
json.dump(out, open(ACTION/"spy_decomposition.json", "w"), indent=2, default=float)
pd.DataFrame(ladder).to_csv(ACTION/"spy_decomposition.csv", index=False)
print("\nwrote", ACTION/"spy_decomposition.json", "+ .csv")
