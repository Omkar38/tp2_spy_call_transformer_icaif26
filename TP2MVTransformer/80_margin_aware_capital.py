"""brief-19: margin-aware capital (Reg-T-style spread margin) for the cash-budgeted backtest.
Reg-T per 1-contract rectangle (x100 multiplier): short legs C,D (sold calls) each require
max(0.20*U - OTM, 0.10*U) + premium; long legs A,B are premium debits. U = spy_close.
The cash-budget then reserves MARGIN, not premium: a deployed dollar of margin controls
premium/margin of notional, so effective_return = realized_return * (entry_gross_exposure/margin).
Reuses script 42's committed trade sets + backtest. GATE: premium-based reproduces Table 3.
Clearly a Reg-T conservative approximation (naked short legs; OCC spread netting would be lower).
Artifact + CSV.
"""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent; ROOT = OUT.parent; ACTION = OUT/"tables"/"action_list"
s = importlib.util.spec_from_file_location("cap42m", OUT/"42_capital_scaled_portfolio_backtest.py")
cap = importlib.util.module_from_spec(s); s.loader.exec_module(cap)
MULT = 100.0

# underlying per date
u = pd.read_parquet(ROOT/"data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet", columns=["date", "spy_close"]).drop_duplicates("date")
u["date"] = pd.to_datetime(u["date"])
# per-episode legs for margin
lg = pd.read_parquet(ROOT/"data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet",
                     columns=["episode_id", "strategy", "signal_date", "exit_policy", "is_valid_backtest",
                              "feature_A_mid", "feature_B_mid", "feature_C_mid", "feature_D_mid",
                              "feature_C_strike", "feature_D_strike"])
lg = lg[(lg.exit_policy == "exit_selected_leg_expiration_payoff") & lg.is_valid_backtest & lg.strategy.isin(["T1", "K2"])].copy()
lg["signal_date"] = pd.to_datetime(lg.signal_date); lg = lg.merge(u, left_on="signal_date", right_on="date", how="left")
U = lg["spy_close"].to_numpy(float)
def regt_short(strike, mid):
    otm = np.maximum(0.0, strike - U)
    return (np.maximum(0.20 * U - otm, 0.10 * U) + mid) * MULT
lg["margin"] = (regt_short(lg.feature_C_strike.to_numpy(float), lg.feature_C_mid.to_numpy(float))
                + regt_short(lg.feature_D_strike.to_numpy(float), lg.feature_D_mid.to_numpy(float))
                + (lg.feature_A_mid.to_numpy(float) + lg.feature_B_mid.to_numpy(float)) * MULT)
margin = lg.groupby("episode_id")["margin"].first()   # per-episode (same rectangle for T1/K2)

labels = cap.load_labels(); tcal = cap.load_trading_calendar()
GATE = {"LightGBM": 0.146, "MLP": 0.172, "Transformer": 0.006}
result = {}; csv_rows = []
print(f"{'model':13} {'premium CAGR':>14} {'(Table3)':>10} {'margin CAGR':>13} {'median margin/premium':>22}")
for name in ["LightGBM", "MLP", "Transformer"]:
    tr = cap.build_trade_set(name, labels).copy()
    tr = tr.merge(margin.rename("margin"), on="episode_id", how="left")
    tr = tr[tr["margin"].notna() & (tr["margin"] > 0)]
    ratio = (tr["entry_gross_exposure"] / tr["margin"]).median()
    frac = cap.run_capital_scaled_backtest(tr, tcal)                       # premium-based (Table 3 gate)
    assert abs(frac["cagr"] - GATE[name]) < 0.01, f"{name} premium gate FAILED {frac['cagr']:.3f}"
    trm = tr.copy(); trm["realized_return"] = trm["realized_return"] * (trm["entry_gross_exposure"] / trm["margin"])
    marg = cap.run_capital_scaled_backtest(trm, tcal)                      # margin-aware
    result[name] = {"premium_cagr": frac["cagr"], "margin_cagr": marg["cagr"],
                    "premium_ending": frac["ending_capital"], "margin_ending": marg["ending_capital"],
                    "median_premium_over_margin": float(ratio)}
    csv_rows.append({"model": name, "premium_cagr": frac["cagr"], "margin_cagr": marg["cagr"], "median_premium_over_margin": float(ratio)})
    print(f"{name:13} {frac['cagr']*100:>13.1f}% {'('+str(GATE[name]*100)+'%)':>10} {marg['cagr']*100:>12.1f}% {ratio:>22.3f}")

json.dump({"models": result, "margin_rule": "Reg-T conservative: short C,D = max(0.20U-OTM,0.10U)+premium; long A,B = premium debit; x100",
           "note": "brief-19. Premium-based gate reproduces Table 3. Margin-aware reserves Reg-T margin, so "
                   "effective return = realized_return * premium/margin (margin > premium dilutes return on capital). "
                   "Reg-T conservative (naked short legs); OCC spread netting would give a lower margin / higher CAGR. "
                   "Reuses script 42. Underlying = spy_close."},
          open(ACTION/"margin_aware_capital.json", "w"), indent=2, default=float)
pd.DataFrame(csv_rows).to_csv(ACTION/"margin_aware_capital.csv", index=False)
print("\nwrote", ACTION/"margin_aware_capital.json", "+ .csv")
