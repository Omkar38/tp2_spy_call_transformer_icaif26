"""P3.3: paired exit-policy check. For selected trades that have a valid reversion exit, compare the
SAME trades' P&L under hold-to-expiry (the chosen policy) vs exit-at-reversion. Paired (same episodes),
so it isolates the exit-policy effect. Confirms whether hold-to-expiry beats the reversion exit.
Uses new-model MLP-5 selected trades (and the union across new models). NORM=1000. Artifact + CSV.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent; ROOT = OUT.parent; ACTION = OUT/"tables"/"action_list"; NESTED = ROOT/"FeatureDataDocs_2026-08-04"/"nested"
NORM = 1000.0; ANN = np.sqrt(252)
lab = pd.read_parquet(ROOT/"data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet",
                      columns=["episode_id", "strategy", "signal_date", "exit_policy", "is_valid_backtest", "realized_return"])
lab = lab[lab.strategy.isin(["T1", "K2"])].copy(); lab["signal_date"] = pd.to_datetime(lab.signal_date)
exp = lab[lab.exit_policy == "exit_selected_leg_expiration_payoff"][["episode_id", "strategy", "signal_date", "realized_return", "is_valid_backtest"]].rename(columns={"realized_return": "ret_expiry", "is_valid_backtest": "valid_expiry"})
rev = lab[lab.exit_policy == "exit_at_reversion_day"][["episode_id", "strategy", "realized_return", "is_valid_backtest"]].rename(columns={"realized_return": "ret_reversion", "is_valid_backtest": "valid_reversion"})
pair = exp.merge(rev, on=["episode_id", "strategy"], how="inner")
pair = pair[pair.valid_expiry & pair.valid_reversion]   # both policies valid

def paired(sel_eps, name):
    d = pair.merge(sel_eps, on=["episode_id", "strategy"], how="inner")
    d = d[d.signal_date.dt.year >= 2021]
    if len(d) == 0: return {"n": 0}
    pe = (d.ret_expiry * NORM); pr = (d.ret_reversion * NORM)
    def sh(x, dt):
        daily = pd.Series(x.to_numpy(), index=dt).groupby(level=0).sum(); return float(daily.mean()/daily.std()*ANN) if daily.std() > 0 else float("nan")
    out = {"n": int(len(d)), "expiry_pnl": float(pe.sum()), "reversion_pnl": float(pr.sum()),
           "paired_diff_expiry_minus_reversion": float((pe - pr).sum()),
           "expiry_win_vs_reversion": float((pe > pr).mean()),
           "expiry_sharpe": sh(pe, d.signal_date), "reversion_sharpe": sh(pr, d.signal_date)}
    print(f"{name}: n={out['n']:,} | expiry=${out['expiry_pnl']:,.0f} (Sh {out['expiry_sharpe']:.2f}) vs reversion=${out['reversion_pnl']:,.0f} (Sh {out['reversion_sharpe']:.2f}) | "
          f"expiry-reversion=${out['paired_diff_expiry_minus_reversion']:,.0f}, expiry>reversion {out['expiry_win_vs_reversion']*100:.0f}%")
    return out

result = {}
for name, f in [("MLP-5(new)", "trades_mlp5.parquet"), ("Transformer-ALiBi-5(new)", "trades_transformer_alibi.parquet")]:
    sel = pd.read_parquet(NESTED/f)[["episode_id", "strategy"]].drop_duplicates()
    result[name] = paired(sel, name)
# union of all new-model selections
allsel = pd.concat([pd.read_parquet(NESTED/f)[["episode_id", "strategy"]] for f in ["trades_mlp5.parquet", "trades_transformer_alibi.parquet", "trades_transformer_plain.parquet", "trades_logistic.parquet"]]).drop_duplicates()
result["union_new_models"] = paired(allsel, "union new models")

json.dump({"models": result, "note": "P3.3 paired exit-policy check (same trades, expiry vs reversion), on trades with "
           "both policies valid, 2021-26. Positive expiry-minus-reversion confirms hold-to-expiry is the better choice. "
           "The committed exit_policy_comparison.json reported the reversion side only (171 trades, -$2,697)."},
          open(ACTION/"exit_policy_paired.json", "w"), indent=2, default=float)
pd.DataFrame([{"model": k, **v} for k, v in result.items()]).to_csv(ACTION/"exit_policy_paired.csv", index=False)
print("wrote", ACTION/"exit_policy_paired.json", "+ .csv")
