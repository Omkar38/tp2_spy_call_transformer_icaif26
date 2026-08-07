"""Task 2.4 / brief-15 remaining: per-model selected-trade P&L by theoretical regime (R1/R2/R3),
on the NEW models (LightGBM old-149). Regimes from data/ml/episode_regime_labels.parquet (per episode):
R1 = certified-European (no_div + small_div), R2 = dividend-robust, R3 = unresolved. Also the unconditional
R1/R2-only P&L (closes the decomposition rung-5). metrics() daily convention. Artifact + CSV.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent; ROOT = OUT.parent; DATA = OUT/"data"; ACTION = OUT/"tables"/"action_list"; NESTED = ROOT/"FeatureDataDocs_2026-08-04"/"nested"
ANN = np.sqrt(252)
reg = pd.read_parquet(ROOT/"data/ml/episode_regime_labels.parquet", columns=["episode_id", "regime"]).drop_duplicates("episode_id")
reg["R"] = reg["regime"].map({"R1_no_div": "R1", "R1_small_div": "R1", "R2_dividend_robust": "R2", "R3_unresolved": "R3"})
R = reg.set_index("episode_id")["R"]

def mets(x, col):
    if len(x) == 0: return {"n": 0, "pnl": 0.0, "sharpe": float("nan"), "win_rate": float("nan")}
    daily = x.groupby("signal_date")[col].sum(); sh = daily.mean()/daily.std()*ANN if daily.std() > 0 else float("nan")
    return {"n": int(len(x)), "pnl": float(x[col].sum()), "sharpe": round(float(sh), 3), "win_rate": round(float((x[col] > 0).mean()), 3)}

MODELS = {"LightGBM(149,old)": (DATA/"lgbm_final_t1k2_selected_trades.parquet", "norm_pnl"),
          "MLP-5(new)": (NESTED/"trades_mlp5.parquet", "target_norm_pnl"),
          "Transformer-plain-5(new)": (NESTED/"trades_transformer_plain.parquet", "target_norm_pnl"),
          "Transformer-ALiBi-5(new)": (NESTED/"trades_transformer_alibi.parquet", "target_norm_pnl"),
          "Logistic(new)": (NESTED/"trades_logistic.parquet", "target_norm_pnl")}
result = {}; csv_rows = []
print(f"{'model':26} {'R1 PnL':>12} {'R2 PnL':>10} {'R3 PnL':>12}  (share of P&L)")
for name, (f, col) in MODELS.items():
    d = pd.read_parquet(f); d["signal_date"] = pd.to_datetime(d["signal_date"]); d["R"] = d["episode_id"].map(R).fillna("R_unknown")
    tot = d[col].sum(); by = {}
    for rg in ["R1", "R2", "R3", "R_unknown"]:
        by[rg] = mets(d[d["R"] == rg], col); by[rg]["pnl_share"] = round(float(by[rg]["pnl"]/tot), 3) if tot else float("nan")
    result[name] = {"total_pnl": float(tot), "by_regime": by}
    csv_rows.append({"model": name, "R1_pnl": by["R1"]["pnl"], "R2_pnl": by["R2"]["pnl"], "R3_pnl": by["R3"]["pnl"],
                     "R1_share": by["R1"]["pnl_share"], "R3_share": by["R3"]["pnl_share"]})
    print(f"{name:26} ${by['R1']['pnl']:>11,.0f} ${by['R2']['pnl']:>9,.0f} ${by['R3']['pnl']:>11,.0f}  "
          f"(R1 {by['R1']['pnl_share']*100:.0f}% / R3 {by['R3']['pnl_share']*100:.0f}%)")

# unconditional R1/R2-only (decomposition rung-5)
lab = pd.read_parquet(ROOT/"data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet",
                      columns=["episode_id", "strategy", "signal_date", "exit_policy", "is_valid_backtest", "net_pnl", "entry_gross_exposure"])
lab = lab[(lab.exit_policy == "exit_selected_leg_expiration_payoff") & lab.is_valid_backtest & lab.strategy.isin(["T1", "K2"])].copy()
lab["signal_date"] = pd.to_datetime(lab.signal_date); lab = lab[lab.signal_date.dt.year >= 2021]
lab["net_norm"] = lab.net_pnl/lab.entry_gross_exposure*1000; lab["R"] = lab["episode_id"].map(R).fillna("R_unknown")
uncond_r1r2 = mets(lab[lab.R.isin(["R1", "R2"])], "net_norm")
print(f"\nunconditional R1/R2-only net: N={uncond_r1r2['n']:,} PnL=${uncond_r1r2['pnl']:,.0f} (vs all -$12.52M)")

json.dump({"per_model_by_regime": result, "unconditional_R1R2_only": uncond_r1r2,
           "regime_map": "R1=no_div+small_div (certified-European), R2=dividend_robust, R3=unresolved",
           "note": "Task 2.4 / brief-15 remaining, on the NEW models (LightGBM old-149). Closes decomposition rung-5."},
          open(ACTION/"regime_stratified_performance.json", "w"), indent=2, default=float)
pd.DataFrame(csv_rows).to_csv(ACTION/"regime_stratified_performance.csv", index=False)
print("wrote", ACTION/"regime_stratified_performance.json", "+ .csv")
