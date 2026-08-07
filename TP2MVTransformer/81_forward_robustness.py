"""brief-14: forward-estimator robustness. Replaces the parity-implied forward with an independent
construction F(T) = (spot - PV(div before T)) * e^{rT} and checks whether the forward-adjusted strike
TARGETS move enough to change the detected rectangle. Retention = both adjusted-strike targets shift
< 0.5 strike (i.e., round to the same listed strike). P&L-sign check on the NEW-model selected trades
restricted to the retained episodes. Isolates the forward effect on the existing episodes (no full
chain re-enumeration). Single-next-dividend approximation (near maturities usually span <=1 ex-date).
Artifact + CSV.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent; ROOT = OUT.parent; ACTION = OUT/"tables"/"action_list"; NESTED = ROOT/"FeatureDataDocs_2026-08-04"/"nested"
e = pd.read_parquet(ROOT/"data/ml/archive2_ml_episode_dataset_2013_2026.parquet",
    columns=["episode_id", "signal_date", "feature_A_strike", "feature_B_strike", "feature_F_T1", "feature_F_T2", "feature_T1_DTE", "feature_T2_DTE"])
e["signal_date"] = pd.to_datetime(e["signal_date"]); e["year"] = e.signal_date.dt.year; e = e[e.year >= 2021].copy()
r = pd.read_parquet(ROOT/"data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet",
    columns=["date", "spy_close", "risk_free_rate", "next_ex_div_date", "next_div_amount"]).drop_duplicates("date")
r["date"] = pd.to_datetime(r["date"]); r["next_ex_div_date"] = pd.to_datetime(r["next_ex_div_date"])
e = e.merge(r, left_on="signal_date", right_on="date", how="left")

spot = e.spy_close.to_numpy(float); rate = e.risk_free_rate.fillna(0.04).to_numpy(float)
T1 = e.feature_T1_DTE.to_numpy(float)/365.0; T2 = e.feature_T2_DTE.to_numpy(float)/365.0
tdiv = (e.next_ex_div_date - e.signal_date).dt.days.to_numpy(float)/365.0
div = e.next_div_amount.fillna(0.0).to_numpy(float)
def pv_div(T): return np.where((tdiv > 0) & (tdiv <= T), div*np.exp(-rate*np.maximum(tdiv, 0)), 0.0)
F1i = (spot - pv_div(T1))*np.exp(rate*T1); F2i = (spot - pv_div(T2))*np.exp(rate*T2)
K1 = e.feature_A_strike.to_numpy(float); K2 = e.feature_B_strike.to_numpy(float)
Fp1 = e.feature_F_T1.to_numpy(float); Fp2 = e.feature_F_T2.to_numpy(float)
ok = np.isfinite(F1i) & np.isfinite(F2i) & (F1i > 0) & (F2i > 0) & np.isfinite(Fp1) & np.isfinite(Fp2) & (Fp1 > 0) & (Fp2 > 0)
k1_ind = K1*F2i/F1i; k2_ind = K2*F1i/F2i; k1_par = K1*Fp2/Fp1; k2_par = K2*Fp1/Fp2
shift1 = np.abs(k1_ind - k1_par); shift2 = np.abs(k2_ind - k2_par)
retained = ok & (shift1 < 0.5) & (shift2 < 0.5)
e["retained"] = retained
ret_pct = float(retained[ok].mean())
print(f"episodes (2021-26): {ok.sum():,} valid | forward-robust retention: {ret_pct*100:.1f}%")
print(f"  median |K1-target shift|: {np.nanmedian(shift1[ok]):.4f}  |K2-target shift|: {np.nanmedian(shift2[ok]):.4f} (strikes)")

# P&L-sign check on NEW-model selected trades restricted to retained episodes
retset = set(e.loc[e.retained, "episode_id"])
pnl_rows = {}
for name, f in [("MLP-5(new)", "trades_mlp5.parquet"), ("Transformer-plain-5(new)", "trades_transformer_plain.parquet"),
                ("Transformer-ALiBi-5(new)", "trades_transformer_alibi.parquet"), ("Logistic(new)", "trades_logistic.parquet")]:
    d = pd.read_parquet(NESTED/f); full = float(d.target_norm_pnl.sum()); ret = float(d[d.episode_id.isin(retset)].target_norm_pnl.sum())
    pnl_rows[name] = {"full_pnl": full, "retained_pnl": ret, "sign_unchanged": bool(np.sign(full) == np.sign(ret))}
    print(f"  {name:26} full=${full:>11,.0f}  retained=${ret:>11,.0f}  sign {'UNCHANGED' if np.sign(full)==np.sign(ret) else 'CHANGED'}")

json.dump({"n_valid_episodes": int(ok.sum()), "retention_pct": ret_pct,
           "median_K1_target_shift": float(np.nanmedian(shift1[ok])), "median_K2_target_shift": float(np.nanmedian(shift2[ok])),
           "pnl_sign_check": pnl_rows,
           "note": "brief-14. Independent forward F(T)=(spot-PV(div<T))e^{rT} vs parity-implied. Retention = both "
                   "adjusted-strike targets shift <0.5 strike (same listed contract). P&L-sign on new-model selected "
                   "trades restricted to retained episodes. Single-next-dividend approx; isolates forward on existing "
                   "episodes (no full chain re-enumeration -- that needs the detector on 4.9M quotes)."},
          open(ACTION/"forward_robustness.json", "w"), indent=2, default=float)
pd.DataFrame([{"model": k, **v} for k, v in pnl_rows.items()]).to_csv(ACTION/"forward_robustness.csv", index=False)
print("\nwrote", ACTION/"forward_robustness.json", "+ .csv")
