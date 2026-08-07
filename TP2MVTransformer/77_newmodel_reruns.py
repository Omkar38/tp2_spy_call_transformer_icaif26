"""New-model reruns (per user directive): redo the professor's analyses on the NEW perfect-pipeline
models -- LightGBM stays OLD 149-feature (no leak), the rest are the frozen-F* new pipeline.
Redoes: Table 2 (new), ex-div exclusion, clustered inference, quote-quality ladder -- all on the
new-model selected trades. Calibration (Brier/ECE) needs full-population predictions the new pipeline
did not save -> selected-tail (mean p, win) done where predicted_prob is present; noted otherwise.
metrics()/ex-div/quote logic mirror scripts 60/72/75. Artifacts *_newmodel.json + CSV.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent; ROOT = OUT.parent; ACTION = OUT/"tables"/"action_list"
NESTED = ROOT/"FeatureDataDocs_2026-08-04"/"nested"
LABELS = ROOT/"data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet"
DIVDATE = ROOT/"data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet"
EPISODES = ROOT/"data/ml/archive2_ml_episode_dataset_2013_2026.parquet"
ANN = np.sqrt(252); rng = np.random.default_rng(42); NB = 2000; BLOCK = 10

def metrics(df, pnl):   # verbatim from 60
    r = df[pnl].astype(float); daily = df.groupby("signal_date")[pnl].sum()
    sharpe = daily.mean()/daily.std()*ANN if daily.std() > 0 else float("nan")
    dn = daily[daily < 0]; sortino = daily.mean()/dn.std()*ANN if len(dn) > 1 and dn.std() > 0 else float("nan")
    yr = df.groupby(df["signal_date"].dt.year)[pnl].sum()
    return {"n": int(len(df)), "pnl": float(r.sum()), "sharpe": float(sharpe), "sortino": float(sortino),
            "win_rate": float((r > 0).mean()), "skew": float(stats.skew(r)) if len(r) > 2 else float("nan"),
            "pos_years": int((yr > 0).sum())}

# ---- the new-model set (LightGBM = old 149; rest new) ----
MODELS = {
    "LightGBM(149,old)": (OUT/"data/lgbm_final_t1k2_selected_trades.parquet", "norm_pnl"),
    "MLP-5(new)":        (NESTED/"trades_mlp5.parquet", "target_norm_pnl"),
    "Transformer-plain-5(new)": (NESTED/"trades_transformer_plain.parquet", "target_norm_pnl"),
    "Transformer-ALiBi-5(new)": (NESTED/"trades_transformer_alibi.parquet", "target_norm_pnl"),
    "Logistic(new)":     (NESTED/"trades_logistic.parquet", "target_norm_pnl"),
}
def load(p, col):
    d = pd.read_parquet(p); d = d[["episode_id", "signal_date", "strategy", col] + (["predicted_prob"] if "predicted_prob" in d.columns else [])].copy()
    d = d.rename(columns={col: "pnl"}); d["signal_date"] = pd.to_datetime(d["signal_date"]); return d
trades = {k: load(p, c) for k, (p, c) in MODELS.items()}

# ex-div flag (mirror 72)
lab = pd.read_parquet(LABELS, columns=["episode_id", "strategy", "signal_date", "exit_policy", "feature_D_expiration"])
lab = lab[(lab.exit_policy == "exit_selected_leg_expiration_payoff") & lab.strategy.isin(["T1", "K2"])].copy()
lab["signal_date"] = pd.to_datetime(lab.signal_date); lab["feature_D_expiration"] = pd.to_datetime(lab.feature_D_expiration)
dv = pd.read_parquet(DIVDATE, columns=["date", "next_ex_div_date"]).drop_duplicates("date")
dv["date"] = pd.to_datetime(dv.date); dv["next_ex_div_date"] = pd.to_datetime(dv.next_ex_div_date)
lab = lab.merge(dv, left_on="signal_date", right_on="date", how="left")
lab["exdiv_exposed"] = lab.next_ex_div_date.notna() & (lab.next_ex_div_date <= lab.feature_D_expiration)
exflag = lab[["episode_id", "strategy", "exdiv_exposed"]].drop_duplicates(["episode_id", "strategy"])

# quote quality (mirror 75)
legs = ["A", "B", "C", "D"]; qcols = ["episode_id"] + [f"feature_{l}_{q}" for l in legs for q in ("spread_pct", "open_interest", "volume")]
ep = pd.read_parquet(EPISODES, columns=qcols)
ep["max_spread_pct"] = ep[[f"feature_{l}_spread_pct" for l in legs]].max(axis=1)
ep["min_oi"] = ep[[f"feature_{l}_open_interest" for l in legs]].min(axis=1)
ep["min_vol"] = ep[[f"feature_{l}_volume" for l in legs]].min(axis=1)
qq = ep[["episode_id", "max_spread_pct", "min_oi", "min_vol"]]

def block_boot(daily):
    x = daily.to_numpy(); n = len(x); nb = int(np.ceil(n/BLOCK)); maxs = n-BLOCK; pnl = np.empty(NB); sh = np.empty(NB)
    for i in range(NB):
        idx = np.concatenate([np.arange(s, s+BLOCK) for s in rng.integers(0, maxs+1, size=nb)])[:n]
        r = x[idx]; pnl[i] = r.sum(); sh[i] = r.mean()/r.std()*ANN if r.std() > 0 else np.nan
    return [float(np.nanpercentile(pnl, 2.5)), float(np.nanpercentile(pnl, 97.5))], float((pnl > 0).mean())

RUNGS = [("base", lambda x: x.index == x.index), ("spread<=10%", lambda x: x.max_spread_pct <= 0.10),
         ("min_OI>=10", lambda x: x.min_oi >= 10), ("all(spread<=10%,OI>=10,vol>=1)", lambda x: (x.max_spread_pct <= 0.10) & (x.min_oi >= 10) & (x.min_vol >= 1))]

table2, exdiv, clustered, quote, seltail = {}, {}, {}, {}, {}
print(f"{'model':28} {'N':>7} {'PnL':>12} {'Sharpe':>7} {'+yrs':>5} {'exdiv%':>7} {'excl Sh':>8} {'PnL CI incl 0?':>14}")
for name, d in trades.items():
    m = metrics(d, "pnl"); table2[name] = m
    de = d.merge(exflag, on=["episode_id", "strategy"], how="left"); de["exdiv_exposed"] = de.exdiv_exposed.fillna(False)
    exm = metrics(de[~de.exdiv_exposed], "pnl"); exdiv[name] = {"full": m, "excl_exdiv": exm, "pct_exposed": float(de.exdiv_exposed.mean())}
    ci, pgt0 = block_boot(d.groupby("signal_date")["pnl"].sum().sort_index()); clustered[name] = {"pnl_point": m["pnl"], "pnl_ci_dateblock": ci, "pnl_gt0_prob": pgt0}
    dq = d.merge(qq, on="episode_id", how="left"); quote[name] = {lab_: metrics(dq[pred(dq)], "pnl") for lab_, pred in RUNGS}
    if "predicted_prob" in d.columns:
        c = d[d.predicted_prob >= 0.5]; seltail[name] = {"mean_selected_p": float(c.predicted_prob.mean()) if len(c) else float("nan"),
                                                          "selected_win_rate": float((c.pnl > 0).mean()) if len(c) else float("nan")}
    print(f"{name:28} {m['n']:>7,} ${m['pnl']:>11,.0f} {m['sharpe']:>7.2f} {m['pos_years']:>4}/6 "
          f"{exdiv[name]['pct_exposed']*100:>6.1f}% {exm['sharpe']:>8.2f} {'YES' if ci[0] <= 0 <= ci[1] else 'no':>14}")

out = {"model_set": "LightGBM=old 149 (no leak); MLP-5/plain-5/ALiBi-5/logistic = new frozen-F* pipeline",
       "table2_newmodel": table2, "exdiv_newmodel": exdiv, "clustered_newmodel": clustered,
       "quote_ladder_newmodel": quote, "selected_tail_newmodel": seltail,
       "calibration_note": "full-population Brier/ECE need full predictions (new pipeline saved only selected trades); "
                           "selected-tail (mean p, win) provided where predicted_prob present.",
       "note": "New-model reruns per user directive. metrics()/ex-div/quote logic mirror 60/72/75."}
json.dump(out, open(ACTION/"newmodel_reruns.json", "w"), indent=2, default=float)
pd.DataFrame([{"model": k, **v} for k, v in table2.items()]).to_csv(ACTION/"newmodel_table2.csv", index=False)
print("\nwrote", ACTION/"newmodel_reruns.json", "+ newmodel_table2.csv")
