"""brief-13: same-input LightGBM-on-48 F* row for Table 2.
Replicates the committed LightGBM pipeline (script 27) exactly -- imports its constants,
LGBM_PARAMS, FOLDS, and fit_isotonic_calibration -- and runs it on (i) the full 149 candidate
set as a GATE (must reproduce 19,636 / $682K / 1.25) and (ii) the 48-feature F* as the
same-input control. metrics()/sel_by_thresh() are copied VERBATIM from 60_action_list_compute.py.
Artifact: tables/action_list/lgbm_f48_comparison.json. A failed gate stops the task.
"""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import lightgbm as lgb

OUT = Path(__file__).resolve().parent
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
m27 = _load("lgbm27", OUT/"27_train_lightgbm_final.py")
FEAT_PATH, LABEL_PATH = m27.FEAT_PATH, m27.LABEL_PATH
STRATEGIES, COMP, NORM = m27.STRATEGIES, m27.COMPARISON_STRATEGIES, m27.NORM_EXPOSURE
THR, FOLDS, LGBM_PARAMS = m27.SELECTION_THRESHOLD, m27.FOLDS, m27.LGBM_PARAMS
fit_iso = m27.fit_isotonic_calibration
ACTION = OUT/"tables"/"action_list"

# ---------- metrics() + sel_by_thresh() VERBATIM from 60_action_list_compute.py ----------
ANN = np.sqrt(252)
def metrics(df, pnl):
    r = df[pnl].astype(float)
    daily = df.groupby("signal_date")[pnl].sum()
    sharpe = daily.mean() / daily.std() * ANN if daily.std() > 0 else float("nan")
    dn = daily[daily < 0]
    sortino = daily.mean() / dn.std() * ANN if len(dn) > 1 and dn.std() > 0 else float("nan")
    eq = daily.sort_index().cumsum()
    maxdd = float((eq - eq.cummax()).min()) if len(eq) else float("nan")
    yr = df.groupby(df["signal_date"].dt.year)[pnl].sum()
    return {"n": int(len(df)), "pnl": float(r.sum()), "sharpe": float(sharpe), "sortino": float(sortino),
            "maxdd": maxdd, "win_rate": float((r > 0).mean()), "mean_ret_pct": float(r.mean() / 10.0),
            "skew": float(stats.skew(r)) if len(r) > 2 else float("nan"),
            "pos_years": int((yr > 0).sum()), "n_years": int(len(yr))}
def sel_by_thresh(pred, thr, valid_filter=True):
    df = pred.copy()
    if "is_valid" in df.columns and valid_filter: df = df[df["is_valid"].astype(bool)]
    if "strategy" in df.columns: df = df[df["strategy"].isin(["T1", "K2"])]
    cand = df[df["predicted_prob"] >= thr]
    if cand.empty or "episode_id" not in cand.columns: return cand
    return (cand.sort_values(["predicted_prob", "strategy"], ascending=[False, True])
                .groupby("episode_id", as_index=False).first())

# ---------- data loading (replicates 27 main lines 128-157) ----------
feat = pd.read_parquet(FEAT_PATH); feat["signal_year"] = pd.to_datetime(feat["signal_date"]).dt.year
raw = pd.read_parquet(LABEL_PATH)
expiry = raw[raw["exit_policy"] == "exit_selected_leg_expiration_payoff"].copy()
ew = expiry.pivot(index="episode_id", columns="strategy", values=["realized_return", "net_pnl", "is_valid_backtest"])
ew.columns = [f"expiry_{v}_{s}" for v, s in ew.columns]; ew = ew.reset_index()
for s in STRATEGIES:
    vc, rc = f"expiry_is_valid_backtest_{s}", f"expiry_realized_return_{s}"
    ew[f"label_expiry_{s}_profitable"] = (ew[vc].fillna(False) & (ew[rc].fillna(-1) > 0)).astype(int)
    ew[f"norm_pnl_{s}"] = ew[rc].fillna(0) * NORM
data = feat.merge(ew, on="episode_id", how="inner"); data = data[data["signal_year"] >= 2014].copy()

FEAT_ALL = ([c for c in data.columns if c.startswith("feature_")] + [c for c in data.columns if c.startswith("lc_")]
            + ["start_normalized_severity", "signal_month", "signal_dayofweek"])
FEAT_ALL = [c for c in FEAT_ALL if c in data.columns and c != "signal_year"]
F48 = pd.read_csv(OUT/"tables"/"feature_dictionary.csv")
F48 = F48[F48["selected_for_transformer"].astype(bool)]["feature"].tolist()
F48 = [c for c in F48 if c in data.columns]
print(f"149-set: {len(FEAT_ALL)} cols | 48-set: {len(F48)} cols (present in data)")

def run(feature_cols):
    test_rows = []
    for s in STRATEGIES:
        lc, nc = f"label_expiry_{s}_profitable", f"norm_pnl_{s}"
        sr = data[["episode_id", "signal_date", "signal_year", lc, nc] + feature_cols].dropna(subset=[lc]).copy()
        for fold in FOLDS:
            trm = sr["signal_year"].isin(fold["train"]); vm = sr["signal_year"].isin(fold["val"]); tem = sr["signal_year"].isin(fold["test"])
            Xtr, ytr = sr.loc[trm, feature_cols], sr.loc[trm, lc]; Xval, yval = sr.loc[vm, feature_cols], sr.loc[vm, lc]; Xte = sr.loc[tem, feature_cols]
            if Xtr.empty or Xval.empty or Xte.empty or ytr.nunique() < 2: continue
            mdl = lgb.LGBMClassifier(**LGBM_PARAMS); mdl.fit(Xtr, ytr, eval_set=[(Xval, yval)])
            iso = fit_iso(mdl.predict_proba(Xval)[:, 1], yval.to_numpy())
            to = sr.loc[tem, ["episode_id", "signal_date", "signal_year", nc]].copy()
            to["strategy"] = s; to["predicted_prob"] = iso.predict(mdl.predict_proba(Xte)[:, 1]); to = to.rename(columns={nc: "norm_pnl"})
            test_rows.append(to)
    allpred = pd.concat(test_rows, ignore_index=True); allpred["signal_date"] = pd.to_datetime(allpred["signal_date"])
    sel = sel_by_thresh(allpred[allpred["strategy"].isin(COMP)], THR)
    return metrics(sel, "norm_pnl")

print("\n=== GATE: LightGBM on 149 (must reproduce 19,636 / $682K / 1.25) ===")
g = run(FEAT_ALL); print(f"  149: N={g['n']:,} PnL=${g['pnl']:,.0f} Sharpe={g['sharpe']:.2f}")
gate_ok = (g["n"] == 19636) and (abs(g["pnl"] - 682411) < 5000) and (abs(g["sharpe"] - 1.25) < 0.03)
print(f"  GATE {'PASS' if gate_ok else '*** FAIL ***'}")
assert gate_ok, "149 gate failed -- stopping (do not trust the 48 row)"

print("\n=== brief-13: same-input LightGBM on 48 F* ===")
c48 = run(F48); print(f"  48: N={c48['n']:,} PnL=${c48['pnl']:,.0f} Sharpe={c48['sharpe']:.2f} Sortino={c48['sortino']:.2f} win={c48['win_rate']*100:.1f}% +yrs={c48['pos_years']}/6")

out = {"gate_149": g, "lgbm_f48": c48, "n_features_149": len(FEAT_ALL), "n_features_48": len(F48),
       "note": "Same-input control: LightGBM retrained on the 48 F* (MLP/transformer feature set). "
               "149 gate reproduced exactly. metrics()/sel_by_thresh copied verbatim from 60_action_list_compute.py.",
       "config": {k: (v if not isinstance(v, float) else v) for k, v in LGBM_PARAMS.items()}}
json.dump(out, open(ACTION/"lgbm_f48_comparison.json", "w"), indent=2, default=float)
print("\nwrote", ACTION/"lgbm_f48_comparison.json")
