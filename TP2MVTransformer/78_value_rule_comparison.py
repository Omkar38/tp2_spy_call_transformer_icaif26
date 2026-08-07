"""brief-20: value-sensitive selection rule vs the p>=0.5 probability rule.
Trains a LightGBM REGRESSOR of E[R|X] (expected normalized return) per strategy per fold on the
149-candidate set (same folds/data as the committed LightGBM, script 27), selects the episode's
higher-E[R] denomination when E[R]>0 (one action per episode), and compares to the committed
LightGBM p>=0.5 classification rule (gate 19,636/$682K/1.25). metrics() verbatim from 60.
Artifact + CSV.
"""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import lightgbm as lgb

OUT = Path(__file__).resolve().parent; ACTION = OUT/"tables"/"action_list"
def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m27 = _load("lgbm27b", OUT/"27_train_lightgbm_final.py")
FEAT_PATH, LABEL_PATH, STRATEGIES, COMP, NORM, FOLDS = m27.FEAT_PATH, m27.LABEL_PATH, m27.STRATEGIES, m27.COMPARISON_STRATEGIES, m27.NORM_EXPOSURE, m27.FOLDS
ANN = np.sqrt(252)
def metrics(df, pnl):   # verbatim from 60
    r = df[pnl].astype(float); daily = df.groupby("signal_date")[pnl].sum()
    sharpe = daily.mean()/daily.std()*ANN if daily.std() > 0 else float("nan")
    dn = daily[daily < 0]; sortino = daily.mean()/dn.std()*ANN if len(dn) > 1 and dn.std() > 0 else float("nan")
    yr = df.groupby(df["signal_date"].dt.year)[pnl].sum()
    return {"n": int(len(df)), "pnl": float(r.sum()), "sharpe": float(sharpe), "sortino": float(sortino),
            "win_rate": float((r > 0).mean()), "pos_years": int((yr > 0).sum())}

# data (mirror 27/69)
feat = pd.read_parquet(FEAT_PATH); feat["signal_year"] = pd.to_datetime(feat["signal_date"]).dt.year
raw = pd.read_parquet(LABEL_PATH); expiry = raw[raw["exit_policy"] == "exit_selected_leg_expiration_payoff"].copy()
ew = expiry.pivot(index="episode_id", columns="strategy", values=["realized_return", "is_valid_backtest"])
ew.columns = [f"expiry_{v}_{s}" for v, s in ew.columns]; ew = ew.reset_index()
for s in STRATEGIES:
    ret = pd.to_numeric(ew[f"expiry_realized_return_{s}"], errors="coerce")
    ew[f"ret_{s}"] = ret.astype(float)                      # continuous return target
    ew[f"norm_pnl_{s}"] = ret.fillna(0).astype(float) * NORM
data = feat.merge(ew, on="episode_id", how="inner"); data = data[data["signal_year"] >= 2014].copy()
FEAT_ALL = [c for c in data.columns if c.startswith("feature_")] + [c for c in data.columns if c.startswith("lc_")] + ["start_normalized_severity", "signal_month", "signal_dayofweek"]
FEAT_ALL = [c for c in FEAT_ALL if c in data.columns and c != "signal_year"]
REG = {"objective": "regression", "n_estimators": 350, "learning_rate": 0.04, "num_leaves": 31,
       "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": -1, "verbose": -1, "random_state": 42}

rows = []
for s in COMP:
    sr = data[["episode_id", "signal_date", "signal_year", f"ret_{s}", f"norm_pnl_{s}"] + FEAT_ALL].dropna(subset=[f"ret_{s}"]).copy()
    for fold in FOLDS:
        trm = sr.signal_year.isin(fold["train"]); tem = sr.signal_year.isin(fold["test"])
        if sr[trm].empty or sr[tem].empty: continue
        mdl = lgb.LGBMRegressor(**REG); mdl.fit(sr.loc[trm, FEAT_ALL], sr.loc[trm, f"ret_{s}"])
        to = sr.loc[tem, ["episode_id", "signal_date", f"norm_pnl_{s}"]].copy()
        to["strategy"] = s; to["pred_ret"] = mdl.predict(sr.loc[tem, FEAT_ALL]); to = to.rename(columns={f"norm_pnl_{s}": "norm_pnl"})
        rows.append(to)
allp = pd.concat(rows, ignore_index=True); allp["signal_date"] = pd.to_datetime(allp["signal_date"])

# value rule: E[R]>0, one action/episode (highest predicted E[R])
cand = allp[allp["pred_ret"] > 0]
value_sel = cand.sort_values(["pred_ret", "strategy"], ascending=[False, True]).groupby("episode_id", as_index=False).first()
vm = metrics(value_sel, "norm_pnl")
print(f"value rule (E[R|X]>0, one action/episode): N={vm['n']:,} PnL=${vm['pnl']:,.0f} Sharpe={vm['sharpe']:.2f} +yrs={vm['pos_years']}/6")
print(f"  vs committed LightGBM p>=0.5: N=19,636 PnL=$682,411 Sharpe=1.25")

out = {"value_rule": vm, "reference_p05_lightgbm": {"n": 19636, "pnl": 682411, "sharpe": 1.25},
       "config": REG, "rule": "LightGBM regressor E[R|X]; trade higher-E[R] denomination when E[R]>0, one action/episode",
       "note": "brief-20. Value-sensitive rule beside the p>=0.5 classification rule. Same 149 features/folds as script 27. "
               "metrics() verbatim from 60."}
json.dump(out, open(ACTION/"value_rule_comparison.json", "w"), indent=2, default=float)
pd.DataFrame([{"rule": "value E[R]>0", **vm}, {"rule": "p>=0.5 (committed)", "n": 19636, "pnl": 682411, "sharpe": 1.25}]).to_csv(ACTION/"value_rule_comparison.csv", index=False)
print("wrote", ACTION/"value_rule_comparison.json", "+ .csv")
