"""[guide P0.2 logistic] Committed L2-logistic linear floor on F* (the leaky/committed 48 features).
Same protocol as the other models: per-fold train standardization, validation-year isotonic
calibration, causal p>=0.5, one action per episode. metrics()/sel_by_thresh() verbatim from 60.

This REPLACES the paper's un-artifacted 2.27 (round-1 script never committed) with a reproducible number.
Artifacts: data/logistic_l2_test_predictions.parquet, data/logistic_l2_selected_trades.parquet,
tables/action_list/logistic_l2_summary.json
"""
import json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
TP2 = ROOT / "TP2MVTransformer"; D = TP2 / "data"; OUT = TP2 / "tables" / "action_list"
ANN = np.sqrt(252)
spec = importlib.util.spec_from_file_location("bfd", TP2 / "01_build_feature_dictionary.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
FOLDS = mod.FOLDS

def metrics(df, pnl="target_norm_pnl"):  # verbatim from 60
    r = df[pnl].astype(float); daily = df.groupby("signal_date")[pnl].sum()
    sharpe = daily.mean()/daily.std()*ANN if daily.std() > 0 else float("nan")
    dn = daily[daily < 0]; sortino = daily.mean()/dn.std()*ANN if len(dn) > 1 and dn.std() > 0 else float("nan")
    eq = daily.sort_index().cumsum(); maxdd = float((eq-eq.cummax()).min()) if len(eq) else float("nan")
    yr = df.groupby(df["signal_date"].dt.year)[pnl].sum()
    return {"n": int(len(df)), "pnl": float(r.sum()), "sharpe": round(float(sharpe),4), "sortino": round(float(sortino),4),
            "maxdd": maxdd, "win_rate": round(float((r > 0).mean()),4), "mean_ret_pct": round(float(r.mean()/10.0),4),
            "skew": round(float(stats.skew(r)),4) if len(r) > 2 else float("nan"),
            "pos_years": int((yr > 0).sum()), "n_years": int(len(yr))}

def sel_by_thresh(df, thr=0.5):  # verbatim from 60
    c = df[df["predicted_prob"] >= thr]
    if c.empty: return c
    return c.sort_values(["predicted_prob","strategy"], ascending=[False,True]).groupby("episode_id", as_index=False).first()

# ---- source the LEAKY/committed 48 F* explicitly from the pre-swap backup (robust to any live dict swap) ----
BK = ROOT / "FeatureDataDocs_2026-08-04/stage2_strict/_backup/feature_dictionary.csv"
LIVE = TP2 / "tables" / "feature_dictionary.csv"
dict_src = BK if BK.exists() else LIVE
feats = pd.read_csv(dict_src).query("selected_for_transformer")["feature"].tolist()
assert len(feats) == 48, len(feats)
# assert this is the committed/leaky set (contains severity + avg_iv, the full-span signature)
assert "feature_severity" in feats and "feature_avg_iv" in feats, "not the committed F*"
print(f"F* source: {dict_src.name} | {len(feats)} features (committed/leaky, contains severity+avg_iv)")

data = pd.read_parquet(D / "tp2_t1k2_token_dataset.parquet"); data["signal_date"] = pd.to_datetime(data["signal_date"])
data = data[data.strategy.isin(["T1","K2"])]
def num(df): return df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)

CONFIG = dict(penalty="l2", C=1.0, class_weight="balanced", max_iter=5000)
preds = []
for fold in FOLDS:
    tr = data[data.signal_year.isin(fold["train"])]; va = data[data.signal_year.isin(fold["val"])]; te = data[data.signal_year.isin(fold["test"])]
    if tr.empty or va.empty or te.empty: continue
    sc = StandardScaler().fit(num(tr))
    clf = LogisticRegression(**CONFIG).fit(sc.transform(num(tr)), tr.target_profitable.astype(int))
    iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(sc.transform(num(va)))[:,1], va.target_profitable.astype(int).to_numpy())
    t = te[["episode_id","signal_date","signal_year","strategy","target_profitable","target_return","target_norm_pnl"]].copy()
    t["predicted_prob"] = iso.predict(clf.predict_proba(sc.transform(num(te)))[:,1])
    t["fold"] = fold["fold"]; preds.append(t)
pred = pd.concat(preds, ignore_index=True)
selected = sel_by_thresh(pred, 0.5)
m = metrics(selected)

pred.to_parquet(D / "logistic_l2_test_predictions.parquet", index=False)
selected.to_parquet(D / "logistic_l2_selected_trades.parquet", index=False)
summary = {"model": "L2 logistic on F* (committed/leaky 48 features)",
           "config": "penalty=l2, C=1.0, class_weight=balanced; per-fold StandardScaler; val isotonic; causal p>=0.5; one action/episode",
           "feature_set": "committed full-span F* (48)", "replaces": "paper Table 2 Logistic-L2 row (13,841/$2.02M/2.27), which had no committed source",
           **m}
json.dump(summary, open(OUT / "logistic_l2_summary.json", "w"), indent=2, default=float)
print(f"\nLogistic-L2 floor (committed F*): N={m['n']:,}  PnL=${m['pnl']:,.0f}  Sharpe={m['sharpe']:.2f}  Sortino={m['sortino']:.2f}  win={m['win_rate']*100:.1f}%  +yrs={m['pos_years']}/{m['n_years']}")
print("wrote data/logistic_l2_{test_predictions,selected_trades}.parquet and tables/action_list/logistic_l2_summary.json")
