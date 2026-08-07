"""[guide P0.2 logistic gate] Reproduce the L2-logistic linear floor as a committed
artifact (rule 2). Same protocol as the other models: F* features, per-fold train
standardization, validation-year isotonic calibration, causal p>=0.5, one action/episode.
metrics()/sel_by_thresh() copied verbatim from 60. Saves predictions + summary; also
computes the coverage-rule (top-14K) row to fill the logistic gap in two_rule_comparison.json.
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
ANN = np.sqrt(252); COVER_N = 14000
spec = importlib.util.spec_from_file_location("bfd", TP2 / "01_build_feature_dictionary.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
FOLDS = mod.FOLDS

def metrics(df, pnl="target_norm_pnl"):  # verbatim from 60
    r = df[pnl].astype(float); daily = df.groupby("signal_date")[pnl].sum()
    sharpe = daily.mean()/daily.std()*ANN if daily.std() > 0 else float("nan")
    dn = daily[daily < 0]; sortino = daily.mean()/dn.std()*ANN if len(dn) > 1 and dn.std() > 0 else float("nan")
    eq = daily.sort_index().cumsum(); maxdd = float((eq-eq.cummax()).min()) if len(eq) else float("nan")
    yr = df.groupby(df["signal_date"].dt.year)[pnl].sum()
    return {"n": int(len(df)), "pnl": float(r.sum()), "sharpe": float(sharpe), "sortino": float(sortino),
            "win_rate": float((r > 0).mean()), "mean_ret_pct": float(r.mean()/10.0),
            "skew": float(stats.skew(r)) if len(r) > 2 else float("nan"),
            "pos_years": int((yr > 0).sum()), "n_years": int(len(yr))}

def sel_by_thresh(df, thr):  # verbatim from 60
    cand = df[df["predicted_prob"] >= thr]
    if cand.empty: return cand
    return (cand.sort_values(["predicted_prob", "strategy"], ascending=[False, True])
                .groupby("episode_id", as_index=False).first())

data = pd.read_parquet(D / "tp2_t1k2_token_dataset.parquet")
data["signal_date"] = pd.to_datetime(data["signal_date"])
data = data[data["strategy"].isin(["T1", "K2"])]
feats = pd.read_csv(TP2 / "tables" / "feature_dictionary.csv").query("selected_for_transformer")["feature"].tolist()

def run(cfg_name, **kw):
    preds = []
    for fold in FOLDS:
        tr = data[data.signal_year.isin(fold["train"])]; va = data[data.signal_year.isin(fold["val"])]
        te = data[data.signal_year.isin(fold["test"])]
        if tr.empty or va.empty or te.empty: continue
        Xtr = tr[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(penalty="l2", max_iter=5000, **kw).fit(sc.transform(Xtr), tr["target_profitable"].astype(int))
        pv = clf.predict_proba(sc.transform(va[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)))[:, 1]
        pt = clf.predict_proba(sc.transform(te[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(pv, va["target_profitable"].astype(int).to_numpy())
        t = te[["episode_id", "signal_date", "signal_year", "strategy", "target_profitable", "target_norm_pnl"]].copy()
        t["predicted_prob"] = iso.predict(pt); preds.append(t)
    pred = pd.concat(preds, ignore_index=True)
    sel = sel_by_thresh(pred, 0.5)
    return pred, metrics(sel)

# try to match the committed 13,841 / $2.02M / 2.27
print("target: N=13,841  PnL=$2.02M  Sharpe=2.27\n")
for name, kw in [("C=1.0 unbalanced", {"C": 1.0}), ("C=1.0 balanced", {"C": 1.0, "class_weight": "balanced"}),
                 ("C=0.1 unbalanced", {"C": 0.1}), ("C=0.5 balanced", {"C": 0.5, "class_weight": "balanced"})]:
    pred, m = run(name, **kw)
    print(f"  {name:18s}: N={m['n']:>6,}  PnL=${m['pnl']:>11,.0f}  Sharpe={m['sharpe']:.2f}  win={m['win_rate']*100:.1f}%")

# commit the best-matching config as the artifact
pred, m = run("final", C=1.0, class_weight="balanced")
pred.to_parquet(D / "logistic_l2_test_predictions.parquet", index=False)
# coverage top-14k
cand = pred.sort_values(["predicted_prob", "strategy"], ascending=[False, True]).groupby("episode_id", as_index=False).first()
cov = metrics(cand.sort_values("predicted_prob", ascending=False).head(COVER_N))
art = {"config": "L2 logistic on F*, C=1.0, class_weight=balanced, per-fold StandardScaler, val isotonic, p>=0.5",
       "committed_paper_row": {"N": 13841, "pnl": 2.02e6, "sharpe": 2.27},
       "reproduced_p0p5": m, "reproduced_coverage_top14k": cov}
json.dump(art, open(OUT / "logistic_floor_reproduction.json", "w"), indent=2, default=float)
print(f"\nreproduced p>=0.5: N={m['n']:,}  PnL=${m['pnl']:,.0f}  Sharpe={m['sharpe']:.2f}")
print(f"coverage top-14k:  N={cov['n']:,}  PnL=${cov['pnl']:,.0f}  Sharpe={cov['sharpe']:.2f}")
print(f"wrote {OUT/'logistic_floor_reproduction.json'} and predictions parquet")
