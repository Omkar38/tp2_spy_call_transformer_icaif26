"""Perfect leak-free pipeline for LightGBM (trading model) and L2 logistic.
Reuses the SAME precomputed per-fold ensemble F*_k (nested/fk_ensemble_perfect.json)
that fed the MLP and both transformers. Only the feature set per fold changes; each
model keeps its committed training recipe (LightGBM cfg from script 27, logistic from 74).
Per fold: fit scaler on train (logistic) / raw (lgbm), calibrate isotonic on val year,
select p>=0.5 one-action/episode. Torch-free -> no libomp clash, fast.
Artifacts: nested/perfect_pipeline_lgbm.json , nested/perfect_pipeline_logistic.json
"""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/jyotirpatel/Downloads/FinalQuant copy 2"); TP2 = ROOT/"TP2MVTransformer"; DAT = TP2/"data"
OUTD = Path(__file__).resolve().parent/"nested"
ANN = np.sqrt(252)
FK = json.load(open(OUTD/"fk_ensemble_perfect.json"))["fk_by_year"]

spec = importlib.util.spec_from_file_location("bfd_ll", TP2/"01_build_feature_dictionary.py")
bfd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bfd)
CANON = bfd.FOLDS

# committed recipes
LGBM_PARAMS = {"objective": "binary", "metric": "binary_logloss", "n_estimators": 350, "learning_rate": 0.04,
               "num_leaves": 31, "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.8,
               "class_weight": "balanced", "n_jobs": -1, "verbose": -1, "random_state": 42}
LOGIT_CFG = dict(penalty="l2", C=1.0, class_weight="balanced", max_iter=5000)

data = pd.read_parquet(DAT/"tp2_t1k2_token_dataset.parquet"); data["signal_date"] = pd.to_datetime(data["signal_date"])
data["signal_year"] = data["signal_date"].dt.year
data = data[data.strategy.isin(["T1", "K2"])]
def num(df, cols): return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

def metrics(df):
    r = df["target_norm_pnl"].astype(float); daily = df.groupby("signal_date")["target_norm_pnl"].sum()
    sh = daily.mean()/daily.std()*ANN if daily.std() > 0 else float("nan")
    dn = daily[daily < 0]; so = daily.mean()/dn.std()*ANN if len(dn) > 1 and dn.std() > 0 else float("nan")
    yr = df.groupby(df["signal_date"].dt.year)["target_norm_pnl"].sum()
    return {"n": int(len(df)), "pnl": float(r.sum()), "sharpe": round(float(sh),4), "sortino": round(float(so),4),
            "win_rate": round(float((r > 0).mean()),4), "pos_years": int((yr > 0).sum())}
def selp(df):
    c = df[df.predicted_prob >= 0.5]
    if c.empty: return c
    return c.sort_values(["predicted_prob","strategy"], ascending=[False,True]).groupby("episode_id", as_index=False).first()

def run(kind):
    allsel = []; per_fold = []
    for fold in CANON:
        yr = fold["test"][0]; Fk = list(FK[str(yr)])
        tr = data[data.signal_year.isin(fold["train"])]; va = data[data.signal_year.isin(fold["val"])]; te = data[data.signal_year.isin(fold["test"])]
        ytr = tr.target_profitable.astype(int); yva = va.target_profitable.astype(int).to_numpy()
        if kind == "lgbm":
            m = lgb.LGBMClassifier(**LGBM_PARAMS).fit(num(tr, Fk), ytr)
            vp = m.predict_proba(num(va, Fk))[:, 1]; tp = m.predict_proba(num(te, Fk))[:, 1]
        else:
            sc = StandardScaler().fit(num(tr, Fk))
            m = LogisticRegression(**LOGIT_CFG).fit(sc.transform(num(tr, Fk)), ytr)
            vp = m.predict_proba(sc.transform(num(va, Fk)))[:, 1]; tp = m.predict_proba(sc.transform(num(te, Fk)))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(vp, yva)
        b = te[["episode_id","signal_date","signal_year","strategy","target_norm_pnl"]].copy()
        b["predicted_prob"] = iso.predict(tp)
        s = selp(b); allsel.append(s); m_ = metrics(s)
        per_fold.append({"test_year": yr, **m_})
        print(f"  [{kind}] {yr}: N={m_['n']:>6,} PnL=${m_['pnl']:>11,.0f} Sharpe={m_['sharpe']:.2f}", flush=True)
    agg = metrics(pd.concat(allsel))
    out = {"model": kind, "aggregate": agg, "per_fold": per_fold,
           "config": (json.dumps(LGBM_PARAMS) if kind == "lgbm" else json.dumps(LOGIT_CFG)),
           "pipeline": "per-fold ensemble deterministic F* (train-years-only), val isotonic, p>=0.5"}
    dest = OUTD/f"perfect_pipeline_{kind}.json"; json.dump(out, open(dest, "w"), indent=2, default=float)
    print(f"=== PERFECT {kind.upper()}: N={agg['n']:,} PnL=${agg['pnl']:,.0f} Sharpe={agg['sharpe']:.2f} "
          f"Sortino={agg['sortino']:.2f} +yrs={agg['pos_years']}/6 ===\n", flush=True)

for kind in ["lgbm", "logistic"]:
    run(kind)
