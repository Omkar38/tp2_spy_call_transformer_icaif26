"""Recompute Sharpe PROPERLY for the fast, deterministic models (LightGBM, logistic, MLP-5),
using the SAME per-fold F*_k (nested/fk_ensemble_perfect.json). Saves each model's selected
trades to parquet (so results are never 'lost' again). Computes Sharpe THREE ways to expose
the active-days inflation, and dissects why MLP (more profit) has a lower Sharpe than LightGBM.

Sharpe conventions (all x sqrt(252), std convention as in the reported metrics()):
  active   : only days with >=1 selected trade   (the number reported so far -- inflated)
  signalday: zero-filled over every day the data had a TP2 signal (strategy opportunity set)
  calendar : zero-filled over every NYSE business day in the test span (standard portfolio Sharpe)

Transformers are NOT rerun (too slow, per user); their active-day Sharpe is read from JSON for the table.
Artifact: nested/sharpe_recompute.json  +  nested/trades_{lgbm,logistic,mlp5}.parquet
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/jyotirpatel/Downloads/FinalQuant copy 2"); TP2 = ROOT/"TP2MVTransformer"; DAT = TP2/"data"
OUTD = Path(__file__).resolve().parent/"nested"
ANN = np.sqrt(252)
FK = json.load(open(OUTD/"fk_ensemble_perfect.json"))["fk_by_year"]

import importlib.util
spec = importlib.util.spec_from_file_location("bfd_rs", TP2/"01_build_feature_dictionary.py")
bfd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bfd)
CANON = bfd.FOLDS

LGBM_PARAMS = {"objective":"binary","metric":"binary_logloss","n_estimators":350,"learning_rate":0.04,
               "num_leaves":31,"min_child_samples":30,"subsample":0.8,"colsample_bytree":0.8,
               "class_weight":"balanced","n_jobs":-1,"verbose":-1,"random_state":42}
LOGIT_CFG = dict(penalty="l2", C=1.0, class_weight="balanced", max_iter=5000)
MLP_SEEDS = [42,101,202,303,404]

data = pd.read_parquet(DAT/"tp2_t1k2_token_dataset.parquet"); data["signal_date"]=pd.to_datetime(data["signal_date"]).dt.normalize()
data["signal_year"]=data["signal_date"].dt.year
data = data[data.strategy.isin(["T1","K2"])]
def num(df,cols): return df[cols].apply(pd.to_numeric,errors="coerce").fillna(0.0)
def selp(df):
    c=df[df.predicted_prob>=0.5]
    if c.empty: return c
    return c.sort_values(["predicted_prob","strategy"],ascending=[False,True]).groupby("episode_id",as_index=False).first()

# trading-day universes (over the test span 2021..2026)
test_lo, test_hi = data[data.signal_year>=2021]["signal_date"].min(), data["signal_date"].max()
SIG_DAYS = pd.DatetimeIndex(sorted(data[data.signal_year>=2021]["signal_date"].unique()))
CAL_DAYS = pd.bdate_range(test_lo, test_hi)

def daily_sharpe(daily):  # daily is a Series indexed by date
    return float(daily.mean()/daily.std()*ANN) if daily.std()>0 else float("nan")

def variants(sel):
    daily = sel.groupby("signal_date")["target_norm_pnl"].sum()
    d_sig = daily.reindex(SIG_DAYS, fill_value=0.0)
    d_cal = daily.reindex(CAL_DAYS, fill_value=0.0)
    yr = sel.groupby(sel.signal_date.dt.year)["target_norm_pnl"].sum()
    return {
        "n_trades": int(len(sel)), "pnl": float(sel.target_norm_pnl.sum()),
        "mean_pnl_per_trade": float(sel.target_norm_pnl.mean()),
        "win_rate": float((sel.target_norm_pnl>0).mean()),
        "n_active_days": int(len(daily)), "n_signal_days": int(len(SIG_DAYS)), "n_calendar_days": int(len(CAL_DAYS)),
        "mean_daily_active": float(daily.mean()), "std_daily_active": float(daily.std()),
        "std_daily_calendar": float(d_cal.std()),
        "sharpe_active": round(daily_sharpe(daily),4),
        "sharpe_signalday": round(daily_sharpe(d_sig),4),
        "sharpe_calendar": round(daily_sharpe(d_cal),4),
        "pos_years": int((yr>0).sum()),
    }

def run_lgbm(tr,va,te,Fk,ytr,yva):
    m=lgb.LGBMClassifier(**LGBM_PARAMS).fit(num(tr,Fk),ytr)
    return m.predict_proba(num(va,Fk))[:,1], m.predict_proba(num(te,Fk))[:,1]
def run_logit(tr,va,te,Fk,ytr,yva):
    sc=StandardScaler().fit(num(tr,Fk)); m=LogisticRegression(**LOGIT_CFG).fit(sc.transform(num(tr,Fk)),ytr)
    return m.predict_proba(sc.transform(num(va,Fk)))[:,1], m.predict_proba(sc.transform(num(te,Fk)))[:,1]
def run_mlp(tr,va,te,Fk,ytr,yva):
    sc=StandardScaler().fit(num(tr,Fk)); Xtr,Xva,Xte=sc.transform(num(tr,Fk)),sc.transform(num(va,Fk)),sc.transform(num(te,Fk))
    vp,tp=[],[]
    for s in MLP_SEEDS:
        mm=MLPClassifier(hidden_layer_sizes=(64,32),activation="relu",alpha=1e-4,batch_size=1024,learning_rate_init=1e-3,
                         max_iter=80,early_stopping=True,validation_fraction=0.15,n_iter_no_change=8,random_state=s).fit(Xtr,ytr)
        vp.append(mm.predict_proba(Xva)[:,1]); tp.append(mm.predict_proba(Xte)[:,1])
    return np.mean(vp,0), np.mean(tp,0)

RUNNERS={"lgbm":run_lgbm,"logistic":run_logit,"mlp5":run_mlp}
results={}
for kind,fn in RUNNERS.items():
    allsel=[]
    for fold in CANON:
        yr=fold["test"][0]; Fk=list(FK[str(yr)])
        tr=data[data.signal_year.isin(fold["train"])]; va=data[data.signal_year.isin(fold["val"])]; te=data[data.signal_year.isin(fold["test"])]
        ytr=tr.target_profitable.astype(int); yva=va.target_profitable.astype(int).to_numpy()
        vp,tp=fn(tr,va,te,Fk,ytr,yva)
        iso=IsotonicRegression(out_of_bounds="clip").fit(vp,yva)
        b=te[["episode_id","signal_date","signal_year","strategy","target_norm_pnl"]].copy(); b["predicted_prob"]=iso.predict(tp)
        allsel.append(selp(b))
    sel=pd.concat(allsel).reset_index(drop=True)
    sel.to_parquet(OUTD/f"trades_{kind}.parquet")
    results[kind]=variants(sel)
    r=results[kind]
    print(f"{kind:9} PnL=${r['pnl']:>10,.0f} N={r['n_trades']:>6,} | Sharpe active={r['sharpe_active']:.2f} "
          f"signalday={r['sharpe_signalday']:.2f} calendar={r['sharpe_calendar']:.2f} | "
          f"active_days={r['n_active_days']} mean/day=${r['mean_daily_active']:,.0f} std/day=${r['std_daily_active']:,.0f}", flush=True)

# transformers: active-day sharpe from JSON (not rerun)
for v in ["alibi","plain"]:
    a=json.load(open(OUTD/f"perfect_pipeline_transformer_{v}.json"))["aggregate"]
    results[f"transformer_{v}"]={"n_trades":a["n"],"pnl":a["pnl"],"sharpe_active":a["sharpe"],
                                 "sharpe_signalday":None,"sharpe_calendar":None,"note":"not rerun; active-day only"}

out={"universes":{"n_signal_days":int(len(SIG_DAYS)),"n_calendar_days":int(len(CAL_DAYS)),
                  "test_span":[str(test_lo.date()),str(test_hi.date())]},
     "convention_note":"active=days-with-trades only (inflated); signalday=zero-filled over TP2-signal days; calendar=zero-filled over NYSE business days. All x sqrt(252).",
     "results":results}
json.dump(out,open(OUTD/"sharpe_recompute.json","w"),indent=2,default=float)
print("\nwrote nested/sharpe_recompute.json + trades_{lgbm,logistic,mlp5}.parquet")
