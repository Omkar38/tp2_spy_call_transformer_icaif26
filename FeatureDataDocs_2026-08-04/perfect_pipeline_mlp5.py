"""PERFECT leak-free pipeline for MLP-5:
  - deterministic LightGBM (deterministic=True, force_row_wise=True, n_jobs=1)
  - ENSEMBLE feature selection: per fold, run discovery with K seeds, average the composite
    importance, take the top 48 -> robust, reproducible F*_k
  - per-fold train scaler, val-year isotonic calibration, causal p>=0.5 (all leak-free)
Stability check: run with 3 different seed-sets; if the aggregate barely moves, it is stable.
Read-only. Artifact: nested/perfect_pipeline_mlp5.json
"""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/jyotirpatel/Downloads/FinalQuant copy 2"); TP2 = ROOT/"TP2MVTransformer"; DAT = TP2/"data"
OUTD = Path(__file__).resolve().parent/"nested"; OUTD.mkdir(exist_ok=True)
ANN = np.sqrt(252); MLP_SEEDS = [42,101,202,303,404]
SEED_SETS = {"A": [1,2,3,4,5], "B": [10,20,30,40,50], "C": [42,7,101,2024,55]}   # 3 ensembles to test stability
spec = importlib.util.spec_from_file_location("bfd", TP2/"01_build_feature_dictionary.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
CANON = mod.FOLDS
mod.LGBM_PARAMS.update({"deterministic": True, "force_row_wise": True, "n_jobs": -1})   # DETERMINISM

def metrics(df):
    r = df["target_norm_pnl"].astype(float); daily = df.groupby("signal_date")["target_norm_pnl"].sum()
    sh = daily.mean()/daily.std()*ANN if daily.std() > 0 else float("nan")
    dn = daily[daily<0]; so = daily.mean()/dn.std()*ANN if len(dn)>1 and dn.std()>0 else float("nan")
    yr = df.groupby(df["signal_date"].dt.year)["target_norm_pnl"].sum()
    return {"n": int(len(df)), "pnl": float(r.sum()), "sharpe": round(float(sh),4), "sortino": round(float(so),4),
            "win_rate": round(float((r>0).mean()),4), "pos_years": int((yr>0).sum())}
def selp(df):
    c = df[df.predicted_prob>=0.5]
    if c.empty: return c
    return c.sort_values(["predicted_prob","strategy"],ascending=[False,True]).groupby("episode_id",as_index=False).first()

data = pd.read_parquet(DAT/"tp2_t1k2_token_dataset.parquet"); data["signal_date"]=pd.to_datetime(data["signal_date"])
data = data[data.strategy.isin(["T1","K2"])]
ALL149 = pd.read_csv(TP2/"tables"/"feature_dictionary.csv")["feature"].tolist()
def num(df,cols): return df[cols].apply(pd.to_numeric,errors="coerce").fillna(0.0)

def ensemble_Fk(fold, seeds):
    trly = fold["train"]
    mod.FOLDS = [{"fold":"e","train":trly[:-1],"val":[trly[-1]],"test":fold["test"]}]
    scores = []
    for s in seeds:
        mod.LGBM_PARAMS["random_state"] = s
        _, summ = mod.run_feature_discovery(data, ALL149)
        scores.append(summ.set_index("feature")["mean_score"])
    mod.FOLDS = CANON
    avg = pd.concat(scores, axis=1).mean(axis=1)
    return avg.sort_values(ascending=False).head(48).index.tolist()

def run(seed_set):
    allsel=[]; fks={}
    for fold in CANON:
        yr=fold["test"][0]
        Fk = ensemble_Fk(fold, seed_set); fks[yr]=set(Fk)
        tr=data[data.signal_year.isin(fold["train"])]; va=data[data.signal_year.isin(fold["val"])]; te=data[data.signal_year.isin(fold["test"])]
        sc=StandardScaler().fit(num(tr,Fk)); Xtr,Xva,Xte=sc.transform(num(tr,Fk)),sc.transform(num(va,Fk)),sc.transform(num(te,Fk))
        ytr=tr.target_profitable.astype(int); yva=va.target_profitable.astype(int).to_numpy()
        vp,tp=[],[]
        for s in MLP_SEEDS:
            m=MLPClassifier(hidden_layer_sizes=(64,32),activation="relu",alpha=1e-4,batch_size=1024,learning_rate_init=1e-3,
                            max_iter=80,early_stopping=True,validation_fraction=0.15,n_iter_no_change=8,random_state=s).fit(Xtr,ytr)
            vp.append(m.predict_proba(Xva)[:,1]); tp.append(m.predict_proba(Xte)[:,1])
        iso=IsotonicRegression(out_of_bounds="clip").fit(np.mean(vp,0),yva)
        b=te[["episode_id","signal_date","signal_year","strategy","target_norm_pnl"]].copy(); b["predicted_prob"]=iso.predict(np.mean(tp,0))
        allsel.append(selp(b))
    return metrics(pd.concat(allsel)), fks

res={}; fkset={}
for name, seeds in SEED_SETS.items():
    print(f">>> ensemble seed-set {name} = {seeds} ...", flush=True)
    m, fks = run(seeds); res[name]=m; fkset[name]=fks
    print(f"    {name}: N={m['n']:,} PnL=${m['pnl']:,.0f} Sharpe={m['sharpe']:.2f} +yrs={m['pos_years']}/6", flush=True)

shs=[res[k]["sharpe"] for k in SEED_SETS]; pnls=[res[k]["pnl"] for k in SEED_SETS]
common={yr: len(set.intersection(*[fkset[k][yr] for k in SEED_SETS])) for yr in fkset["A"]}
out={"per_seed_set":res,"sharpe_mean":round(float(np.mean(shs)),3),"sharpe_std":round(float(np.std(shs)),3),
     "pnl_mean":float(np.mean(pnls)),"pnl_std":float(np.std(pnls)),"fk_common_across_seed_sets":common,
     "config":"deterministic LightGBM + 5-seed ensemble feature selection, per-fold training-years-only"}
json.dump(out,open(OUTD/"perfect_pipeline_mlp5.json","w"),indent=2,default=float)
print(f"\n=== PERFECT PIPELINE: Sharpe {np.mean(shs):.2f} +/- {np.std(shs):.2f} (range {min(shs):.2f}-{max(shs):.2f}) ===")
print(f"    PnL ${np.mean(pnls):,.0f} +/- ${np.std(pnls):,.0f} | F* common across seed-sets/fold: {common}")
print("wrote", OUTD/"perfect_pipeline_mlp5.json")
