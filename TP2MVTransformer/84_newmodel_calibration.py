"""New-model full-population calibration (closes the last item). Re-trains MLP-5 and logistic on the
per-fold F* (fk_ensemble_perfect), predicts the FULL test population, and computes PR-AUC / Brier / ECE
(+ selected-tail). LightGBM (old 149) taken from committed calibration_metrics.json. Transformers read
fullpred_transformer_{alibi,plain}.parquet from the full-pred re-run (skipped with a note if not yet present).
expected_calibration_error imported verbatim from script 50. Artifact + CSV.
"""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss

OUT = Path(__file__).resolve().parent; ROOT = OUT.parent; DATA = OUT/"data"; ACTION = OUT/"tables"/"action_list"; NESTED = ROOT/"FeatureDataDocs_2026-08-04"/"nested"
def _load(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ece = _load("cal50c", OUT/"50_calibration_metrics.py").expected_calibration_error
disc = _load("bfd84", OUT/"01_build_feature_dictionary.py"); CANON = disc.FOLDS
FK = json.load(open(NESTED/"fk_ensemble_perfect.json"))["fk_by_year"]
MLP_SEEDS = [42, 101, 202, 303, 404]

data = pd.read_parquet(DATA/"tp2_t1k2_token_dataset.parquet"); data["signal_date"] = pd.to_datetime(data["signal_date"])
data["signal_year"] = data["signal_date"].dt.year; data = data[data.strategy.isin(["T1", "K2"])]
def num(df, cols): return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

def full_predictions(kind):
    preds = []
    for fold in CANON:
        yr = fold["test"][0]; Fk = list(FK[str(yr)])
        tr = data[data.signal_year.isin(fold["train"])]; va = data[data.signal_year.isin(fold["val"])]; te = data[data.signal_year.isin(fold["test"])]
        ytr = tr.target_profitable.astype(int); yva = va.target_profitable.astype(int).to_numpy()
        if kind == "logistic":
            sc = StandardScaler().fit(num(tr, Fk)); mdl = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(num(tr, Fk)), ytr)
            vp = mdl.predict_proba(sc.transform(num(va, Fk)))[:, 1]; tp = mdl.predict_proba(sc.transform(num(te, Fk)))[:, 1]
        else:
            sc = StandardScaler().fit(num(tr, Fk)); vps, tps = [], []
            for s in MLP_SEEDS:
                mm = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4, batch_size=1024, learning_rate_init=1e-3,
                                   max_iter=80, early_stopping=True, validation_fraction=0.15, n_iter_no_change=8, random_state=s).fit(sc.transform(num(tr, Fk)), ytr)
                vps.append(mm.predict_proba(sc.transform(num(va, Fk)))[:, 1]); tps.append(mm.predict_proba(sc.transform(num(te, Fk)))[:, 1])
            vp = np.mean(vps, 0); tp = np.mean(tps, 0)
        iso = IsotonicRegression(out_of_bounds="clip").fit(vp, yva)
        b = te[["episode_id", "signal_date", "strategy", "target_profitable"]].copy(); b["predicted_prob"] = iso.predict(tp); preds.append(b)
    return pd.concat(preds, ignore_index=True)

def calib(y, p):
    sel = p >= 0.5
    return {"n": int(len(y)), "base_rate": float(y.mean()), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "ece": float(ece(y, p)),
            "n_selected": int(sel.sum()), "mean_selected_p": float(p[sel].mean()) if sel.any() else float("nan"),
            "selected_win_rate": float(y[sel].mean()) if sel.any() else float("nan")}

result = {}
# LightGBM (old 149) from committed
cm = json.load(open(OUT/"tables"/"calibration_metrics.json"))
result["LightGBM(149,old)"] = {k: cm["LightGBM"][k] for k in ("n", "base_rate", "pr_auc", "brier", "ece")}
print(f"LightGBM(149,old) [committed]  PR-AUC={result['LightGBM(149,old)']['pr_auc']:.3f} Brier={result['LightGBM(149,old)']['brier']:.4f} ECE={result['LightGBM(149,old)']['ece']:.4f}")
# MLP-5, logistic (new) re-trained
for kind, name in [("mlp5", "MLP-5(new)"), ("logistic", "Logistic(new)")]:
    fp = full_predictions(kind); y = fp.target_profitable.astype(int).to_numpy(); p = fp.predicted_prob.to_numpy()
    result[name] = calib(y, p)
    print(f"{name:20} PR-AUC={result[name]['pr_auc']:.3f} Brier={result[name]['brier']:.4f} ECE={result[name]['ece']:.4f} | sel mean_p={result[name]['mean_selected_p']:.3f} win={result[name]['selected_win_rate']*100:.1f}%")
# transformers (new) from fullpred re-run
for v, name in [("alibi", "Transformer-ALiBi-5(new)"), ("plain", "Transformer-plain-5(new)")]:
    f = NESTED/f"fullpred_transformer_{v}.parquet"
    if f.exists():
        d = pd.read_parquet(f); y = d.target_profitable.astype(int).to_numpy(); p = d.predicted_prob.to_numpy()
        result[name] = calib(y, p)
        print(f"{name:26} PR-AUC={result[name]['pr_auc']:.3f} Brier={result[name]['brier']:.4f} ECE={result[name]['ece']:.4f}")
    else:
        result[name] = {"status": "PENDING full-pred re-run"}; print(f"{name:26} PENDING (fullpred not yet saved)")

json.dump({"models": result, "ece_bins": 10, "note": "New-model full-population calibration. MLP-5/logistic re-trained on "
           "per-fold F* saving full predictions; LightGBM(old 149) from committed calibration_metrics.json; transformers "
           "from fullpred_transformer_*.parquet (full-pred re-run). expected_calibration_error verbatim from 50."},
          open(ACTION/"newmodel_calibration.json", "w"), indent=2, default=float)
pd.DataFrame([{"model": k, **v} for k, v in result.items()]).to_csv(ACTION/"newmodel_calibration.csv", index=False)
print("\nwrote", ACTION/"newmodel_calibration.json", "+ .csv")
