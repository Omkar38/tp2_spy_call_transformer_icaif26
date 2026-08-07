"""Lightweight finalizer: add the transformers' full-population calibration (Brier/ECE/PR-AUC) to the
existing newmodel_calibration.json by READING fullpred_transformer_{alibi,plain}.parquet -- NO re-training.
Run after the transformer full-pred re-run finishes.
"""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

OUT = Path(__file__).resolve().parent; ACTION = OUT/"tables"/"action_list"; NESTED = OUT.parent/"FeatureDataDocs_2026-08-04"/"nested"
ece = importlib.util.module_from_spec(importlib.util.spec_from_file_location("c50f", OUT/"50_calibration_metrics.py"))
importlib.util.spec_from_file_location("c50f", OUT/"50_calibration_metrics.py").loader.exec_module(ece); ece = ece.expected_calibration_error

cal = json.load(open(ACTION/"newmodel_calibration.json")); res = cal["models"]
def calib(y, p):
    sel = p >= 0.5
    return {"n": int(len(y)), "base_rate": float(y.mean()), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "ece": float(ece(y, p)),
            "n_selected": int(sel.sum()), "mean_selected_p": float(p[sel].mean()) if sel.any() else float("nan"),
            "selected_win_rate": float(y[sel].mean()) if sel.any() else float("nan")}
done = True
for v, name in [("alibi", "Transformer-ALiBi-5(new)"), ("plain", "Transformer-plain-5(new)")]:
    f = NESTED/f"fullpred_transformer_{v}.parquet"
    if f.exists():
        d = pd.read_parquet(f); y = d.target_profitable.astype(int).to_numpy(); p = d.predicted_prob.to_numpy()
        res[name] = calib(y, p)
        print(f"{name:26} PR-AUC={res[name]['pr_auc']:.3f} Brier={res[name]['brier']:.4f} ECE={res[name]['ece']:.4f} | sel mean_p={res[name]['mean_selected_p']:.3f} win={res[name]['selected_win_rate']*100:.1f}%")
    else:
        print(f"{name}: fullpred NOT found ({f}) -- still pending"); done = False
json.dump(cal, open(ACTION/"newmodel_calibration.json", "w"), indent=2, default=float)
pd.DataFrame([{"model": k, **v} for k, v in res.items()]).to_csv(ACTION/"newmodel_calibration.csv", index=False)
print("finalized" if done else "partial -- transformers still pending", "-> newmodel_calibration.json")
