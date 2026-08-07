"""brief-21 + rev2-P0.9 (0.4b): per-year calibration and selected-tail columns.
Per model x year: base rate, mean calibrated p among SELECTED trades, selected win rate,
Brier, ECE. Uses script 50's conventions (imported expected_calibration_error; LightGBM
label = norm_pnl>0 with is_valid filter; others target_profitable) and sel_by_thresh from 60.
GATE: pooled Brier/ECE reproduce tables/calibration_metrics.json for LightGBM/plain/ALiBi.
Artifact: tables/action_list/calibration_by_year.json.
"""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

OUT = Path(__file__).resolve().parent; DATA = OUT/"data"; ACTION = OUT/"tables"/"action_list"
def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m50 = _load("cal50", OUT/"50_calibration_metrics.py")
ece = m50.expected_calibration_error   # 10 equal-width bins, verbatim

def sel_by_thresh(pred, thr=0.5):   # verbatim from 60
    df = pred.copy()
    if "is_valid" in df.columns: df = df[df["is_valid"].astype(bool)]
    if "strategy" in df.columns: df = df[df["strategy"].isin(["T1", "K2"])]
    cand = df[df["predicted_prob"] >= thr]
    if cand.empty or "episode_id" not in cand.columns: return cand
    return cand.sort_values(["predicted_prob", "strategy"], ascending=[False, True]).groupby("episode_id", as_index=False).first()

def load(name):
    if name == "LightGBM":
        d = pd.read_parquet(DATA/"lgbm_final_test_predictions.parquet")
        d = d[d["strategy"].isin(["T1", "K2"]) & d["is_valid"]].copy(); d["y"] = (d["norm_pnl"] > 0).astype(int)
    else:
        f = {"MLP-5": "mlp_5seed_ensemble_test_predictions.parquet",
             "Transformer-plain-5": "mv_no_alibi_5seed_test_predictions.parquet",
             "Transformer-ALiBi-5": "mv_alibi_transformer_v4final_test_predictions.parquet",
             "Logistic": "logistic_l2_test_predictions.parquet"}[name]
        d = pd.read_parquet(DATA/f); d = d[d["strategy"].isin(["T1", "K2"])].copy(); d["y"] = d["target_profitable"].astype(int)
    d["signal_date"] = pd.to_datetime(d["signal_date"]); d["year"] = d["signal_date"].dt.year
    return d

MODELS = ["LightGBM", "MLP-5", "Transformer-plain-5", "Transformer-ALiBi-5", "Logistic"]
GATE = {"LightGBM": (0.0865, 0.0494), "Transformer-plain-5": (0.0830, 0.0243), "Transformer-ALiBi-5": (0.0831, 0.0276)}  # (brier,ece) from calibration_metrics.json

def cal_block(y, p):
    return {"n": int(len(y)), "base_rate": float(y.mean()), "brier": float(brier_score_loss(y, p)),
            "ece": float(ece(y.to_numpy() if hasattr(y, "to_numpy") else y, p.to_numpy() if hasattr(p, "to_numpy") else p))}

result = {}; print("=== GATE: pooled Brier/ECE vs calibration_metrics.json ===")
for name in MODELS:
    d = load(name); y, p = d["y"], d["predicted_prob"]
    pooled = cal_block(y, p); pooled["pr_auc"] = float(average_precision_score(y, p))
    if name in GATE:
        gb, ge = GATE[name]; ok = abs(pooled["brier"] - gb) < 5e-4 and abs(pooled["ece"] - ge) < 5e-4
        print(f"  {name:22} Brier={pooled['brier']:.4f} (gate {gb}) ECE={pooled['ece']:.4f} (gate {ge})  {'OK' if ok else '*** FAIL ***'}")
        assert ok, f"{name} calibration gate failed"
    # selected-tail (0.4b): mean calibrated p + win rate among p>=0.5 one-action selections
    sel = sel_by_thresh(d)
    seltail = {"n_selected": int(len(sel)), "mean_selected_p": float(sel["predicted_prob"].mean()) if len(sel) else float("nan"),
               "selected_win_rate": float(sel["y"].mean()) if len(sel) else float("nan")}
    byyear = {}
    for yr, g in d.groupby("year"):
        gy, gp = g["y"], g["predicted_prob"]; s = sel_by_thresh(g)
        byyear[int(yr)] = {**cal_block(gy, gp),
                           "mean_selected_p": float(s["predicted_prob"].mean()) if len(s) else float("nan"),
                           "selected_win_rate": float(s["y"].mean()) if len(s) else float("nan"),
                           "n_selected": int(len(s))}
    result[name] = {"pooled": pooled, "selected_tail": seltail, "by_year": byyear}
    print(f"  {name:22} pooled base={pooled['base_rate']:.3f} | selected: N={seltail['n_selected']:,} "
          f"mean_p={seltail['mean_selected_p']:.3f} win={seltail['selected_win_rate']*100:.1f}%")

json.dump({"models": result, "ece_bins": 10,
           "note": "brief-21 per-year calibration + 0.4b selected-tail columns. LightGBM label=norm_pnl>0 (is_valid); "
                   "others target_profitable. expected_calibration_error imported verbatim from 50; sel_by_thresh from 60. "
                   "MLP row is the 5-seed ensemble (resolves the 0.4b MLP inconsistency)."},
          open(ACTION/"calibration_by_year.json", "w"), indent=2, default=float)
print("\nwrote", ACTION/"calibration_by_year.json")
