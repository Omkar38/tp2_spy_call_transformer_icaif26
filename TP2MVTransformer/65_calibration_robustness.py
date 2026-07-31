"""#18 calibration robustness: (a) Platt vs isotonic calibration, fit per-fold on
the validation year and applied to the test year, compared on selected N, Sharpe,
and test-set calibration; (b) a block-bootstrap CI on the selected trade count N_m.

Uses LightGBM's raw val/test scores (raw predicted_prob, continuous) for (a) --
its val_predictions carry the raw model probability and the label is derived from
realized P&L sign. For (b) we resample trading dates for each model's shipped
calibrated selection.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "TP2MVTransformer" / "data"
OUT = ROOT / "TP2MVTransformer" / "tables" / "action_list"
OUT.mkdir(parents=True, exist_ok=True)
ANN = np.sqrt(252)

def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def episode_select(df, thr=0.5):
    c = df[df["cal_prob"] >= thr]
    if c.empty: return c
    return c.sort_values(["cal_prob", "strategy"], ascending=[False, True]).groupby("episode_id", as_index=False).first()

def sharpe(sel, pnl):
    if len(sel) == 0: return float("nan")
    d = sel.groupby("signal_date")[pnl].sum()
    return float(d.mean() / d.std() * ANN) if d.std() > 0 else float("nan")

# ---------- (a) Platt vs isotonic on LightGBM ----------
print("=" * 70); print("#18a  PLATT vs ISOTONIC (LightGBM, fit on val year, applied to test)"); print("=" * 70)
val = pd.read_parquet(D / "lgbm_final_val_predictions.parquet")
test = pd.read_parquet(D / "lgbm_final_test_predictions.parquet")
for df in (val, test):
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["label"] = (df["norm_pnl"] > 0).astype(int)          # profitable-trade label
val = val[val["is_valid"] & val["strategy"].isin(["T1", "K2"])]
test = test[test["is_valid"] & test["strategy"].isin(["T1", "K2"])]

rows = {"isotonic": [], "platt": []}
per_method_selected = {"isotonic": [], "platt": []}
for fold, tg in test.groupby("fold"):
    vg = val[val["fold"] == fold]
    if len(vg) < 50: continue
    iso = IsotonicRegression(out_of_bounds="clip").fit(vg["predicted_prob"], vg["label"])
    platt = LogisticRegression(C=1e6, solver="lbfgs").fit(vg[["predicted_prob"]], vg["label"])
    for name, cal in [("isotonic", iso.predict(tg["predicted_prob"])),
                      ("platt", platt.predict_proba(tg[["predicted_prob"]])[:, 1])]:
        t = tg.copy(); t["cal_prob"] = cal
        sel = episode_select(t)
        per_method_selected[name].append(sel)
        rows[name].append({"fold": fold, "n": len(sel), "pnl": float(sel["norm_pnl"].sum()),
                           "test_ece": ece(t["label"].values, cal), "test_brier": brier_score_loss(t["label"], cal)})

summary_a = {}
for name in ("isotonic", "platt"):
    allsel = pd.concat(per_method_selected[name], ignore_index=True)
    r = pd.DataFrame(rows[name])
    summary_a[name] = {"total_n": int(r["n"].sum()), "total_pnl": float(r["pnl"].sum()),
                       "sharpe": sharpe(allsel, "norm_pnl"),
                       "mean_test_ece": float(r["test_ece"].mean()), "mean_test_brier": float(r["test_brier"].mean())}
    s = summary_a[name]
    print(f"  {name:9s}: N={s['total_n']:,} PnL=${s['total_pnl']:,.0f} Sharpe={s['sharpe']:.2f} "
          f"ECE={s['mean_test_ece']:.3f} Brier={s['mean_test_brier']:.3f}")
json.dump(summary_a, open(OUT / "calibration_platt_vs_isotonic.json", "w"), indent=2, default=float)

# ---------- (b) block-bootstrap CI on selected N_m ----------
print("\n" + "=" * 70); print("#18b  BLOCK-BOOTSTRAP CI ON SELECTED N (resample trading dates)"); print("=" * 70)
SEL = {
    "LightGBM (1)": ("lgbm_final_t1k2_selected_trades.parquet", "norm_pnl"),
    "MLP (5)": ("mlp_5seed_ensemble_selected_trades.parquet", "target_norm_pnl"),
    "Transformer (5)": ("mv_no_alibi_5seed_selected_trades.parquet", "target_norm_pnl"),
}
rng = np.random.default_rng(20260731); B = 5000
nm_ci = {}
for name, (f, pnl) in SEL.items():
    df = pd.read_parquet(D / f)
    if "selected" in df.columns: df = df[df["selected"].astype(bool)]
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    by_date = df.groupby("signal_date").size()
    dates = by_date.index.to_numpy(); counts = by_date.to_numpy()
    draws = np.array([counts[rng.integers(0, len(counts), len(counts))].sum() for _ in range(B)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    nm_ci[name] = {"N_observed": int(len(df)), "ci_lo": int(lo), "ci_hi": int(hi), "rel_width": float((hi - lo) / len(df))}
    print(f"  {name:16s} N={len(df):,}  95% CI [{int(lo):,}, {int(hi):,}]  (+/-{nm_ci[name]['rel_width']*50:.0f}%)")
json.dump(nm_ci, open(OUT / "calibration_Nm_bootstrap.json", "w"), indent=2, default=float)
print("\nDONE ->", OUT)
