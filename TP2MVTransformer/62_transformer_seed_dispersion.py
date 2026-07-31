"""#5 (transformer arm): single-seed Sharpe dispersion for the PLAIN (no-ALiBi)
primary transformer across the same 5 seeds x 6 folds used for its ensemble.

Reuses 40's exact machinery (v4.train_one_model with base.MVAlibiClassifier
monkeypatched to the no-ALiBi subclass), but instead of averaging seeds it
calibrates and selects EACH seed on its own -- the correct way to report what a
single-seed model would score -- then aggregates each seed's per-fold test P&L
into one 6-year Sharpe per seed. Reports min / IQR / median / IQM / max.

Writes tables/transformer_seed_dispersion.{csv,json}. No existing script modified.
"""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TP2MVTransformer"
TABLE_DIR = OUT / "tables"

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

v4 = load_module("tp2_v4_disp", OUT / "03_train_mv_alibi_transformer_v4_multiseed.py")
noalibi = load_module("tp2_noalibi_disp", OUT / "37_train_transformer_no_alibi_ablation.py")
base = v4.base
SEEDS = v4.SEEDS
THR = 0.5
base.MVAlibiClassifier = noalibi.NoALiBiClassifier  # plain attention, identical to shipped primary

def sharpe(sel):
    if len(sel) == 0: return float("nan")
    d = sel.groupby("signal_date")["target_norm_pnl"].sum()
    return float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else float("nan")

def main():
    device = base.choose_device()
    print(f"device={device} | plain-attention per-seed dispersion | SEEDS={SEEDS}", flush=True)
    data = pd.read_parquet(base.TOKEN_PATH)
    data["signal_date"] = pd.to_datetime(data["signal_date"])
    data["signal_year"] = data["signal_date"].dt.year
    data = data.sort_values(["signal_date", "episode_id", "strategy"]).reset_index(drop=True)
    fd = pd.read_csv(base.FEATURE_DICT_PATH)
    feature_cols = fd[fd["selected_for_transformer"].astype(bool)].sort_values("selected_rank")["feature"].tolist()
    block_indices = {b: np.array([i for i, c in enumerate(feature_cols) if base.block_for_feature(c) == b], dtype=int)
                     for b in ["surface", "short_d", "liquidity", "lifecycle"]}

    per_seed_trades = {s: [] for s in SEEDS}
    perfold = []
    for fold in base.FOLDS:
        fn, ty = fold["fold"], fold["test"][0]
        print(f"Fold {fn}", flush=True)
        train = data[data["signal_year"].isin(fold["train"])].copy()
        val = data[data["signal_year"].isin(fold["val"])].copy()
        test = data[data["signal_year"].isin(fold["test"])].copy()
        scaler = base.StandardScaler(); scaler.fit(base.safe_numeric_frame(train, feature_cols))
        tr_a = base.make_arrays(train, train, feature_cols, scaler, block_indices)
        va_a = base.make_arrays(val, data[data["signal_year"].isin(fold["train"])].copy(), feature_cols, scaler, block_indices)
        te_a = base.make_arrays(test, data[data["signal_year"].isin(fold["train"] + fold["val"])].copy(), feature_cols, scaler, block_indices)
        for seed in SEEDS:
            val_p, test_p, best_loss, best_epoch, _ = v4.train_one_model(seed, tr_a, va_a, te_a, feature_cols, device)
            iso = base.fit_isotonic_calibration(val_p, va_a.y)
            test_cal = base.apply_calibration(iso, test_p)
            tt = te_a.meta.copy(); tt["predicted_prob"] = test_cal
            sel = base.select_episode_actions(tt, THR)
            per_seed_trades[seed].append(sel)
            perfold.append({"fold": fn, "test_year": ty, "seed": seed,
                            "n": int(len(sel)), "pnl": float(sel["target_norm_pnl"].sum()),
                            "sharpe": sharpe(sel), "best_epoch": best_epoch, "best_val_loss": best_loss})
            print(f"  seed={seed} n={len(sel):,} pnl=${sel['target_norm_pnl'].sum():,.0f} sharpe={sharpe(sel):.2f}", flush=True)

    # aggregate each seed across all 6 folds -> one 6-year Sharpe per seed
    seed_sharpe = {}
    for s in SEEDS:
        allt = pd.concat(per_seed_trades[s], ignore_index=True)
        seed_sharpe[s] = {"sharpe": sharpe(allt), "n": int(len(allt)), "pnl": float(allt["target_norm_pnl"].sum())}
    sh = np.array([seed_sharpe[s]["sharpe"] for s in SEEDS])
    q1, q3 = np.percentile(sh, [25, 75])
    iqm = sh[(sh >= q1) & (sh <= q3)].mean() if ((sh >= q1) & (sh <= q3)).any() else sh.mean()
    summary = {
        "model": "transformer_plain_no_alibi", "seeds": list(SEEDS),
        "per_seed_sharpe": {str(s): seed_sharpe[s]["sharpe"] for s in SEEDS},
        "min": float(sh.min()), "q1": float(q1), "median": float(np.median(sh)),
        "iqm": float(iqm), "q3": float(q3), "max": float(sh.max()), "std": float(sh.std()),
        "ensemble_reference_sharpe": 2.18,
    }
    pd.DataFrame(perfold).to_csv(TABLE_DIR / "transformer_seed_dispersion_perfold.csv", index=False)
    json.dump(summary, open(TABLE_DIR / "transformer_seed_dispersion.json", "w"), indent=2, default=float)
    print("\n[dispersion] per-seed 6yr Sharpe:", {s: round(seed_sharpe[s]["sharpe"], 2) for s in SEEDS}, flush=True)
    print(f"[dispersion] min={sh.min():.2f} q1={q1:.2f} median={np.median(sh):.2f} iqm={iqm:.2f} q3={q3:.2f} max={sh.max():.2f}", flush=True)

if __name__ == "__main__":
    main()
