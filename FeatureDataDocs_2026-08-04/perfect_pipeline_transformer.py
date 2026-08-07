"""PERFECT leak-free pipeline for the 5-seed transformers (ALiBi and Plain).

Same protocol validated on MLP-5:
  Step 1  per fold, ensemble deterministic LightGBM x5 seeds on TRAINING YEARS ONLY
          -> average importance -> top 48 = F*_k   (NO calibration-year leak)
  Step 2  per-fold train scaler, 5-seed transformer, average probs
  Step 3  isotonic calibration fit on the calibration (val) year only
  Step 4  causal p>=0.5, one action/episode

The only structural change vs script 24/40: feature_cols + block_indices are
rebuilt PER FOLD from F*_k (precomputed by precompute_fk.py in a torch-free process,
since LightGBM and torch both bundle libomp on macOS and segfault if co-loaded)
instead of read once from the frozen dictionary. Everything else (make_arrays /
train_one_model / calibration / selection) is the committed transformer code, unmodified.

Usage:  python precompute_fk.py            (once; writes nested/fk_ensemble_perfect.json)
        python perfect_pipeline_transformer.py {alibi|plain}
Read-only w.r.t. the repo. Artifact: nested/perfect_pipeline_transformer_<variant>.json
"""
from __future__ import annotations
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

VARIANT = (sys.argv[1] if len(sys.argv) > 1 else "alibi").lower()
assert VARIANT in ("alibi", "plain"), "variant must be alibi or plain"
SMOKE = os.environ.get("SMOKE") == "1"

ROOT = Path("/Users/jyotirpatel/Downloads/FinalQuant copy 2"); TP2 = ROOT/"TP2MVTransformer"
OUTD = Path(__file__).resolve().parent/"nested"; OUTD.mkdir(exist_ok=True)
ANN = np.sqrt(252)

# ---- precomputed per-fold ensemble F*_k (model-independent; same for both variants) ----
FK = json.load(open(OUTD/"fk_ensemble_perfect.json"))["fk_by_year"]

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

# ---- transformer stack (v4 multiseed = train_one_model; base = shared helpers) ----
v4 = load_module(f"tp2_v4_pf_{VARIANT}", TP2/"03_train_mv_alibi_transformer_v4_multiseed.py")
v3 = v4.v3
base = v4.base
SEEDS = v4.SEEDS
SELECTION_THRESHOLD = 0.5
if VARIANT == "plain":
    noalibi = load_module("tp2_noalibi_pf", TP2/"37_train_transformer_no_alibi_ablation.py")
    base.MVAlibiClassifier = noalibi.NoALiBiClassifier      # plain attention, exactly as script 40

CANON = base.FOLDS
if SMOKE:
    CANON = CANON[:1]; SEEDS = SEEDS[:1]

# ---- data ----
data = pd.read_parquet(base.TOKEN_PATH)
data["signal_date"] = pd.to_datetime(data["signal_date"])
data["signal_year"] = data["signal_date"].dt.year
data = data.sort_values(["signal_date", "episode_id", "strategy"]).reset_index(drop=True)

def metrics(df):
    r = df["target_norm_pnl"].astype(float); daily = df.groupby("signal_date")["target_norm_pnl"].sum()
    sh = daily.mean()/daily.std()*ANN if daily.std() > 0 else float("nan")
    dn = daily[daily < 0]; so = daily.mean()/dn.std()*ANN if len(dn) > 1 and dn.std() > 0 else float("nan")
    yr = df.groupby(df["signal_date"].dt.year)["target_norm_pnl"].sum()
    return {"n": int(len(df)), "pnl": float(r.sum()), "sharpe": round(float(sh),4), "sortino": round(float(so),4),
            "win_rate": round(float((r > 0).mean()),4), "pos_years": int((yr > 0).sum())}

def run():
    device = base.choose_device()
    print(f"[perfect-{VARIANT}] device={device} SEEDS={SEEDS} "
          f"model={'NoALiBi' if VARIANT=='plain' else 'MVAlibi'}", flush=True)
    allsel = []; allpred = []; per_fold = []; fks = {}
    for fold in CANON:
        t0 = time.time(); fold_name = fold["fold"]; test_year = fold["test"][0]
        feature_cols = list(FK[str(test_year)]); fks[test_year] = feature_cols
        if "signal_year" in feature_cols:
            raise RuntimeError("signal_year leaked into feature_cols")
        block_indices = {b: np.array([i for i, c in enumerate(feature_cols) if base.block_for_feature(c) == b], dtype=int)
                         for b in ["surface", "short_d", "liquidity", "lifecycle"]}

        train = data[data["signal_year"].isin(fold["train"])].copy()
        val = data[data["signal_year"].isin(fold["val"])].copy()
        test = data[data["signal_year"].isin(fold["test"])].copy()
        scaler = base.StandardScaler(); scaler.fit(base.safe_numeric_frame(train, feature_cols))
        train_arrays = base.make_arrays(train, train, feature_cols, scaler, block_indices)
        val_history = data[data["signal_year"].isin(fold["train"])].copy()
        val_arrays = base.make_arrays(val, val_history, feature_cols, scaler, block_indices)
        test_history = data[data["signal_year"].isin(fold["train"] + fold["val"])].copy()
        test_arrays = base.make_arrays(test, test_history, feature_cols, scaler, block_indices)

        vps, tps, losses = [], [], []
        for seed in SEEDS:
            val_p, test_p, best_loss, best_epoch, _ = v4.train_one_model(
                seed, train_arrays, val_arrays, test_arrays, feature_cols, device)
            vps.append(val_p); tps.append(test_p); losses.append(best_loss)
            print(f"    fold {fold_name} seed={seed} best_epoch={best_epoch} val_loss={best_loss:.4f}", flush=True)
        val_avg = np.mean(vps, axis=0); test_avg = np.mean(tps, axis=0)

        iso = base.fit_isotonic_calibration(val_avg, val_arrays.y)
        test_cal = base.apply_calibration(iso, test_avg)
        tok = test_arrays.meta.copy(); tok["predicted_prob"] = test_cal
        allpred.append(tok.copy())   # full test-population predictions (for calibration Brier/ECE)
        selected = base.select_episode_actions(tok, SELECTION_THRESHOLD)
        allsel.append(selected)
        try:
            auc = float(roc_auc_score(test_arrays.y, test_cal))
        except ValueError:
            auc = float("nan")
        m = metrics(selected)
        per_fold.append({"fold": fold_name, "test_year": test_year, "auc": round(auc,4),
                         "mean_val_loss": round(float(np.mean(losses)),4), **m})
        print(f"  == fold {fold_name} test={test_year}: N={m['n']:,} PnL=${m['pnl']:,.0f} "
              f"Sharpe={m['sharpe']:.2f} AUC={auc:.3f} ({time.time()-t0:.0f}s)", flush=True)

    all_trades = pd.concat(allsel).reset_index(drop=True)
    agg = metrics(all_trades)
    if not SMOKE:   # persist per-trade selections so Sharpe/etc. can be recomputed WITHOUT a rerun
        cols = [c for c in ["episode_id","signal_date","signal_year","strategy","target_norm_pnl","predicted_prob"] if c in all_trades.columns]
        all_trades[cols].to_parquet(OUTD/f"trades_transformer_{VARIANT}.parquet")
        all_trades[cols].to_csv(OUTD/f"trades_transformer_{VARIANT}.csv", index=False)
        fp = pd.concat(allpred).reset_index(drop=True)   # full test-population predictions
        fpcols = [c for c in ["episode_id","signal_date","signal_year","strategy","target_profitable","target_norm_pnl","predicted_prob"] if c in fp.columns]
        fp[fpcols].to_parquet(OUTD/f"fullpred_transformer_{VARIANT}.parquet")
        print(f"  saved trades_transformer_{VARIANT} ({len(all_trades):,}) + fullpred ({len(fp):,})", flush=True)
    out = {"variant": VARIANT, "config": "per-fold ensemble deterministic F* (train-years-only), 5-seed transformer, val isotonic, p>=0.5",
           "disc_seeds": json.load(open(OUTD/"fk_ensemble_perfect.json"))["disc_seeds"],
           "model_seeds": SEEDS, "aggregate": agg,
           "per_fold": per_fold, "fk_by_year": {str(y): f for y, f in fks.items()}}
    tag = "smoke_" if SMOKE else ""
    dest = OUTD/f"perfect_pipeline_transformer_{tag}{VARIANT}.json"
    json.dump(out, open(dest, "w"), indent=2, default=float)
    print(f"\n=== PERFECT {VARIANT.upper()} TRANSFORMER: N={agg['n']:,} PnL=${agg['pnl']:,.0f} "
          f"Sharpe={agg['sharpe']:.2f} Sortino={agg['sortino']:.2f} +yrs={agg['pos_years']}/6 ===", flush=True)
    print("wrote", dest, flush=True)

if __name__ == "__main__":
    run()
