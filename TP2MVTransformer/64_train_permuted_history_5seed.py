"""#13 permuted-history control at identical parameter count.

The no-history ablation (47/58) removes ~21k parameters, so a gap there could in
principle be a capacity effect. This control keeps the FULL plain-attention
transformer (all 37,086 params) but replaces each token's history -- instead of
its 64 most-recent strictly-earlier tokens, it attends to 64 *randomly sampled*
strictly-earlier tokens (same count, same causality, distances recomputed for the
sampled tokens). If historical attention works by attending to relevant recent
episodes, performance should fall toward the no-history level; if it were merely
extra parameters, it would not.

Reuses the plain-attention 5-seed pipeline (40's approach) by monkeypatching
base.build_history_arrays with a permuted-sampling version. No existing script
is modified. Writes only new *_permuted_history_5seed_* outputs.
"""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TP2MVTransformer"
DATA_DIR = OUT / "data"; TABLE_DIR = OUT / "tables"

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m

v4 = load_module("tp2_v4_perm", OUT / "03_train_mv_alibi_transformer_v4_multiseed.py")
noalibi = load_module("tp2_noalibi_perm", OUT / "37_train_transformer_no_alibi_ablation.py")
base = v4.base
SEEDS = v4.SEEDS
THR = 0.5
MAX_HISTORY = base.MAX_HISTORY
base.MVAlibiClassifier = noalibi.NoALiBiClassifier  # plain attention, matches primary

_PERM_RNG = np.random.default_rng(20260731)  # fixed: permutation is a property of the control, not of a model seed

def build_history_arrays_permuted(current, history, x_current, x_history, feature_cols, block_indices):
    """Identical to base.build_history_arrays but samples 64 RANDOM strictly-earlier
    tokens instead of the 64 most-recent ones."""
    hist_dates = base.date_to_day_number(history["signal_date"])
    cur_dates = base.date_to_day_number(current["signal_date"])
    hist_strategy = history["strategy_id"].to_numpy(dtype=np.float32)
    cur_strategy = current["strategy_id"].to_numpy(dtype=np.float32)

    hist_x = np.zeros((len(current), MAX_HISTORY, len(feature_cols)), dtype=np.float32)
    hist_mask = np.zeros((len(current), MAX_HISTORY), dtype=bool)
    distances = np.zeros((len(current), MAX_HISTORY, 6), dtype=np.float32)

    for row_idx, day in enumerate(cur_dates):
        end = np.searchsorted(hist_dates, day, side="left")
        if end <= 0:
            continue
        pool = np.arange(0, end)                       # all strictly-earlier tokens (causal)
        k = min(MAX_HISTORY, len(pool))
        idx = np.sort(_PERM_RNG.choice(pool, size=k, replace=False))  # random earlier tokens
        target_slice = slice(MAX_HISTORY - k, MAX_HISTORY)
        hx = x_history[idx]
        hist_x[row_idx, target_slice, :] = hx
        hist_mask[row_idx, target_slice] = True
        distances[row_idx, target_slice, 0] = np.log1p(np.maximum(day - hist_dates[idx], 0))
        for dist_pos, block_name in enumerate(["surface", "short_d", "liquidity", "lifecycle"], start=1):
            cols = block_indices.get(block_name, np.array([], dtype=int))
            if len(cols) == 0:
                continue
            distances[row_idx, target_slice, dist_pos] = np.abs(x_current[row_idx, cols] - hx[:, cols]).mean(axis=1)
        distances[row_idx, target_slice, 5] = (cur_strategy[row_idx] == hist_strategy[idx]).astype(np.float32)
    return hist_x, hist_mask, distances

base.build_history_arrays = build_history_arrays_permuted  # the one deliberate change

def sharpe(sel):
    if len(sel) == 0: return float("nan")
    d = sel.groupby("signal_date")["target_norm_pnl"].sum()
    return float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else float("nan")

def main():
    device = base.choose_device()
    print(f"[permuted-history-5seed] device={device} SEEDS={SEEDS} model=NoALiBiClassifier (plain), history=RANDOM earlier tokens", flush=True)
    data = pd.read_parquet(base.TOKEN_PATH)
    data["signal_date"] = pd.to_datetime(data["signal_date"]); data["signal_year"] = data["signal_date"].dt.year
    data = data.sort_values(["signal_date", "episode_id", "strategy"]).reset_index(drop=True)
    fd = pd.read_csv(base.FEATURE_DICT_PATH)
    feature_cols = fd[fd["selected_for_transformer"].astype(bool)].sort_values("selected_rank")["feature"].tolist()
    block_indices = {b: np.array([i for i, c in enumerate(feature_cols) if base.block_for_feature(c) == b], dtype=int)
                     for b in ["surface", "short_d", "liquidity", "lifecycle"]}

    fold_rows, prediction_rows, seed_diag_rows = [], [], []
    for fold in base.FOLDS:
        fn, ty = fold["fold"], fold["test"][0]
        print(f"Fold {fn}", flush=True)
        train = data[data["signal_year"].isin(fold["train"])].copy()
        val = data[data["signal_year"].isin(fold["val"])].copy()
        test = data[data["signal_year"].isin(fold["test"])].copy()
        scaler = base.StandardScaler(); scaler.fit(base.safe_numeric_frame(train, feature_cols))
        tr = base.make_arrays(train, train, feature_cols, scaler, block_indices)
        va = base.make_arrays(val, data[data["signal_year"].isin(fold["train"])].copy(), feature_cols, scaler, block_indices)
        te = base.make_arrays(test, data[data["signal_year"].isin(fold["train"] + fold["val"])].copy(), feature_cols, scaler, block_indices)
        val_ps, test_ps = [], []
        for seed in SEEDS:
            vp, tp, bl, be, _ = v4.train_one_model(seed, tr, va, te, feature_cols, device)
            val_ps.append(vp); test_ps.append(tp)
            seed_diag_rows.append({"fold": fn, "test_year": ty, "seed": seed, "best_epoch": be, "best_val_loss": bl})
            print(f"  seed={seed} best_epoch={be} best_val_loss={bl:.4f}", flush=True)
        va_avg = np.mean(val_ps, axis=0); te_avg = np.mean(test_ps, axis=0)
        iso = base.fit_isotonic_calibration(va_avg, va.y)
        tt = te.meta.copy(); tt["predicted_prob"] = base.apply_calibration(iso, te_avg)
        sel = base.select_episode_actions(tt, THR)
        m = base.summarize(sel, THR)
        try: auc = float(roc_auc_score(te.y, tt["predicted_prob"]))
        except ValueError: auc = float("nan")
        fold_rows.append({"test_year": ty, "n_seeds": len(SEEDS), "test_auc": auc,
                          **{f"test_{k}": v_ for k, v_ in m.items()}})
        tt["fold"] = fn
        tt["selected"] = tt["episode_id"].isin(sel["episode_id"]) & tt["strategy"].eq(
            tt["episode_id"].map(sel.set_index("episode_id")["strategy"].to_dict()))
        prediction_rows.append(tt)
        print(f"  pnl=${m['norm_total_pnl']:,.0f} n={m['n_selected']:,} sharpe={m['daily_sharpe']:.2f}", flush=True)

    results = pd.DataFrame(fold_rows)
    preds = pd.concat(prediction_rows, ignore_index=True)
    selected = preds[preds["selected"]].copy()
    pd.DataFrame(seed_diag_rows).to_csv(TABLE_DIR / "permuted_history_5seed_seed_diagnostics.csv", index=False)
    results.to_csv(TABLE_DIR / "permuted_history_5seed_yearly_diagnostics.csv", index=False)
    selected.to_parquet(DATA_DIR / "permuted_history_5seed_selected_trades.parquet", index=False)
    total = float(selected["target_norm_pnl"].sum())
    pos = int((results["test_norm_total_pnl"] > 0).sum())
    summary = {"model": "permuted_history_5seed", "total_pnl": total, "sharpe": sharpe(selected),
               "n_selected": int(len(selected)), "pos_years": pos,
               "per_year_pnl": results.set_index("test_year")["test_norm_total_pnl"].round(0).to_dict()}
    json.dump(summary, open(TABLE_DIR / "permuted_history_5seed_summary.json", "w"), indent=2, default=float)
    print(f"\n[permuted-history-5seed] TOTAL pnl=${total:,.0f} sharpe={sharpe(selected):.2f} pos_years={pos}/6 n={len(selected):,}", flush=True)

if __name__ == "__main__":
    main()
