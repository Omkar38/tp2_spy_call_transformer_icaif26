"""
TP2-MV-ALiBi transformer, 10-seed ensemble, final: the 10-seed sensitivity
check (03_train_mv_alibi_transformer_v4b_10seed_sensitivity.py) was only
ever run under the OLD, non-causal fraction-selection rule, before the
integrity-check fix and before p>=0.5 was shipped as the actual default.
It has never been re-run under the current, shipped methodology -- this
script closes that gap (user request, 2026-07-17).

Same seeds as the original v4b: the 5 already used in v4-final
(42, 101, 202, 303, 404) plus 5 more (505, 606, 707, 808, 909). Same
protocol as v3-final/v4-final (val-loss checkpoint, warm-up, grad clip
1.0, LR=3e-5) and the same causal p>=0.5 selection. Unlike v4-final, this
script also saves each seed's raw probabilities to disk (not just the
10-seed average), so a future seed-count analysis (e.g. 5 vs 10) doesn't
require retraining -- v4-final did not persist this, which is why this
script cannot simply reuse its already-trained 5 seeds and had to retrain
all 10 from scratch.

Imports v4's train_one_model via importlib (does not modify v4, v4-final,
or the original v4b script). Writes only *_10seedfinal_* outputs.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TP2MVTransformer"
DATA_DIR = OUT / "data"
TABLE_DIR = OUT / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

V4_PATH = OUT / "03_train_mv_alibi_transformer_v4_multiseed.py"


def load_v4_module():
    spec = importlib.util.spec_from_file_location("tp2_mv_alibi_v4_for_10seed_final", V4_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v4 = load_v4_module()
v3 = v4.v3
base = v4.base
SEEDS_5 = v4.SEEDS                        # [42, 101, 202, 303, 404]
SEEDS_EXTRA = [505, 606, 707, 808, 909]
SEEDS_10 = SEEDS_5 + SEEDS_EXTRA
SELECTION_THRESHOLD = 0.5


def score(test_tokens: pd.DataFrame, val_tokens: pd.DataFrame, val_y: np.ndarray, val_prob_avg: np.ndarray, test_prob_avg: np.ndarray) -> dict:
    iso = base.fit_isotonic_calibration(val_prob_avg, val_y)
    val_prob_cal = base.apply_calibration(iso, val_prob_avg)
    test_prob_cal = base.apply_calibration(iso, test_prob_avg)
    v_tokens = val_tokens.copy()
    t_tokens = test_tokens.copy()
    v_tokens["predicted_prob"] = val_prob_cal
    t_tokens["predicted_prob"] = test_prob_cal
    test_selected = base.select_episode_actions(t_tokens, SELECTION_THRESHOLD)
    metrics = base.summarize(test_selected, SELECTION_THRESHOLD)
    return metrics, test_selected, t_tokens


def main() -> None:
    device = base.choose_device()
    print(f"Using device: {device}", flush=True)
    print(f"[10seed-final] SEEDS_10={SEEDS_10} | selection=fixed p>={SELECTION_THRESHOLD}", flush=True)

    data = pd.read_parquet(base.TOKEN_PATH)
    data["signal_date"] = pd.to_datetime(data["signal_date"])
    data["signal_year"] = data["signal_date"].dt.year
    data = data.sort_values(["signal_date", "episode_id", "strategy"]).reset_index(drop=True)

    feature_dict = pd.read_csv(base.FEATURE_DICT_PATH)
    feature_cols = (
        feature_dict[feature_dict["selected_for_transformer"].astype(bool)]
        .sort_values("selected_rank")["feature"]
        .tolist()
    )
    block_indices = {
        block: np.array([i for i, col in enumerate(feature_cols) if base.block_for_feature(col) == block], dtype=int)
        for block in ["surface", "short_d", "liquidity", "lifecycle"]
    }

    fold_rows_10, fold_rows_5 = [], []
    prediction_rows_10 = []
    seed_diag_rows = []
    seed_prob_rows = []   # per-seed raw probabilities, persisted this time

    for fold in base.FOLDS:
        fold_name = fold["fold"]
        test_year = fold["test"][0]
        print(f"\n{'='*60}\nFold {fold_name}", flush=True)

        train = data[data["signal_year"].isin(fold["train"])].copy()
        val = data[data["signal_year"].isin(fold["val"])].copy()
        test = data[data["signal_year"].isin(fold["test"])].copy()

        scaler = base.StandardScaler()
        scaler.fit(base.safe_numeric_frame(train, feature_cols))
        train_arrays = base.make_arrays(train, train, feature_cols, scaler, block_indices)
        val_history = data[data["signal_year"].isin(fold["train"])].copy()
        val_arrays = base.make_arrays(val, val_history, feature_cols, scaler, block_indices)
        test_history = data[data["signal_year"].isin(fold["train"] + fold["val"])].copy()
        test_arrays = base.make_arrays(test, test_history, feature_cols, scaler, block_indices)

        val_probs_by_seed, test_probs_by_seed = {}, {}
        for seed in SEEDS_10:
            val_p, test_p, best_loss, best_epoch, _ = v4.train_one_model(
                seed, train_arrays, val_arrays, test_arrays, feature_cols, device
            )
            val_probs_by_seed[seed] = val_p
            test_probs_by_seed[seed] = test_p
            seed_diag_rows.append({"fold": fold_name, "test_year": test_year, "seed": seed, "best_epoch": best_epoch, "best_val_loss": best_loss})
            print(f"  seed={seed} best_epoch={best_epoch} best_val_loss={best_loss:.4f}", flush=True)

            seed_prob_rows.append(pd.DataFrame({
                "fold": fold_name, "test_year": test_year, "seed": seed,
                "episode_id": test_arrays.meta["episode_id"].to_numpy(),
                "strategy": test_arrays.meta["strategy"].to_numpy(),
                "test_prob_raw": test_p,
            }))

        val_tokens_base = val_arrays.meta.copy()
        test_tokens_base = test_arrays.meta.copy()

        val_10_avg = np.mean([val_probs_by_seed[s] for s in SEEDS_10], axis=0)
        test_10_avg = np.mean([test_probs_by_seed[s] for s in SEEDS_10], axis=0)
        metrics_10, selected_10, test_tokens_10 = score(test_tokens_base, val_tokens_base, val_arrays.y, val_10_avg, test_10_avg)
        test_tokens_10["fold"] = fold_name
        test_tokens_10["selected"] = test_tokens_10["episode_id"].isin(selected_10["episode_id"]) & (
            test_tokens_10["strategy"].eq(test_tokens_10["episode_id"].map(selected_10.set_index("episode_id")["strategy"].to_dict()))
        )
        prediction_rows_10.append(test_tokens_10)

        val_5_avg = np.mean([val_probs_by_seed[s] for s in SEEDS_5], axis=0)
        test_5_avg = np.mean([test_probs_by_seed[s] for s in SEEDS_5], axis=0)
        metrics_5, _, _ = score(test_tokens_base, val_tokens_base, val_arrays.y, val_5_avg, test_5_avg)

        fold_rows_10.append({"fold": fold_name, "test_year": test_year, "n_seeds": 10, **{f"test_{k}": v for k, v in metrics_10.items()}})
        fold_rows_5.append({"fold": fold_name, "test_year": test_year, "n_seeds": 5, **{f"test_{k}": v for k, v in metrics_5.items()}})

        print(
            f"  [10-seed] test_pnl=${metrics_10['norm_total_pnl']:,.0f} test_n={metrics_10['n_selected']:,} sharpe={metrics_10['daily_sharpe']:.2f}  |  "
            f"[5-seed, same run] test_pnl=${metrics_5['norm_total_pnl']:,.0f} test_n={metrics_5['n_selected']:,} sharpe={metrics_5['daily_sharpe']:.2f}",
            flush=True,
        )

    results_10 = pd.DataFrame(fold_rows_10)
    results_5 = pd.DataFrame(fold_rows_5)
    predictions_10 = pd.concat(prediction_rows_10, ignore_index=True)
    selected_10_all = predictions_10[predictions_10["selected"]].copy()
    seed_diagnostics = pd.DataFrame(seed_diag_rows)
    seed_probs = pd.concat(seed_prob_rows, ignore_index=True)

    results_10.to_csv(TABLE_DIR / "mv_alibi_10seedfinal_yearly_diagnostics.csv", index=False)
    results_5.to_csv(TABLE_DIR / "mv_alibi_10seedfinal_5seed_same_run_yearly_diagnostics.csv", index=False)
    seed_diagnostics.to_csv(TABLE_DIR / "mv_alibi_10seedfinal_seed_diagnostics.csv", index=False)
    seed_probs.to_parquet(DATA_DIR / "mv_alibi_10seedfinal_per_seed_test_probs.parquet", index=False)
    predictions_10.to_parquet(DATA_DIR / "mv_alibi_transformer_10seedfinal_test_predictions.parquet", index=False)

    summary_10 = {
        "method": "TP2-MV-ALiBi transformer, final (10-seed ensemble, causal p>=0.5 selection)",
        "seeds": SEEDS_10,
        "selection_rule": f"fixed p>={SELECTION_THRESHOLD}, zero tuning, fully causal",
        "total_selected": int(len(selected_10_all)),
        "total_norm_pnl": float(selected_10_all["target_norm_pnl"].sum()) if len(selected_10_all) else 0.0,
        "positive_years": int((results_10["test_norm_total_pnl"] > 0).sum()), "total_years": int(len(results_10)),
        "combined_daily_sharpe": base.daily_sharpe(selected_10_all), "combined_daily_sortino": base.daily_sortino(selected_10_all),
        "combined_max_drawdown": base.daily_drawdown(selected_10_all),
        "combined_win_rate": float((selected_10_all["target_norm_pnl"] > 0).mean()) if len(selected_10_all) else float("nan"),
    }
    with open(TABLE_DIR / "mv_alibi_transformer_10seedfinal_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary_10, fh, indent=2)

    print("\n" + "=" * 60)
    print("10-seed final summary:")
    print(json.dumps(summary_10, indent=2))
    print(f"\nWrote {TABLE_DIR / 'mv_alibi_transformer_10seedfinal_summary.json'}")


if __name__ == "__main__":
    main()
