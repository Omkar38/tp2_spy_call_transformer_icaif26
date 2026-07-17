"""
TP2-MV-ALiBi v4-final: v4's confirmed 5-seed ensemble protocol with the
trade-selection rule replaced by the causal p>=0.5 rule (trust the
calibrated, seed-averaged model directly, zero tuning) -- the "shipped"
selection-rule decision. See 23_train_mv_alibi_transformer_v3_final.py's
docstring for the full justification (post-hoc validation, held-out
split-sample check).

Imports v4's already-built `train_one_model` function via importlib
(does not modify v4) -- reuses the exact 5-seed training loop unchanged.
Only the selection step at the end of each fold differs: v4's
`tune_selection_fraction`/`apply_selection_fraction` (non-causal, quantile
over the entire test period) is replaced with a fixed constant, 0.5.

Writes only *_v4final_* outputs; no v2/v3/v4/v3final artifact is modified.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TP2MVTransformer"
DATA_DIR = OUT / "data"
TABLE_DIR = OUT / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

V4_PATH = OUT / "03_train_mv_alibi_transformer_v4_multiseed.py"


def load_v4_module():
    spec = importlib.util.spec_from_file_location("tp2_mv_alibi_v4_for_final", V4_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v4 = load_v4_module()
v3 = v4.v3
base = v4.base
SEEDS = v4.SEEDS
SELECTION_THRESHOLD = 0.5


def main() -> None:
    device = base.choose_device()
    print(f"Using device: {device}", flush=True)
    print(
        f"[v4-final] SEEDS={SEEDS} | LR={v3.LR} | GRAD_CLIP_NORM={v3.GRAD_CLIP_NORM} | "
        f"checkpoint_metric=val_loss | selection=fixed p>={SELECTION_THRESHOLD} (causal, zero tuning)",
        flush=True,
    )

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
    if "signal_year" in feature_cols:
        raise RuntimeError("signal_year leaked into feature_cols")

    block_indices = {
        block: np.array([i for i, col in enumerate(feature_cols) if base.block_for_feature(col) == block], dtype=int)
        for block in ["surface", "short_d", "liquidity", "lifecycle"]
    }

    fold_rows = []
    prediction_rows = []
    val_prediction_rows = []
    seed_diag_rows = []
    training_history_rows = []

    for fold in base.FOLDS:
        fold_name = fold["fold"]
        test_year = fold["test"][0]
        print(f"\n{'='*60}", flush=True)
        print(f"Fold {fold_name}", flush=True)

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
        print(
            f"  history coverage: train={train_arrays.hist_mask.mean():.3f} "
            f"val={val_arrays.hist_mask.mean():.3f} test={test_arrays.hist_mask.mean():.3f}",
            flush=True,
        )

        val_probs_by_seed, test_probs_by_seed = [], []
        for seed in SEEDS:
            val_p, test_p, best_loss, best_epoch, seed_history = v4.train_one_model(
                seed, train_arrays, val_arrays, test_arrays, feature_cols, device
            )
            val_probs_by_seed.append(val_p)
            test_probs_by_seed.append(test_p)
            seed_diag_rows.append({
                "fold": fold_name, "test_year": test_year, "seed": seed,
                "best_epoch": best_epoch, "best_val_loss": best_loss,
            })
            for r in seed_history:
                r["fold"] = fold_name
                r["test_year"] = test_year
            training_history_rows.extend(seed_history)
            print(f"  seed={seed} best_epoch={best_epoch} best_val_loss={best_loss:.4f}", flush=True)

        val_prob_avg = np.mean(val_probs_by_seed, axis=0)
        test_prob_avg = np.mean(test_probs_by_seed, axis=0)
        seed_losses = [r["best_val_loss"] for r in seed_diag_rows if r["fold"] == fold_name]
        print(
            f"  ensemble across {len(SEEDS)} seeds: mean best_val_loss={np.mean(seed_losses):.4f} "
            f"std={np.std(seed_losses):.4f}",
            flush=True,
        )

        iso = base.fit_isotonic_calibration(val_prob_avg, val_arrays.y)
        val_prob_cal = base.apply_calibration(iso, val_prob_avg)
        test_prob_cal = base.apply_calibration(iso, test_prob_avg)

        val_tokens = val_arrays.meta.copy()
        test_tokens = test_arrays.meta.copy()
        val_tokens["predicted_prob"] = val_prob_cal
        val_tokens["predicted_prob_raw"] = val_prob_avg
        test_tokens["predicted_prob"] = test_prob_cal
        test_tokens["predicted_prob_raw"] = test_prob_avg

        val_selected = base.select_episode_actions(val_tokens, SELECTION_THRESHOLD)
        test_selected = base.select_episode_actions(test_tokens, SELECTION_THRESHOLD)
        val_metrics = base.summarize(val_selected, SELECTION_THRESHOLD)
        test_metrics = base.summarize(test_selected, SELECTION_THRESHOLD)
        try:
            test_auc = float(roc_auc_score(test_arrays.y, test_prob_cal))
        except ValueError:
            test_auc = float("nan")

        row = {
            "fold": fold_name, "train_years": ",".join(map(str, fold["train"])),
            "val_year": fold["val"][0], "test_year": test_year, "n_seeds": len(SEEDS),
            "mean_best_val_loss": float(np.mean(seed_losses)), "std_best_val_loss": float(np.std(seed_losses)),
            "test_auc": test_auc, "selection_threshold": SELECTION_THRESHOLD,
            **{f"val_{k}": v_ for k, v_ in val_metrics.items()},
            **{f"test_{k}": v_ for k, v_ in test_metrics.items()},
        }
        fold_rows.append(row)

        test_tokens["fold"] = fold_name
        test_tokens["selected"] = test_tokens["episode_id"].isin(test_selected["episode_id"]) & (
            test_tokens["strategy"].eq(
                test_tokens["episode_id"].map(test_selected.set_index("episode_id")["strategy"].to_dict())
            )
        )
        prediction_rows.append(test_tokens)
        val_tokens["fold"] = fold_name
        val_prediction_rows.append(val_tokens)

        print(
            f"  threshold={SELECTION_THRESHOLD} test_pnl=${test_metrics['norm_total_pnl']:,.0f} "
            f"test_n={test_metrics['n_selected']:,} sharpe={test_metrics['daily_sharpe']:.2f} "
            f"win_rate={test_metrics['win_rate']:.3f}",
            flush=True,
        )

    results = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    val_predictions = pd.concat(val_prediction_rows, ignore_index=True)
    selected = predictions[predictions["selected"]].copy()
    seed_diagnostics = pd.DataFrame(seed_diag_rows)
    training_history = pd.DataFrame(training_history_rows)

    selected_daily = (
        selected.groupby("signal_date", as_index=False)["target_norm_pnl"].sum()
        if len(selected) else pd.DataFrame(columns=["signal_date", "target_norm_pnl"])
    )

    yearly = results[[
        "test_year", "n_seeds", "mean_best_val_loss", "std_best_val_loss", "selection_threshold",
        "test_n_selected", "test_norm_total_pnl", "test_win_rate", "test_daily_sharpe",
        "test_daily_sortino", "test_max_drawdown", "test_t1_selected", "test_k2_selected", "test_auc",
    ]].copy()
    yearly.to_csv(TABLE_DIR / "mv_alibi_v4final_yearly_diagnostics.csv", index=False)
    seed_diagnostics.to_csv(TABLE_DIR / "mv_alibi_v4final_seed_diagnostics.csv", index=False)
    training_history.to_csv(TABLE_DIR / "mv_alibi_v4final_training_history.csv", index=False)
    selected.to_parquet(DATA_DIR / "mv_alibi_transformer_v4final_selected_trades.parquet", index=False)
    selected_daily.to_csv(TABLE_DIR / "mv_alibi_transformer_v4final_selected_daily_pnl.csv", index=False)

    summary = {
        "method": f"TP2-MV-ALiBi transformer, final ({len(SEEDS)}-seed ensemble, causal p>=0.5 selection)",
        "seeds": SEEDS,
        "base_protocol": "v3 (val-loss checkpoint, warm-up, grad clip 1.0, LR=3e-5)",
        "selection_rule": f"fixed p>={SELECTION_THRESHOLD}, zero tuning, fully causal",
        "total_selected": int(len(selected)), "total_norm_pnl": float(selected["target_norm_pnl"].sum()) if len(selected) else 0.0,
        "positive_years": int((results["test_norm_total_pnl"] > 0).sum()), "total_years": int(len(results)),
        "combined_daily_sharpe": base.daily_sharpe(selected), "combined_daily_sortino": base.daily_sortino(selected),
        "combined_max_drawdown": base.daily_drawdown(selected),
        "combined_win_rate": float((selected["target_norm_pnl"] > 0).mean()) if len(selected) else float("nan"),
        "device": str(device),
        "notes": [
            "Selection rule is a fixed constant (p>=0.5) compared per-trade against that trade's own "
            "seed-averaged calibrated probability -- no validation-year tuning, no reference to any "
            "other trade's prediction. Fully causal and live-deployable as-is.",
            f"Training: {len(SEEDS)} independently-seeded models per fold, v3's exact protocol, raw "
            "probabilities averaged before one isotonic calibration per fold.",
            "This script writes only *_v4final_* outputs; no v2/v3/v4/v3final artifact is modified.",
        ],
    }

    results.to_csv(TABLE_DIR / "mv_alibi_transformer_v4final_results.csv", index=False)
    predictions.to_parquet(DATA_DIR / "mv_alibi_transformer_v4final_test_predictions.parquet", index=False)
    val_predictions.to_parquet(DATA_DIR / "mv_alibi_transformer_v4final_val_predictions.parquet", index=False)
    with open(TABLE_DIR / "mv_alibi_transformer_v4final_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 60)
    print("TP2-MV-ALiBi v4-final summary:")
    print(json.dumps(summary, indent=2))
    print("\nYearly diagnostics:")
    print(yearly.to_string(index=False))
    print(f"\nWrote {TABLE_DIR / 'mv_alibi_transformer_v4final_summary.json'}")


if __name__ == "__main__":
    main()
