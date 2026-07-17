"""
TP2-MV-ALiBi v3-final: v3's confirmed training protocol (val_loss checkpoint,
warm-up, grad clip 1.0, LR=3e-5) with the trade-selection rule replaced by
the causal p>=0.5 rule (trust the calibrated model directly, zero tuning).

This is the "shipped" selection-rule decision, not a post-hoc rescore. It
was validated post-hoc first (17_causal_selection_comparison.py,
22_arch_wd_sweep_causal_rescore.py) and confirmed to hold under a
held-out split-sample check (early years 2021-2023 vs. late years
2024-2026: p>=0.5 wins on Sharpe/drawdown/consistency using ONLY early
years' evidence, then holds up unchanged on the untouched late years) --
see the conversation record / project memory for that check. This script
makes the decision real: replaces v3's `tune_selection_fraction` /
`apply_selection_fraction` (which computes its cutoff as a quantile over
the entire test period -- non-causal, not live-deployable) with a single
fixed constant, 0.5, applied identically to every trade with no reference
to any other trade's prediction.

Imports v3's protocol via importlib (does not modify v3). Runs all 6
walk-forward folds (2021-2026 test years), single seed=42, matching v3's
exact training loop -- only the selection step at the end of each fold
differs. Full per-epoch train_loss/val_loss/val_auc history is saved for
every fold so nothing needs to be re-run to inspect training dynamics.

Outputs are written under new *_v3final_* filenames; no v2/v3/v4 artifact
is modified.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TP2MVTransformer"
DATA_DIR = OUT / "data"
TABLE_DIR = OUT / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

V3_PATH = OUT / "03_train_mv_alibi_transformer_v3.py"


def load_v3_module():
    spec = importlib.util.spec_from_file_location("tp2_mv_alibi_v3_for_final", V3_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v3 = load_v3_module()
base = v3.base

SELECTION_THRESHOLD = 0.5


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    device = base.choose_device()
    print(f"Using device: {device}", flush=True)
    print(
        f"[v3-final] MAX_HISTORY={base.MAX_HISTORY} | EPOCHS={base.EPOCHS} | PATIENCE={base.PATIENCE} | "
        f"hidden=64 | LAMBDA_RETURN={base.LAMBDA_RETURN} | WARMUP_FRACTION={v3.WARMUP_FRACTION} | "
        f"GRAD_CLIP_NORM={v3.GRAD_CLIP_NORM} | LR={v3.LR} | checkpoint_metric=val_loss | "
        f"selection=fixed p>={SELECTION_THRESHOLD} (causal, zero tuning)",
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
    training_history_rows = []
    distance_rows = []
    attention_rows = []

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
        distance_rows.extend(base.distance_summary(train_arrays, fold_name, "train"))
        distance_rows.extend(base.distance_summary(val_arrays, fold_name, "validation"))
        distance_rows.extend(base.distance_summary(test_arrays, fold_name, "test"))
        print(
            f"  history coverage: train={train_arrays.hist_mask.mean():.3f} "
            f"val={val_arrays.hist_mask.mean():.3f} "
            f"test={test_arrays.hist_mask.mean():.3f}",
            flush=True,
        )

        pos_rate = float(train["target_profitable"].mean())
        pos_weight = torch.tensor([(1.0 - pos_rate) / max(pos_rate, 1e-6)], dtype=torch.float32, device=device)

        model = base.MVAlibiClassifier(
            n_features=len(feature_cols), hidden=64, heads=4, use_return_head=(base.LAMBDA_RETURN > 0),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=v3.LR, weight_decay=base.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        huber_criterion = nn.HuberLoss(delta=0.5)

        train_loader = DataLoader(base.TP2HistoryDataset(train_arrays), batch_size=base.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(base.TP2HistoryDataset(val_arrays), batch_size=base.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(base.TP2HistoryDataset(test_arrays), batch_size=base.BATCH_SIZE, shuffle=False)

        warmup_epochs = max(1, round(base.EPOCHS * v3.WARMUP_FRACTION))
        best_val_loss = float("inf")
        best_val_auc_at_best = float("nan")
        best_state = None
        best_epoch = 0
        stale = 0
        for epoch in range(1, base.EPOCHS + 1):
            if epoch <= warmup_epochs:
                warmup_lr = v3.LR * epoch / warmup_epochs
                for group in optimizer.param_groups:
                    group["lr"] = warmup_lr
                lr_phase = "warmup"
            else:
                lr_phase = "plateau"

            train_loss, grad_norm = v3.train_epoch_clip1(
                model, train_loader, optimizer, bce_criterion, huber_criterion, device, base.LAMBDA_RETURN
            )
            val_loss = base.evaluate_loss(model, val_loader, bce_criterion, huber_criterion, device, base.LAMBDA_RETURN)
            val_prob, _ = base.predict(model, val_loader, device)
            try:
                val_auc = float(roc_auc_score(val_arrays.y, val_prob))
            except ValueError:
                val_auc = float("nan")

            training_history_rows.append({
                "fold": fold_name, "test_year": test_year, "epoch": epoch, "lr_phase": lr_phase,
                "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc,
                "mean_grad_norm_pre_clip": grad_norm, "learning_rate": float(optimizer.param_groups[0]["lr"]),
            })
            print(
                f"  epoch={epoch:02d} [{lr_phase}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_auc={val_auc:.4f} grad_norm={grad_norm:.3f} lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

            if epoch > warmup_epochs and np.isfinite(val_loss):
                scheduler.step(val_loss)

            if np.isfinite(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_auc_at_best = val_auc
                best_epoch = epoch
                best_state = {k: v_.detach().cpu().clone() for k, v_ in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= base.PATIENCE:
                    print(f"  Early stopping at epoch {epoch}", flush=True)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        attention_rows.extend(base.attention_parameter_rows(model, fold_name, test_year))

        val_prob_raw, val_ret_pred = base.predict(model, val_loader, device)
        test_prob_raw, test_ret_pred = base.predict(model, test_loader, device)
        iso = base.fit_isotonic_calibration(val_prob_raw, val_arrays.y)
        val_prob_cal = base.apply_calibration(iso, val_prob_raw)
        test_prob_cal = base.apply_calibration(iso, test_prob_raw)

        val_tokens = val_arrays.meta.copy()
        test_tokens = test_arrays.meta.copy()
        val_tokens["predicted_prob"] = val_prob_cal
        val_tokens["predicted_prob_raw"] = val_prob_raw
        val_tokens["predicted_return"] = val_ret_pred
        test_tokens["predicted_prob"] = test_prob_cal
        test_tokens["predicted_prob_raw"] = test_prob_raw
        test_tokens["predicted_return"] = test_ret_pred

        # Causal, zero-tuning selection: a fixed constant compared against
        # each trade's own predicted probability -- no validation search,
        # no reference to any other trade's prediction.
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
            "val_year": fold["val"][0], "test_year": test_year, "feature_count": len(feature_cols),
            "max_history_tokens": base.MAX_HISTORY, "best_epoch": best_epoch, "best_val_loss": best_val_loss,
            "val_auc_at_best_epoch": best_val_auc_at_best, "test_auc": test_auc,
            "selection_threshold": SELECTION_THRESHOLD,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
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
    selected = predictions[predictions["selected"]].copy()
    training_history = pd.DataFrame(training_history_rows)
    distance_diagnostics = pd.DataFrame(distance_rows)
    attention_diagnostics = pd.DataFrame(attention_rows)

    selected_daily = (
        selected.groupby("signal_date", as_index=False)["target_norm_pnl"].sum()
        if len(selected) else pd.DataFrame(columns=["signal_date", "target_norm_pnl"])
    )

    yearly = results[[
        "test_year", "selection_threshold", "test_n_selected", "test_norm_total_pnl", "test_win_rate",
        "test_daily_sharpe", "test_daily_sortino", "test_max_drawdown", "test_t1_selected", "test_k2_selected",
        "val_auc_at_best_epoch", "test_auc", "best_epoch", "best_val_loss",
    ]].copy()
    yearly.to_csv(TABLE_DIR / "mv_alibi_v3final_yearly_diagnostics.csv", index=False)
    training_history.to_csv(TABLE_DIR / "mv_alibi_v3final_training_history.csv", index=False)
    distance_diagnostics.to_csv(TABLE_DIR / "mv_alibi_v3final_distance_diagnostics.csv", index=False)
    attention_diagnostics.to_csv(TABLE_DIR / "mv_alibi_v3final_attention_parameters.csv", index=False)
    selected.to_parquet(DATA_DIR / "mv_alibi_transformer_v3final_selected_trades.parquet", index=False)
    selected_daily.to_csv(TABLE_DIR / "mv_alibi_transformer_v3final_selected_daily_pnl.csv", index=False)

    summary = {
        "method": "TP2-MV-ALiBi transformer, final (val-loss checkpoint + warm-up + causal p>=0.5 selection)",
        "feature_count": len(feature_cols), "max_history_tokens": base.MAX_HISTORY, "hidden_dim": 64,
        "epochs_max": base.EPOCHS, "warmup_fraction": v3.WARMUP_FRACTION, "grad_clip_norm": v3.GRAD_CLIP_NORM,
        "checkpoint_metric": "val_loss (min)", "selection_rule": f"fixed p>={SELECTION_THRESHOLD}, zero tuning, fully causal",
        "lambda_return": base.LAMBDA_RETURN, "calibration": "isotonic on val probs",
        "total_selected": int(len(selected)), "total_norm_pnl": float(selected["target_norm_pnl"].sum()) if len(selected) else 0.0,
        "positive_years": int((results["test_norm_total_pnl"] > 0).sum()), "total_years": int(len(results)),
        "combined_daily_sharpe": base.daily_sharpe(selected), "combined_daily_sortino": base.daily_sortino(selected),
        "combined_max_drawdown": base.daily_drawdown(selected),
        "combined_win_rate": float((selected["target_norm_pnl"] > 0).mean()) if len(selected) else float("nan"),
        "device": str(device),
        "notes": [
            "Selection rule is a fixed constant (p>=0.5) compared per-trade against that trade's own "
            "calibrated probability -- no validation-year tuning, no reference to any other trade's "
            "prediction, no quantile computed over any period. Fully causal and live-deployable as-is.",
            "Training protocol (LR=3e-5, warm-up, grad clip 1.0, checkpoint on val_loss) is unchanged from v3.",
            "This script writes only *_v3final_* outputs; no v2/v3/v4 artifact is modified.",
        ],
    }

    val_predictions = pd.concat(val_prediction_rows, ignore_index=True)
    results.to_csv(TABLE_DIR / "mv_alibi_transformer_v3final_results.csv", index=False)
    predictions.to_parquet(DATA_DIR / "mv_alibi_transformer_v3final_test_predictions.parquet", index=False)
    val_predictions.to_parquet(DATA_DIR / "mv_alibi_transformer_v3final_val_predictions.parquet", index=False)
    with open(TABLE_DIR / "mv_alibi_transformer_v3final_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 60)
    print("TP2-MV-ALiBi v3-final summary:")
    print(json.dumps(summary, indent=2))
    print("\nYearly diagnostics:")
    print(yearly.to_string(index=False))
    print(f"\nWrote {TABLE_DIR / 'mv_alibi_transformer_v3final_summary.json'}")


if __name__ == "__main__":
    main()
