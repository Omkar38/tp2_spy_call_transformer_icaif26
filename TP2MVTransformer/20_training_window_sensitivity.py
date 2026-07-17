"""
Training-window-length sensitivity: from the professor's checklist,
"experiment with different training window lengths (e.g., recent 1, 3, 6,
12 months, or all history)." Never tried -- the pipeline has always used a
fixed 3-year training window per fold. This script tests that directly.

New, separate script -- does not modify v3. Imports v3's protocol via
importlib (LR=3e-5, warm-up, grad clip 1.0, checkpoint on val_loss) and
re-implements only the FOLD CONSTRUCTION with a variable train-window
length. Restricted to 3 representative test years (2022, 2024, 2026 --
matching the architecture/weight-decay sweep) to keep runtime tractable.

Trade selection: causal p>=0.5 (trust the calibrated model directly, zero
tuning) -- NOT v3's rank-based fraction selection, which computes its
cutoff as a quantile over the entire test period and is therefore
non-causal. An earlier version of this script reused that fraction
mechanism and its results were confounded by the same tuning-noise issue
already diagnosed for v2's original threshold rule (see integrity check,
2026-07-14); switched to p>=0.5 for a clean, trustworthy comparison across
window lengths. The 3yr default-window comparison point is likewise
recomputed with p>=0.5 from v3's already-saved test predictions (pure
post-processing, not re-run) rather than reusing v3's original
fraction-selected numbers, so all four window lengths in this comparison
use the identical selection rule.

Window lengths tested: 1 year, 2 years, "all history" (every year from
2014 up to val_year-1). 3-year (current default) is not re-run -- already
have it.

Writes only window-suffixed outputs; no v2/v3/v4 artifact is modified.
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
EARLIEST_YEAR = 2014


def load_v3_module():
    spec = importlib.util.spec_from_file_location("tp2_mv_alibi_v3_for_window", V3_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v3 = load_v3_module()
base = v3.base

# (test_year, val_year) pairs matching base.FOLDS for the 3 representative years
TARGET_FOLDS = [
    (2022, 2021),
    (2024, 2023),
    (2026, 2025),
]

WINDOW_LENGTHS = {
    "1yr": lambda val_year: [val_year - 1],
    "2yr": lambda val_year: [val_year - 2, val_year - 1],
    "all_history": lambda val_year: list(range(EARLIEST_YEAR, val_year)),
}


def run_one_fold(train_years: list[int], val_year: int, test_year: int, data, feature_cols, block_indices, device):
    train = data[data["signal_year"].isin(train_years)].copy()
    val = data[data["signal_year"].eq(val_year)].copy()
    test = data[data["signal_year"].eq(test_year)].copy()
    if train.empty or val.empty or test.empty:
        return None

    scaler = base.StandardScaler()
    scaler.fit(base.safe_numeric_frame(train, feature_cols))
    train_arrays = base.make_arrays(train, train, feature_cols, scaler, block_indices)
    val_history = data[data["signal_year"].isin(train_years)].copy()
    val_arrays = base.make_arrays(val, val_history, feature_cols, scaler, block_indices)
    test_history = data[data["signal_year"].isin(train_years + [val_year])].copy()
    test_arrays = base.make_arrays(test, test_history, feature_cols, scaler, block_indices)

    torch.manual_seed(42)
    np.random.seed(42)
    pos_rate = float(train_arrays.y.mean())
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
    best_state = None
    stale = 0
    epochs_run = 0
    for epoch in range(1, base.EPOCHS + 1):
        epochs_run = epoch
        if epoch <= warmup_epochs:
            for group in optimizer.param_groups:
                group["lr"] = v3.LR * epoch / warmup_epochs
        v3.train_epoch_clip1(model, train_loader, optimizer, bce_criterion, huber_criterion, device, base.LAMBDA_RETURN)
        val_loss = base.evaluate_loss(model, val_loader, bce_criterion, huber_criterion, device, base.LAMBDA_RETURN)
        if epoch > warmup_epochs and np.isfinite(val_loss):
            scheduler.step(val_loss)
        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v_.detach().cpu().clone() for k, v_ in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= base.PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    val_prob_raw, _ = base.predict(model, val_loader, device)
    test_prob_raw, _ = base.predict(model, test_loader, device)
    iso = base.fit_isotonic_calibration(val_prob_raw, val_arrays.y)
    val_prob_cal = base.apply_calibration(iso, val_prob_raw)
    test_prob_cal = base.apply_calibration(iso, test_prob_raw)

    val_tokens = val_arrays.meta.copy()
    test_tokens = test_arrays.meta.copy()
    val_tokens["predicted_prob"] = val_prob_cal
    test_tokens["predicted_prob"] = test_prob_cal

    # Causal p>=0.5 selection (trust the calibrated model directly, zero
    # tuning) -- established elsewhere in this project
    # (17_causal_selection_comparison.py) as the best, most causal rule;
    # avoids the non-causal fraction-of-test-period quantile this script
    # used previously.
    test_selected = base.select_episode_actions(test_tokens, 0.5)
    stats = base.summarize(test_selected, 0.5)

    try:
        test_auc = float(roc_auc_score(test_arrays.y, test_prob_cal))
    except ValueError:
        test_auc = float("nan")

    return {
        "n_train_years": len(train_years), "train_years": ",".join(map(str, train_years)),
        "n_train_rows": len(train), "epochs_run": epochs_run, "best_val_loss": best_val_loss,
        "test_auc": test_auc, **stats,
    }, test_tokens


def main() -> None:
    device = base.choose_device()
    print(f"Using device: {device}", flush=True)

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

    rows = []
    all_test_tokens = []
    for test_year, val_year in TARGET_FOLDS:
        for window_name, window_fn in WINDOW_LENGTHS.items():
            train_years = window_fn(val_year)
            print(f"\n{'='*60}\ntest_year={test_year} window={window_name} train_years={train_years}", flush=True)
            outcome = run_one_fold(train_years, val_year, test_year, data, feature_cols, block_indices, device)
            if outcome is None:
                print("  SKIPPED (empty split)", flush=True)
                continue
            result, test_tokens = outcome
            result["test_year"] = test_year
            result["val_year"] = val_year
            result["window"] = window_name
            rows.append(result)
            test_tokens = test_tokens.copy()
            test_tokens["test_year"] = test_year
            test_tokens["window"] = window_name
            all_test_tokens.append(test_tokens)
            print(
                f"  n_train_years={result['n_train_years']} n_train_rows={result['n_train_rows']:,} "
                f"epochs={result['epochs_run']} best_val_loss={result['best_val_loss']:.4f} "
                f"test_auc={result['test_auc']:.4f} trades={result['n_selected']:,} "
                f"pnl=${result['norm_total_pnl']:,.0f} sharpe={result['daily_sharpe']:.2f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "training_window_sensitivity.csv", index=False)
    if all_test_tokens:
        pd.concat(all_test_tokens, ignore_index=True).to_parquet(
            DATA_DIR / "training_window_sensitivity_test_predictions.parquet", index=False
        )
        print(f"Wrote {DATA_DIR / 'training_window_sensitivity_test_predictions.parquet'}")
    print("\n" + "=" * 60)
    print(df.to_string(index=False))
    print(f"\nWrote {TABLE_DIR / 'training_window_sensitivity.csv'}")


if __name__ == "__main__":
    main()
