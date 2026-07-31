"""
Professor action-list #3: no-history ablation at 5 seeds.

Takes the single-seed NoHistoryClassifier ablation (47_train_no_history_ablation.py)
and runs it at the same 5 seeds (42,101,202,303,404) as every other ensemble in
the paper, averaging raw probabilities across seeds before one isotonic calibration
per fold -- identical ensembling protocol to 40_train_no_alibi_5seed_ensemble.py.

This closes the matched-seed-count gap the professor flagged: §4.7's no-history
result was single-seed (42) while it is compared against the 5-seed transformer.

Reuses 47's NoHistoryClassifier and v3's training helpers via importlib. Does not
modify any existing script. Writes only new *_no_history_5seed_* outputs.
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

NOHIST_PATH = OUT / "47_train_no_history_ablation.py"

SEEDS = [42, 101, 202, 303, 404]
SELECTION_THRESHOLD = 0.5


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nohist = load_module("tp2_no_history_for_5seed", NOHIST_PATH)
v3 = nohist.v3
base = v3.base
NoHistoryClassifier = nohist.NoHistoryClassifier


def train_one_seed(seed, train_arrays, val_arrays, test_arrays, feature_cols, pos_weight, device):
    """One NoHistory model at a given seed; returns raw (uncalibrated) val/test probs.
    Mirrors 47_train_no_history_ablation.py's per-fold loop exactly."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = NoHistoryClassifier(
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
    best_epoch = 0
    stale = 0
    for epoch in range(1, base.EPOCHS + 1):
        if epoch <= warmup_epochs:
            warmup_lr = v3.LR * epoch / warmup_epochs
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr
        train_loss, grad_norm = v3.train_epoch_clip1(
            model, train_loader, optimizer, bce_criterion, huber_criterion, device, base.LAMBDA_RETURN
        )
        val_loss = base.evaluate_loss(model, val_loader, bce_criterion, huber_criterion, device, base.LAMBDA_RETURN)
        if epoch > warmup_epochs and np.isfinite(val_loss):
            scheduler.step(val_loss)
        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
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
    return val_prob_raw, test_prob_raw, best_val_loss, best_epoch


def main() -> None:
    device = base.choose_device()
    print(f"Using device: {device}", flush=True)
    print(
        f"[no-history-5seed] SEEDS={SEEDS} | EPOCHS={base.EPOCHS} | PATIENCE={base.PATIENCE} | "
        f"LR={v3.LR} | checkpoint_metric=val_loss | selection=fixed p>={SELECTION_THRESHOLD} | "
        f"model=NoHistoryClassifier (current-token-only, no attention)",
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
    seed_diag_rows = []

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

        pos_rate = float(train["target_profitable"].mean())
        pos_weight = torch.tensor([(1.0 - pos_rate) / max(pos_rate, 1e-6)], dtype=torch.float32, device=device)

        val_probs_by_seed, test_probs_by_seed = [], []
        for seed in SEEDS:
            val_p, test_p, best_loss, best_epoch = train_one_seed(
                seed, train_arrays, val_arrays, test_arrays, feature_cols, pos_weight, device
            )
            val_probs_by_seed.append(val_p)
            test_probs_by_seed.append(test_p)
            seed_diag_rows.append({
                "fold": fold_name, "test_year": test_year, "seed": seed,
                "best_epoch": best_epoch, "best_val_loss": best_loss,
            })
            print(f"  seed={seed} best_epoch={best_epoch} best_val_loss={best_loss:.4f}", flush=True)

        val_prob_avg = np.mean(val_probs_by_seed, axis=0)
        test_prob_avg = np.mean(test_probs_by_seed, axis=0)
        seed_losses = [r["best_val_loss"] for r in seed_diag_rows if r["fold"] == fold_name]
        print(f"  ensemble {len(SEEDS)} seeds: mean best_val_loss={np.mean(seed_losses):.4f} std={np.std(seed_losses):.4f}", flush=True)

        iso = base.fit_isotonic_calibration(val_prob_avg, val_arrays.y)
        val_prob_cal = base.apply_calibration(iso, val_prob_avg)
        test_prob_cal = base.apply_calibration(iso, test_prob_avg)

        val_tokens = val_arrays.meta.copy()
        test_tokens = test_arrays.meta.copy()
        val_tokens["predicted_prob"] = val_prob_cal
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
        print(
            f"  test_pnl=${test_metrics['norm_total_pnl']:,.0f} test_n={test_metrics['n_selected']:,} "
            f"sharpe={test_metrics['daily_sharpe']:.2f} sortino={test_metrics['daily_sortino']:.2f} "
            f"win_rate={test_metrics['win_rate']:.3f}",
            flush=True,
        )

    results = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    selected = predictions[predictions["selected"]].copy()
    seed_diagnostics = pd.DataFrame(seed_diag_rows)

    yearly = results[[
        "test_year", "n_seeds", "mean_best_val_loss", "std_best_val_loss", "selection_threshold",
        "test_n_selected", "test_norm_total_pnl", "test_win_rate", "test_daily_sharpe",
        "test_daily_sortino", "test_max_drawdown", "test_t1_selected", "test_k2_selected", "test_auc",
    ]].copy()
    yearly.to_csv(TABLE_DIR / "mv_no_history_5seed_yearly_diagnostics.csv", index=False)
    seed_diagnostics.to_csv(TABLE_DIR / "mv_no_history_5seed_seed_diagnostics.csv", index=False)
    predictions.to_parquet(DATA_DIR / "no_history_5seed_test_predictions.parquet", index=False)
    selected.to_parquet(DATA_DIR / "no_history_5seed_selected_trades.parquet", index=False)

    total_pnl = float(selected["target_norm_pnl"].sum()) if len(selected) else 0.0
    daily = selected.groupby("signal_date")["target_norm_pnl"].sum() if len(selected) else pd.Series(dtype=float)
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 1 and daily.std() > 0 else float("nan")
    pos_years = int((results["test_norm_total_pnl"] > 0).sum())
    summary = {
        "model": "no_history_5seed", "seeds": SEEDS, "total_pnl": total_pnl,
        "sharpe": sharpe, "n_selected": int(len(selected)), "pos_years": pos_years,
        "per_year_pnl": results.set_index("test_year")["test_norm_total_pnl"].round(0).to_dict(),
    }
    with open(TABLE_DIR / "mv_no_history_5seed_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[no-history-5seed] TOTAL pnl=${total_pnl:,.0f} sharpe={sharpe:.2f} pos_years={pos_years}/6 n={len(selected):,}", flush=True)


if __name__ == "__main__":
    main()
