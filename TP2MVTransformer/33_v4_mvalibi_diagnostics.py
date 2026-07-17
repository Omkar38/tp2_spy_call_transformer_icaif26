"""
MV-ALiBi distance/attention diagnostics for the v4 5-seed ensemble --
the remaining gap flagged in final_comprehensive_report.pdf ("Not yet
done for v4/v4b -- would need per-seed export changes").

v4's train_one_model() (03_train_mv_alibi_transformer_v4_multiseed.py)
returns only predictions, not the trained model object, so it cannot be
reused as-is to extract each seed's learned attention parameters. This
script inlines the same training loop (imported pieces: base.MVAlibiClassifier,
v3.train_epoch_clip1, base.evaluate_loss, base.predict, base.attention_parameter_rows
-- all via importlib, none modified) so each seed's model is available
right after training, purely to extract its learned MV-ALiBi slopes
before discarding the model. No predictions, calibration, selection, or
trading results are recomputed here -- those already exist in
mv_alibi_transformer_v4final_*. Raw distance scales (base.distance_summary)
are model-independent (computed from the data, not the model) and were
already produced for v3 in 18_mvalibi_diagnostics_v3.py -- v3 and v4 share
the exact same data/array construction, so those numbers are identical
and not recomputed here; only the per-seed learned attention slopes are
new.

Writes only mv_alibi_v4_seed_attention_parameters.csv and a comparison
figure; no v3/v4/v4final artifact is modified.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TP2MVTransformer"
DATA_DIR = OUT / "data"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MPL_DIR = OUT / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

V4_PATH = OUT / "03_train_mv_alibi_transformer_v4_multiseed.py"


def load_v4_module():
    spec = importlib.util.spec_from_file_location("tp2_v4_for_diag", V4_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v4 = load_v4_module()
v3 = v4.v3
base = v4.base
SEEDS = v4.SEEDS
DISTANCE_ORDER = ["time", "surface", "short_d", "liquidity", "lifecycle"]


def train_one_model_for_diagnostics(seed, train_arrays, val_arrays, test_arrays, feature_cols, device):
    """Same training loop as v4.train_one_model, inlined to expose the
    trained model object for attention-parameter extraction."""
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    warmup_epochs = max(1, round(base.EPOCHS * v3.WARMUP_FRACTION))
    best_val_loss = float("inf")
    best_state = None
    stale = 0
    for epoch in range(1, base.EPOCHS + 1):
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
    return model, best_val_loss


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

    attention_rows = []
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

        for seed in SEEDS:
            model, best_val_loss = train_one_model_for_diagnostics(seed, train_arrays, val_arrays, test_arrays, feature_cols, device)
            rows = base.attention_parameter_rows(model, fold_name, test_year)
            for r in rows:
                r["seed"] = seed
            attention_rows.extend(rows)
            slope_str = ", ".join(f"{d}={next(r[f'slope_{d}'] for r in rows):.3f}" for d in DISTANCE_ORDER)
            print(f"  seed={seed} best_val_loss={best_val_loss:.4f}  slopes: {slope_str}", flush=True)
            del model

    attn_df = pd.DataFrame(attention_rows)
    attn_df.to_csv(TABLE_DIR / "mv_alibi_v4_seed_attention_parameters.csv", index=False)
    print(f"\nWrote {TABLE_DIR / 'mv_alibi_v4_seed_attention_parameters.csv'}")

    slope_cols = [f"slope_{d}" for d in DISTANCE_ORDER]
    summary = attn_df.groupby("seed")[slope_cols].mean()
    print("\nMean learned slope per seed:")
    print(summary.to_string())

    overall_mean = attn_df[slope_cols].mean()
    overall_std = attn_df[slope_cols].std()
    print("\nAcross all 5 seeds x 6 folds:")
    for d in DISTANCE_ORDER:
        print(f"  {d}: mean={overall_mean[f'slope_{d}']:.4f} std={overall_std[f'slope_{d}']:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(DISTANCE_ORDER))
    width = 0.15
    for i, seed in enumerate(SEEDS):
        seed_means = attn_df[attn_df["seed"] == seed][slope_cols].mean()
        ax.bar(x + (i - 2) * width, [seed_means[f"slope_{d}"] for d in DISTANCE_ORDER], width, label=f"seed {seed}")
    ax.set_xticks(x)
    ax.set_xticklabels(DISTANCE_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Mean learned slope")
    ax.set_title("v4 ensemble: learned MV-ALiBi slopes, per seed (6-fold mean)")
    ax.legend(frameon=False, ncols=5, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mv_alibi_v4_seed_attention_slopes.pdf")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'mv_alibi_v4_seed_attention_slopes.pdf'}")


if __name__ == "__main__":
    main()
