"""
ml/train.py — Training loop for DS-CAT.

Key decisions for the professor:

  1. Focal Loss (Lin et al., 2017)
     Standard cross-entropy treats all samples equally. With 0.14% positive
     rate, the model learns to predict "slow/never" for everything and gets
     99.86% accuracy — useless. Focal loss down-weights easy negatives
     and focuses on the hard positives.

     FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

     γ = 2-3: focuses more on hard examples
     α = inverse class frequency: upweights the rare positive class

  2. Early stopping on val F1 (not loss, not accuracy)
     Accuracy is meaningless with class imbalance. Loss can decrease
     while F1 plateaus. We care about ranking the POSITIVE class
     correctly — F1 is the right metric.

  3. Gradient clipping
     Transformer gradients can spike, especially with small datasets.
     max_norm=1.0 is standard.

  4. Learning rate warmup + cosine decay
     Warmup prevents early instability. Cosine decay gives the model
     time to converge without abrupt LR drops.
"""

import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger("ml.train")


# ── Focal Loss ────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced binary classification.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)

    When γ > 0, down-weights easy examples and focuses on hard ones.
    When α is set, compensates for class imbalance.

    γ = 0, α = None → equivalent to standard cross-entropy
    γ = 2, α = auto → good starting point for extreme imbalance

    Math:
        p_t = p if y=1 else (1-p)
        FL  = -α_t × (1-p_t)^γ × log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, 2) raw model output
            targets: (B,) long tensor with class indices {0, 1}
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # probability of the true class

        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            # alpha for positive class, (1-alpha) for negative
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_focal_alpha(labels: np.ndarray) -> float:
    """
    Compute α (positive class weight) from training labels.
    α = n_negative / n_total — gives more weight to rare positives.
    """
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    n_total = n_pos + n_neg
    if n_total == 0 or n_pos == 0:
        return 0.5
    alpha = n_neg / n_total  # high alpha → upweight positive class
    return float(np.clip(alpha, 0.1, 0.99))


# ── Evaluation Metrics ────────────────────────────────────────

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Compute classification metrics relevant for rare-event detection.

    Returns dict with: accuracy, precision, recall, f1, auc_proxy
    """
    preds = logits.argmax(dim=1)
    probs = F.softmax(logits, dim=1)[:, 1]  # P(fast reversion)

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()

    total    = tp + fp + fn + tn
    accuracy = (tp + tn) / max(total, 1)

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_total":  total,
        "n_pos":    tp + fn,
        "avg_prob":  round(probs.mean().item(), 4),
    }


# ── Training Loop ─────────────────────────────────────────────

def train(
    model: nn.Module,
    loaders: dict,
    device: str = "cpu",
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    gamma: float = 2.0,
    alpha: float = None,
    patience: int = 10,
    save_dir: str = None,
    grad_clip: float = 1.0,
) -> dict:
    """
    Full training loop with:
      - Focal loss
      - AdamW optimizer + cosine annealing
      - Gradient clipping
      - Early stopping on validation F1
      - Checkpointing best model

    Args:
        model:        DS-CAT model
        loaders:      dict with 'train', optionally 'val', 'test'
        device:       'cuda' or 'cpu'
        epochs:       max training epochs
        lr:           peak learning rate
        weight_decay: L2 regularization
        gamma:        focal loss focusing parameter (2-3 for extreme imbalance)
        alpha:        focal loss class weight (None → auto-compute from data)
        patience:     early stopping patience (epochs without val F1 improvement)
        save_dir:     directory to save checkpoints and logs
        grad_clip:    max gradient norm

    Returns:
        dict with training history and best metrics
    """
    # ── Setup ──
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = Path("checkpoints")
        save_path.mkdir(exist_ok=True)

    # Auto-compute alpha from training data if not provided
    if alpha is None:
        train_labels = np.array([
            label.item() for batch in loaders["train"] for label in batch[3]
        ])
        alpha = compute_focal_alpha(train_labels)
        log.info(f"Auto-computed focal alpha={alpha:.4f}")

    criterion = FocalLoss(gamma=gamma, alpha=alpha)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=max(epochs // 3, 5), T_mult=2, eta_min=lr * 0.01
    )

    # ── Tracking ──
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "lr": []}
    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_state = None
    start_time = time.time()

    log.info(
        f"Training start: epochs={epochs}, lr={lr}, gamma={gamma}, "
        f"alpha={alpha:.4f}, patience={patience}, device={device}"
    )

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss_accum = 0.0
        train_batches = 0

        for temporal, surface, surface_mask, labels in loaders["train"]:
            temporal     = temporal.to(device)
            surface      = surface.to(device)
            surface_mask = surface_mask.to(device)
            labels       = labels.to(device)

            optimizer.zero_grad()
            logits = model(temporal, surface, surface_mask)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            train_loss_accum += loss.item()
            train_batches += 1

        train_loss = train_loss_accum / max(train_batches, 1)
        history["train_loss"].append(train_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        scheduler.step()

        # ── Validate ──
        val_metrics = None
        if "val" in loaders:
            val_loss, val_metrics = evaluate(model, loaders["val"], criterion, device)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_metrics["f1"])

            # Early stopping on F1
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, save_path / "best_model.pt")
            else:
                epochs_no_improve += 1

        # ── Log ──
        val_str = ""
        if val_metrics:
            val_str = (
                f" | val_loss={history['val_loss'][-1]:.4f} "
                f"F1={val_metrics['f1']:.4f} "
                f"P={val_metrics['precision']:.4f} "
                f"R={val_metrics['recall']:.4f}"
            )

        log.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f}{val_str} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        # Early stopping
        if epochs_no_improve >= patience:
            log.info(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    elapsed = time.time() - start_time
    log.info(f"Training complete: {elapsed:.1f}s, best val F1={best_val_f1:.4f}")

    # ── Restore best model ──
    if best_state:
        model.load_state_dict(best_state)
        log.info("Restored best model checkpoint")

    # ── Final test evaluation ──
    test_metrics = None
    if "test" in loaders:
        _, test_metrics = evaluate(model, loaders["test"], criterion, device)
        log.info(
            f"TEST: F1={test_metrics['f1']:.4f} "
            f"P={test_metrics['precision']:.4f} "
            f"R={test_metrics['recall']:.4f} "
            f"Acc={test_metrics['accuracy']:.4f}"
        )
        log.info(
            f"  TP={test_metrics['tp']} FP={test_metrics['fp']} "
            f"FN={test_metrics['fn']} TN={test_metrics['tn']}"
        )

    # ── Save training log ──
    results = {
        "history": history,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "elapsed_seconds": elapsed,
        "config": {
            "epochs": epoch,
            "lr": lr,
            "gamma": gamma,
            "alpha": alpha,
            "patience": patience,
            "grad_clip": grad_clip,
        },
    }
    with open(save_path / "train_log.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


# ── Evaluation ────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
) -> tuple:
    """
    Evaluate model on a data loader.
    Returns: (avg_loss, metrics_dict)
    """
    model.eval()
    all_logits  = []
    all_targets = []
    total_loss  = 0.0
    n_batches   = 0

    for temporal, surface, surface_mask, labels in loader:
        temporal     = temporal.to(device)
        surface      = surface.to(device)
        surface_mask = surface_mask.to(device)
        labels       = labels.to(device)

        logits = model(temporal, surface, surface_mask)
        loss = criterion(logits, labels)

        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())
        total_loss += loss.item()
        n_batches += 1

    all_logits  = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    avg_loss    = total_loss / max(n_batches, 1)
    metrics     = compute_metrics(all_logits, all_targets)

    return avg_loss, metrics


# ── Convenience: end-to-end pipeline ──────────────────────────

def run_pipeline(
    db_path: str = None,
    seq_len: int = 60,
    max_instruments: int = 400,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    gamma: float = 2.0,
    patience: int = 10,
    save_dir: str = "checkpoints",
) -> dict:
    """
    End-to-end: DB → features → dataset → model → train → evaluate.

    This is the function you call from Colab:

        from ml.train import run_pipeline
        results = run_pipeline(
            db_path="data/db/tp2.db",
            epochs=50,
            d_model=64,
        )

    Returns training results dict.
    """
    import sqlite3

    if db_path is None:
        import config as cfg
        db_path = str(cfg.DB_PATH)

    log.info(f"=== DS-CAT Training Pipeline ===")
    log.info(f"DB: {db_path}")

    # ── 1. Connect to DB ──
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # ── 2. Extract features ──
    from ml.features import build_dataset, D_TEMPORAL, D_SURFACE
    dataset_dict = build_dataset(conn, seq_len=seq_len, max_instruments=max_instruments)
    if dataset_dict is None:
        log.error("No data — cannot train")
        return None

    # ── 3. Chronological split ──
    from ml.dataset import chronological_split, make_dataloaders
    train_ds, val_ds, test_ds, norm_stats = chronological_split(dataset_dict)
    loaders = make_dataloaders(train_ds, val_ds, test_ds, batch_size=batch_size)

    # ── 4. Build model ──
    from ml.transformer import build_model
    model, device = build_model(
        d_temporal=D_TEMPORAL,
        d_surface=D_SURFACE,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )

    # ── 5. Train ──
    results = train(
        model, loaders, device=device,
        epochs=epochs, lr=lr, gamma=gamma,
        patience=patience, save_dir=save_dir,
    )

    conn.close()
    log.info("=== Pipeline complete ===")
    return results
