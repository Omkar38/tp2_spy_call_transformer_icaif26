"""
ml/dataset.py — PyTorch Dataset and chronological data splitting.

Key design decisions (all defensible for the Zoom call):

  1. Chronological split, NOT random shuffle.
     Random shuffle leaks future information into training — a violation
     at hour H could end up in training while one at H-1 is in test.
     Financial data must be split by time.

  2. 24-hour gap between train/val and val/test.
     Violations within 24h of each other may share market regime features
     (same IV surface, same spot movement). The gap prevents information
     leakage through correlated features.

  3. No normalization baked into the dataset.
     We compute train-set statistics and pass them in.
     This prevents look-ahead bias from using test-set statistics.

Split: 70% train | 15% val | 15% test (chronological)
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger("ml.dataset")


class TP2Dataset(Dataset):
    """
    PyTorch Dataset for TP2 violation classification.

    Each sample yields:
        temporal:     (seq_len, D_TEMPORAL) float tensor
        surface:      (max_instruments, D_SURFACE) float tensor
        surface_mask: (max_instruments,) float tensor — 1 for real, 0 for pad
        label:        scalar long tensor — 1=fast reversion, 0=slow/never
    """

    def __init__(
        self,
        temporal: np.ndarray,
        surface: np.ndarray,
        surface_mask: np.ndarray,
        labels: np.ndarray,
        normalize_stats: dict = None,
    ):
        """
        Args:
            temporal:        (N, seq_len, D_TEMPORAL)
            surface:         (N, max_instruments, D_SURFACE)
            surface_mask:    (N, max_instruments)
            labels:          (N,)
            normalize_stats: {key: (mean, std)} — computed from TRAINING set only
        """
        self.temporal     = temporal.copy()
        self.surface      = surface.copy()
        self.surface_mask = surface_mask.copy()
        self.labels       = labels.copy()

        # Apply normalization if provided
        if normalize_stats:
            self._normalize(normalize_stats)

    def _normalize(self, stats: dict):
        """
        Z-score normalization using training set statistics.
        Only normalizes continuous features, skips binary/cyclical ones.
        """
        if "temporal" in stats:
            mean, std = stats["temporal"]
            # Avoid division by zero — features with zero std are constant
            std = np.where(std < 1e-8, 1.0, std)
            self.temporal = (self.temporal - mean) / std

        if "surface" in stats:
            mean, std = stats["surface"]
            std = np.where(std < 1e-8, 1.0, std)
            # Don't normalize the is_violation_leg flag (last feature)
            self.surface[:, :, :-1] = (
                (self.surface[:, :, :-1] - mean[:-1]) / std[:-1]
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.temporal[idx]).float(),
            torch.from_numpy(self.surface[idx]).float(),
            torch.from_numpy(self.surface_mask[idx]).float(),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ── Chronological splitting ──────────────────────────────────

def compute_normalize_stats(temporal: np.ndarray, surface: np.ndarray) -> dict:
    """
    Compute mean/std from TRAINING data only.
    These stats are then applied to val/test sets — no look-ahead bias.

    Args:
        temporal: (N_train, seq_len, D_TEMPORAL)
        surface:  (N_train, max_instruments, D_SURFACE)

    Returns:
        dict with 'temporal' and 'surface' keys, each → (mean, std)
    """
    # Temporal: mean/std across samples and timesteps, per feature
    # Shape: (D_TEMPORAL,)
    t_flat = temporal.reshape(-1, temporal.shape[-1])
    t_mean = t_flat.mean(axis=0)
    t_std  = t_flat.std(axis=0)

    # Surface: mean/std across samples and instruments, per feature
    # Only use non-padded positions (where at least one feature != 0)
    s_flat = surface.reshape(-1, surface.shape[-1])
    # Mask out pure-zero rows (padding)
    nonzero = np.any(s_flat != 0, axis=1)
    if nonzero.sum() > 0:
        s_active = s_flat[nonzero]
        s_mean = s_active.mean(axis=0)
        s_std  = s_active.std(axis=0)
    else:
        s_mean = np.zeros(surface.shape[-1])
        s_std  = np.ones(surface.shape[-1])

    return {
        "temporal": (t_mean.astype(np.float32), t_std.astype(np.float32)),
        "surface":  (s_mean.astype(np.float32), s_std.astype(np.float32)),
    }


def chronological_split(
    dataset_dict: dict,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    gap_hours: int = 24,
) -> tuple:
    """
    Split dataset chronologically with a gap between splits.

    The gap prevents information leakage from correlated market conditions.
    Violations within gap_hours of a split boundary are DROPPED, not
    assigned to either side.

    Args:
        dataset_dict: output of features.build_dataset()
        train_frac:   fraction for training
        val_frac:     fraction for validation (remainder = test)
        gap_hours:    hours of gap between train/val and val/test

    Returns:
        (train_ds, val_ds, test_ds, norm_stats)
    """
    metadata = dataset_dict["metadata"]
    N = len(metadata)

    # Parse detection times
    hours = [datetime.fromisoformat(m["snapshot_hour"]) for m in metadata]

    # Data is already sorted chronologically (from features.py ORDER BY)
    # Find split indices
    train_end_idx = int(N * train_frac)
    val_end_idx   = int(N * (train_frac + val_frac))

    # Find gap boundaries
    gap_delta = timedelta(hours=gap_hours)

    train_cutoff = hours[train_end_idx] if train_end_idx < N else hours[-1]
    val_start    = train_cutoff + gap_delta

    val_cutoff = hours[val_end_idx] if val_end_idx < N else hours[-1]
    test_start = val_cutoff + gap_delta

    # Assign indices to splits, dropping those in the gap
    train_idx = [i for i in range(N) if hours[i] < train_cutoff]
    val_idx   = [i for i in range(N) if val_start <= hours[i] < val_cutoff]
    test_idx  = [i for i in range(N) if hours[i] >= test_start]

    dropped = N - len(train_idx) - len(val_idx) - len(test_idx)

    log.info(
        f"Chronological split: "
        f"train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)} | "
        f"gap_dropped={dropped}"
    )

    if len(train_idx) == 0:
        raise ValueError("Training set is empty — not enough data")

    # Extract arrays for each split
    def _slice(indices):
        idx = np.array(indices)
        return (
            dataset_dict["temporal"][idx],
            dataset_dict["surface"][idx],
            dataset_dict["surface_mask"][idx],
            dataset_dict["labels"][idx],
        )

    train_arrays = _slice(train_idx)
    val_arrays   = _slice(val_idx)
    test_arrays  = _slice(test_idx)

    # Compute normalization from training set ONLY
    norm_stats = compute_normalize_stats(train_arrays[0], train_arrays[1])

    # Build datasets
    train_ds = TP2Dataset(*train_arrays, normalize_stats=norm_stats)
    val_ds   = TP2Dataset(*val_arrays,   normalize_stats=norm_stats) if val_idx   else None
    test_ds  = TP2Dataset(*test_arrays,  normalize_stats=norm_stats) if test_idx  else None

    # Log class distribution per split
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        if ds is None:
            continue
        pos = ds.labels.sum()
        total = len(ds)
        log.info(f"  {name}: {total} samples, {pos} positive ({100*pos/max(total,1):.1f}%)")

    return train_ds, val_ds, test_ds, norm_stats


def make_dataloaders(
    train_ds: TP2Dataset,
    val_ds: TP2Dataset = None,
    test_ds: TP2Dataset = None,
    batch_size: int = 32,
    num_workers: int = 0,
    balanced: bool = True,
    pos_neg_ratio: float = 0.3,
) -> dict:
    """
    Wrap datasets in DataLoaders.

    balanced=True uses WeightedRandomSampler so each training batch contains
    ~pos_neg_ratio fraction of positives (default 30%). Without this, with
    1.33% positives the model sees 1 fast violation per 75 slow ones per batch
    and focal loss cannot compensate — it still converges to predict everything
    negative (or everything positive when alpha is high).

    pos_neg_ratio=0.3 means ~30% of each batch is positive. This is aggressive
    but necessary given so few positives (~3K in training). Val/test are never
    resampled — they stay in chronological order with true class proportions.

    num_workers=0 is safest for Colab + SQLite.
    """
    import torch
    from torch.utils.data import WeightedRandomSampler

    pin_memory = torch.cuda.is_available()  # only for CUDA, not MPS

    if balanced:
        labels = train_ds.labels  # numpy array
        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        # Weight each sample so positives are sampled at pos_neg_ratio
        # w_pos / w_neg = pos_neg_ratio / (1 - pos_neg_ratio) * (n_neg / n_pos)
        w_pos = (pos_neg_ratio / max(n_pos, 1))
        w_neg = ((1 - pos_neg_ratio) / max(n_neg, 1))
        weights = np.where(labels == 1, w_pos, w_neg).astype(np.float64)
        weights = weights / weights.sum()  # normalise to sum=1
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=len(train_ds),
            replacement=True,
        )
        log.info(
            f"Balanced sampler: n_pos={n_pos}, n_neg={n_neg}, "
            f"target_pos_rate={pos_neg_ratio:.0%} per batch"
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        )

    loaders = {"train": train_loader}

    if val_ds and len(val_ds) > 0:
        loaders["val"] = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
    if test_ds and len(test_ds) > 0:
        loaders["test"] = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
    return loaders
