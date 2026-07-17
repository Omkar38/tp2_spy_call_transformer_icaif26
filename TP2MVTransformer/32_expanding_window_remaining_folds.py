"""
Fills in the 3 walk-forward folds (test years 2021, 2023, 2025) not
covered by the earlier training-window sensitivity pass (which sampled
2022/2024/2026), so the rolling-vs-expanding comparison the professor
asked about ("see what happens if you use expanding... you use all
previous years for training, not just the previous year") covers all 6
folds, not a 3-fold sample.

Imports run_one_fold() from 20_training_window_sensitivity.py via
importlib (does not modify it) -- same v3 protocol, same causal p>=0.5
selection, same "all_history" (expanding window: every year from 2014 up
to val_year-1) window definition already used and integrity-checked
there. The "3yr" (current rolling default) comparison point for these
same 3 folds already exists in mv_alibi_transformer_v3final_yearly_diagnostics.csv
(v3-final, already shipped, all 6 folds) -- not re-run here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TP2MVTransformer"
TABLE_DIR = OUT / "tables"
DATA_DIR = OUT / "data"

WINDOW_SCRIPT = OUT / "20_training_window_sensitivity.py"


def load_window_module():
    spec = importlib.util.spec_from_file_location("tp2_window_for_remaining", WINDOW_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REMAINING_FOLDS = [
    (2021, 2020),
    (2023, 2022),
    (2025, 2024),
]


def main() -> None:
    wm = load_window_module()
    base = wm.base
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
    for test_year, val_year in REMAINING_FOLDS:
        train_years = wm.WINDOW_LENGTHS["all_history"](val_year)
        print(f"\n{'='*60}\ntest_year={test_year} window=all_history train_years={train_years}", flush=True)
        outcome = wm.run_one_fold(train_years, val_year, test_year, data, feature_cols, block_indices, device)
        if outcome is None:
            print("  SKIPPED (empty split)", flush=True)
            continue
        result, _ = outcome
        result["test_year"] = test_year
        result["val_year"] = val_year
        result["window"] = "all_history"
        rows.append(result)
        print(
            f"  n_train_years={result['n_train_years']} n_train_rows={result['n_train_rows']:,} "
            f"epochs={result['epochs_run']} best_val_loss={result['best_val_loss']:.4f} "
            f"trades={result['n_selected']:,} pnl=${result['norm_total_pnl']:,.0f} "
            f"sharpe={result['daily_sharpe']:.2f}",
            flush=True,
        )

    new_df = pd.DataFrame(rows)
    existing = pd.read_csv(TABLE_DIR / "training_window_sensitivity.csv")
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(TABLE_DIR / "training_window_sensitivity.csv", index=False)
    print(f"\nWrote {TABLE_DIR / 'training_window_sensitivity.csv'} ({len(combined)} rows total)")


if __name__ == "__main__":
    main()
