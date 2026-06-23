#!/usr/bin/env python3
"""Run leakage-safe rolling LightGBM 5-action evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.pipeline import Pipeline

from tp2rr2.config import CONFIG


ACTION_ID_TO_NAME = {0: "skip", 1: "T1", 2: "T2", 3: "K1", 4: "K2"}
STRATEGIES = ["T1", "T2", "K1", "K2"]
TARGETS = {
    "net_pnl": {
        "action_col": "label_best_action",
        "action_id_col": "label_best_action_id",
        "value_col": "label_best_net_pnl",
    },
    "return": {
        "action_col": "label_best_return_action",
        "action_id_col": "label_best_return_action_id",
        "value_col": "label_best_return_value",
    },
    "capped_pnl": {
        "action_col": "label_best_capped_pnl_action",
        "action_id_col": "label_best_capped_pnl_action_id",
        "value_col": "label_best_capped_pnl_value",
    },
}

QUALITY_GATES = [
    "base",
    "volume_and_delta",
    "strict_quality_combo",
]


PRIMARY_FOLDS = [
    {"fold": "2017_2019__val2020__test2021", "train": [2017, 2018, 2019], "val": [2020], "test": [2021]},
    {"fold": "2018_2020__val2021__test2022", "train": [2018, 2019, 2020], "val": [2021], "test": [2022]},
    {"fold": "2019_2021__val2022__test2023", "train": [2019, 2020, 2021], "val": [2022], "test": [2023]},
    {"fold": "2020_2022__val2023__test2024", "train": [2020, 2021, 2022], "val": [2023], "test": [2024]},
    {"fold": "2021_2023__val2024__test2025", "train": [2021, 2022, 2023], "val": [2024], "test": [2025]},
    {"fold": "2022_2024__val2025__test2026", "train": [2022, 2023, 2024], "val": [2025], "test": [2026]},
]


def _load_feature_columns(root: Path, summary_path: Path) -> list[str]:
    summary = json.loads(summary_path.read_text())
    return summary["feature_columns"]


def _add_quality_gate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    leg_volume = [f"feature_{leg}_volume" for leg in "ABCD"]
    leg_oi = [f"feature_{leg}_open_interest" for leg in "ABCD"]
    leg_delta = [f"feature_{leg}_delta" for leg in "ABCD"]
    leg_spread = [f"feature_{leg}_spread_pct" for leg in "ABCD"]

    out["q_all_volume_positive"] = out[leg_volume].fillna(0).gt(0).all(axis=1)
    out["q_all_open_interest_positive"] = out[leg_oi].fillna(0).gt(0).all(axis=1)
    out["q_delta_01_99"] = out[leg_delta].abs().ge(0.01).all(axis=1) & out[leg_delta].abs().le(0.99).all(axis=1)
    out["q_max_spread_le_25pct"] = out[leg_spread].le(0.25).all(axis=1)
    out["q_K1_round_distance"] = (out["feature_K1_forward_rounded"] - out["feature_K1_forward_adjusted"]).abs()
    out["q_K2_round_distance"] = (out["feature_K2_forward_rounded"] - out["feature_K2_forward_adjusted"]).abs()
    out["q_max_round_distance"] = out[["q_K1_round_distance", "q_K2_round_distance"]].max(axis=1)
    out["q_round_distance_le_2_5"] = out["q_max_round_distance"].le(2.5)
    out["q_volume_and_delta"] = out["q_all_volume_positive"] & out["q_delta_01_99"]
    out["q_strict_quality"] = (
        out["q_all_volume_positive"]
        & out["q_all_open_interest_positive"]
        & out["q_delta_01_99"]
        & out["q_max_spread_le_25pct"]
        & out["q_round_distance_le_2_5"]
    )
    return out


def _apply_quality_gate(df: pd.DataFrame, gate: str) -> pd.DataFrame:
    if gate == "base":
        return df.copy()
    gated = _add_quality_gate_columns(df)
    if gate == "volume_and_delta":
        return gated[gated["q_volume_and_delta"]].copy()
    if gate == "strict_quality_combo":
        return gated[gated["q_strict_quality"]].copy()
    raise ValueError(f"Unknown quality gate: {gate}")


def _selected_pnl(df: pd.DataFrame, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pnl = np.zeros(len(df), dtype=float)
    valid = np.zeros(len(df), dtype=bool)
    for strategy_id, strategy in enumerate(STRATEGIES, start=1):
        mask = action == strategy_id
        pnl_col = f"label_net_pnl_{strategy}"
        valid_col = f"label_is_valid_{strategy}"
        valid_strategy = df[valid_col].fillna(False).to_numpy(dtype=bool)
        valid[mask] = valid_strategy[mask]
        values = df[pnl_col].fillna(0.0).to_numpy(dtype=float)
        pnl[mask & valid_strategy] = values[mask & valid_strategy]
    return pnl, valid


def _evaluate_actions(df: pd.DataFrame, action: np.ndarray, fold: str, split: str, model_name: str) -> dict:
    pnl, valid = _selected_pnl(df, action)
    selected = action != 0
    executed = selected & valid
    profitable = executed & (pnl > 0)
    return {
        "fold": fold,
        "split": split,
        "model": model_name,
        "rows": int(len(df)),
        "selected": int(selected.sum()),
        "executed": int(executed.sum()),
        "invalid_selected": int((selected & ~valid).sum()),
        "selection_rate": float(selected.mean()) if len(df) else np.nan,
        "execution_rate_selected": float(executed.sum() / selected.sum()) if selected.sum() else np.nan,
        "hit_rate_executed": float(profitable.sum() / executed.sum()) if executed.sum() else np.nan,
        "net_pnl": float(pnl.sum()),
        "mean_pnl_selected": float(pnl[selected].mean()) if selected.sum() else np.nan,
        "median_pnl_selected": float(np.median(pnl[selected])) if selected.sum() else np.nan,
        "mean_pnl_executed": float(pnl[executed].mean()) if executed.sum() else np.nan,
        "median_pnl_executed": float(np.median(pnl[executed])) if executed.sum() else np.nan,
    }


def _prediction_frame(
    df: pd.DataFrame,
    fold: str,
    split: str,
    proba: np.ndarray,
    threshold: float,
    target: str,
) -> pd.DataFrame:
    target_cols = TARGETS[target]
    non_skip = proba[:, 1:]
    candidate_strategy_offset = np.argmax(non_skip, axis=1)
    candidate_action = candidate_strategy_offset + 1
    candidate_prob = non_skip[np.arange(len(df)), candidate_strategy_offset]
    skip_prob = proba[:, 0]
    action = np.where((candidate_prob >= threshold) & (candidate_prob > skip_prob), candidate_action, 0)
    pnl, valid = _selected_pnl(df, action)
    out = df[
        [
            "episode_id",
            "signal_date",
            target_cols["action_col"],
            target_cols["action_id_col"],
            target_cols["value_col"],
            *[f"label_net_pnl_{s}" for s in STRATEGIES],
            *[f"label_realized_return_{s}" for s in STRATEGIES],
            *[f"label_capped_net_pnl_{s}" for s in STRATEGIES],
            *[f"label_is_valid_{s}" for s in STRATEGIES],
        ]
    ].copy()
    out = out.rename(
        columns={
            target_cols["action_col"]: "label_target_action",
            target_cols["action_id_col"]: "label_target_action_id",
            target_cols["value_col"]: "label_target_value",
        }
    )
    out["fold"] = fold
    out["split"] = split
    out["target"] = target
    out["threshold"] = threshold
    out["predicted_action_id"] = action
    out["predicted_action"] = [ACTION_ID_TO_NAME[int(x)] for x in action]
    out["predicted_trade_probability"] = candidate_prob
    out["predicted_skip_probability"] = skip_prob
    for action_id, name in ACTION_ID_TO_NAME.items():
        out[f"prob_{name}"] = proba[:, action_id]
    out["selected"] = action != 0
    out["executed"] = out["selected"] & valid
    out["realized_net_pnl"] = pnl
    return out


def _tune_threshold(df: pd.DataFrame, proba: np.ndarray, min_trades: int, target: str) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.81, 0.025), 3):
        pred = _prediction_frame(df, "validation", "validation", proba, float(threshold), target)
        metrics = _evaluate_actions(df, pred["predicted_action_id"].to_numpy(), "validation", "validation", "threshold")
        metrics["threshold"] = float(threshold)
        metrics["target"] = target
        rows.append(metrics)
    grid = pd.DataFrame(rows)
    eligible = grid[grid["selected"].ge(min_trades)].copy()
    if eligible.empty:
        eligible = grid.copy()
    best = eligible.sort_values(["net_pnl", "hit_rate_executed", "selected"], ascending=[False, False, False]).iloc[0]
    return float(best["threshold"]), grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling LightGBM 5-action evaluation.")
    parser.add_argument("--dataset", default="data/ml/archive2_ml_episode_dataset_2013_2026.parquet")
    parser.add_argument("--dataset-summary", default="data/ml/archive2_ml_episode_dataset_summary_2013_2026.json")
    parser.add_argument("--predictions-output", default="data/ml/archive2_ml_lightgbm_rolling_predictions.parquet")
    parser.add_argument("--summary-output", default="data/ml/archive2_ml_lightgbm_rolling_summary.csv")
    parser.add_argument("--threshold-grid-output", default="data/ml/archive2_ml_lightgbm_threshold_grid.csv")
    parser.add_argument("--target", choices=sorted(TARGETS), default="net_pnl")
    parser.add_argument("--quality-gate", choices=QUALITY_GATES, default="base")
    parser.add_argument("--min-validation-trades", type=int, default=100)
    args = parser.parse_args()

    root = CONFIG.paths.root
    feature_cols = _load_feature_columns(root, root / args.dataset_summary)
    df = pd.read_parquet(root / args.dataset)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["signal_year"] = df["signal_date"].dt.year
    original_rows = len(df)
    df = _apply_quality_gate(df, args.quality_gate)
    target_cols = TARGETS[args.target]

    predictions = []
    summaries = []
    grids = []

    for fold in PRIMARY_FOLDS:
        train = df[df["signal_year"].isin(fold["train"])].copy()
        val = df[df["signal_year"].isin(fold["val"])].copy()
        test = df[df["signal_year"].isin(fold["test"])].copy()
        if train.empty or val.empty or test.empty:
            continue

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    lgb.LGBMClassifier(
                        objective="multiclass",
                        num_class=5,
                        n_estimators=450,
                        learning_rate=0.035,
                        num_leaves=31,
                        max_depth=-1,
                        min_child_samples=80,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
        X_train = train[feature_cols]
        y_train = train[target_cols["action_id_col"]].astype(int)
        model.fit(X_train, y_train)

        val_proba = model.predict_proba(val[feature_cols])
        test_proba = model.predict_proba(test[feature_cols])
        threshold, grid = _tune_threshold(val, val_proba, args.min_validation_trades, args.target)
        grid["fold"] = fold["fold"]
        grids.append(grid)

        for split_name, split_df, proba in [("validation", val, val_proba), ("test", test, test_proba)]:
            pred = _prediction_frame(split_df, fold["fold"], split_name, proba, threshold, args.target)
            pred["quality_gate"] = args.quality_gate
            predictions.append(pred)
            summaries.append(
                _evaluate_actions(
                    split_df,
                    pred["predicted_action_id"].to_numpy(dtype=int),
                    fold["fold"],
                    split_name,
                    f"lightgbm_5_action_{args.target}",
                )
                | {
                    "target": args.target,
                    "quality_gate": args.quality_gate,
                    "threshold": threshold,
                    "train_years": ",".join(map(str, fold["train"])),
                    "validation_years": ",".join(map(str, fold["val"])),
                    "test_years": ",".join(map(str, fold["test"])),
                    "full_dataset_rows_before_gate": int(original_rows),
                    "dataset_rows_after_gate": int(len(df)),
                    "action_label_accuracy": float(
                        accuracy_score(split_df[target_cols["action_id_col"]], pred["predicted_action_id"])
                    ),
                    "balanced_action_label_accuracy": float(
                        balanced_accuracy_score(split_df[target_cols["action_id_col"]], pred["predicted_action_id"])
                    ),
                    "multiclass_log_loss": float(
                        log_loss(split_df[target_cols["action_id_col"]], proba, labels=[0, 1, 2, 3, 4])
                    ),
                }
            )

    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    grid_df = pd.concat(grids, ignore_index=True) if grids else pd.DataFrame()

    (root / args.predictions_output).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(root / args.predictions_output, index=False)
    summary_df.to_csv(root / args.summary_output, index=False)
    grid_df.to_csv(root / args.threshold_grid_output, index=False)

    print("Wrote", root / args.predictions_output)
    print("Wrote", root / args.summary_output)
    print(summary_df[summary_df["split"].eq("test")].to_string(index=False))


if __name__ == "__main__":
    main()
