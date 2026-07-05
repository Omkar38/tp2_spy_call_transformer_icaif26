"""
Build the TP2-MV-ALiBi transformer token dataset and stable feature dictionary.

This script is the first implementation step from Literature/tp2_mv_alibi_transformer.tex:
- fixed strategy universe: T1 and K2
- corrected expiry labels only
- $1,000 normalized P&L target
- rolling LightGBM used only for feature discovery
- no booster predictions/leaves/scores are passed to downstream transformer data

Outputs are written under TP2MVTransformer/.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TP2MVTransformer"
DATA_DIR = OUT / "data"
TABLE_DIR = OUT / "tables"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_PATH = ROOT / "data/ml/archive2_ml_episode_dataset_2013_2026.parquet"
LABEL_PATH = ROOT / "data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet"

STRATEGIES = ["T1", "K2"]
NORM_EXPOSURE = 1_000.0
TOP_M = 30
MIN_STABILITY = 0.50
MAX_SELECTED_FEATURES = 48
PERMUTATION_SAMPLE = 6000

FOLDS = [
    {"fold": "2017_2019__val2020__test2021", "train": [2017, 2018, 2019], "val": [2020], "test": [2021]},
    {"fold": "2018_2020__val2021__test2022", "train": [2018, 2019, 2020], "val": [2021], "test": [2022]},
    {"fold": "2019_2021__val2022__test2023", "train": [2019, 2020, 2021], "val": [2022], "test": [2023]},
    {"fold": "2020_2022__val2023__test2024", "train": [2020, 2021, 2022], "val": [2023], "test": [2024]},
    {"fold": "2021_2023__val2024__test2025", "train": [2021, 2022, 2023], "val": [2024], "test": [2025]},
    {"fold": "2022_2024__val2025__test2026", "train": [2022, 2023, 2024], "val": [2025], "test": [2026]},
]

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 300,
    "learning_rate": 0.04,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
}


def infer_feature_block(name: str) -> str:
    lower = name.lower()
    if lower in {"strategy_id", "is_t1", "is_k2", "signal_month", "signal_dayofweek"}:
        return "strategy_calendar"
    if "severity" in lower or lower in {"feature_lhs", "feature_rhs", "feature_product_ratio", "start_normalized_severity"}:
        return "tp2_geometry"
    if lower.startswith("lc_"):
        return "lifecycle_pressure"
    if "spread" in lower or "volume" in lower or "open_interest" in lower or "zero_" in lower or "bidask" in lower:
        return "liquidity"
    if "_d_" in lower or "feature_d_" in lower:
        return "short_d_leg"
    if "moneyness" in lower or "dte" in lower or "strike" in lower or "term_gap" in lower or "distance" in lower:
        return "surface_location"
    if "iv" in lower or "delta" in lower or "gamma" in lower or "theta" in lower or "vega" in lower or "rho" in lower:
        return "option_risk"
    return "market"


def safe_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_strategy_features(rows: pd.DataFrame, strategy: str) -> pd.DataFrame:
    out = rows.copy()
    out["strategy"] = strategy
    out["strategy_id"] = 0 if strategy == "T1" else 1
    out["is_t1"] = 1 if strategy == "T1" else 0
    out["is_k2"] = 1 if strategy == "K2" else 0

    legs = ["A", "D"] if strategy == "T1" else ["B", "D"]
    spread_cols = [f"feature_{leg}_spread_pct" for leg in legs]
    volume_cols = [f"feature_{leg}_volume" for leg in legs]
    oi_cols = [f"feature_{leg}_open_interest" for leg in legs]
    mid_cols = [f"feature_{leg}_mid" for leg in legs]

    out["strategy_selected_max_spread_pct"] = out[spread_cols].max(axis=1)
    out["strategy_selected_mean_spread_pct"] = out[spread_cols].mean(axis=1)
    out["strategy_selected_min_volume"] = out[volume_cols].min(axis=1)
    out["strategy_selected_min_open_interest"] = out[oi_cols].min(axis=1)
    out["strategy_selected_mid_sum"] = out[mid_cols].sum(axis=1)
    out["strategy_short_d_spread_pct"] = out["feature_D_spread_pct"]
    out["strategy_short_d_volume"] = out["feature_D_volume"]
    out["strategy_short_d_open_interest"] = out["feature_D_open_interest"]
    return out


def build_token_dataset() -> tuple[pd.DataFrame, list[str]]:
    print("Loading episode features...")
    feat = pd.read_parquet(FEATURE_PATH)
    feat["signal_date"] = pd.to_datetime(feat["signal_date"])
    feat["signal_year"] = feat["signal_date"].dt.year

    print("Loading corrected expiry labels...")
    labels = pd.read_parquet(
        LABEL_PATH,
        columns=[
            "episode_id",
            "signal_date",
            "strategy",
            "exit_policy",
            "is_valid_backtest",
            "realized_return",
            "net_pnl",
            "entry_gross_exposure",
        ],
    )
    labels = labels[
        labels["exit_policy"].eq("exit_selected_leg_expiration_payoff")
        & labels["strategy"].isin(STRATEGIES)
    ].copy()
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    labels["target_profitable"] = (
        labels["is_valid_backtest"].fillna(False)
        & (pd.to_numeric(labels["realized_return"], errors="coerce").fillna(-1.0) > 0)
    ).astype(int)
    labels["target_return"] = pd.to_numeric(labels["realized_return"], errors="coerce").fillna(0.0)
    labels["target_norm_pnl"] = labels["target_return"] * NORM_EXPOSURE

    chunks = []
    for strategy in STRATEGIES:
        chunk = add_strategy_features(feat, strategy)
        chunk = chunk.merge(
            labels[
                [
                    "episode_id",
                    "strategy",
                    "is_valid_backtest",
                    "target_profitable",
                    "target_return",
                    "target_norm_pnl",
                ]
            ],
            on=["episode_id", "strategy"],
            how="inner",
        )
        chunks.append(chunk)
    data = pd.concat(chunks, ignore_index=True)
    data = data[data["signal_year"] >= 2014].copy()
    data = data.sort_values(["signal_date", "episode_id", "strategy"]).reset_index(drop=True)

    forbidden_prefixes = ("label_",)
    forbidden_exact = {
        "sample_role",
        "start_year",
        "start_month",
        "signal_year",
        "target_profitable",
        "target_return",
        "target_norm_pnl",
        "is_valid_backtest",
    }
    candidate_cols = []
    for col in data.columns:
        if col in {"episode_id", "signal_date", "strategy"}:
            continue
        if col in forbidden_exact or col.startswith(forbidden_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(data[col]):
            candidate_cols.append(col)

    # Keep point-in-time numeric features only.
    feature_cols = [
        c for c in candidate_cols
        if not c.startswith("expiry_")
        and "exit" not in c.lower()
        and "pnl" not in c.lower()
        and "return" not in c.lower()
        and "holding" not in c.lower()
        and "valid" not in c.lower()
    ]

    return data, feature_cols


def run_feature_discovery(data: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    print(f"Candidate point-in-time features: {len(feature_cols)}")

    for fold in FOLDS:
        fold_name = fold["fold"]
        train_mask = data["signal_year"].isin(fold["train"])
        val_mask = data["signal_year"].isin(fold["val"])

        for strategy in STRATEGIES:
            strat_mask = data["strategy"].eq(strategy)
            train = data[train_mask & strat_mask]
            val = data[val_mask & strat_mask]
            if train.empty or val.empty or train["target_profitable"].nunique() < 2:
                continue

            X_train = safe_numeric_frame(train, feature_cols)
            y_train = train["target_profitable"].astype(int)
            X_val = safe_numeric_frame(val, feature_cols)
            y_val = val["target_profitable"].astype(int)

            model = lgb.LGBMClassifier(**LGBM_PARAMS)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

            val_prob = model.predict_proba(X_val)[:, 1]
            try:
                auc = float(roc_auc_score(y_val, val_prob))
            except ValueError:
                auc = float("nan")

            gain = model.booster_.feature_importance(importance_type="gain")
            split = model.booster_.feature_importance(importance_type="split")
            gain_norm = gain / gain.sum() if gain.sum() > 0 else np.zeros_like(gain, dtype=float)
            split_norm = split / split.sum() if split.sum() > 0 else np.zeros_like(split, dtype=float)

            rng = np.random.default_rng(42)
            if len(X_val) > PERMUTATION_SAMPLE:
                sample_idx = rng.choice(len(X_val), size=PERMUTATION_SAMPLE, replace=False)
                X_perm = X_val.iloc[sample_idx]
                y_perm = y_val.iloc[sample_idx]
            else:
                X_perm = X_val
                y_perm = y_val
            perm = permutation_importance(
                model,
                X_perm,
                y_perm,
                n_repeats=3,
                random_state=42,
                scoring="roc_auc",
                n_jobs=1,
            )
            perm_mean = np.maximum(perm.importances_mean, 0.0)
            perm_norm = perm_mean / perm_mean.sum() if perm_mean.sum() > 0 else np.zeros_like(perm_mean)

            rank_df = pd.DataFrame({
                "feature": feature_cols,
                "gain_norm": gain_norm,
                "split_norm": split_norm,
                "perm_norm": perm_norm,
            })
            rank_df["fold_importance_score"] = (
                0.55 * rank_df["gain_norm"]
                + 0.20 * rank_df["split_norm"]
                + 0.25 * rank_df["perm_norm"]
            )
            top_features = set(
                rank_df.sort_values("fold_importance_score", ascending=False)
                .head(TOP_M)["feature"]
            )
            rank_df["in_top_m"] = rank_df["feature"].isin(top_features).astype(int)
            rank_df["fold"] = fold_name
            rank_df["strategy"] = strategy
            rank_df["validation_year"] = fold["val"][0]
            rank_df["validation_auc"] = auc
            rows.append(rank_df)
            print(f"  {fold_name} {strategy}: val_auc={auc:.3f}")

    fold_importance = pd.concat(rows, ignore_index=True)
    summary = (
        fold_importance.groupby("feature", as_index=False)
        .agg(
            mean_score=("fold_importance_score", "mean"),
            mean_gain=("gain_norm", "mean"),
            mean_split=("split_norm", "mean"),
            mean_permutation=("perm_norm", "mean"),
            stability=("in_top_m", "mean"),
            folds_seen=("fold", "nunique"),
            strategy_count=("strategy", "nunique"),
            mean_validation_auc=("validation_auc", "mean"),
        )
    )
    summary["feature_block"] = summary["feature"].map(infer_feature_block)
    summary = summary.sort_values(["stability", "mean_score"], ascending=[False, False])
    selected = summary[
        (summary["stability"] >= MIN_STABILITY)
        | (summary["mean_score"].rank(ascending=False, method="first") <= MAX_SELECTED_FEATURES)
    ].head(MAX_SELECTED_FEATURES).copy()
    selected["selected_rank"] = np.arange(1, len(selected) + 1)
    selected["selected_for_transformer"] = True
    summary = summary.merge(
        selected[["feature", "selected_rank", "selected_for_transformer"]],
        on="feature",
        how="left",
    )
    summary["selected_for_transformer"] = summary["selected_for_transformer"].fillna(False)
    return fold_importance, summary


def main() -> None:
    data, feature_cols = build_token_dataset()
    data.to_parquet(DATA_DIR / "tp2_t1k2_token_dataset.parquet", index=False)

    fold_importance, feature_dictionary = run_feature_discovery(data, feature_cols)
    fold_importance.to_csv(TABLE_DIR / "fold_feature_importance.csv", index=False)
    feature_dictionary.to_csv(TABLE_DIR / "feature_dictionary.csv", index=False)

    selected = feature_dictionary[feature_dictionary["selected_for_transformer"]].copy()
    summary = {
        "strategy_universe": STRATEGIES,
        "label_policy": "exit_selected_leg_expiration_payoff",
        "token_rows": int(len(data)),
        "unique_episodes": int(data["episode_id"].nunique()),
        "date_min": str(data["signal_date"].min().date()),
        "date_max": str(data["signal_date"].max().date()),
        "candidate_features": int(len(feature_cols)),
        "selected_features": int(len(selected)),
        "selected_feature_blocks": selected["feature_block"].value_counts().to_dict(),
        "folds": FOLDS,
        "notes": [
            "Boosting is used only for feature discovery.",
            "Booster predictions, leaves, SHAP vectors, and scores are not transformer inputs.",
            "signal_year and all label/future columns are excluded from candidate features.",
            "SHAP is not computed in this environment because shap is not installed; permutation importance is used instead.",
        ],
    }
    with open(TABLE_DIR / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSelected feature dictionary:")
    print(selected[["selected_rank", "feature", "feature_block", "stability", "mean_score"]].to_string(index=False))
    print(f"\nWrote {DATA_DIR / 'tp2_t1k2_token_dataset.parquet'}")
    print(f"Wrote {TABLE_DIR / 'feature_dictionary.csv'}")
    print(f"Wrote {TABLE_DIR / 'build_summary.json'}")


if __name__ == "__main__":
    main()
