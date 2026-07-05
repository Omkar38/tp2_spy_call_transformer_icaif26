"""
Train the first TP2-MV-ALiBi ablation: an MLP on the selected F* features.

This is not the transformer yet. It answers the first ablation question from
Literature/tp2_mv_alibi_transformer.tex:

    Does the selected feature dictionary F* contain usable signal without
    historical attention context?

The evaluation protocol matches the corrected expiry pipeline:
- fixed strategy universe: T1/K2
- expiry-payoff labels
- train 3 years, validate 1 year, test 1 year
- validation year tunes the probability threshold
- test year is evaluated once
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TP2MVTransformer"
DATA_DIR = OUT / "data"
TABLE_DIR = OUT / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

TOKEN_PATH = DATA_DIR / "tp2_t1k2_token_dataset.parquet"
FEATURE_DICT_PATH = TABLE_DIR / "feature_dictionary.csv"
NORM_EXPOSURE = 1_000.0
THRESHOLD_GRID = np.arange(0.05, 0.95, 0.025)
MIN_TRADES_FOR_THRESHOLD = 30

FOLDS = [
    {"fold": "2017_2019__val2020__test2021", "train": [2017, 2018, 2019], "val": [2020], "test": [2021]},
    {"fold": "2018_2020__val2021__test2022", "train": [2018, 2019, 2020], "val": [2021], "test": [2022]},
    {"fold": "2019_2021__val2022__test2023", "train": [2019, 2020, 2021], "val": [2022], "test": [2023]},
    {"fold": "2020_2022__val2023__test2024", "train": [2020, 2021, 2022], "val": [2023], "test": [2024]},
    {"fold": "2021_2023__val2024__test2025", "train": [2021, 2022, 2023], "val": [2024], "test": [2025]},
    {"fold": "2022_2024__val2025__test2026", "train": [2022, 2023, 2024], "val": [2025], "test": [2026]},
]


def safe_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def select_episode_actions(token_rows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    candidates = token_rows[token_rows["predicted_prob"] >= threshold].copy()
    if candidates.empty:
        return candidates
    return (
        candidates.sort_values(["predicted_prob", "strategy"], ascending=[False, True])
        .groupby("episode_id", as_index=False)
        .first()
    )


def pnl_for_threshold(token_rows: pd.DataFrame, threshold: float) -> tuple[float, int]:
    selected = select_episode_actions(token_rows, threshold)
    if selected.empty:
        return 0.0, 0
    return float(selected["target_norm_pnl"].sum()), int(len(selected))


def tune_threshold(validation_tokens: pd.DataFrame) -> float:
    best_threshold = 0.5
    best_pnl = -np.inf
    for threshold in THRESHOLD_GRID:
        pnl, n_selected = pnl_for_threshold(validation_tokens, threshold)
        if n_selected < MIN_TRADES_FOR_THRESHOLD:
            continue
        if pnl > best_pnl:
            best_pnl = pnl
            best_threshold = float(threshold)
    return best_threshold


def daily_sharpe(selected: pd.DataFrame) -> float:
    if selected.empty:
        return float("nan")
    daily = selected.groupby("signal_date")["target_norm_pnl"].sum()
    std = daily.std()
    if std == 0 or pd.isna(std):
        return float("nan")
    return float(daily.mean() / std * np.sqrt(252))


def summarize_selected(selected: pd.DataFrame, threshold: float) -> dict:
    if selected.empty:
        return {
            "threshold": threshold,
            "n_selected": 0,
            "norm_total_pnl": 0.0,
            "win_rate": float("nan"),
            "daily_sharpe": float("nan"),
            "t1_selected": 0,
            "k2_selected": 0,
        }
    dist = selected["strategy"].value_counts().to_dict()
    return {
        "threshold": threshold,
        "n_selected": int(len(selected)),
        "norm_total_pnl": float(selected["target_norm_pnl"].sum()),
        "win_rate": float((selected["target_norm_pnl"] > 0).mean()),
        "daily_sharpe": daily_sharpe(selected),
        "t1_selected": int(dist.get("T1", 0)),
        "k2_selected": int(dist.get("K2", 0)),
    }


def main() -> None:
    print("Loading token dataset and feature dictionary...")
    data = pd.read_parquet(TOKEN_PATH)
    data["signal_date"] = pd.to_datetime(data["signal_date"])
    data["signal_year"] = data["signal_date"].dt.year

    feature_dict = pd.read_csv(FEATURE_DICT_PATH)
    feature_cols = (
        feature_dict[feature_dict["selected_for_transformer"].astype(bool)]
        .sort_values("selected_rank")["feature"]
        .tolist()
    )
    print(f"Selected F* features: {len(feature_cols)}")

    fold_rows = []
    prediction_rows = []

    for fold in FOLDS:
        fold_name = fold["fold"]
        train = data[data["signal_year"].isin(fold["train"])].copy()
        val = data[data["signal_year"].isin(fold["val"])].copy()
        test = data[data["signal_year"].isin(fold["test"])].copy()

        X_train = safe_numeric_frame(train, feature_cols)
        y_train = train["target_profitable"].astype(int)
        X_val = safe_numeric_frame(val, feature_cols)
        y_val = val["target_profitable"].astype(int)
        X_test = safe_numeric_frame(test, feature_cols)
        y_test = test["target_profitable"].astype(int)

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-4,
                        batch_size=1024,
                        learning_rate_init=1e-3,
                        max_iter=80,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=8,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)

        val_tokens = val[[
            "episode_id",
            "signal_date",
            "signal_year",
            "strategy",
            "target_profitable",
            "target_return",
            "target_norm_pnl",
        ]].copy()
        test_tokens = test[[
            "episode_id",
            "signal_date",
            "signal_year",
            "strategy",
            "target_profitable",
            "target_return",
            "target_norm_pnl",
        ]].copy()

        val_tokens["predicted_prob"] = model.predict_proba(X_val)[:, 1]
        test_tokens["predicted_prob"] = model.predict_proba(X_test)[:, 1]

        threshold = tune_threshold(val_tokens)
        val_selected = select_episode_actions(val_tokens, threshold)
        test_selected = select_episode_actions(test_tokens, threshold)

        try:
            val_auc = float(roc_auc_score(y_val, val_tokens["predicted_prob"]))
        except ValueError:
            val_auc = float("nan")
        try:
            test_auc = float(roc_auc_score(y_test, test_tokens["predicted_prob"]))
        except ValueError:
            test_auc = float("nan")

        val_metrics = summarize_selected(val_selected, threshold)
        test_metrics = summarize_selected(test_selected, threshold)

        print(
            f"{fold_name}: threshold={threshold:.3f} "
            f"val_pnl=${val_metrics['norm_total_pnl']:,.0f} "
            f"test_pnl=${test_metrics['norm_total_pnl']:,.0f} "
            f"test_n={test_metrics['n_selected']:,}"
        )

        fold_rows.append({
            "fold": fold_name,
            "train_years": ",".join(map(str, fold["train"])),
            "validation_year": fold["val"][0],
            "test_year": fold["test"][0],
            "val_auc": val_auc,
            "test_auc": test_auc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        })
        test_tokens["fold"] = fold_name
        test_tokens["threshold"] = threshold
        prediction_rows.append(test_tokens)

    results = pd.DataFrame(fold_rows)
    results.to_csv(TABLE_DIR / "mlp_baseline_results.csv", index=False)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_parquet(DATA_DIR / "mlp_baseline_test_predictions.parquet", index=False)

    total_pnl = float(results["test_norm_total_pnl"].sum())
    total_selected = int(results["test_n_selected"].sum())
    positive_years = int((results["test_norm_total_pnl"] > 0).sum())
    selected_all = []
    for _, fold_row in results.iterrows():
        fold_preds = predictions[predictions["fold"].eq(fold_row["fold"])]
        selected_all.append(select_episode_actions(fold_preds, float(fold_row["test_threshold"])))
    selected_all = pd.concat(selected_all, ignore_index=True) if selected_all else pd.DataFrame()
    combined_sharpe = daily_sharpe(selected_all)
    combined_win_rate = float((selected_all["target_norm_pnl"] > 0).mean()) if not selected_all.empty else float("nan")

    summary = {
        "method": "MLP baseline on selected F_star features",
        "feature_count": len(feature_cols),
        "total_selected": total_selected,
        "total_norm_pnl": total_pnl,
        "positive_years": positive_years,
        "total_years": len(FOLDS),
        "combined_daily_sharpe": combined_sharpe,
        "combined_win_rate": combined_win_rate,
        "notes": [
            "This is the first ablation before transformer attention.",
            "Thresholds are tuned on validation years only.",
            "Test years are evaluated once.",
        ],
    }
    with open(TABLE_DIR / "mlp_baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nMLP baseline summary:")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {TABLE_DIR / 'mlp_baseline_results.csv'}")
    print(f"Wrote {DATA_DIR / 'mlp_baseline_test_predictions.parquet'}")


if __name__ == "__main__":
    main()
