"""
MLP baseline, final: trained fresh (not reused from
02_train_mlp_baseline.py's saved output) and scored with the same
methodology now shipped for the transformer and the LightGBM-final
baseline, so all three models are genuinely apples-to-apples:

  - Isotonic calibration fit on validation predictions only -- the original
    script had none.
  - Causal p>=0.5 selection -- the original script used
    tune_threshold (validation-P&L-maximizing search), the same mechanism
    already retired from the transformer pipeline for being unstable.
  - Sortino and max drawdown added -- the original script's daily_sharpe
    was already correctly daily-grouped, but Sortino and drawdown were
    never computed at all.

Same F* feature dictionary and T1+K2 token dataset as the original script
-- no change to features or data scope, only calibration/selection/metrics.

New, separate script -- does not modify 02_train_mlp_baseline.py.
Writes only *_mlp_final_* outputs.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
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
SELECTION_THRESHOLD = 0.5

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


def select_episode_actions(tokens: pd.DataFrame) -> pd.DataFrame:
    candidates = tokens[tokens["predicted_prob"] >= SELECTION_THRESHOLD].copy()
    if candidates.empty:
        return candidates
    return candidates.sort_values(["predicted_prob", "strategy"], ascending=[False, True]).groupby("episode_id", as_index=False).first()


def daily_sharpe(selected: pd.DataFrame) -> float:
    if selected.empty:
        return float("nan")
    daily = selected.groupby("signal_date")["target_norm_pnl"].sum()
    std = daily.std()
    if std == 0 or pd.isna(std):
        return float("nan")
    return float(daily.mean() / std * np.sqrt(252))


def daily_sortino(selected: pd.DataFrame) -> float:
    if selected.empty:
        return float("nan")
    daily = selected.groupby("signal_date")["target_norm_pnl"].sum()
    if len(daily) < 2:
        return float("nan")
    downside = daily[daily < 0.0]
    if downside.empty:
        return float("inf") if daily.mean() > 0 else float("nan")
    downside_std = downside.std()
    if downside_std == 0 or pd.isna(downside_std):
        return float("nan")
    return float(daily.mean() / downside_std * np.sqrt(252))


def daily_drawdown(selected: pd.DataFrame) -> float:
    if selected.empty:
        return float("nan")
    daily = selected.groupby("signal_date")["target_norm_pnl"].sum().sort_index()
    equity = daily.cumsum()
    return float((equity - equity.cummax()).min())


def summarize(selected: pd.DataFrame) -> dict:
    if selected.empty:
        return {"n_selected": 0, "norm_total_pnl": 0.0, "win_rate": float("nan"),
                "daily_sharpe": float("nan"), "daily_sortino": float("nan"), "max_drawdown": float("nan")}
    return {
        "n_selected": int(len(selected)),
        "norm_total_pnl": float(selected["target_norm_pnl"].sum()),
        "win_rate": float((selected["target_norm_pnl"] > 0).mean()),
        "daily_sharpe": daily_sharpe(selected),
        "daily_sortino": daily_sortino(selected),
        "max_drawdown": daily_drawdown(selected),
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

        model = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4, batch_size=1024,
                learning_rate_init=1e-3, max_iter=80, early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=8, random_state=42,
            )),
        ])
        model.fit(X_train, y_train)

        val_prob_raw = model.predict_proba(X_val)[:, 1]
        test_prob_raw = model.predict_proba(X_test)[:, 1]

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_prob_raw, y_val.to_numpy())
        val_prob_cal = iso.predict(val_prob_raw)
        test_prob_cal = iso.predict(test_prob_raw)

        val_tokens = val[["episode_id", "signal_date", "signal_year", "strategy", "target_profitable", "target_return", "target_norm_pnl"]].copy()
        test_tokens = test[["episode_id", "signal_date", "signal_year", "strategy", "target_profitable", "target_return", "target_norm_pnl"]].copy()
        val_tokens["predicted_prob"] = val_prob_cal
        test_tokens["predicted_prob"] = test_prob_cal
        test_tokens["fold"] = fold_name

        try:
            val_auc = float(roc_auc_score(y_val, val_prob_raw))
        except ValueError:
            val_auc = float("nan")
        try:
            test_auc = float(roc_auc_score(y_test, test_prob_raw))
        except ValueError:
            test_auc = float("nan")

        test_selected = select_episode_actions(test_tokens)
        test_metrics = summarize(test_selected)

        print(
            f"{fold_name}: val_auc={val_auc:.3f} test_auc={test_auc:.3f} "
            f"test_pnl=${test_metrics['norm_total_pnl']:,.0f} test_n={test_metrics['n_selected']:,} "
            f"sharpe={test_metrics['daily_sharpe']:.2f}"
        )

        fold_rows.append({
            "fold": fold_name, "train_years": ",".join(map(str, fold["train"])),
            "validation_year": fold["val"][0], "test_year": fold["test"][0],
            "val_auc": val_auc, "test_auc": test_auc,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        })
        prediction_rows.append(test_tokens)

    results = pd.DataFrame(fold_rows)
    results.to_csv(TABLE_DIR / "mlp_final_results.csv", index=False)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_parquet(DATA_DIR / "mlp_final_test_predictions.parquet", index=False)

    selected_all = pd.concat([select_episode_actions(predictions[predictions["fold"] == f]) for f in predictions["fold"].unique()], ignore_index=True)
    combined = summarize(selected_all)

    summary = {
        "method": "MLP on selected F_star features, final: isotonic calibration + causal p>=0.5 selection",
        "selection_rule": f"fixed p>={SELECTION_THRESHOLD}, zero tuning, fully causal",
        "feature_count": len(feature_cols),
        **combined,
        "positive_years": int((results["test_norm_total_pnl"] > 0).sum()),
        "total_years": len(FOLDS),
        "notes": [
            "Trained fresh -- does not reuse 02_train_mlp_baseline.py's saved output.",
            "Isotonic calibration fit on validation predictions only -- the original script had none.",
            "Selection is a fixed p>=0.5 cutoff, matching the transformer and LightGBM-final's shipped "
            "rule -- the original script used tune_threshold (validation-P&L-maximizing search).",
            "Sortino and max drawdown added -- the original script computed neither.",
        ],
    }
    with open(TABLE_DIR / "mlp_final_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\nMLP-final summary:")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {TABLE_DIR / 'mlp_final_summary.json'}")


if __name__ == "__main__":
    main()
