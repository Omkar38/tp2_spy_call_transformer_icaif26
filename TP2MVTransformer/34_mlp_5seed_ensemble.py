"""
MLP, 5-seed ensemble -- closes a gap surfaced by re-reading the meeting
transcript carefully (professor, [12:23]): "You can see a little bit of
that because you are averaging models and only then you get better
performance." The transformer went through v3 (1 seed) -> v4 (5-seed
ensemble average), and v4 is what MLP has been compared against in
model_comparison_report.pdf. MLP itself has NEVER been tested as an
ensemble -- every MLP script (02_train_mlp_baseline.py,
28_train_mlp_final.py, 31_mlp_loss_curve_comparison.py) uses a single
fixed random_state=42. That makes the existing MLP-vs-v4-final comparison
apples-to-oranges on the ensembling axis specifically.

This script trains 5 independently-seeded MLPs per fold (same seeds as
the transformer's v4: 42, 101, 202, 303, 404), averages raw probabilities
before ONE isotonic calibration -- identical ensembling methodology to
03_train_mv_alibi_transformer_v4_multiseed.py, for genuine
apples-to-apples comparability. Same F* features, folds, and causal
p>=0.5 selection as 28_train_mlp_final.py.

New, separate script -- does not modify 28_train_mlp_final.py.
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
SEEDS = [42, 101, 202, 303, 404]   # identical to v4's SEEDS

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


def daily_sharpe(sel: pd.DataFrame) -> float:
    if sel.empty:
        return float("nan")
    daily = sel.groupby("signal_date")["target_norm_pnl"].sum()
    std = daily.std()
    return float(daily.mean() / std * np.sqrt(252)) if std and not pd.isna(std) else float("nan")


def daily_sortino(sel: pd.DataFrame) -> float:
    if sel.empty:
        return float("nan")
    daily = sel.groupby("signal_date")["target_norm_pnl"].sum()
    if len(daily) < 2:
        return float("nan")
    downside = daily[daily < 0.0]
    if downside.empty:
        return float("inf") if daily.mean() > 0 else float("nan")
    dstd = downside.std()
    return float(daily.mean() / dstd * np.sqrt(252)) if dstd and not pd.isna(dstd) else float("nan")


def daily_drawdown(sel: pd.DataFrame) -> float:
    if sel.empty:
        return float("nan")
    daily = sel.groupby("signal_date")["target_norm_pnl"].sum().sort_index()
    equity = daily.cumsum()
    return float((equity - equity.cummax()).min())


def summarize(sel: pd.DataFrame) -> dict:
    if sel.empty:
        return {"n_selected": 0, "norm_total_pnl": 0.0, "win_rate": float("nan"),
                "daily_sharpe": float("nan"), "daily_sortino": float("nan"), "max_drawdown": float("nan")}
    return {
        "n_selected": int(len(sel)),
        "norm_total_pnl": float(sel["target_norm_pnl"].sum()),
        "win_rate": float((sel["target_norm_pnl"] > 0).mean()),
        "daily_sharpe": daily_sharpe(sel),
        "daily_sortino": daily_sortino(sel),
        "max_drawdown": daily_drawdown(sel),
    }


def train_one_seed(seed: int, X_train_s, y_train, X_val_s, X_test_s) -> tuple[np.ndarray, np.ndarray, int]:
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4, batch_size=1024,
        learning_rate_init=1e-3, max_iter=80, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=8, random_state=seed,
    )
    model.fit(X_train_s, y_train)
    return model.predict_proba(X_val_s)[:, 1], model.predict_proba(X_test_s)[:, 1], model.n_iter_


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
    print(f"Selected F* features: {len(feature_cols)} | SEEDS={SEEDS}")

    fold_rows = []
    prediction_rows = []
    seed_diag_rows = []

    for fold in FOLDS:
        fold_name = fold["fold"]
        test_year = fold["test"][0]
        train = data[data["signal_year"].isin(fold["train"])].copy()
        val = data[data["signal_year"].isin(fold["val"])].copy()
        test = data[data["signal_year"].isin(fold["test"])].copy()

        X_train = safe_numeric_frame(train, feature_cols)
        y_train = train["target_profitable"].astype(int)
        X_val = safe_numeric_frame(val, feature_cols)
        y_val = val["target_profitable"].astype(int)
        X_test = safe_numeric_frame(test, feature_cols)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        val_probs_by_seed, test_probs_by_seed = [], []
        for seed in SEEDS:
            val_p, test_p, n_iter = train_one_seed(seed, X_train_s, y_train, X_val_s, X_test_s)
            val_probs_by_seed.append(val_p)
            test_probs_by_seed.append(test_p)
            seed_diag_rows.append({"fold": fold_name, "test_year": test_year, "seed": seed, "n_iter": n_iter})
            print(f"  seed={seed} n_iter={n_iter}", flush=True)

        val_prob_avg = np.mean(val_probs_by_seed, axis=0)
        test_prob_avg = np.mean(test_probs_by_seed, axis=0)

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_prob_avg, y_val.to_numpy())
        test_prob_cal = iso.predict(test_prob_avg)

        test_tokens = test[["episode_id", "signal_date", "signal_year", "strategy", "target_profitable", "target_return", "target_norm_pnl"]].copy()
        test_tokens["predicted_prob"] = test_prob_cal
        test_tokens["fold"] = fold_name
        test_selected = select_episode_actions(test_tokens)
        metrics = summarize(test_selected)

        print(
            f"{fold_name}: test_pnl=${metrics['norm_total_pnl']:,.0f} test_n={metrics['n_selected']:,} "
            f"sharpe={metrics['daily_sharpe']:.2f}",
            flush=True,
        )

        fold_rows.append({
            "fold": fold_name, "test_year": test_year, "n_seeds": len(SEEDS),
            **{f"test_{k}": v for k, v in metrics.items()},
        })
        prediction_rows.append(test_tokens)

    results = pd.DataFrame(fold_rows)
    results.to_csv(TABLE_DIR / "mlp_5seed_ensemble_results.csv", index=False)
    pd.DataFrame(seed_diag_rows).to_csv(TABLE_DIR / "mlp_5seed_seed_diagnostics.csv", index=False)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    selected_all = pd.concat([select_episode_actions(predictions[predictions["fold"] == f]) for f in predictions["fold"].unique()], ignore_index=True)
    combined = summarize(selected_all)

    # comparison against single-seed MLP (already shipped)
    single_seed_summary = json.loads((TABLE_DIR / "mlp_final_summary.json").read_text()) if (TABLE_DIR / "mlp_final_summary.json").exists() else {}

    summary = {
        "method": f"MLP on selected F_star features, {len(SEEDS)}-seed ensemble average + causal p>=0.5 selection",
        "seeds": SEEDS,
        "selection_rule": f"fixed p>={SELECTION_THRESHOLD}, zero tuning, fully causal",
        **combined,
        "positive_years": int((results["test_norm_total_pnl"] > 0).sum()),
        "total_years": len(FOLDS),
        "single_seed_comparison": {
            "single_seed_daily_sharpe": single_seed_summary.get("daily_sharpe"),
            "single_seed_daily_sortino": single_seed_summary.get("daily_sortino"),
            "single_seed_norm_total_pnl": single_seed_summary.get("norm_total_pnl"),
            "ensemble_daily_sharpe": combined["daily_sharpe"],
            "ensemble_daily_sortino": combined["daily_sortino"],
            "ensemble_norm_total_pnl": combined["norm_total_pnl"],
        },
        "notes": [
            "Same ensembling methodology as 03_train_mv_alibi_transformer_v4_multiseed.py: raw "
            "probabilities averaged across 5 seeds before ONE isotonic calibration.",
            "Built to close a gap surfaced by the professor's own comment (meeting transcript, "
            "[12:23]): the existing MLP-vs-transformer comparison compared single-seed MLP against "
            "5-seed-ensembled transformer (v4-final) -- apples-to-oranges on the ensembling axis.",
        ],
    }
    with open(TABLE_DIR / "mlp_5seed_ensemble_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 70)
    print("MLP 5-seed ensemble summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {TABLE_DIR / 'mlp_5seed_ensemble_summary.json'}")


if __name__ == "__main__":
    main()
