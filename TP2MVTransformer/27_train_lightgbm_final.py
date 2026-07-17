"""
LightGBM baseline, final: trained fresh (not reused from
ExpiryMLCorrection/02_parallel_binary_lgbm.py's saved output) and scored
with the exact same methodology now shipped for the transformer, so the
model comparison is genuinely apples-to-apples:

  - Isotonic calibration fit on validation predictions only (the original
    LightGBM script had no calibration step at all -- predict_proba from a
    class_weight="balanced" model is not a well-calibrated probability).
  - Causal p>=0.5 selection (trust the calibrated model directly, zero
    tuning) -- the original script used a validation-P&L-maximizing
    threshold search (tune_threshold), the same "lottery ticket" mechanism
    already retired from the transformer pipeline for being unstable.
  - Correctly-computed daily Sharpe, Sortino, and max drawdown -- the
    original script computed Sharpe from trade-level (not daily-grouped)
    returns, which is a different and incorrect statistic (its own comment
    admitted this: "approximate using trade-level std (not daily grouped
    here, simple version)"). Sortino and max drawdown were not computed at
    all.

Trains all 4 parallel binary strategies (T1/T2/K1/K2, matching the original
script's role as the feature-discovery model), but the final comparison
metrics restrict to T1+K2 only, matching the transformer's scope and the
paper's structural short-D-leg prior.

New, separate script -- does not modify ExpiryMLCorrection/02_parallel_binary_lgbm.py.
Writes only *_lgbm_final_* outputs under TP2MVTransformer/tables and /data.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TP2MVTransformer"
DATA_DIR = OUT / "data"
TABLE_DIR = OUT / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

FEAT_PATH = ROOT / "data/ml/archive2_ml_episode_dataset_2013_2026.parquet"
LABEL_PATH = ROOT / "data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet"

STRATEGIES = ["T1", "T2", "K1", "K2"]
COMPARISON_STRATEGIES = ["T1", "K2"]  # structural short-D-leg prior, matches the transformer's scope
NORM_EXPOSURE = 1_000.0
SELECTION_THRESHOLD = 0.5

FOLDS = [
    {"fold": "2017_2019__val2020__test2021", "train": [2017, 2018, 2019], "val": [2020], "test": [2021]},
    {"fold": "2018_2020__val2021__test2022", "train": [2018, 2019, 2020], "val": [2021], "test": [2022]},
    {"fold": "2019_2021__val2022__test2023", "train": [2019, 2020, 2021], "val": [2022], "test": [2023]},
    {"fold": "2020_2022__val2023__test2024", "train": [2020, 2021, 2022], "val": [2023], "test": [2024]},
    {"fold": "2021_2023__val2024__test2025", "train": [2021, 2022, 2023], "val": [2024], "test": [2025]},
    {"fold": "2022_2024__val2025__test2026", "train": [2022, 2023, 2024], "val": [2025], "test": [2026]},
]

LGBM_PARAMS = {
    "objective": "binary", "metric": "binary_logloss", "n_estimators": 350, "learning_rate": 0.04,
    "num_leaves": 31, "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.8,
    "class_weight": "balanced", "n_jobs": -1, "verbose": -1, "random_state": 42,
}


# ── metrics, identical formulas to 03_train_mv_alibi_transformer.py ─────────────

def daily_sharpe(selected: pd.DataFrame) -> float:
    if selected.empty:
        return float("nan")
    daily = selected.groupby("signal_date")["norm_pnl"].sum()
    std = daily.std()
    if std == 0 or pd.isna(std):
        return float("nan")
    return float(daily.mean() / std * np.sqrt(252))


def daily_sortino(selected: pd.DataFrame) -> float:
    if selected.empty:
        return float("nan")
    daily = selected.groupby("signal_date")["norm_pnl"].sum()
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
    daily = selected.groupby("signal_date")["norm_pnl"].sum().sort_index()
    equity = daily.cumsum()
    return float((equity - equity.cummax()).min())


def summarize(selected: pd.DataFrame) -> dict:
    if selected.empty:
        return {"n_selected": 0, "norm_total_pnl": 0.0, "win_rate": float("nan"),
                "daily_sharpe": float("nan"), "daily_sortino": float("nan"), "max_drawdown": float("nan")}
    return {
        "n_selected": int(len(selected)),
        "norm_total_pnl": float(selected["norm_pnl"].sum()),
        "win_rate": float((selected["norm_pnl"] > 0).mean()),
        "daily_sharpe": daily_sharpe(selected),
        "daily_sortino": daily_sortino(selected),
        "max_drawdown": daily_drawdown(selected),
    }


def fit_isotonic_calibration(val_probs: np.ndarray, val_labels: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_probs, val_labels)
    return iso


def main() -> None:
    print("Loading features and labels...")
    feat = pd.read_parquet(FEAT_PATH)
    feat["signal_year"] = pd.to_datetime(feat["signal_date"]).dt.year

    raw_labels = pd.read_parquet(LABEL_PATH)
    expiry = raw_labels[raw_labels["exit_policy"] == "exit_selected_leg_expiration_payoff"].copy()
    expiry_wide = expiry.pivot(index="episode_id", columns="strategy", values=["realized_return", "net_pnl", "is_valid_backtest"])
    expiry_wide.columns = [f"expiry_{v}_{s}" for v, s in expiry_wide.columns]
    expiry_wide = expiry_wide.reset_index()

    for strat in STRATEGIES:
        valid_col = f"expiry_is_valid_backtest_{strat}"
        ret_col = f"expiry_realized_return_{strat}"
        expiry_wide[f"label_expiry_{strat}_profitable"] = (
            expiry_wide[valid_col].fillna(False) & (expiry_wide[ret_col].fillna(-1) > 0)
        ).astype(int)
        expiry_wide[f"norm_pnl_{strat}"] = expiry_wide[ret_col].fillna(0) * NORM_EXPOSURE

    data = feat.merge(expiry_wide, on="episode_id", how="inner")
    data = data[data["signal_year"] >= 2014].copy()
    print(f"  Episodes after 2013 exclusion: {len(data):,}")

    feature_cols = (
        [c for c in data.columns if c.startswith("feature_")]
        + [c for c in data.columns if c.startswith("lc_")]
        + ["start_normalized_severity", "signal_month", "signal_dayofweek"]
    )
    feature_cols = [c for c in feature_cols if c in data.columns and c != "signal_year"]
    print(f"  Feature columns: {len(feature_cols)}")

    all_test_rows = []  # per-strategy, per-fold, calibrated test predictions
    all_val_rows = []
    fold_val_auc_rows = []

    for strat in STRATEGIES:
        label_col = f"label_expiry_{strat}_profitable"
        norm_pnl_col = f"norm_pnl_{strat}"
        valid_col = f"expiry_is_valid_backtest_{strat}"
        strat_rows = data[["episode_id", "signal_date", "signal_year", label_col, norm_pnl_col, valid_col] + feature_cols].dropna(subset=[label_col]).copy()

        print(f"\n{'='*60}\nStrategy: {strat} | episodes: {len(strat_rows):,} | base rate: {strat_rows[label_col].mean():.3f}")

        for fold in FOLDS:
            tr_mask = strat_rows["signal_year"].isin(fold["train"])
            val_mask = strat_rows["signal_year"].isin(fold["val"])
            te_mask = strat_rows["signal_year"].isin(fold["test"])

            X_tr, y_tr = strat_rows.loc[tr_mask, feature_cols], strat_rows.loc[tr_mask, label_col]
            X_val, y_val = strat_rows.loc[val_mask, feature_cols], strat_rows.loc[val_mask, label_col]
            X_te = strat_rows.loc[te_mask, feature_cols]

            if X_tr.empty or X_val.empty or X_te.empty or y_tr.nunique() < 2:
                print(f"  {fold['fold']}: SKIPPED (insufficient data)")
                continue

            model = lgb.LGBMClassifier(**LGBM_PARAMS)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

            val_prob_raw = model.predict_proba(X_val)[:, 1]
            test_prob_raw = model.predict_proba(X_te)[:, 1]

            iso = fit_isotonic_calibration(val_prob_raw, y_val.to_numpy())
            val_prob_cal = iso.predict(val_prob_raw)
            test_prob_cal = iso.predict(test_prob_raw)

            try:
                val_auc = float(roc_auc_score(y_val, val_prob_raw))
            except ValueError:
                val_auc = float("nan")
            try:
                test_auc = float(roc_auc_score(strat_rows.loc[te_mask, label_col], test_prob_raw))
            except ValueError:
                test_auc = float("nan")
            fold_val_auc_rows.append({"strategy": strat, "fold": fold["fold"], "test_year": fold["test"][0], "val_auc": val_auc, "test_auc": test_auc})

            val_out = strat_rows.loc[val_mask, ["episode_id", "signal_date", "signal_year", norm_pnl_col, valid_col]].copy()
            val_out = val_out.rename(columns={norm_pnl_col: "norm_pnl", valid_col: "is_valid"})
            val_out["strategy"] = strat
            val_out["fold"] = fold["fold"]
            val_out["predicted_prob"] = val_prob_cal
            all_val_rows.append(val_out)

            test_out = strat_rows.loc[te_mask, ["episode_id", "signal_date", "signal_year", norm_pnl_col, valid_col]].copy()
            test_out = test_out.rename(columns={norm_pnl_col: "norm_pnl", valid_col: "is_valid"})
            test_out["strategy"] = strat
            test_out["fold"] = fold["fold"]
            test_out["test_year"] = fold["test"][0]
            test_out["predicted_prob"] = test_prob_cal
            all_test_rows.append(test_out)

            print(f"  {fold['fold']} | val_auc={val_auc:.3f} test_auc={test_auc:.3f}")

    val_predictions = pd.concat(all_val_rows, ignore_index=True)
    test_predictions = pd.concat(all_test_rows, ignore_index=True)
    val_predictions.to_parquet(DATA_DIR / "lgbm_final_val_predictions.parquet", index=False)
    test_predictions.to_parquet(DATA_DIR / "lgbm_final_test_predictions.parquet", index=False)
    pd.DataFrame(fold_val_auc_rows).to_csv(TABLE_DIR / "lgbm_final_auc_by_fold.csv", index=False)

    # ── causal p>=0.5 selection, one action per episode per fold (rank ties broken by strategy name) ──
    print("\n" + "=" * 70)
    print("Applying causal p>=0.5 selection (all 4 strategies, then T1+K2 subset)...")

    def select_episode_actions(tokens: pd.DataFrame) -> pd.DataFrame:
        candidates = tokens[tokens["predicted_prob"] >= SELECTION_THRESHOLD].copy()
        if candidates.empty:
            return candidates
        return candidates.sort_values(["predicted_prob", "strategy"], ascending=[False, True]).groupby(["fold", "episode_id"], as_index=False).first()

    selected_all4 = select_episode_actions(test_predictions)
    selected_t1k2 = select_episode_actions(test_predictions[test_predictions["strategy"].isin(COMPARISON_STRATEGIES)])

    yearly_rows = []
    for test_year in sorted(test_predictions["test_year"].unique()):
        y_all4 = selected_all4[selected_all4["test_year"] == test_year]
        y_t1k2 = selected_t1k2[selected_t1k2["test_year"] == test_year]
        s_all4 = summarize(y_all4)
        s_t1k2 = summarize(y_t1k2)
        yearly_rows.append({"test_year": int(test_year), **{f"all4_{k}": v for k, v in s_all4.items()}, **{f"t1k2_{k}": v for k, v in s_t1k2.items()}})
        print(f"  {int(test_year)}: [all4] n={s_all4['n_selected']:,} pnl=${s_all4['norm_total_pnl']:,.0f} sharpe={s_all4['daily_sharpe']:.2f}  |  [T1+K2] n={s_t1k2['n_selected']:,} pnl=${s_t1k2['norm_total_pnl']:,.0f} sharpe={s_t1k2['daily_sharpe']:.2f}")

    yearly_df = pd.DataFrame(yearly_rows)
    yearly_df.to_csv(TABLE_DIR / "lgbm_final_yearly_diagnostics.csv", index=False)

    combined_all4 = summarize(selected_all4)
    combined_t1k2 = summarize(selected_t1k2)
    summary = {
        "method": "LightGBM parallel binary classifiers (T1/T2/K1/K2), final: isotonic calibration + causal p>=0.5 selection",
        "selection_rule": f"fixed p>={SELECTION_THRESHOLD}, zero tuning, fully causal",
        "feature_count": len(feature_cols),
        "all4_strategies": {**combined_all4, "positive_years": int((yearly_df["all4_norm_total_pnl"] > 0).sum()), "total_years": len(yearly_df)},
        "t1_k2_only": {**combined_t1k2, "positive_years": int((yearly_df["t1k2_norm_total_pnl"] > 0).sum()), "total_years": len(yearly_df)},
        "notes": [
            "Trained fresh -- does not reuse ExpiryMLCorrection/02_parallel_binary_lgbm.py's saved predictions.",
            "Isotonic calibration fit on validation predictions only, then applied unchanged to test -- "
            "the original script had no calibration step.",
            "Selection is a fixed p>=0.5 cutoff, matching the transformer's shipped rule -- the original "
            "script used a validation-P&L-maximizing threshold search (tune_threshold), already retired "
            "from the transformer pipeline for producing unstable results.",
            "Sharpe/Sortino/drawdown are computed from daily-grouped P&L, matching the transformer's "
            "methodology -- the original script's Sharpe was computed from trade-level (not daily) "
            "returns, a different and incorrect statistic; Sortino and max drawdown were not computed at all.",
            "t1_k2_only restricts to the T1+K2 structural short-D-leg prior, matching the transformer's scope.",
        ],
    }
    with open(TABLE_DIR / "lgbm_final_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    selected_t1k2.to_parquet(DATA_DIR / "lgbm_final_t1k2_selected_trades.parquet", index=False)

    print("\n" + "=" * 70)
    print("LightGBM final summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {TABLE_DIR / 'lgbm_final_summary.json'}")


if __name__ == "__main__":
    main()
