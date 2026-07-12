#!/usr/bin/env python3
"""Audit feature drift, label drift, and probability calibration for TP2 models."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TP2MVTransformer"
TABLE_DIR = OUT / "tables"
AUDIT_DIR = OUT / "ablation_runs" / "_audit"
AUDIT_TABLE_DIR = AUDIT_DIR / "tables"
AUDIT_TABLE_DIR.mkdir(parents=True, exist_ok=True)


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (prob >= left) & (prob < right if right < 1.0 else prob <= right)
        if not np.any(mask):
            continue
        ece += float(mask.mean() * abs(prob[mask].mean() - y_true[mask].mean()))
    return ece


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    actual = pd.to_numeric(actual, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(expected) < 20 or len(actual) < 20:
        return float("nan")
    quantiles = np.unique(np.nanquantile(expected, np.linspace(0.0, 1.0, bins + 1)))
    if len(quantiles) <= 2:
        return float("nan")
    exp_counts = pd.cut(expected, quantiles, include_lowest=True, duplicates="drop").value_counts(normalize=True)
    act_counts = pd.cut(actual, quantiles, include_lowest=True, duplicates="drop").value_counts(normalize=True)
    idx = exp_counts.index.union(act_counts.index)
    exp_pct = exp_counts.reindex(idx, fill_value=0.0).clip(lower=1e-6)
    act_pct = act_counts.reindex(idx, fill_value=0.0).clip(lower=1e-6)
    return float(((act_pct - exp_pct) * np.log(act_pct / exp_pct)).sum())


def calibration_table(preds: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows = []
    for year, g in preds.groupby("signal_year"):
        y = g["target_profitable"].astype(int).to_numpy()
        p = g["predicted_prob"].astype(float).clip(0, 1).to_numpy()
        brier = float(brier_score_loss(y, p)) if len(np.unique(y)) > 1 else float("nan")
        rows.append({
            "year": int(year),
            "rows": int(len(g)),
            "positive_rate": float(y.mean()),
            "mean_probability": float(np.mean(p)),
            "brier": brier,
            "ece": expected_calibration_error(y, p, bins=bins),
        })
    return pd.DataFrame(rows)


def calibration_bins(preds: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for year, g in preds.groupby("signal_year"):
        p = g["predicted_prob"].astype(float).clip(0, 1)
        y = g["target_profitable"].astype(int)
        bucket = pd.cut(p, edges, include_lowest=True)
        tmp = pd.DataFrame({"bucket": bucket, "prob": p, "label": y})
        for bucket_value, b in tmp.groupby("bucket", observed=False):
            if len(b) == 0:
                continue
            rows.append({
                "year": int(year),
                "bucket": str(bucket_value),
                "rows": int(len(b)),
                "mean_probability": float(b["prob"].mean()),
                "realized_positive_rate": float(b["label"].mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TP2 transformer drift/calibration audit.")
    parser.add_argument("--predictions", default="TP2MVTransformer/data/mv_alibi_transformer_v2_test_predictions.parquet")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    token_path = OUT / "data" / "tp2_t1k2_token_dataset.parquet"
    feature_dict = pd.read_csv(TABLE_DIR / "feature_dictionary.csv")
    feature_cols = (
        feature_dict[feature_dict["selected_for_transformer"].astype(bool)]
        .sort_values("selected_rank")["feature"]
        .tolist()
    )
    cols = ["signal_date", "strategy", "target_profitable", *feature_cols]
    data = pd.read_parquet(token_path, columns=cols)
    data["signal_date"] = pd.to_datetime(data["signal_date"])
    data["year"] = data["signal_date"].dt.year

    label_rows = []
    for (year, strategy), g in data.groupby(["year", "strategy"]):
        label_rows.append({
            "year": int(year),
            "strategy": strategy,
            "rows": int(len(g)),
            "target_positive_rate": float(g["target_profitable"].mean()),
        })
    pd.DataFrame(label_rows).to_csv(AUDIT_TABLE_DIR / "target_rate_drift_by_year_strategy.csv", index=False)

    baseline = data[data["year"].between(2017, 2020)]
    drift_rows = []
    for year, g in data.groupby("year"):
        for feature in feature_cols:
            drift_rows.append({
                "year": int(year),
                "feature": feature,
                "psi_vs_2017_2020": psi(baseline[feature], g[feature], bins=args.bins),
                "mean": float(pd.to_numeric(g[feature], errors="coerce").mean()),
                "missing_rate": float(pd.to_numeric(g[feature], errors="coerce").isna().mean()),
            })
    drift = pd.DataFrame(drift_rows)
    drift.to_csv(AUDIT_TABLE_DIR / "feature_drift_psi_by_year.csv", index=False)
    drift.sort_values("psi_vs_2017_2020", ascending=False).head(100).to_csv(
        AUDIT_TABLE_DIR / "top_feature_drift_psi.csv", index=False
    )

    pred_path = ROOT / args.predictions
    if pred_path.exists():
        preds = pd.read_parquet(pred_path)
        preds["signal_year"] = pd.to_datetime(preds["signal_date"]).dt.year
        calibration_table(preds, args.bins).to_csv(AUDIT_TABLE_DIR / "probability_calibration_by_year.csv", index=False)
        calibration_bins(preds, args.bins).to_csv(AUDIT_TABLE_DIR / "probability_calibration_bins.csv", index=False)

    print(f"Wrote audit tables to {AUDIT_TABLE_DIR}")


if __name__ == "__main__":
    main()
