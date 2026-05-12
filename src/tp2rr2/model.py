from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DROP_COLUMNS = {
    "realized_return",
    "net_pnl",
    "is_valid_label",
    "exit_date",
    "exit_reason",
    "label_horizon",
    "source_file",
}

ID_COLUMNS = {
    "violation_id",
    "date",
    "T1_expiration",
    "T2_expiration",
    "A_contract_id",
    "B_contract_id",
    "C_contract_id",
    "D_contract_id",
    "A_expiration",
    "B_expiration",
    "C_expiration",
    "D_expiration",
}


def _lightgbm_regressor():
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("LightGBM is required. Install requirements.txt first.") from exc
    return LGBMRegressor(
        objective="regression",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def split_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    date = pd.to_datetime(df["date"])
    return df[(date >= pd.Timestamp(start)) & (date <= pd.Timestamp(end))].copy()


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    candidates = [c for c in df.columns if c not in DROP_COLUMNS and c not in ID_COLUMNS]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in candidates if c not in numeric]
    return numeric, categorical


def make_pipeline(train_df: pd.DataFrame) -> Pipeline:
    numeric, categorical = feature_columns(train_df)
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    return Pipeline([("preprocess", pre), ("model", _lightgbm_regressor())])


def train_model(train_df: pd.DataFrame, validation_df: pd.DataFrame, output_path: Path) -> dict:
    train_df = train_df[train_df["is_valid_label"]].copy()
    validation_df = validation_df[validation_df["is_valid_label"]].copy()
    if train_df.empty or validation_df.empty:
        raise ValueError("Train and validation data must both contain valid labels.")

    y_train = train_df["realized_return"].astype(float)
    y_val = validation_df["realized_return"].astype(float)
    pipe = make_pipeline(train_df)
    pipe.fit(train_df, y_train)
    pred = pipe.predict(validation_df)
    metrics = {
        "validation_mae": float(mean_absolute_error(y_val, pred)),
        "validation_rmse": float(np.sqrt(mean_squared_error(y_val, pred))),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "metrics": metrics}, output_path)
    return metrics


def load_model(path: Path):
    return joblib.load(path)["pipeline"]
