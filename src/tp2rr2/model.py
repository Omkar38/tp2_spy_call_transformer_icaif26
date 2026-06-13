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
    "actual_holding_days",
    "exit_normalized_severity",
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
    "A_option_symbol",
    "B_option_symbol",
    "C_option_symbol",
    "D_option_symbol",
    "A_expiration",
    "B_expiration",
    "C_expiration",
    "D_expiration",
}


MODEL_TYPES = {"lightgbm", "catboost", "catboost_native"}


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


def _catboost_regressor(loss_function: str = "RMSE", depth: int = 8):
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError("CatBoost is required. Install requirements.txt first.") from exc
    return CatBoostRegressor(
        loss_function=loss_function,
        iterations=600,
        learning_rate=0.03,
        depth=depth,
        subsample=0.85,
        random_seed=42,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )


class CatBoostStrategyRegressor:
    def __init__(self, loss_function: str = "RMSE", depth: int = 8):
        self.loss_function = loss_function
        self.depth = depth

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.numeric_columns_, self.categorical_columns_ = feature_columns(X)
        self.feature_columns_ = self.numeric_columns_ + self.categorical_columns_
        self.numeric_medians_ = X[self.numeric_columns_].median(numeric_only=True)
        train = self._prepare(X)
        self.model_ = _catboost_regressor(loss_function=self.loss_function, depth=self.depth)
        cat_features = [train.columns.get_loc(col) for col in self.categorical_columns_]
        self.model_.fit(train, y, cat_features=cat_features)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._prepare(X))

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.reindex(columns=self.feature_columns_).copy()
        if self.numeric_columns_:
            out[self.numeric_columns_] = out[self.numeric_columns_].apply(pd.to_numeric, errors="coerce")
            out[self.numeric_columns_] = out[self.numeric_columns_].fillna(self.numeric_medians_)
        if self.categorical_columns_:
            out[self.categorical_columns_] = out[self.categorical_columns_].fillna("__missing__").astype(str)
        return out


def make_regressor(model_type: str):
    model_type = model_type.lower()
    if model_type == "lightgbm":
        return _lightgbm_regressor()
    if model_type == "catboost":
        return _catboost_regressor()
    raise ValueError(f"Unknown model type {model_type!r}. Expected one of {sorted(MODEL_TYPES)}.")


def split_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    date = pd.to_datetime(df["date"])
    return df[(date >= pd.Timestamp(start)) & (date <= pd.Timestamp(end))].copy()


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    candidates = [c for c in df.columns if c not in DROP_COLUMNS and c not in ID_COLUMNS]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in candidates if c not in numeric]
    return numeric, categorical


def make_pipeline(train_df: pd.DataFrame, model_type: str = "lightgbm"):
    if model_type == "catboost_native":
        return CatBoostStrategyRegressor()

    numeric, categorical = feature_columns(train_df)
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    return Pipeline([("preprocess", pre), ("model", make_regressor(model_type))])


def train_model(train_df: pd.DataFrame, validation_df: pd.DataFrame, output_path: Path, model_type: str = "lightgbm") -> dict:
    train_df = train_df[train_df["is_valid_label"]].copy()
    validation_df = validation_df[validation_df["is_valid_label"]].copy()
    if train_df.empty or validation_df.empty:
        raise ValueError("Train and validation data must both contain valid labels.")

    y_train = train_df["realized_return"].astype(float)
    y_val = validation_df["realized_return"].astype(float)
    pipe = make_pipeline(train_df, model_type)
    pipe.fit(train_df, y_train)
    pred = pipe.predict(validation_df)
    metrics = {
        "model_type": model_type.lower(),
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
