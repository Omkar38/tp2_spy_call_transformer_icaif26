from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path}")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")
