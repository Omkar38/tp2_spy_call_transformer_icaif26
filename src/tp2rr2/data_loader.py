from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


NUMERIC_FLOAT_COLUMNS = [
    "strike",
    "last",
    "mark",
    "bid",
    "ask",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
]

NUMERIC_INT_COLUMNS = ["bid_size", "ask_size", "volume", "open_interest"]


def flatten_archive_file(path: Path) -> pd.DataFrame:
    """Load one yearly JSON file whose top level is a list of daily option-chain lists."""
    with path.open("r") as f:
        days = json.load(f)
    rows = [row for day in days for row in day]
    df = pd.DataFrame(rows)
    df["source_file"] = path.name
    return df


def load_archive(archive_dir: Path, years: list[int] | None = None) -> pd.DataFrame:
    files = sorted(archive_dir.glob("spy_options_data_*.json"))
    if years is not None:
        suffixes = {str(year)[-2:] for year in years}
        files = [p for p in files if p.stem.rsplit("_", 1)[-1] in suffixes]
    if not files:
        raise FileNotFoundError(f"No archive JSON files found in {archive_dir}")
    return pd.concat([flatten_archive_file(path) for path in files], ignore_index=True)


def normalize_raw_options(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["expiration"] = pd.to_datetime(out["expiration"]).dt.normalize()
    out["type"] = out["type"].astype(str).str.lower()
    out["symbol"] = out["symbol"].astype(str).str.upper()

    for column in NUMERIC_FLOAT_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in NUMERIC_INT_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out["dte"] = (out["expiration"] - out["date"]).dt.days
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    out["spread"] = out["ask"] - out["bid"]
    out["spread_pct"] = out["spread"] / out["mid"]
    return out
