#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import pandas as pd

from tp2rr2.config import CONFIG
from tp2rr2.io import read_table, write_table


LEGS = ["A", "B", "C", "D"]


def _bucket(moneyness: pd.Series) -> pd.Series:
    out = pd.Series("atm", index=moneyness.index, dtype="string")
    out = out.mask(moneyness < 0.98, "itm")
    out = out.mask(moneyness > 1.02, "otm")
    return out


def _rectangle_type(row: pd.Series) -> str:
    buckets = sorted({str(row[f"{leg}_moneyness_bucket"]) for leg in LEGS if pd.notna(row[f"{leg}_moneyness_bucket"])})
    if len(buckets) == 1:
        return f"pure_{buckets[0]}"
    if buckets == ["atm", "itm"]:
        return "mixed_itm_atm"
    if buckets == ["atm", "otm"]:
        return "mixed_atm_otm"
    if buckets == ["itm", "otm"]:
        return "mixed_itm_otm"
    if buckets == ["atm", "itm", "otm"]:
        return "mixed_itm_atm_otm"
    return "unknown"


def add_rectangle_moneyness(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    if "underlying_price" not in out:
        raise ValueError("features must contain underlying_price")

    for leg in LEGS:
        strike_col = f"{leg}_strike"
        if strike_col not in out:
            raise ValueError(f"features must contain {strike_col}")
        out[f"{leg}_moneyness"] = out[strike_col].astype(float) / out["underlying_price"].astype(float)
        out[f"{leg}_moneyness_bucket"] = _bucket(out[f"{leg}_moneyness"])

    out["rectangle_moneyness_signature"] = (
        out["A_moneyness_bucket"].astype(str)
        + "-"
        + out["B_moneyness_bucket"].astype(str)
        + "-"
        + out["C_moneyness_bucket"].astype(str)
        + "-"
        + out["D_moneyness_bucket"].astype(str)
    )
    out["rectangle_moneyness_type"] = out.apply(_rectangle_type, axis=1).astype("string")
    out["is_pure_moneyness_rectangle"] = out["rectangle_moneyness_type"].str.startswith("pure_")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Add leg-level and rectangle-level moneyness classes to TP2 features.")
    parser.add_argument("--input", default="data/features/archive2_moneyness/tp2_call_forward_features_all_calls_2013_2026.parquet")
    parser.add_argument("--output", default="data/features/archive2_moneyness/tp2_call_forward_features_all_calls_rect_moneyness_2013_2026.parquet")
    parser.add_argument("--summary-output", default="data/features/archive2_moneyness/tp2_call_forward_summary_all_calls_rect_moneyness_2013_2026.json")
    args = parser.parse_args()

    root = CONFIG.paths.root
    features = read_table(root / args.input)
    out = add_rectangle_moneyness(features)
    write_table(out, root / args.output)

    counts = out["rectangle_moneyness_type"].value_counts(dropna=False).sort_index()
    summary = {
        "input": args.input,
        "output": args.output,
        "rows": int(len(out)),
        "rectangle_moneyness_type_counts": {str(k): int(v) for k, v in counts.items()},
        "signature_counts_top25": {str(k): int(v) for k, v in out["rectangle_moneyness_signature"].value_counts().head(25).items()},
    }
    summary_path = root / args.summary_output
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
