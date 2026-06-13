#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tp2rr2.config import CONFIG
from tp2rr2.io import read_table, write_table


BUCKETS = {
    "itm": "moneyness < 0.98",
    "atm": "moneyness >= 0.98 and moneyness <= 1.02",
    "otm": "moneyness > 1.02",
    "all_calls": "moneyness == moneyness",
}


def _bucket_frame(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    if bucket == "itm":
        out = df[df["moneyness"] < 0.98].copy()
    elif bucket == "atm":
        out = df[(df["moneyness"] >= 0.98) & (df["moneyness"] <= 1.02)].copy()
    elif bucket == "otm":
        out = df[df["moneyness"] > 1.02].copy()
    elif bucket == "all_calls":
        out = df.copy()
    else:
        raise ValueError(f"Unknown bucket {bucket!r}")
    out["moneyness_bucket"] = bucket
    return out


def _summary(df: pd.DataFrame, bucket: str, output: Path) -> dict:
    years = pd.to_datetime(df["date"]).dt.year if not df.empty else pd.Series(dtype=int)
    return {
        "bucket": bucket,
        "definition": BUCKETS[bucket],
        "output": str(output),
        "rows": int(len(df)),
        "sessions": int(df["date"].nunique()) if not df.empty else 0,
        "unique_strikes": int(df["strike"].nunique()) if not df.empty else 0,
        "unique_expirations": int(df["expiration"].nunique()) if not df.empty else 0,
        "unique_contracts": int(df["contractID"].nunique()) if "contractID" in df and not df.empty else 0,
        "total_volume": float(df["volume"].fillna(0).sum()) if "volume" in df else None,
        "total_open_interest": float(df["open_interest"].fillna(0).sum()) if "open_interest" in df else None,
        "avg_spread_pct": float(df["spread_pct"].mean()) if "spread_pct" in df and not df.empty else None,
        "rows_by_year": {str(int(k)): int(v) for k, v in df.groupby(years).size().items()} if not df.empty else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Split cleaned Archive-2 SPY calls into moneyness universes.")
    parser.add_argument("--input", default="data/processed/archive2_spy_call_allmoneyness_clean_2013_2026.parquet")
    parser.add_argument("--output-dir", default="data/processed/archive2_moneyness")
    parser.add_argument("--prefix", default="archive2_spy_call")
    args = parser.parse_args()

    root = CONFIG.paths.root
    source = read_table(root / args.input)
    if "moneyness" not in source:
        source = source.copy()
        source["moneyness"] = source["strike"] / source["underlying_price"]

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for bucket in BUCKETS:
        frame = _bucket_frame(source, bucket)
        output = output_dir / f"{args.prefix}_{bucket}_clean_2013_2026.parquet"
        write_table(frame, output)
        summary = _summary(frame, bucket, output)
        summaries.append(summary)
        (output_dir / f"{args.prefix}_{bucket}_clean_2013_2026_summary.json").write_text(json.dumps(summary, indent=2))
        print(
            f"{bucket}: rows={summary['rows']:,} sessions={summary['sessions']:,} "
            f"strikes={summary['unique_strikes']:,} expiries={summary['unique_expirations']:,}",
            flush=True,
        )

    pd.DataFrame(summaries).to_csv(output_dir / "archive2_moneyness_universe_summary.csv", index=False)
    (output_dir / "archive2_moneyness_universe_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"Wrote moneyness universe summaries to {output_dir}")


if __name__ == "__main__":
    main()
