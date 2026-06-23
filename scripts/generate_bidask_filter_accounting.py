#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TEXT = (35, 39, 47)
GRID = (220, 225, 232)
BLUE = (76, 120, 168)
ORANGE = (245, 133, 24)
GREEN = (84, 162, 75)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_yearly_accounting(full_path: Path, conservative_path: Path) -> pd.DataFrame:
    full = pd.read_parquet(full_path, columns=["episode_id", "signal_date"])
    conservative = pd.read_parquet(
        conservative_path,
        columns=["episode_id", "signal_date", "conservative_bidask_normalized_severity"],
    )
    full["year"] = pd.to_datetime(full["signal_date"]).dt.year
    conservative["year"] = pd.to_datetime(conservative["signal_date"]).dt.year

    full_year = (
        full.groupby("year", dropna=False)
        .agg(mid_price_episode_starts=("episode_id", "size"))
        .reset_index()
    )
    conservative_year = (
        conservative.groupby("year", dropna=False)
        .agg(
            conservative_bidask_episode_starts=("episode_id", "size"),
            median_conservative_bidask_severity=("conservative_bidask_normalized_severity", "median"),
            mean_conservative_bidask_severity=("conservative_bidask_normalized_severity", "mean"),
        )
        .reset_index()
    )
    out = full_year.merge(conservative_year, on="year", how="left")
    out["conservative_bidask_episode_starts"] = (
        out["conservative_bidask_episode_starts"].fillna(0).astype(int)
    )
    out["filtered_out_episode_starts"] = (
        out["mid_price_episode_starts"] - out["conservative_bidask_episode_starts"]
    )
    out["conservative_bidask_share"] = (
        out["conservative_bidask_episode_starts"] / out["mid_price_episode_starts"]
    )
    out["filtered_out_share"] = 1.0 - out["conservative_bidask_share"]
    return out.sort_values("year").reset_index(drop=True)


def save_accounting_chart(accounting: pd.DataFrame, path: Path) -> None:
    img = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(img)
    draw.text((60, 35), "Episode TP2 Violation Accounting By Year", fill=TEXT, font=_font(34, True))
    draw.text(
        (60, 82),
        "Mid-price episode starts versus conservative bid/ask executable TP2 subset.",
        fill=(95, 103, 115),
        font=_font(20),
    )

    left, top, right, bottom = 115, 165, 1460, 690
    max_count = float(accounting["mid_price_episode_starts"].max()) * 1.15
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)
    draw.line((left, top, left, bottom), fill=TEXT, width=2)
    for i in range(6):
        y = bottom - (bottom - top) * i / 5
        value = max_count * i / 5
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((22, y - 10), f"{value/1000:.0f}k", fill=(92, 99, 112), font=_font(15))

    years = accounting["year"].astype(str).tolist()
    slot = (right - left) / len(years)
    bar_w = slot * 0.33
    for i, row in accounting.reset_index(drop=True).iterrows():
        cx = left + slot * i + slot / 2
        mid_h = float(row["mid_price_episode_starts"]) / max_count * (bottom - top)
        bid_h = float(row["conservative_bidask_episode_starts"]) / max_count * (bottom - top)
        draw.rectangle((cx - bar_w - 4, bottom - mid_h, cx - 4, bottom), fill=BLUE)
        draw.rectangle((cx + 4, bottom - bid_h, cx + bar_w + 4, bottom), fill=ORANGE)
        draw.text((cx - 24, bottom + 14), years[i], fill=TEXT, font=_font(14))
        share = float(row["conservative_bidask_share"]) * 100
        draw.text((cx - 24, bottom - bid_h - 24), f"{share:.0f}%", fill=TEXT, font=_font(12))

    draw.rectangle((1040, 42, 1062, 58), fill=BLUE)
    draw.text((1070, 36), "mid-price episode starts", fill=TEXT, font=_font(17))
    draw.rectangle((1040, 72, 1062, 88), fill=ORANGE)
    draw.text((1070, 66), "bid/ask-conservative starts", fill=TEXT, font=_font(17))
    draw.text((1040, 102), "Percent labels show kept share.", fill=(95, 103, 115), font=_font(15))
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate yearly mid-vs-bidask TP2 episode accounting.")
    parser.add_argument("--full-labels", default="data/labels/archive2_episode_start_labels_2013_2026.parquet")
    parser.add_argument(
        "--conservative-labels",
        default="data/labels/archive2_episode_start_labels_conservative_bidask_2013_2026.parquet",
    )
    parser.add_argument(
        "--output-csv",
        default="reports/raw_archive2_eda/tables/conservative_bidask_subset/mid_vs_bidask_episode_accounting_by_year.csv",
    )
    parser.add_argument(
        "--output-figure",
        default="reports/figures/mid_vs_bidask_episode_accounting_by_year.png",
    )
    args = parser.parse_args()

    accounting = build_yearly_accounting(ROOT / args.full_labels, ROOT / args.conservative_labels)
    csv_path = ROOT / args.output_csv
    fig_path = ROOT / args.output_figure
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    accounting.to_csv(csv_path, index=False)
    save_accounting_chart(accounting, fig_path)
    print(accounting.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
