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
GREEN = (84, 162, 75)
RED = (228, 87, 86)
ORANGE = (245, 133, 24)
PURPLE = (178, 121, 162)


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


def _canvas(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (1500, 850), "white")
    draw = ImageDraw.Draw(img)
    draw.text((60, 35), title, fill=TEXT, font=_font(34, bold=True))
    if subtitle:
        draw.text((60, 82), subtitle, fill=(95, 103, 115), font=_font(20))
    return img, draw


def _y(value: float, low: float, high: float, top: int, bottom: int) -> float:
    if high == low:
        return (top + bottom) / 2
    return bottom - ((value - low) / (high - low)) * (bottom - top)


def save_valid_share_and_return(by_strategy: pd.DataFrame, path: Path, subtitle: str) -> None:
    img, draw = _canvas(
        "Episode Mini-Backtest By Strategy",
        subtitle,
    )
    left, top, right, bottom = 115, 160, 1420, 675
    strategies = by_strategy["strategy"].tolist()
    slot = (right - left) / len(strategies)
    bar_w = slot * 0.32
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)
    draw.line((left, top, left, bottom), fill=TEXT, width=2)
    for i in range(6):
        y = bottom - (bottom - top) * i / 5
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((38, y - 10), f"{20*i}%", fill=(92, 99, 112), font=_font(15))

    for i, row in by_strategy.reset_index(drop=True).iterrows():
        cx = left + slot * i + slot / 2
        valid_h = float(row["valid_share"]) * (bottom - top)
        y1 = bottom - valid_h
        draw.rectangle((cx - bar_w - 8, y1, cx - 8, bottom), fill=BLUE)
        ret = float(row["median_return"]) * 100
        ret_y = _y(ret, -20, 5, top, bottom)
        draw.rectangle((cx + 8, min(ret_y, bottom), cx + bar_w + 8, max(ret_y, bottom)), fill=RED)
        draw.text((cx - 26, bottom + 15), row["strategy"], fill=TEXT, font=_font(16, bold=True))
        draw.text((cx - 70, y1 - 24), f"{row['valid_share']*100:.1f}%", fill=TEXT, font=_font(14))
        draw.text((cx + 8, ret_y - 22), f"{ret:.1f}%", fill=TEXT, font=_font(14))

    draw.rectangle((980, 44, 1000, 58), fill=BLUE)
    draw.text((1008, 39), "valid share", fill=TEXT, font=_font(16))
    draw.rectangle((980, 72, 1000, 86), fill=RED)
    draw.text((1008, 67), "median return", fill=TEXT, font=_font(16))
    img.save(path)


def save_exit_reason_counts(counts: pd.DataFrame, path: Path, subtitle: str) -> None:
    img, draw = _canvas(
        "Mini-Backtest Exit Reason Counts",
        subtitle,
    )
    counts = counts.sort_values("strategy_rows", ascending=False).reset_index(drop=True)
    left, top, right, bottom = 145, 160, 1430, 675
    y_max = float(counts["strategy_rows"].max()) * 1.12
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)
    draw.line((left, top, left, bottom), fill=TEXT, width=2)
    for i in range(6):
        y = bottom - (bottom - top) * i / 5
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((25, y - 10), f"{y_max*i/5/1_000_000:.1f}M", fill=(92, 99, 112), font=_font(15))
    colors = [GREEN, ORANGE, PURPLE, RED, BLUE, (150, 150, 150), (114, 183, 178)]
    slot = (right - left) / len(counts)
    bar_w = slot * 0.55
    for i, row in counts.iterrows():
        cx = left + slot * i + slot / 2
        y = bottom - float(row["strategy_rows"]) / y_max * (bottom - top)
        draw.rectangle((cx - bar_w / 2, y, cx + bar_w / 2, bottom), fill=colors[i % len(colors)])
        draw.text((cx - 42, y - 25), f"{row['strategy_rows']/1_000_000:.2f}M", fill=TEXT, font=_font(14))
        for j, part in enumerate(row["exit_reason"].replace("_", "\n").split("\n")):
            draw.text((cx - 58, bottom + 12 + j * 17), part, fill=TEXT, font=_font(13))
    img.save(path)


def save_yearly_net_pnl(by_year: pd.DataFrame, path: Path, subtitle: str) -> None:
    yearly = by_year.groupby("year", as_index=False)["net_pnl"].sum()
    img, draw = _canvas(
        "Episode Mini-Backtest Net P&L By Signal Year",
        subtitle,
    )
    left, top, right, bottom = 105, 155, 1430, 690
    low = min(float(yearly["net_pnl"].min()), 0.0)
    high = max(float(yearly["net_pnl"].max()), 0.0)
    pad = (high - low) * 0.08 if high > low else 1.0
    low -= pad
    high += pad
    zero_y = _y(0, low, high, top, bottom)
    draw.line((left, zero_y, right, zero_y), fill=TEXT, width=2)
    draw.line((left, top, left, bottom), fill=TEXT, width=2)
    for i in range(6):
        value = low + (high - low) * i / 5
        y = _y(value, low, high, top, bottom)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((20, y - 10), f"{value/1_000_000:.0f}M", fill=(92, 99, 112), font=_font(15))
    years = yearly["year"].astype(str).tolist()
    slot = (right - left) / len(years)
    bar_w = slot * 0.65
    for i, row in yearly.reset_index(drop=True).iterrows():
        cx = left + slot * i + slot / 2
        y = _y(float(row["net_pnl"]), low, high, top, bottom)
        color = GREEN if row["net_pnl"] >= 0 else RED
        draw.rectangle((cx - bar_w / 2, min(y, zero_y), cx + bar_w / 2, max(y, zero_y)), fill=color)
        draw.text((cx - 24, bottom + 14), years[i], fill=TEXT, font=_font(15))
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate episode mini-backtest figures.")
    parser.add_argument(
        "--table-dir",
        default="reports/raw_archive2_eda/tables/episode_mini_backtest",
    )
    parser.add_argument("--figure-dir", default="reports/figures")
    parser.add_argument("--prefix", default="episode_mini_backtest")
    parser.add_argument(
        "--subtitle",
        default="Conservative policy: enter at episode start, exit only on clean true reversion.",
    )
    parser.add_argument(
        "--yearly-subtitle",
        default="Active strategies combined; valid true-reversion exits only.",
    )
    args = parser.parse_args()

    table_dir = ROOT / args.table_dir
    figure_dir = ROOT / args.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    by_strategy = pd.read_csv(table_dir / "episode_mini_backtest_by_strategy.csv")
    counts = pd.read_csv(table_dir / "episode_mini_backtest_exit_reason_counts.csv")
    by_year = pd.read_csv(table_dir / "episode_mini_backtest_by_year_strategy.csv")
    save_valid_share_and_return(by_strategy, figure_dir / f"{args.prefix}_by_strategy.png", args.subtitle)
    save_exit_reason_counts(counts, figure_dir / f"{args.prefix}_exit_reasons.png", "Invalid or unobservable outcomes are retained instead of silently dropped.")
    save_yearly_net_pnl(by_year, figure_dir / f"{args.prefix}_yearly_net_pnl.png", args.yearly_subtitle)
    print(f"Wrote mini-backtest figures to {figure_dir}")


if __name__ == "__main__":
    main()
