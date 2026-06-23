#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EPISODE_DIR = ROOT / "data" / "features" / "archive2_episodes"
TABLE_DIR = ROOT / "reports" / "raw_archive2_eda" / "tables" / "episodes"
FIG_DIR = ROOT / "reports" / "figures"

BLUE = (72, 116, 168)
ORANGE = (245, 133, 24)
GREEN = (84, 162, 75)
RED = (228, 87, 86)
PURPLE = (178, 121, 162)
GRID = (218, 224, 232)
TEXT = (35, 39, 47)


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


def _draw_axes(draw: ImageDraw.ImageDraw, left: int, top: int, right: int, bottom: int, y_max: float, y_label: str) -> None:
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)
    draw.line((left, top, left, bottom), fill=TEXT, width=2)
    for i in range(6):
        y = bottom - (bottom - top) * i / 5
        value = y_max * i / 5
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((15, y - 10), f"{value:,.0f}", fill=(92, 99, 112), font=_font(15))
    draw.text((left, top - 34), y_label, fill=TEXT, font=_font(17, bold=True))


def _bar_y(value: float, y_max: float, top: int, bottom: int) -> float:
    return bottom - (value / y_max) * (bottom - top)


def _save_rows_vs_episodes(yearly: pd.DataFrame, path: Path) -> None:
    img, draw = _canvas(
        "Daily TP2 Violation Rows Versus Pair Episodes",
        "Episode tracking groups consecutive appearances of the same A/B pair; no daily rows are deleted.",
    )
    left, top, right, bottom = 95, 145, 1430, 705
    y_max = float(yearly[["daily_violation_rows", "episode_starts"]].max().max()) * 1.12
    _draw_axes(draw, left, top, right, bottom, y_max, "Count")

    years = yearly["year"].astype(str).tolist()
    slot = (right - left) / len(years)
    bar_w = slot * 0.28
    for i, row in yearly.reset_index(drop=True).iterrows():
        cx = left + slot * i + slot / 2
        y1 = _bar_y(row["daily_violation_rows"], y_max, top, bottom)
        y2 = _bar_y(row["episode_starts"], y_max, top, bottom)
        draw.rectangle((cx - bar_w, y1, cx - 2, bottom), fill=BLUE)
        draw.rectangle((cx + 2, y2, cx + bar_w, bottom), fill=ORANGE)
        draw.text((cx - 26, bottom + 12), years[i], fill=TEXT, font=_font(15))

    draw.rectangle((1015, 72, 1040, 91), fill=BLUE)
    draw.text((1050, 68), "Daily violation rows", fill=TEXT, font=_font(18))
    draw.rectangle((1215, 72, 1240, 91), fill=ORANGE)
    draw.text((1250, 68), "Pair episode starts", fill=TEXT, font=_font(18))
    img.save(path)


def _save_duration_distribution(duration: pd.DataFrame, path: Path) -> None:
    img, draw = _canvas(
        "Pair Episode Duration Distribution",
        "Most pair episodes are one-session events; the y-axis is log-scaled to show the long tail.",
    )
    left, top, right, bottom = 115, 145, 1430, 705
    max_log = math.log10(float(duration["episode_count"].max()))
    _draw_axes(draw, left, top, right, bottom, max_log, "log10(Episode count)")

    slot = (right - left) / len(duration)
    bar_w = slot * 0.58
    for i, row in duration.reset_index(drop=True).iterrows():
        value = math.log10(float(row["episode_count"]))
        y = _bar_y(value, max_log, top, bottom)
        cx = left + slot * i + slot / 2
        draw.rectangle((cx - bar_w / 2, y, cx + bar_w / 2, bottom), fill=(114, 183, 178))
        draw.text((cx - 9, bottom + 12), str(int(row["duration_sessions"])), fill=TEXT, font=_font(16))
    draw.text((left, bottom + 48), "Duration in trading sessions", fill=TEXT, font=_font(18, bold=True))
    img.save(path)


def _polyline_points(values: list[float], left: int, top: int, right: int, bottom: int, y_max: float) -> list[tuple[float, float]]:
    slot = (right - left) / max(1, len(values) - 1)
    return [(left + i * slot, _bar_y(value, y_max, top, bottom)) for i, value in enumerate(values)]


def _save_quality_diagnostics(yearly: pd.DataFrame, path: Path) -> None:
    plot = yearly.copy()
    plot["single_day_share"] = plot["single_day_episodes"] / plot["episode_starts"] * 100
    plot["rectangle_change_share"] = plot["exact_rectangle_change_episodes"] / plot["episode_starts"] * 100

    img, draw = _canvas(
        "Episode Composition Diagnostics By Year",
        "Single-day share describes persistence; rectangle-change share captures forward-rounding movement inside pair episodes.",
    )
    left, top, right, bottom = 95, 145, 1430, 705
    y_max = 100.0
    _draw_axes(draw, left, top, right, bottom, y_max, "Share of pair episodes (%)")

    years = plot["year"].astype(str).tolist()
    pts1 = _polyline_points(plot["single_day_share"].tolist(), left, top, right, bottom, y_max)
    pts2 = _polyline_points(plot["rectangle_change_share"].tolist(), left, top, right, bottom, y_max)
    draw.line(pts1, fill=RED, width=4)
    draw.line(pts2, fill=PURPLE, width=4)
    for x, y in pts1:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=RED)
    for x, y in pts2:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=PURPLE)
    for i, year in enumerate(years):
        x = left + i * (right - left) / max(1, len(years) - 1)
        draw.text((x - 24, bottom + 12), year, fill=TEXT, font=_font(15))

    draw.line((910, 72, 950, 72), fill=RED, width=4)
    draw.text((960, 62), "Single-day episode share", fill=TEXT, font=_font(18))
    draw.line((1160, 72, 1200, 72), fill=PURPLE, width=4)
    draw.text((1210, 62), "Rectangle-change share", fill=TEXT, font=_font(18))
    img.save(path)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads((EPISODE_DIR / "tp2_call_episode_summary_2013_2026.json").read_text())
    yearly = pd.read_csv(TABLE_DIR / "episode_starts_by_year.csv")
    duration = pd.read_csv(TABLE_DIR / "episode_duration_distribution.csv")

    _save_rows_vs_episodes(yearly, FIG_DIR / "episode_rows_vs_pair_episodes_by_year.png")
    _save_duration_distribution(duration, FIG_DIR / "episode_duration_distribution.png")
    _save_quality_diagnostics(yearly, FIG_DIR / "episode_quality_diagnostics_by_year.png")

    report_table = yearly[
        [
            "year",
            "daily_violation_rows",
            "episode_starts",
            "daily_rows_per_episode_start",
            "median_duration_sessions",
            "single_day_episodes",
            "exact_rectangle_change_episodes",
        ]
    ].copy()
    report_table["daily_rows_per_episode_start"] = report_table["daily_rows_per_episode_start"].round(3)
    report_table.to_csv(FIG_DIR / "episode_yearly_report_table.csv", index=False)

    print(f"Daily violation rows: {summary['daily_violation_rows']:,}")
    print(f"Pair episodes: {summary['episode_rows']:,}")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
