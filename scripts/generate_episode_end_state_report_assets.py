#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "data/features/archive2_episodes/tp2_call_pair_episode_end_state_summary_2013_2026.json"
TABLE_DIR = ROOT / "reports/raw_archive2_eda/tables/episode_end_states"
FIG_DIR = ROOT / "reports/figures"

COLORS = {
    "true_reversion": (76, 120, 168),
    "right_leg_missing_or_expired": (245, 133, 24),
    "liquidity_filter_failure": (84, 162, 75),
    "left_leg_expired_or_missing": (228, 87, 86),
    "still_violation_or_reappeared": (178, 121, 162),
    "sample_ended": (114, 183, 178),
}
TEXT = (35, 39, 47)
GRID = (220, 225, 232)


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


def _bar_y(value: float, y_max: float, top: int, bottom: int) -> float:
    return bottom - (value / y_max) * (bottom - top)


def _draw_count_grid(draw: ImageDraw.ImageDraw, left: int, top: int, right: int, bottom: int, y_max: float) -> None:
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)
    draw.line((left, top, left, bottom), fill=TEXT, width=2)
    for i in range(6):
        y = bottom - (bottom - top) * i / 5
        value = y_max * i / 5
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((15, y - 10), f"{value:,.0f}", fill=(92, 99, 112), font=_font(15))


def save_overall_counts(counts: pd.DataFrame, path: Path) -> None:
    img, draw = _canvas(
        "Episode End-State Classification Counts",
        "Each pair episode receives exactly one next-session end-state label.",
    )
    left, top, right, bottom = 130, 155, 1430, 690
    counts = counts.sort_values("episode_count", ascending=False).reset_index(drop=True)
    y_max = float(counts["episode_count"].max()) * 1.12
    _draw_count_grid(draw, left, top, right, bottom, y_max)

    slot = (right - left) / len(counts)
    bar_w = slot * 0.55
    for i, row in counts.iterrows():
        state = row["end_state"]
        color = COLORS.get(state, (150, 150, 150))
        cx = left + slot * i + slot / 2
        y = _bar_y(float(row["episode_count"]), y_max, top, bottom)
        draw.rectangle((cx - bar_w / 2, y, cx + bar_w / 2, bottom), fill=color)
        draw.text((cx - 38, y - 28), f"{int(row['episode_count']):,}", fill=TEXT, font=_font(15))
        label = state.replace("_", "\n")
        for j, part in enumerate(label.split("\n")):
            draw.text((cx - 54, bottom + 14 + j * 18), part, fill=TEXT, font=_font(14))
    img.save(path)


def save_yearly_true_reversion(by_year: pd.DataFrame, path: Path) -> None:
    pivot = by_year.pivot_table(index="year", columns="end_state", values="episode_count", fill_value=0, aggfunc="sum")
    totals = pivot.sum(axis=1)
    true_share = (pivot.get("true_reversion", 0) / totals * 100).reset_index(name="true_reversion_share")

    img, draw = _canvas(
        "True Reversion Share By Episode End Year",
        "True reversion requires all four next-session legs to be observable and lhs >= rhs.",
    )
    left, top, right, bottom = 95, 155, 1430, 690
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)
    draw.line((left, top, left, bottom), fill=TEXT, width=2)
    for i in range(6):
        y = bottom - (bottom - top) * i / 5
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((32, y - 10), f"{20*i}%", fill=(92, 99, 112), font=_font(15))

    values = true_share["true_reversion_share"].tolist()
    years = true_share["year"].astype(str).tolist()
    slot = (right - left) / max(1, len(values) - 1)
    points = [(left + i * slot, _bar_y(v, 100, top, bottom)) for i, v in enumerate(values)]
    draw.line(points, fill=COLORS["true_reversion"], width=4)
    for i, (x, y) in enumerate(points):
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=COLORS["true_reversion"])
        draw.text((x - 24, bottom + 14), years[i], fill=TEXT, font=_font(15))
    img.save(path)


def save_yearly_stacked(by_year: pd.DataFrame, path: Path) -> None:
    states = [
        "true_reversion",
        "right_leg_missing_or_expired",
        "liquidity_filter_failure",
        "left_leg_expired_or_missing",
        "still_violation_or_reappeared",
        "sample_ended",
    ]
    pivot = by_year.pivot_table(index="year", columns="end_state", values="episode_count", fill_value=0, aggfunc="sum")
    for state in states:
        if state not in pivot:
            pivot[state] = 0
    shares = pivot[states].div(pivot[states].sum(axis=1), axis=0) * 100

    img, draw = _canvas(
        "Episode End-State Mix By Year",
        "Shares sum to 100% within each episode end year.",
    )
    left, top, right, bottom = 95, 155, 1430, 690
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)
    draw.line((left, top, left, bottom), fill=TEXT, width=2)
    for i in range(6):
        y = bottom - (bottom - top) * i / 5
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((32, y - 10), f"{20*i}%", fill=(92, 99, 112), font=_font(15))

    years = shares.index.astype(str).tolist()
    slot = (right - left) / len(years)
    bar_w = slot * 0.68
    for i, year in enumerate(years):
        cx = left + slot * i + slot / 2
        y_cursor = bottom
        for state in states:
            h = float(shares.loc[int(year), state]) / 100 * (bottom - top)
            y_next = y_cursor - h
            draw.rectangle((cx - bar_w / 2, y_next, cx + bar_w / 2, y_cursor), fill=COLORS[state])
            y_cursor = y_next
        draw.text((cx - 24, bottom + 14), year, fill=TEXT, font=_font(15))

    lx, ly = 930, 40
    for i, state in enumerate(states):
        y = ly + i * 24
        draw.rectangle((lx, y, lx + 20, y + 14), fill=COLORS[state])
        draw.text((lx + 28, y - 3), state.replace("_", " "), fill=TEXT, font=_font(16))
    img.save(path)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text())
    counts = pd.read_csv(TABLE_DIR / "end_state_counts.csv")
    by_year = pd.read_csv(TABLE_DIR / "end_state_counts_by_year.csv")

    save_overall_counts(counts, FIG_DIR / "episode_end_state_counts.png")
    save_yearly_true_reversion(by_year, FIG_DIR / "episode_end_state_true_reversion_share_by_year.png")
    save_yearly_stacked(by_year, FIG_DIR / "episode_end_state_mix_by_year.png")

    print(f"Episode rows: {summary['episode_rows']:,}")
    print(f"True reversion share: {summary['true_reversion_share']:.2%}")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
