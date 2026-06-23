#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "data/labels/archive2_episode_start_label_summary_2013_2026.json"
TABLE_DIR = ROOT / "reports/raw_archive2_eda/tables/episode_start_labels"
FIG_DIR = ROOT / "reports/figures"

COLORS = {
    "clean_positive_true_reversion": (76, 120, 168),
    "invalid_missing_expired_or_liquidity": (245, 133, 24),
    "clean_negative_still_violation": (178, 121, 162),
    "censored_sample_end": (114, 183, 178),
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


def _grid(draw: ImageDraw.ImageDraw, left: int, top: int, right: int, bottom: int, y_max: float, suffix: str = "") -> None:
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)
    draw.line((left, top, left, bottom), fill=TEXT, width=2)
    for i in range(6):
        y = bottom - (bottom - top) * i / 5
        value = y_max * i / 5
        draw.line((left, y, right, y), fill=GRID, width=1)
        label = f"{value:,.0f}{suffix}" if suffix != "%" else f"{value:.0f}%"
        draw.text((18, y - 10), label, fill=(92, 99, 112), font=_font(15))


def save_quality_counts(summary: dict, path: Path) -> None:
    counts = pd.DataFrame(
        [
            {"label_quality_group": key, "episode_count": value}
            for key, value in summary["label_quality_group_counts"].items()
        ]
    ).sort_values("episode_count", ascending=False)
    img, draw = _canvas(
        "Episode-Start Label Quality",
        "One row per pair episode start; only observable rows receive a binary ML target.",
    )
    left, top, right, bottom = 130, 155, 1430, 690
    y_max = float(counts["episode_count"].max()) * 1.12
    _grid(draw, left, top, right, bottom, y_max)
    slot = (right - left) / len(counts)
    bar_w = slot * 0.55
    for i, row in counts.reset_index(drop=True).iterrows():
        group = row["label_quality_group"]
        color = COLORS.get(group, (150, 150, 150))
        cx = left + slot * i + slot / 2
        y = _bar_y(float(row["episode_count"]), y_max, top, bottom)
        draw.rectangle((cx - bar_w / 2, y, cx + bar_w / 2, bottom), fill=color)
        draw.text((cx - 52, y - 28), f"{int(row['episode_count']):,}", fill=TEXT, font=_font(15))
        label = group.replace("_", "\n")
        for j, part in enumerate(label.split("\n")):
            draw.text((cx - 70, bottom + 14 + j * 18), part, fill=TEXT, font=_font(14))
    img.save(path)


def save_ml_readiness(readiness: pd.DataFrame, path: Path) -> None:
    img, draw = _canvas(
        "Episode-Start ML Readiness By Year",
        "Observable share is the fraction with a clean next-state label; true reversion share is measured over all starts.",
    )
    left, top, right, bottom = 95, 155, 1430, 690
    _grid(draw, left, top, right, bottom, 100, suffix="%")
    years = readiness["year"].astype(str).tolist()
    slot = (right - left) / max(1, len(years) - 1)
    series = [
        ("observable_share", (76, 120, 168), "observable share"),
        ("true_reversion_share_all", (84, 162, 75), "true reversion share, all starts"),
    ]
    for col, color, _ in series:
        values = (readiness[col].astype(float) * 100).tolist()
        points = [(left + i * slot, _bar_y(value, 100, top, bottom)) for i, value in enumerate(values)]
        draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    for i, year in enumerate(years):
        x = left + i * slot
        draw.text((x - 24, bottom + 14), year, fill=TEXT, font=_font(15))
    lx, ly = 955, 46
    for i, (_, color, label) in enumerate(series):
        y = ly + i * 28
        draw.line((lx, y + 8, lx + 34, y + 8), fill=color, width=4)
        draw.text((lx + 44, y - 2), label, fill=TEXT, font=_font(17))
    img.save(path)


def save_yearly_stack(by_year: pd.DataFrame, path: Path) -> None:
    state_to_group = {
        "true_reversion": "clean_positive_true_reversion",
        "still_violation_or_reappeared": "clean_negative_still_violation",
        "sample_ended": "censored_sample_end",
        "left_leg_expired_or_missing": "invalid_missing_expired_or_liquidity",
        "right_leg_missing_or_expired": "invalid_missing_expired_or_liquidity",
        "liquidity_filter_failure": "invalid_missing_expired_or_liquidity",
    }
    grouped = by_year.copy()
    grouped["label_quality_group"] = grouped["label_end_state"].map(state_to_group)
    grouped = grouped.groupby(["start_year", "label_quality_group"], as_index=False)["episode_count"].sum()
    groups = [
        "clean_positive_true_reversion",
        "invalid_missing_expired_or_liquidity",
        "clean_negative_still_violation",
        "censored_sample_end",
    ]
    pivot = grouped.pivot_table(index="start_year", columns="label_quality_group", values="episode_count", fill_value=0)
    for group in groups:
        if group not in pivot:
            pivot[group] = 0
    shares = pivot[groups].div(pivot[groups].sum(axis=1), axis=0) * 100

    img, draw = _canvas(
        "Episode-Start Label Mix By Signal Year",
        "Shares sum to 100% within each episode start year.",
    )
    left, top, right, bottom = 95, 155, 1430, 690
    _grid(draw, left, top, right, bottom, 100, suffix="%")
    years = shares.index.astype(str).tolist()
    slot = (right - left) / len(years)
    bar_w = slot * 0.68
    for i, year in enumerate(years):
        cx = left + slot * i + slot / 2
        y_cursor = bottom
        for group in groups:
            h = float(shares.loc[int(year), group]) / 100 * (bottom - top)
            y_next = y_cursor - h
            draw.rectangle((cx - bar_w / 2, y_next, cx + bar_w / 2, y_cursor), fill=COLORS[group])
            y_cursor = y_next
        draw.text((cx - 24, bottom + 14), year, fill=TEXT, font=_font(15))

    lx, ly = 925, 42
    for i, group in enumerate(groups):
        y = ly + i * 24
        draw.rectangle((lx, y, lx + 20, y + 14), fill=COLORS[group])
        draw.text((lx + 28, y - 3), group.replace("_", " "), fill=TEXT, font=_font(16))
    img.save(path)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text())
    readiness = pd.read_csv(TABLE_DIR / "episode_start_ml_readiness_by_year.csv")
    by_year = pd.read_csv(TABLE_DIR / "episode_start_labels_by_year.csv")

    save_quality_counts(summary, FIG_DIR / "episode_start_label_quality_counts.png")
    save_ml_readiness(readiness, FIG_DIR / "episode_start_ml_readiness_by_year.png")
    save_yearly_stack(by_year, FIG_DIR / "episode_start_label_mix_by_year.png")

    print(f"Episode-start label rows: {summary['episode_start_label_rows']:,}")
    print(f"Observable share: {summary['observable_next_state_share']:.2%}")
    print(f"True reversion share among observable rows: {summary['true_reversion_share_observable']:.2%}")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
