#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "raw_archive2_eda" / "tables"
FIGS = ROOT / "reports" / "raw_archive2_eda" / "figures"
SUMMARY_PATH = ROOT / "data" / "features" / "archive2_lastquote_forward" / "tp2_call_forward_summary_lastquote_2013_2026.json"
FEATURES_PATH = ROOT / "data" / "features" / "archive2_lastquote_forward" / "tp2_call_forward_features_lastquote_2013_2026.parquet"
CLEAN_PATH = ROOT / "data" / "processed" / "archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet"


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_year_chart(df: pd.DataFrame, out_path: Path) -> None:
    width, height = 1800, 900
    ml, mr, mt, mb = 150, 130, 125, 135
    plot_w, plot_h = width - ml - mr, height - mt - mb
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_font = _font(40, True)
    tick_font = _font(23)
    small_font = _font(22)
    label_font = _font(26, True)
    bar_color = "#2f6f9f"
    line_color = "#c74632"

    years = df["year"].astype(int).tolist()
    bars = df["violation_rows"].astype(float).tolist()
    rates = df["violation_rows_per_100k_clean_rows"].astype(float).tolist()
    max_bar = max(bars) * 1.12
    min_rate, max_rate = min(rates), max(rates)
    rate_pad = (max_rate - min_rate) * 0.12 if max_rate > min_rate else 1.0
    min_rate -= rate_pad
    max_rate += rate_pad

    title = "TP2 Violation Rows by Year on Active Last-Available Dataset"
    d.text(((width - d.textbbox((0, 0), title, font=title_font)[2]) / 2, 38), title, fill="#111", font=title_font)
    d.rounded_rectangle((ml, mt, ml + plot_w, mt + plot_h), radius=18, fill="#fbfbfb", outline="#d0d0d0", width=2)

    for i in range(6):
        y = mt + plot_h - i * plot_h / 5
        d.line((ml, y, ml + plot_w, y), fill="#e5e5e5", width=1)
        label = f"{max_bar * i / 5 / 1000:.0f}k"
        bbox = d.textbbox((0, 0), label, font=tick_font)
        d.text((ml - 20 - (bbox[2] - bbox[0]), y - 12), label, fill="#333", font=tick_font)

    n = len(years)
    gap = plot_w / n
    bar_w = gap * 0.58
    line_points = []
    for i, (year, bar, rate) in enumerate(zip(years, bars, rates)):
        cx = ml + gap * i + gap / 2
        bh = bar / max_bar * plot_h
        d.rounded_rectangle((cx - bar_w / 2, mt + plot_h - bh, cx + bar_w / 2, mt + plot_h), radius=7, fill=bar_color)
        label = str(year)
        bbox = d.textbbox((0, 0), label, font=tick_font)
        d.text((cx - (bbox[2] - bbox[0]) / 2, mt + plot_h + 25), label, fill="#222", font=tick_font)
        y_rate = mt + plot_h - (rate - min_rate) / (max_rate - min_rate) * plot_h
        line_points.append((cx, y_rate))

    for a, b in zip(line_points, line_points[1:]):
        d.line((a[0], a[1], b[0], b[1]), fill=line_color, width=6)
    for x, y in line_points:
        d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=line_color, outline="white", width=3)

    d.rounded_rectangle((ml + 25, mt + 25, ml + 665, mt + 105), radius=14, fill="white", outline="#cccccc")
    d.rectangle((ml + 45, mt + 50, ml + 83, mt + 74), fill=bar_color)
    d.text((ml + 98, mt + 40), "Violation rows", fill="#222", font=small_font)
    d.line((ml + 325, mt + 62, ml + 375, mt + 62), fill=line_color, width=6)
    d.ellipse((ml + 342, mt + 54, ml + 360, mt + 72), fill=line_color, outline="white", width=2)
    d.text((ml + 392, mt + 40), "Rows per 100k clean rows", fill="#222", font=small_font)
    d.text((ml + plot_w / 2 - 45, height - 65), "Year", fill="#111", font=label_font)
    d.text((35, mt + plot_h / 2 - 20), "Rows", fill="#111", font=label_font)
    img.save(out_path)


def _draw_month_chart(df: pd.DataFrame, out_path: Path) -> None:
    width, height = 2200, 900
    ml, mr, mt, mb = 150, 70, 125, 150
    plot_w, plot_h = width - ml - mr, height - mt - mb
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_font = _font(40, True)
    tick_font = _font(20)
    label_font = _font(26, True)
    bar_color = "#2f6f9f"

    values = df["violation_rows"].astype(float).tolist()
    labels = df["month"].astype(str).tolist()
    max_v = max(values) * 1.12
    title = "Monthly TP2 Violation Rows"
    d.text(((width - d.textbbox((0, 0), title, font=title_font)[2]) / 2, 38), title, fill="#111", font=title_font)
    d.rounded_rectangle((ml, mt, ml + plot_w, mt + plot_h), radius=18, fill="#fbfbfb", outline="#d0d0d0", width=2)

    for i in range(6):
        y = mt + plot_h - i * plot_h / 5
        d.line((ml, y, ml + plot_w, y), fill="#e5e5e5", width=1)
        label = f"{max_v * i / 5 / 1000:.0f}k"
        bbox = d.textbbox((0, 0), label, font=tick_font)
        d.text((ml - 20 - (bbox[2] - bbox[0]), y - 10), label, fill="#333", font=tick_font)

    n = len(values)
    gap = plot_w / max(n, 1)
    bar_w = max(3, gap * 0.74)
    for i, value in enumerate(values):
        cx = ml + gap * i + gap / 2
        bh = value / max_v * plot_h if max_v else 0
        d.rectangle((cx - bar_w / 2, mt + plot_h - bh, cx + bar_w / 2, mt + plot_h), fill=bar_color)
        if i % 12 == 0:
            label = labels[i][:4]
            bbox = d.textbbox((0, 0), label, font=tick_font)
            d.text((cx - (bbox[2] - bbox[0]) / 2, mt + plot_h + 22), label, fill="#222", font=tick_font)

    d.text((ml + plot_w / 2 - 55, height - 65), "Month", fill="#111", font=label_font)
    d.text((35, mt + plot_h / 2 - 20), "Rows", fill="#111", font=label_font)
    img.save(out_path)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    summary = json.loads(SUMMARY_PATH.read_text())
    year = pd.DataFrame(summary["years"])
    year = year.rename(
        columns={
            "rows_input": "clean_contract_day_rows",
            "sessions_input": "clean_sessions",
            "rows_features": "violation_rows",
        }
    )
    year["violation_rows_per_session"] = year["violation_rows"] / year["clean_sessions"]
    year["violation_rows_per_100k_clean_rows"] = year["violation_rows"] / year["clean_contract_day_rows"] * 100_000
    year["partial_year"] = year["year"].eq(2026)
    year_out = year[
        [
            "year",
            "clean_contract_day_rows",
            "clean_sessions",
            "violation_rows",
            "sessions_with_violations",
            "violation_rows_per_session",
            "violation_rows_per_100k_clean_rows",
            "partial_year",
        ]
    ].copy()
    year_out.to_csv(TABLES / "lastquote_tp2_violation_counts_by_year.csv", index=False)

    clean_dates = pd.read_parquet(CLEAN_PATH, columns=["date"])
    clean_dates["date"] = pd.to_datetime(clean_dates["date"])
    clean_dates["month"] = clean_dates["date"].dt.to_period("M").astype(str)
    clean_month = clean_dates.groupby("month").agg(clean_contract_day_rows=("date", "size"), clean_sessions=("date", "nunique")).reset_index()

    feat = pd.read_parquet(FEATURES_PATH, columns=["date", "normalized_severity"])
    feat["date"] = pd.to_datetime(feat["date"])
    feat["month"] = feat["date"].dt.to_period("M").astype(str)
    viol_month = (
        feat.groupby("month")
        .agg(
            violation_rows=("date", "size"),
            violation_sessions=("date", "nunique"),
            mean_normalized_severity=("normalized_severity", "mean"),
            median_normalized_severity=("normalized_severity", "median"),
        )
        .reset_index()
    )
    month = clean_month.merge(viol_month, on="month", how="left")
    month[["violation_rows", "violation_sessions"]] = month[["violation_rows", "violation_sessions"]].fillna(0).astype(int)
    month["violation_rows_per_session"] = month["violation_rows"] / month["clean_sessions"]
    month["violation_rows_per_100k_clean_rows"] = month["violation_rows"] / month["clean_contract_day_rows"] * 100_000
    month.to_csv(TABLES / "lastquote_tp2_violation_counts_by_month.csv", index=False)

    _draw_year_chart(year_out, FIGS / "lastquote_tp2_violations_by_year.png")
    _draw_month_chart(month, FIGS / "lastquote_tp2_violations_by_month.png")

    print(TABLES / "lastquote_tp2_violation_counts_by_year.csv")
    print(TABLES / "lastquote_tp2_violation_counts_by_month.csv")
    print(FIGS / "lastquote_tp2_violations_by_year.png")
    print(FIGS / "lastquote_tp2_violations_by_month.png")


if __name__ == "__main__":
    main()
