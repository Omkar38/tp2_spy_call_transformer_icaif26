#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = ROOT / "reports" / "raw_archive2_eda"
TABLES = EDA_DIR / "tables"
FIGS = EDA_DIR / "figures"
REPORT_TEX = EDA_DIR / "archive2_raw_data_eda_report.tex"


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


def _money_int(value: float) -> str:
    return f"{value:,.0f}"


def _pct(value: float) -> str:
    return f"{value:.1f}\\%"


def _latex_escape(text: object) -> str:
    s = str(text)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


def _draw_bar_line_chart(
    df: pd.DataFrame,
    x_col: str,
    bar_col: str,
    line_col: str,
    title: str,
    bar_label: str,
    line_label: str,
    out_path: Path,
    bar_color: str = "#2f6f9f",
    line_color: str = "#d17a22",
) -> None:
    width, height = 1800, 900
    ml, mr, mt, mb = 150, 130, 130, 130
    plot_w, plot_h = width - ml - mr, height - mt - mb
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_font = _font(40, True)
    tick_font = _font(23)
    label_font = _font(26, True)
    small_font = _font(22)

    years = df[x_col].astype(int).tolist()
    bars = df[bar_col].astype(float).tolist()
    lines = df[line_col].astype(float).tolist()
    max_bar = max(bars) * 1.12
    min_line, max_line = min(lines), max(lines)
    pad_line = (max_line - min_line) * 0.15 if max_line > min_line else 1
    min_line -= pad_line
    max_line += pad_line

    d.text(((width - d.textbbox((0, 0), title, font=title_font)[2]) / 2, 40), title, fill="#111", font=title_font)
    d.rounded_rectangle((ml, mt, ml + plot_w, mt + plot_h), radius=18, fill="#fbfbfb", outline="#d0d0d0", width=2)

    for i in range(6):
        y = mt + plot_h - i * plot_h / 5
        d.line((ml, y, ml + plot_w, y), fill="#e5e5e5", width=1)
        label = f"{max_bar * i / 5 / 1_000_000:.1f}M"
        bbox = d.textbbox((0, 0), label, font=tick_font)
        d.text((ml - 20 - (bbox[2] - bbox[0]), y - 12), label, fill="#333", font=tick_font)

    n = len(years)
    gap = plot_w / n
    bar_w = gap * 0.58
    points = []
    for i, (year, bar, line) in enumerate(zip(years, bars, lines)):
        cx = ml + gap * i + gap / 2
        bh = bar / max_bar * plot_h
        d.rounded_rectangle((cx - bar_w / 2, mt + plot_h - bh, cx + bar_w / 2, mt + plot_h), radius=7, fill=bar_color)
        label = str(year)
        bbox = d.textbbox((0, 0), label, font=tick_font)
        d.text((cx - (bbox[2] - bbox[0]) / 2, mt + plot_h + 25), label, fill="#222", font=tick_font)
        y_line = mt + plot_h - (line - min_line) / (max_line - min_line) * plot_h
        points.append((cx, y_line))

    for a, b in zip(points, points[1:]):
        d.line((a[0], a[1], b[0], b[1]), fill=line_color, width=6)
    for x, y in points:
        d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=line_color, outline="white", width=3)

    d.rounded_rectangle((ml + 25, mt + 25, ml + 560, mt + 100), radius=14, fill="white", outline="#cccccc")
    d.rectangle((ml + 45, mt + 47, ml + 83, mt + 72), fill=bar_color)
    d.text((ml + 98, mt + 38), bar_label, fill="#222", font=small_font)
    d.line((ml + 315, mt + 60, ml + 365, mt + 60), fill=line_color, width=6)
    d.ellipse((ml + 335, mt + 52, ml + 353, mt + 70), fill=line_color, outline="white", width=2)
    d.text((ml + 380, mt + 38), line_label, fill="#222", font=small_font)
    d.text((ml + plot_w / 2 - 45, height - 65), "Year", fill="#111", font=label_font)
    d.text((35, mt + plot_h / 2 - 20), "Rows", fill="#111", font=label_font)
    img.save(out_path)


def _draw_multi_line_chart(df: pd.DataFrame, cols: list[str], labels: list[str], title: str, out_path: Path) -> None:
    width, height = 1800, 900
    ml, mr, mt, mb = 140, 80, 130, 120
    plot_w, plot_h = width - ml - mr, height - mt - mb
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_font = _font(40, True)
    tick_font = _font(23)
    small_font = _font(22, True)
    years = df["year"].astype(int).tolist()
    values = [df[c].astype(float).tolist() for c in cols]
    max_v = max(max(v) for v in values) * 1.15
    min_v = min(min(v) for v in values)
    min_v = min(0, min_v)
    colors = ["#2f6f9f", "#c74632", "#4d8f3a", "#7b4fa1"]

    d.text(((width - d.textbbox((0, 0), title, font=title_font)[2]) / 2, 40), title, fill="#111", font=title_font)
    d.rounded_rectangle((ml, mt, ml + plot_w, mt + plot_h), radius=18, fill="#fbfbfb", outline="#d0d0d0", width=2)
    for i in range(6):
        y = mt + plot_h - i * plot_h / 5
        d.line((ml, y, ml + plot_w, y), fill="#e5e5e5", width=1)
        label = f"{max_v * i / 5:.0f}%"
        bbox = d.textbbox((0, 0), label, font=tick_font)
        d.text((ml - 20 - (bbox[2] - bbox[0]), y - 12), label, fill="#333", font=tick_font)

    n = len(years)
    for ci, vals in enumerate(values):
        pts = []
        for i, val in enumerate(vals):
            x = ml + i * plot_w / (n - 1)
            y = mt + plot_h - (val - min_v) / (max_v - min_v) * plot_h
            pts.append((x, y))
        for a, b in zip(pts, pts[1:]):
            d.line((a[0], a[1], b[0], b[1]), fill=colors[ci], width=6)
        for x, y in pts:
            d.ellipse((x - 8, y - 8, x + 8, y + 8), fill=colors[ci], outline="white", width=2)

    for i, year in enumerate(years):
        x = ml + i * plot_w / (n - 1)
        label = str(year)
        bbox = d.textbbox((0, 0), label, font=tick_font)
        d.text((x - (bbox[2] - bbox[0]) / 2, mt + plot_h + 22), label, fill="#222", font=tick_font)

    lx, ly = ml + 30, mt + 25
    for ci, label in enumerate(labels):
        d.line((lx, ly + ci * 40, lx + 48, ly + ci * 40), fill=colors[ci], width=6)
        d.text((lx + 65, ly + ci * 40 - 14), label, fill="#222", font=small_font)
    img.save(out_path)


def _draw_breadth_chart(df: pd.DataFrame, out_path: Path) -> None:
    width, height = 1800, 900
    ml, mr, mt, mb = 150, 80, 130, 130
    plot_w, plot_h = width - ml - mr, height - mt - mb
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_font = _font(40, True)
    tick_font = _font(23)
    small_font = _font(22, True)
    years = df["year"].astype(int).tolist()
    cols = ["avg_contracts_per_day", "avg_strikes_per_day", "avg_expirations_per_day"]
    labels = ["Avg contracts/day", "Avg strikes/day", "Avg expirations/day"]
    colors = ["#2f6f9f", "#d17a22", "#4d8f3a"]
    vals = [df[c].astype(float).tolist() for c in cols]
    max_v = max(max(v) for v in vals) * 1.12

    d.text(((width - d.textbbox((0, 0), "Option Chain Breadth by Year", font=title_font)[2]) / 2, 40), "Option Chain Breadth by Year", fill="#111", font=title_font)
    d.rounded_rectangle((ml, mt, ml + plot_w, mt + plot_h), radius=18, fill="#fbfbfb", outline="#d0d0d0", width=2)
    for i in range(6):
        y = mt + plot_h - i * plot_h / 5
        d.line((ml, y, ml + plot_w, y), fill="#e5e5e5", width=1)
        label = f"{max_v * i / 5:,.0f}"
        bbox = d.textbbox((0, 0), label, font=tick_font)
        d.text((ml - 20 - (bbox[2] - bbox[0]), y - 12), label, fill="#333", font=tick_font)

    n = len(years)
    for ci, series in enumerate(vals):
        pts = []
        for i, val in enumerate(series):
            x = ml + i * plot_w / (n - 1)
            y = mt + plot_h - val / max_v * plot_h
            pts.append((x, y))
        for a, b in zip(pts, pts[1:]):
            d.line((a[0], a[1], b[0], b[1]), fill=colors[ci], width=6)
        for x, y in pts:
            d.ellipse((x - 8, y - 8, x + 8, y + 8), fill=colors[ci], outline="white", width=2)

    for i, year in enumerate(years):
        x = ml + i * plot_w / (n - 1)
        label = str(year)
        bbox = d.textbbox((0, 0), label, font=tick_font)
        d.text((x - (bbox[2] - bbox[0]) / 2, mt + plot_h + 22), label, fill="#222", font=tick_font)

    lx, ly = ml + 30, mt + 25
    for ci, label in enumerate(labels):
        d.line((lx, ly + ci * 40, lx + 48, ly + ci * 40), fill=colors[ci], width=6)
        d.text((lx + 65, ly + ci * 40 - 14), label, fill="#222", font=small_font)
    img.save(out_path)


def _maybe_presence_summary() -> tuple[int, int]:
    path = TABLES / "raw_skipped_dates_presence_check_fast.csv"
    if not path.exists():
        return 0, 0
    df = pd.read_csv(path)
    present = int(df["present_in_raw_daily_summary"].astype(bool).sum()) if "present_in_raw_daily_summary" in df else 0
    absent = int(len(df) - present)
    return present, absent


def _table_rows(df: pd.DataFrame, cols: list[str], fmts: dict[str, str] | None = None) -> str:
    fmts = fmts or {}
    lines = []
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if c in fmts:
                vals.append(fmts[c].format(v))
            elif isinstance(v, float):
                vals.append(f"{v:,.2f}")
            elif isinstance(v, int):
                vals.append(f"{v:,}")
            else:
                vals.append(_latex_escape(v))
        lines.append(" & ".join(vals) + r" \\")
    return "\n".join(lines)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    year = pd.read_csv(TABLES / "raw_yearly_summary.csv")
    dup = pd.read_csv(TABLES / "raw_duplicate_summary.csv")
    miss = pd.read_csv(TABLES / "raw_missingness_summary.csv")
    qt = pd.read_csv(TABLES / "raw_quote_time_summary.csv")
    mat = pd.read_csv(TABLES / "raw_dte_bucket_summary.csv")
    overall = json.loads((TABLES / "raw_eda_overall_summary.json").read_text())
    detailed_summary_path = TABLES / "raw_detailed_eda_summary.json"
    detailed = json.loads(detailed_summary_path.read_text()) if detailed_summary_path.exists() else {}
    skipped_sessions = pd.read_csv(TABLES / "raw_missing_or_skipped_sessions.csv") if (TABLES / "raw_missing_or_skipped_sessions.csv").exists() else pd.DataFrame()
    top_dup_dates = pd.read_csv(TABLES / "raw_top_duplicate_dates.csv") if (TABLES / "raw_top_duplicate_dates.csv").exists() else pd.DataFrame()
    dup_examples = pd.read_csv(TABLES / "raw_duplicate_examples_top.csv") if (TABLES / "raw_duplicate_examples_top.csv").exists() else pd.DataFrame()
    top_invalid_dates = pd.read_csv(TABLES / "raw_top_invalid_bidask_dates.csv") if (TABLES / "raw_top_invalid_bidask_dates.csv").exists() else pd.DataFrame()
    top_wide_dates = pd.read_csv(TABLES / "raw_top_wide_spread_dates.csv") if (TABLES / "raw_top_wide_spread_dates.csv").exists() else pd.DataFrame()
    top_missing_col_dates = pd.read_csv(TABLES / "raw_top_missing_column_dates.csv") if (TABLES / "raw_top_missing_column_dates.csv").exists() else pd.DataFrame()
    clean_versions = pd.read_csv(TABLES / "clean_dataset_version_comparison.csv") if (TABLES / "clean_dataset_version_comparison.csv").exists() else pd.DataFrame()
    tp2_year = pd.read_csv(TABLES / "lastquote_tp2_violation_counts_by_year.csv") if (TABLES / "lastquote_tp2_violation_counts_by_year.csv").exists() else pd.DataFrame()
    tp2_month = pd.read_csv(TABLES / "lastquote_tp2_violation_counts_by_month.csv") if (TABLES / "lastquote_tp2_violation_counts_by_month.csv").exists() else pd.DataFrame()
    skipped_present, skipped_absent = _maybe_presence_summary()

    _draw_bar_line_chart(
        year,
        "year",
        "rows",
        "sessions",
        "Raw Row Count and Trading Sessions",
        "Rows",
        "Sessions",
        FIGS / "raw_rows_sessions_by_year.png",
    )
    _draw_multi_line_chart(
        year,
        ["valid_bidask_pct", "zero_bid_pct", "wide_spread_50pct_pct"],
        ["Valid bid/ask", "Zero bid", "Spread > 50%"],
        "Quote Quality by Year",
        FIGS / "raw_quote_quality_by_year.png",
    )
    _draw_breadth_chart(year, FIGS / "raw_chain_breadth_by_year.png")

    miss_pivot = (
        miss[miss["column"].isin(["implied_vol", "delta", "theta", "vega", "open_interest", "risk_free_rate"])]
        .pivot(index="year", columns="column", values="missing_pct")
        .reset_index()
    )
    qt_summary = qt.pivot_table(index="year", columns="quote_time_bucket", values="pct", aggfunc="sum").fillna(0).reset_index()
    dte_total = mat.groupby("dte_bucket", dropna=False)["rows"].sum().reset_index()
    dte_total["pct"] = dte_total["rows"] / dte_total["rows"].sum() * 100
    skipped_preview = skipped_sessions.copy()
    if not skipped_preview.empty:
        skipped_preview["date"] = skipped_preview["date"].astype(str)
        skipped_preview = skipped_preview.head(80)
    top_dup_dates = top_dup_dates.head(15).copy()
    top_invalid_dates = top_invalid_dates.head(15).copy()
    top_wide_dates = top_wide_dates.head(15).copy()
    top_missing_col_dates = top_missing_col_dates.head(20).copy()
    dup_examples = dup_examples.head(20).copy()
    tp2_month_top = tp2_month.sort_values("violation_rows", ascending=False).head(24).copy() if not tp2_month.empty else pd.DataFrame()

    summary_table = year[
        [
            "year",
            "rows",
            "sessions",
            "unique_option_symbols",
            "unique_expirations",
            "unique_strikes",
            "avg_contracts_per_day",
            "valid_bidask_pct",
            "zero_bid_pct",
            "wide_spread_50pct_pct",
        ]
    ].copy()
    int_formats = {
        "year": "{:.0f}",
        "rows": "{:,.0f}",
        "sessions": "{:,.0f}",
        "unique_option_symbols": "{:,.0f}",
        "unique_expirations": "{:,.0f}",
        "unique_strikes": "{:,.0f}",
        "duplicate_contract_date_pairs": "{:,.0f}",
        "duplicate_extra_rows": "{:,.0f}",
        "max_rows_same_contract_date": "{:,.0f}",
    }

    manifest_rows = overall.get("rows_total_manifest")
    observed_rows = overall.get("rows_total_observed")
    manifest_text = (
        f"The manifest reports {manifest_rows:,} rows, while the parquet files currently contain {observed_rows:,} rows."
        if isinstance(manifest_rows, int)
        else f"The parquet files currently contain {observed_rows:,} rows."
    )

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{graphicx}}
\usepackage{{array}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{6pt}}

\title{{Archive-2 Raw SPY Options Data EDA}}
\author{{ML-Based Trading of TP2-Inspired Violations in SPY Call Options}}
\date{{Generated from professor-provided raw parquet files}}

\begin{{document}}
\maketitle

\section*{{Purpose}}

This report describes the raw professor-provided Archive-2 dataset before our
project-specific cleaning, TP2 detection, labeling, or ML backtest. The goal is
to understand what data exists, how broad the option chain is, and which raw
data issues must be controlled before using the data for publishable research.

\section*{{Raw Source}}

\begin{{tabular}}{{ll}}
\toprule
Directory & \texttt{{Archive-2/spy\_options\_daily/year=*/data.parquet}} \\
Dataset in manifest & {_latex_escape(overall.get("dataset"))} \\
Parent symbol & {_latex_escape(overall.get("parent_symbol"))} \\
Date range in files & {overall.get("first_date")} to {overall.get("last_date")} \\
Observed trading sessions & {overall.get("sessions_total"):,} \\
Observed raw rows & {observed_rows:,} \\
Manifest raw rows & {manifest_rows:,} \\
Known vendor gap & {_latex_escape(", ".join(overall.get("known_gaps", [])))} \\
Skipped-day list size & {overall.get("skipped_days_count"):,} dates; {len(overall.get("known_gaps", [])):,} separate known vendor gap \\
\bottomrule
\end{{tabular}}

\textbf{{Important note.}} {manifest_text} The report uses the actual parquet
file contents as the source of truth. The difference should be disclosed as a
metadata inconsistency, not as a trading signal.

\section*{{What One Raw Row Means}}

One raw row is an option quote observation for a SPY option contract on a quote
date. The row contains contract identity, strike, expiration, option type,
bid/ask/mid, quote size, quote timestamp, volume, open interest, SPY close,
dividend metadata, implied volatility, Greeks, forward, and risk-free rate.
The raw files contain multiple rows for many \texttt{{quote\_date,
contract\_id}} pairs. Therefore, raw row count is not the same thing as
contract-day count. Our final pipeline collapses these raw observations to one
latest-available quote per contract-day before TP2 detection.

\section*{{Yearly Coverage and Breadth}}

\begin{{center}}
\includegraphics[width=0.98\textwidth]{{figures/raw_rows_sessions_by_year.png}}
\end{{center}}

\begin{{center}}
\includegraphics[width=0.98\textwidth]{{figures/raw_chain_breadth_by_year.png}}
\end{{center}}

\begingroup
\scriptsize
\begin{{longtable}}{{rrrrrrrrrr}}
\toprule
Year & Rows & Sessions & Symbols & Exp. & Strikes & Avg contracts/day & Valid BA & Zero bid & Wide 50 \\
\midrule
{_table_rows(summary_table, ["year", "rows", "sessions", "unique_option_symbols", "unique_expirations", "unique_strikes", "avg_contracts_per_day", "valid_bidask_pct", "zero_bid_pct", "wide_spread_50pct_pct"], {**int_formats, "avg_contracts_per_day": "{:,.0f}", "valid_bidask_pct": "{:.1f}\\%", "zero_bid_pct": "{:.1f}\\%", "wide_spread_50pct_pct": "{:.1f}\\%"})}
\bottomrule
\end{{longtable}}
\endgroup

\section*{{Quote Quality}}

\begin{{center}}
\includegraphics[width=0.98\textwidth]{{figures/raw_quote_quality_by_year.png}}
\end{{center}}

Basic bid/ask validity is high in most years, generally above 94\%, but it
drops in 2023 and remains lower in 2026. Wide spreads and zero bids are
material for option data and must be filtered before constructing arbitrage
signals. Crossed quotes are rare, but they exist and should be excluded.

\begingroup
\scriptsize
\begin{{longtable}}{{rrrrr}}
\toprule
Year & Duplicate contract-date pairs & Extra raw rows & Extra raw rows pct & Max rows / contract-day \\
\midrule
{_table_rows(dup, ["year", "duplicate_contract_date_pairs", "duplicate_extra_rows", "duplicate_extra_rows_pct", "max_rows_same_contract_date"], {**int_formats, "duplicate_extra_rows_pct": "{:.1f}\\%"})}
\bottomrule
\end{{longtable}}
\endgroup

The duplicate table should be read as \textbf{{multiple raw observations per
contract-day}}, not automatically as bad data. It explains why an ingestion
step is necessary: TP2 detection needs one consistent quote per contract-day.

\section*{{Detailed Duplicate Diagnostics}}

Across the raw parquet files, duplicate contract-day observations appear on
{detailed.get("dates_with_duplicate_contract_days", 0):,} quote dates. The
total number of extra duplicate rows is {detailed.get("total_duplicate_extra_rows", 0):,}.
These are not deleted from the professor data; our ingestion writes a separate
clean research table that collapses each \texttt{{quote\_date, contract\_id}}
to one final contract-day quote.

\begingroup
\scriptsize
\begin{{longtable}}{{rrrrr}}
\toprule
Date & Raw rows & Duplicate pairs & Extra duplicate rows & Max rows same contract-day \\
\midrule
{_table_rows(top_dup_dates, ["date", "raw_rows", "duplicate_contract_date_pairs", "duplicate_extra_rows", "max_rows_same_contract_date"], {"raw_rows": "{:,.0f}", "duplicate_contract_date_pairs": "{:,.0f}", "duplicate_extra_rows": "{:,.0f}", "max_rows_same_contract_date": "{:,.0f}"}) if not top_dup_dates.empty else "No duplicate dates found."}
\bottomrule
\end{{longtable}}
\endgroup

\begingroup
\scriptsize
\begin{{longtable}}{{rrrrr}}
\toprule
Year & Date & Contract ID & Option symbol & Raw rows same contract-day \\
\midrule
{_table_rows(dup_examples, ["year", "date", "contract_id", "option_symbol", "raw_rows_same_contract_date"], {"year": "{:.0f}", "contract_id": "{:.0f}", "raw_rows_same_contract_date": "{:.0f}"}) if not dup_examples.empty else "No duplicate examples found."}
\bottomrule
\end{{longtable}}
\endgroup

\section*{{Quote Time Distribution}}

\begingroup
\scriptsize
\begin{{longtable}}{{rrrrrr}}
\toprule
Year & 09:30--10:00 & 10:00--12:00 & 12:00--15:00 & 15:00--15:55 & 15:55--16:00 \\
\midrule
{_table_rows(qt_summary, ["year", "0930_1000", "1000_1200", "1200_1500", "1500_1555", "1555_1600"], {"year": "{:.0f}", "0930_1000": "{:.1f}\\%", "1000_1200": "{:.1f}\\%", "1200_1500": "{:.1f}\\%", "1500_1555": "{:.1f}\\%", "1555_1600": "{:.1f}\\%"})}
\bottomrule
\end{{longtable}}
\endgroup

The raw archive is not a single clean close snapshot. Many rows have quote
times before the final five minutes, and some rows have missing quote-time
bucket values. After reviewing this behavior, the active clean dataset now uses
the latest available quote per contract-day rather than a hard 15:55 cutoff.

\section*{{Missing or Skipped Sessions}}

The raw archive metadata lists {len(skipped_sessions):,} skipped or missing
session dates. Some are normal early-close dates, and one is explicitly listed
as a known vendor gap. A direct raw-date presence check found {skipped_present:,}
of these dates present and {skipped_absent:,} absent in the raw daily summary,
with no 15:55 filter applied. The project does not fabricate missing sessions.

\begingroup
\scriptsize
\begin{{longtable}}{{rll}}
\toprule
Date & Known vendor gap & Note \\
\midrule
{_table_rows(skipped_preview, ["date", "known_vendor_gap", "note"]) if not skipped_preview.empty else "No skipped sessions listed."}
\bottomrule
\end{{longtable}}
\endgroup

\section*{{Date-Level Quote Problems}}

The next tables identify the most affected dates in the raw archive. These are
diagnostic tables for data understanding; the final trading pipeline handles
them through quote cleaning, bid/ask filters, and invalid-exit accounting.

\begingroup
\scriptsize
\begin{{longtable}}{{rrrrr}}
\toprule
Date & Raw rows & Invalid bid/ask rows & Invalid bid/ask pct & Zero bid pct \\
\midrule
{_table_rows(top_invalid_dates, ["date", "raw_rows", "invalid_bidask_rows", "invalid_bidask_pct", "zero_bid_pct"], {"raw_rows": "{:,.0f}", "invalid_bidask_rows": "{:,.0f}", "invalid_bidask_pct": "{:.1f}\\%", "zero_bid_pct": "{:.1f}\\%"}) if not top_invalid_dates.empty else "No invalid bid/ask dates found."}
\bottomrule
\end{{longtable}}
\endgroup

\begingroup
\scriptsize
\begin{{longtable}}{{rrrrr}}
\toprule
Date & Raw rows & Wide 50 rows & Wide 50 pct & Crossed rows \\
\midrule
{_table_rows(top_wide_dates, ["date", "raw_rows", "wide_spread_50pct_rows", "wide_spread_50pct_pct", "crossed_rows"], {"raw_rows": "{:,.0f}", "wide_spread_50pct_rows": "{:,.0f}", "wide_spread_50pct_pct": "{:.1f}\\%", "crossed_rows": "{:,.0f}"}) if not top_wide_dates.empty else "No wide-spread dates found."}
\bottomrule
\end{{longtable}}
\endgroup

\section*{{Maturity Distribution}}

\begin{{center}}
\begin{{tabular}}{{lrr}}
\toprule
DTE bucket & Rows & Percent \\
\midrule
{_table_rows(dte_total, ["dte_bucket", "rows", "pct"], {"rows": "{:,.0f}", "pct": "{:.1f}\\%"})}
\bottomrule
\end{{tabular}}
\end{{center}}

The archive covers short-dated and long-dated SPY options. Our TP2 research
pipeline restricts the usable universe to a controlled DTE range so that
illiquid long tails and near-expiration edge cases do not dominate the signal.

\section*{{Missingness in Model-Relevant Columns}}

\begingroup
\scriptsize
\begin{{longtable}}{{rrrrrrr}}
\toprule
Year & IV miss & Delta miss & Theta miss & Vega miss & OI miss & Rate miss \\
\midrule
{_table_rows(miss_pivot, ["year", "implied_vol", "delta", "theta", "vega", "open_interest", "risk_free_rate"], {"year": "{:.0f}", "implied_vol": "{:.1f}\\%", "delta": "{:.1f}\\%", "theta": "{:.1f}\\%", "vega": "{:.1f}\\%", "open_interest": "{:.1f}\\%", "risk_free_rate": "{:.1f}\\%"})}
\bottomrule
\end{{longtable}}
\endgroup

Greeks and implied volatility are useful model features, but they are not
always available. They should be treated as optional/derived features with
missingness controls, not as mandatory raw fields for defining option prices or
TP2 violations.

\section*{{Date-Level Missingness}}

At least one key field is missing on {detailed.get("dates_with_any_missing_key_column", 0):,}
raw sessions. This is mostly because implied volatility, Greeks, risk-free
rate, and open interest are not complete for every contract. Bid, ask, mid,
volume, and SPY close are much more complete and are the required fields for
pricing and TP2 construction.

\begingroup
\scriptsize
\begin{{longtable}}{{rlrrr}}
\toprule
Date & Column & Raw rows & Missing rows & Missing pct \\
\midrule
{_table_rows(top_missing_col_dates, ["date", "column", "raw_rows", "missing_rows", "missing_pct"], {"raw_rows": "{:,.0f}", "missing_rows": "{:,.0f}", "missing_pct": "{:.1f}\\%"}) if not top_missing_col_dates.empty else "No key-column missingness found."}
\bottomrule
\end{{longtable}}
\endgroup

\section*{{Clean Dataset Version for Next Pipeline}}

After reviewing the raw quote-time distribution, the next pipeline version
should use the latest available raw quote for each contract-day rather than
requiring the quote time to be at or after 15:55. The data provider appears to
store the latest scraped quote available for that contract-day, even when the
timestamp is earlier than 15:55. Therefore the new clean dataset keeps the
last available quote after duplicate collapse, while preserving the same
research filters: SPY calls only, DTE 1--90, valid bid/ask, positive bid, and
spread percentage no greater than 50\%.

\begingroup
\scriptsize
\begin{{longtable}}{{lrrrrrr}}
\toprule
Dataset version & Rows & Sessions & Duplicate rows & Call rows & Min DTE & Max DTE \\
\midrule
{_table_rows(clean_versions, ["dataset_version", "rows", "sessions", "duplicate_date_contract_rows", "call_rows", "min_dte", "max_dte"], {"rows": "{:,.0f}", "sessions": "{:,.0f}", "duplicate_date_contract_rows": "{:,.0f}", "call_rows": "{:,.0f}", "min_dte": "{:.0f}", "max_dte": "{:.0f}"}) if not clean_versions.empty else "No clean dataset comparison found."}
\bottomrule
\end{{longtable}}
\endgroup

The active reconstructed file for the next structural pipeline work is:
\begin{{quote}}
\small\path{{data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet}}
\end{{quote}}

\section*{{TP2 Inequality Audit Before Violation Charts}}

Before adding violation counts by year or month, we verified the call-side TP2
condition against the Glasserman--Li--Pirjol formulation. For calls with
$T_1<T_2$ and $K_1/F_{{T_1}}<K_2/F_{{T_2}}$, the no-violation condition is

\[
C(K_1,T_1)C(K_2,T_2)
\geq
C(K_1F_{{T_2}}/F_{{T_1}},T_2)C(K_2F_{{T_1}}/F_{{T_2}},T_1).
\]

Therefore, the call violation condition is

\[
\mathrm{{LHS}} < \mathrm{{RHS}}.
\]

The current forward-adjusted detector implements exactly this direction:
\texttt{{lhs = A\_mid * B\_mid}},
\texttt{{rhs = C\_mid * D\_mid}}, and keeps call rows with
\texttt{{lhs < rhs}} and positive normalized severity. The January 2--5, 2024
smoke test on the last-available clean dataset produced 661 violation rows;
all 661 satisfied \texttt{{lhs < rhs}}, all satisfied
$K_1/F_{{T_1}}<K_2/F_{{T_2}}$, and all had $T_1<T_2$. This means yearly and monthly
violation charts can be added after running the full detector on the new active
dataset.

\section*{{TP2 Violation Counts on Active Last-Available Dataset}}

The full forward-adjusted TP2 detector was run on the active clean dataset:
\begin{{quote}}
\small\path{{data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet}}
\end{{quote}}

Detector settings were all SPY calls, DTE 1--90, positive bid, valid bid/ask,
spread percentage no greater than 50\%, latest available quote per
contract-day, maximum 80 strikes per expiration, and minimum normalized
severity of 0.0001. The detector output contains 2,430,220 daily TP2 violation
feature rows across 2013--2026. A validation pass over the 14 yearly parquet
outputs found zero rows with \texttt{{lhs >= rhs}}, zero rows with invalid
expiration ordering, and zero duplicate \texttt{{violation\_id}} values.

\textbf{{Interpretation.}} These are daily rectangle-level violation rows, not
unique arbitrage episodes and not executed trades. The same economic rectangle
can appear on multiple adjacent trading days. The next structural pipeline step
should therefore convert daily rows into episode-level observations before ML
labeling and trading backtests.

\begin{{center}}
\includegraphics[width=0.98\textwidth]{{figures/lastquote_tp2_violations_by_year.png}}
\end{{center}}

\begin{{center}}
\includegraphics[width=0.98\textwidth]{{figures/lastquote_tp2_violations_by_month.png}}
\end{{center}}

\begingroup
\scriptsize
\begin{{longtable}}{{rrrrrrr}}
\toprule
Year & Clean rows & Sessions & Violation rows & Violation sessions & Rows/session & Rows/100k clean \\
\midrule
{_table_rows(tp2_year, ["year", "clean_contract_day_rows", "clean_sessions", "violation_rows", "sessions_with_violations", "violation_rows_per_session", "violation_rows_per_100k_clean_rows"], {"year": "{:.0f}", "clean_contract_day_rows": "{:,.0f}", "clean_sessions": "{:,.0f}", "violation_rows": "{:,.0f}", "sessions_with_violations": "{:,.0f}", "violation_rows_per_session": "{:,.1f}", "violation_rows_per_100k_clean_rows": "{:,.0f}"}) if not tp2_year.empty else "TP2 year table not generated."}
\bottomrule
\end{{longtable}}
\endgroup

\begingroup
\scriptsize
\begin{{longtable}}{{rrrrrr}}
\toprule
Month & Clean rows & Sessions & Violation rows & Violation sessions & Median severity \\
\midrule
{_table_rows(tp2_month_top, ["month", "clean_contract_day_rows", "clean_sessions", "violation_rows", "violation_sessions", "median_normalized_severity"], {"clean_contract_day_rows": "{:,.0f}", "clean_sessions": "{:,.0f}", "violation_rows": "{:,.0f}", "violation_sessions": "{:,.0f}", "median_normalized_severity": "{:.4f}"}) if not tp2_month_top.empty else "TP2 month table not generated."}
\bottomrule
\end{{longtable}}
\endgroup

The table above lists the 24 highest-count months. The complete monthly table is
written to:
\begin{{quote}}
\small\path{{reports/raw_archive2_eda/tables/lastquote_tp2_violation_counts_by_month.csv}}
\end{{quote}}

\section*{{Project Takeaways}}

\begin{{enumerate}}
  \item The raw Archive-2 data is broad enough for the project: it spans 2013--2026,
  includes calls and puts, and grows substantially in option-chain breadth over
  time.
  \item Raw row count is not equal to final research row count. Multiple raw
  quote observations per contract-day must be reduced to one quote before TP2
  rectangle construction.
  \item Quote quality filters are necessary. Zero bids, wide spreads, locked or
  crossed quotes, stale quote times, and missing exit quotes can otherwise
  create false arbitrage opportunities.
  \item The dataset is suitable for publishable research only after transparent
  cleaning and disclosure. The raw metadata discrepancy between manifest row
  count and observed parquet row count should be mentioned in internal data
  notes.
  \item For the final American-call TP2 project, the required clean dataset is
  one latest-available, valid bid/ask quote per SPY call contract-day, with strike,
  expiration, quote prices, volume/open interest, SPY close, and dividend
  metadata preserved.
\end{{enumerate}}

\section*{{Generated Files}}

\begin{{itemize}}
  \item \texttt{{reports/raw\_archive2\_eda/tables/raw\_yearly\_summary.csv}}
  \item \texttt{{reports/raw\_archive2\_eda/tables/raw\_daily\_summary.csv}}
  \item \texttt{{reports/raw\_archive2\_eda/tables/raw\_option\_type\_summary.csv}}
  \item \texttt{{reports/raw\_archive2\_eda/tables/raw\_dte\_bucket\_summary.csv}}
  \item \texttt{{reports/raw\_archive2\_eda/tables/raw\_missingness\_summary.csv}}
  \item \texttt{{reports/raw\_archive2\_eda/tables/raw\_quote\_time\_summary.csv}}
  \item \texttt{{reports/raw\_archive2\_eda/tables/raw\_duplicate\_summary.csv}}
  \item \texttt{{reports/raw\_archive2\_eda/tables/raw\_duplicate\_by\_date.csv}}
  \item \texttt{{reports/raw\_archive2\_eda/tables/raw\_duplicate\_examples\_top.csv}}
  \item \texttt{{reports/raw\_archive2\_eda/tables/raw\_missingness\_by\_date.csv}}
  \item \texttt{{reports/raw\_archive2\_eda/tables/raw\_quote\_quality\_by\_date.csv}}
  \item \texttt{{reports/raw\_archive2\_eda/tables/raw\_missing\_or\_skipped\_sessions.csv}}
  \item \texttt{{reports/raw\_archive2\_eda/tables/clean\_dataset\_version\_comparison.csv}}
  \item \texttt{{reports/raw\_archive2\_eda/tables/lastquote\_tp2\_violation\_counts\_by\_year.csv}}
  \item \texttt{{reports/raw\_archive2\_eda/tables/lastquote\_tp2\_violation\_counts\_by\_month.csv}}
\end{{itemize}}

\end{{document}}
"""
    REPORT_TEX.write_text(tex)
    print(REPORT_TEX)


if __name__ == "__main__":
    main()
