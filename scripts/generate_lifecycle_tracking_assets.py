#!/usr/bin/env python3
"""Generate lifecycle tracking tables, figures, and PDF report."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from tp2rr2.config import CONFIG


def _money(value: float) -> str:
    if pd.isna(value):
        return "NA"
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    if value >= 1e6:
        return f"{sign}${value/1e6:.2f}M"
    if value >= 1e3:
        return f"{sign}${value/1e3:.1f}K"
    return f"{sign}${value:.0f}"


def _plot_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    fig.text(0.06, 0.93, title, fontsize=20, fontweight="bold", ha="left", va="top")
    y = 0.86
    for line in lines:
        if line.startswith("## "):
            y -= 0.035
            fig.text(0.06, y, line[3:], fontsize=13, fontweight="bold", ha="left", va="top")
            y -= 0.035
        elif line.strip() == "":
            y -= 0.025
        else:
            fig.text(0.07, y, line, fontsize=10.5, ha="left", va="top")
            y -= 0.026
        if y < 0.08:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(11, 8.5))
            plt.axis("off")
            y = 0.93
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _save_current(fig, pdf: PdfPages, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lifecycle tracking assets.")
    parser.add_argument(
        "--input",
        default="data/features/archive2_lifecycle/tp2_conservative_episode_lifecycle_2013_2026.parquet",
    )
    parser.add_argument(
        "--summary-input",
        default="data/features/archive2_lifecycle/tp2_conservative_episode_lifecycle_summary_2013_2026.json",
    )
    parser.add_argument(
        "--tables-dir",
        default="reports/raw_archive2_eda/tables/lifecycle_tracking",
    )
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--pdf-output", default="reports/lifecycle_tracking_report.pdf")
    args = parser.parse_args()

    root = CONFIG.paths.root
    input_path = root / args.input
    summary_path = root / args.summary_input
    tables_dir = root / args.tables_dir
    figures_dir = root / args.figures_dir
    pdf_path = root / args.pdf_output
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "episode_id",
        "signal_date",
        "lifecycle_date",
        "event_day_index",
        "days_to_T1",
        "status",
        "label_end_state",
        "normalized_pressure",
        "log_pressure",
        "violation_pressure",
        "is_violation",
        "is_conservative_bidask_violation",
        "all_legs_tradable_under_50pct_spread",
        "product_is_valid_backtest_T1",
        "product_is_valid_backtest_K1",
        "product_net_pnl_T1",
        "product_net_pnl_K1",
        "product_realized_return_T1",
        "product_realized_return_K1",
        "unit_is_valid_backtest_T1",
        "unit_is_valid_backtest_K1",
        "unit_net_pnl_T1",
        "unit_net_pnl_K1",
        "unit_realized_return_T1",
        "unit_realized_return_K1",
    ]
    df = pd.read_parquet(input_path, columns=cols)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["lifecycle_date"] = pd.to_datetime(df["lifecycle_date"])
    df["has_pressure"] = df["normalized_pressure"].notna()
    df["pressure_group"] = np.where(
        df["normalized_pressure"].gt(0), "violating", "non_violating"
    )
    df.loc[~df["has_pressure"], "pressure_group"] = "not_evaluable"

    for strat in ["T1", "K1"]:
        valid_col = f"product_is_valid_backtest_{strat}"
        pnl_col = f"product_net_pnl_{strat}"
        group_col = f"{strat}_outcome_group"
        df[group_col] = "invalid_or_unobservable"
        df.loc[df[valid_col].fillna(False) & df[pnl_col].gt(0), group_col] = f"{strat}_profitable"
        df.loc[df[valid_col].fillna(False) & df[pnl_col].le(0), group_col] = f"{strat}_unprofitable"

    # Event-time pressure summary.
    event = (
        df.groupby("event_day_index")
        .agg(
            lifecycle_rows=("episode_id", "size"),
            unique_episodes=("episode_id", "nunique"),
            pressure_rows=("has_pressure", "sum"),
            median_pressure=("normalized_pressure", "median"),
            mean_pressure=("normalized_pressure", "mean"),
            p10_pressure=("normalized_pressure", lambda x: x.quantile(0.10)),
            p25_pressure=("normalized_pressure", lambda x: x.quantile(0.25)),
            p75_pressure=("normalized_pressure", lambda x: x.quantile(0.75)),
            p90_pressure=("normalized_pressure", lambda x: x.quantile(0.90)),
            violation_share=("is_violation", "mean"),
            conservative_violation_share=("is_conservative_bidask_violation", "mean"),
            tradable_50pct_share=("all_legs_tradable_under_50pct_spread", "mean"),
        )
        .reset_index()
    )
    event.to_csv(tables_dir / "lifecycle_event_time_pressure_summary.csv", index=False)

    status_counts = (
        df.groupby(["event_day_index", "status"])
        .size()
        .rename("rows")
        .reset_index()
    )
    status_counts["share"] = status_counts["rows"] / status_counts.groupby("event_day_index")[
        "rows"
    ].transform("sum")
    status_counts.to_csv(tables_dir / "lifecycle_status_by_event_day.csv", index=False)

    outcome = (
        df[df["has_pressure"]]
        .groupby(["event_day_index", "T1_outcome_group"])
        .agg(
            rows=("episode_id", "size"),
            median_pressure=("normalized_pressure", "median"),
            p25_pressure=("normalized_pressure", lambda x: x.quantile(0.25)),
            p75_pressure=("normalized_pressure", lambda x: x.quantile(0.75)),
            violation_share=("is_violation", "mean"),
        )
        .reset_index()
    )
    outcome.to_csv(tables_dir / "lifecycle_pressure_by_T1_outcome.csv", index=False)

    episode_level = (
        df[df["event_day_index"].eq(0)]
        .drop_duplicates("episode_id")
        .copy()
    )
    episode_summary = {
        "episodes": int(episode_level["episode_id"].nunique()),
        "T1_valid": int(episode_level["product_is_valid_backtest_T1"].fillna(False).sum()),
        "T1_profitable": int(
            (
                episode_level["product_is_valid_backtest_T1"].fillna(False)
                & episode_level["product_net_pnl_T1"].gt(0)
            ).sum()
        ),
        "K1_valid": int(episode_level["product_is_valid_backtest_K1"].fillna(False).sum()),
        "K1_profitable": int(
            (
                episode_level["product_is_valid_backtest_K1"].fillna(False)
                & episode_level["product_net_pnl_K1"].gt(0)
            ).sum()
        ),
        "T1_net_pnl": float(episode_level["product_net_pnl_T1"].sum()),
        "K1_net_pnl": float(episode_level["product_net_pnl_K1"].sum()),
    }
    (tables_dir / "lifecycle_episode_level_summary.json").write_text(
        json.dumps(episode_summary, indent=2)
    )

    status_pivot = status_counts.pivot(
        index="event_day_index", columns="status", values="share"
    ).fillna(0)
    status_order = [
        "observable",
        "liquidity_failure_wide_spread",
        "missing_A_or_B",
        "missing_C_or_D",
        "missing_forward",
        "not_forward_ordered",
    ]
    status_pivot = status_pivot[[c for c in status_order if c in status_pivot.columns]]

    with PdfPages(pdf_path) as pdf:
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        _plot_text_page(
            pdf,
            "TP2 Violation Lifecycle Tracking",
            [
                "## What This Report Tracks",
                "One row is one conservative bid/ask TP2 episode observed on one lifecycle date.",
                "Event day 0 is the first violation day. Negative event days are pre-violation.",
                "Forward tracking stops at T1 expiration because the short-maturity leg disappears.",
                "",
                "## Core Counts",
                f"Episodes tracked: {summary.get('episode_count', episode_level['episode_id'].nunique()):,}",
                f"Lifecycle rows: {summary.get('lifecycle_rows', len(df)):,}",
                f"Event-day range: {summary.get('event_day_min', int(df['event_day_index'].min()))} to {summary.get('event_day_max', int(df['event_day_index'].max()))}",
                f"Rows with evaluable pressure: {int(df['has_pressure'].sum()):,}",
                f"Rows with raw TP2 violation pressure > 0: {int(df['is_violation'].fillna(False).sum()):,}",
                f"Rows with conservative bid/ask violation: {int(df['is_conservative_bidask_violation'].fillna(False).sum()):,}",
                "",
                "## P&L Context",
                f"T1 valid product-weighted episodes: {episode_summary['T1_valid']:,}",
                f"T1 profitable product-weighted episodes: {episode_summary['T1_profitable']:,}",
                f"T1 product-weighted net P&L: {_money(episode_summary['T1_net_pnl'])}",
                f"K1 valid product-weighted episodes: {episode_summary['K1_valid']:,}",
                f"K1 profitable product-weighted episodes: {episode_summary['K1_profitable']:,}",
                f"K1 product-weighted net P&L: {_money(episode_summary['K1_net_pnl'])}",
                "",
                "## Reading The Charts",
                "Normalized pressure = (rhs - lhs) / (rhs + lhs).",
                "Positive pressure means TP2 violation. Negative pressure means safe margin.",
                "Median paths show typical behavior; percentile bands show dispersion.",
            ],
        )

        fig, ax = plt.subplots(figsize=(11, 6))
        e = event.sort_values("event_day_index")
        ax.fill_between(
            e["event_day_index"],
            e["p10_pressure"],
            e["p90_pressure"],
            alpha=0.15,
            label="10-90%",
        )
        ax.fill_between(
            e["event_day_index"],
            e["p25_pressure"],
            e["p75_pressure"],
            alpha=0.25,
            label="25-75%",
        )
        ax.plot(e["event_day_index"], e["median_pressure"], color="black", lw=2, label="Median")
        ax.axhline(0, color="red", lw=1, linestyle="--")
        ax.axvline(0, color="blue", lw=1, linestyle="--")
        ax.set_title("Lifecycle Path Of Normalized TP2 Pressure")
        ax.set_xlabel("Event day index, 0 = first violation day")
        ax.set_ylabel("Normalized pressure")
        ax.legend()
        ax.grid(alpha=0.25)
        _save_current(fig, pdf, figures_dir / "lifecycle_normalized_pressure_band.png")

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(e["event_day_index"], e["violation_share"], label="mid-price violation share", lw=2)
        ax.plot(
            e["event_day_index"],
            e["conservative_violation_share"],
            label="conservative bid/ask violation share",
            lw=2,
        )
        ax.plot(e["event_day_index"], e["tradable_50pct_share"], label="all legs <=50% spread", lw=2)
        ax.axvline(0, color="blue", lw=1, linestyle="--")
        ax.set_ylim(0, 1.02)
        ax.set_title("Violation Persistence And Tradability By Event Day")
        ax.set_xlabel("Event day index")
        ax.set_ylabel("Share of lifecycle rows")
        ax.legend()
        ax.grid(alpha=0.25)
        _save_current(fig, pdf, figures_dir / "lifecycle_violation_tradability_share.png")

        fig, ax = plt.subplots(figsize=(11, 6))
        status_pivot.plot.area(ax=ax, alpha=0.85)
        ax.axvline(0, color="black", lw=1, linestyle="--")
        ax.set_ylim(0, 1)
        ax.set_title("Lifecycle Row Status Mix")
        ax.set_xlabel("Event day index")
        ax.set_ylabel("Share")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.grid(alpha=0.2)
        _save_current(fig, pdf, figures_dir / "lifecycle_status_mix_by_event_day.png")

        fig, ax = plt.subplots(figsize=(11, 6))
        for group, label, color in [
            ("T1_profitable", "T1 profitable", "green"),
            ("T1_unprofitable", "T1 unprofitable", "red"),
            ("invalid_or_unobservable", "T1 invalid/unobservable", "gray"),
        ]:
            sub = outcome[outcome["T1_outcome_group"].eq(group)].sort_values("event_day_index")
            if sub.empty:
                continue
            ax.plot(sub["event_day_index"], sub["median_pressure"], label=label, lw=2, color=color)
        ax.axhline(0, color="black", lw=1, linestyle="--")
        ax.axvline(0, color="blue", lw=1, linestyle="--")
        ax.set_title("Median Pressure Path Split By T1 Product-Weighted Outcome")
        ax.set_xlabel("Event day index")
        ax.set_ylabel("Median normalized pressure")
        ax.legend()
        ax.grid(alpha=0.25)
        _save_current(fig, pdf, figures_dir / "lifecycle_pressure_by_T1_outcome.png")

        heat = df[df["has_pressure"] & df["event_day_index"].between(-20, 40)].copy()
        bins = np.linspace(-0.25, 0.25, 51)
        heat["pressure_bin"] = pd.cut(heat["normalized_pressure"].clip(-0.25, 0.25), bins=bins)
        heat_pivot = (
            heat.groupby(["pressure_bin", "event_day_index"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        heat_pivot = heat_pivot.div(heat_pivot.sum(axis=0).replace(0, np.nan), axis=1)
        fig, ax = plt.subplots(figsize=(12, 7))
        im = ax.imshow(
            heat_pivot.to_numpy(),
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[
                heat_pivot.columns.min(),
                heat_pivot.columns.max(),
                bins[0],
                bins[-1],
            ],
        )
        ax.axvline(0, color="white", lw=1, linestyle="--")
        ax.axhline(0, color="red", lw=1, linestyle="--")
        ax.set_title("Heatmap Of Normalized Pressure Distribution")
        ax.set_xlabel("Event day index")
        ax.set_ylabel("Normalized pressure bin")
        fig.colorbar(im, ax=ax, label="Within-day share")
        _save_current(fig, pdf, figures_dir / "lifecycle_pressure_heatmap.png")

        _plot_text_page(
            pdf,
            "Interpretation Notes",
            [
                "## Main Reading",
                "If the median pressure is below zero before day 0 and jumps above zero on day 0,",
                "then violations are usually sudden local dislocations rather than always slowly building trends.",
                "If pressure falls back below zero quickly after day 0, the market usually corrects fast.",
                "",
                "## What Matters For Trading",
                "The future path itself is not a live model feature.",
                "But the past path can become a model feature: prior pressure slope, near-violation count,",
                "pre-event volatility of pressure, and whether the pair was recently close to zero.",
                "",
                "## Why Status Mix Matters",
                "Missing A/B rows usually mean the pair was not listed during some lookback dates.",
                "Missing C/D or missing forward means the adjusted rectangle could not be reconstructed.",
                "Liquidity failure means the gap is observable but at least one leg is wider than the 50% spread rule.",
                "",
                "## Next Modeling Step",
                "Build entry-time lifecycle features using only event days <= 0.",
                "Then test whether those features improve rolling ML selection for T1/K1 product-weighted P&L.",
            ],
        )

    print(f"Wrote tables to {tables_dir}", flush=True)
    print(f"Wrote figures to {figures_dir}", flush=True)
    print(f"Wrote PDF report to {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
