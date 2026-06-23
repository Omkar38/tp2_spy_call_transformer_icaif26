#!/usr/bin/env python3
"""Generate report assets for the liquid-exit TP2 rebuild."""
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
    if value >= 1e9:
        return f"{sign}${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{sign}${value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{sign}${value / 1e3:.1f}K"
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


def _save(fig, pdf: PdfPages, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _table_page(pdf: PdfPages, title: str, table: pd.DataFrame, note: str = "") -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=16)
    tbl = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)
    if note:
        fig.text(0.06, 0.06, note, ha="left", va="bottom", fontsize=9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate liquid-exit report.")
    parser.add_argument("--input", default="data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet")
    parser.add_argument("--summary-input", default="data/labels/archive2_liquid_exit_strategy_labels_summary_2013_2026.json")
    parser.add_argument("--tables-dir", default="reports/raw_archive2_eda/tables/liquid_exit_rebuild")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--pdf-output", default="reports/liquid_exit_rebuild_report.pdf")
    args = parser.parse_args()

    root = CONFIG.paths.root
    tables_dir = root / args.tables_dir
    figures_dir = root / args.figures_dir
    pdf_path = root / args.pdf_output
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "episode_id",
        "signal_date",
        "strategy",
        "exit_policy",
        "first_reversion_date",
        "first_reversion_event_day",
        "exit_date",
        "exit_reason",
        "is_valid_backtest",
        "entry_gross_exposure",
        "net_pnl",
        "realized_return",
        "actual_holding_days",
        "feature_normalized_severity",
        "feature_max_spread_pct",
        "feature_total_volume",
        "feature_total_open_interest",
    ]
    df = pd.read_parquet(root / args.input, columns=cols)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["first_reversion_date"] = pd.to_datetime(df["first_reversion_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["year"] = df["signal_date"].dt.year
    df["wait_days_after_reversion"] = (df["exit_date"] - df["first_reversion_date"]).dt.days

    summary = json.loads((root / args.summary_input).read_text()) if (root / args.summary_input).exists() else {}

    by_policy = (
        df.groupby(["exit_policy", "strategy"])
        .agg(
            selected=("episode_id", "count"),
            valid=("is_valid_backtest", "sum"),
            profitable=("net_pnl", lambda x: (x > 0).sum()),
            net_pnl=("net_pnl", "sum"),
            median_return=("realized_return", "median"),
            mean_holding_days=("actual_holding_days", "mean"),
        )
        .reset_index()
    )
    by_policy["valid_rate"] = by_policy["valid"] / by_policy["selected"]
    by_policy["hit_rate"] = by_policy["profitable"] / by_policy["valid"].replace(0, np.nan)
    by_policy.to_csv(tables_dir / "liquid_exit_by_policy_strategy.csv", index=False)

    by_year = (
        df.groupby(["year", "exit_policy", "strategy"])
        .agg(
            selected=("episode_id", "count"),
            valid=("is_valid_backtest", "sum"),
            profitable=("net_pnl", lambda x: (x > 0).sum()),
            net_pnl=("net_pnl", "sum"),
            median_return=("realized_return", "median"),
        )
        .reset_index()
    )
    by_year["hit_rate"] = by_year["profitable"] / by_year["valid"].replace(0, np.nan)
    by_year.to_csv(tables_dir / "liquid_exit_by_year_policy_strategy.csv", index=False)

    exit_reasons = (
        df.groupby(["exit_policy", "strategy", "exit_reason"])
        .size()
        .rename("rows")
        .reset_index()
    )
    exit_reasons.to_csv(tables_dir / "liquid_exit_reason_counts.csv", index=False)

    reversion = (
        df.drop_duplicates("episode_id")
        .assign(has_reversion=lambda x: x["first_reversion_date"].notna())
        .groupby("year")
        .agg(
            episodes=("episode_id", "count"),
            reversion_episodes=("has_reversion", "sum"),
            median_reversion_event_day=("first_reversion_event_day", "median"),
            p75_reversion_event_day=("first_reversion_event_day", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )
    reversion["reversion_share"] = reversion["reversion_episodes"] / reversion["episodes"]
    reversion.to_csv(tables_dir / "liquid_exit_reversion_by_year.csv", index=False)

    primary = df[df["exit_policy"].eq("exit_first_liquid_on_or_after_reversion")].copy()
    wait_summary = (
        primary[primary["is_valid_backtest"]]
        .groupby("strategy")
        .agg(
            valid=("episode_id", "count"),
            same_day_exit=("wait_days_after_reversion", lambda x: (x == 0).sum()),
            median_wait_days=("wait_days_after_reversion", "median"),
            p90_wait_days=("wait_days_after_reversion", lambda x: x.quantile(0.90)),
        )
        .reset_index()
    )
    wait_summary["same_day_exit_share"] = wait_summary["same_day_exit"] / wait_summary["valid"]
    wait_summary.to_csv(tables_dir / "liquid_exit_wait_summary.csv", index=False)

    display_policy = by_policy.copy()
    display_policy["net_pnl"] = display_policy["net_pnl"].map(_money)
    for col in ["valid_rate", "hit_rate", "median_return"]:
        display_policy[col] = display_policy[col].map(lambda x: "NA" if pd.isna(x) else f"{x:.2%}")
    display_policy["mean_holding_days"] = display_policy["mean_holding_days"].map(lambda x: "NA" if pd.isna(x) else f"{x:.2f}")

    with PdfPages(pdf_path) as pdf:
        _plot_text_page(
            pdf,
            "Liquid-Exit TP2 Rebuild Report",
            [
                "## What Changed",
                "The rebuild keeps no-spread quote availability for lifecycle and reversion tracking.",
                "Liquidity is now enforced when a trade is entered or exited, not used to erase the TP2 path.",
                "The main new policy exits at the first liquid date on or after true TP2 reversion.",
                "",
                "## Dataset",
                f"Episodes: {summary.get('episodes', df['episode_id'].nunique()):,}",
                f"Strategy-policy rows: {summary.get('rows', len(df)):,}",
                f"Valid strategy-policy rows: {summary.get('valid_rows', int(df['is_valid_backtest'].sum())):,}",
                f"Total valid net P&L across all policies: {_money(summary.get('valid_net_pnl', df['net_pnl'].sum()))}",
                "",
                "## Interpretation",
                "Exit-at-reversion is strict: the selected legs must be liquid exactly on the reversion date.",
                "First-liquid-after-reversion is more realistic: if reversion occurs but the close is not executable,",
                "the strategy waits for the first later liquid close before selected-leg expiration.",
                "Fixed +1/+3/+5 exits are robustness tests, not necessarily the preferred live rule.",
                "Expiration payoff is included separately as a Glasserman-style benchmark.",
            ],
        )

        _table_page(
            pdf,
            "Policy x Strategy Summary",
            display_policy[
                [
                    "exit_policy",
                    "strategy",
                    "selected",
                    "valid",
                    "valid_rate",
                    "hit_rate",
                    "net_pnl",
                    "median_return",
                    "mean_holding_days",
                ]
            ],
            "Product-weighted Glasserman-style quantities; bid/ask execution; $0.65 per contract per side.",
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        pivot = by_policy.pivot(index="exit_policy", columns="strategy", values="valid")
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Executable Rows By Exit Policy And Strategy")
        ax.set_xlabel("Exit policy")
        ax.set_ylabel("Valid strategy-policy rows")
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(axis="y", alpha=0.25)
        _save(fig, pdf, figures_dir / "liquid_exit_valid_rows_by_policy_strategy.png")

        fig, ax = plt.subplots(figsize=(12, 6))
        pivot = by_policy.pivot(index="exit_policy", columns="strategy", values="net_pnl") / 1e6
        pivot.plot(kind="bar", ax=ax)
        ax.axhline(0, color="black", lw=1)
        ax.set_title("Net P&L By Exit Policy And Strategy")
        ax.set_xlabel("Exit policy")
        ax.set_ylabel("Net P&L ($ millions)")
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(axis="y", alpha=0.25)
        _save(fig, pdf, figures_dir / "liquid_exit_net_pnl_by_policy_strategy.png")

        fig, ax = plt.subplots(figsize=(12, 6))
        primary_year = by_year[
            by_year["exit_policy"].isin(["exit_at_reversion_day", "exit_first_liquid_on_or_after_reversion"])
            & by_year["strategy"].isin(["T1", "K1"])
        ].copy()
        for (policy, strategy), sub in primary_year.groupby(["exit_policy", "strategy"]):
            ax.plot(sub["year"], sub["net_pnl"] / 1e6, marker="o", label=f"{strategy}: {policy}")
        ax.axhline(0, color="black", lw=1)
        ax.set_title("Yearly Net P&L For Primary Strategies")
        ax.set_xlabel("Signal year")
        ax.set_ylabel("Net P&L ($ millions)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        _save(fig, pdf, figures_dir / "liquid_exit_yearly_primary_pnl.png")

        fig, ax = plt.subplots(figsize=(11, 6))
        reason_pivot = (
            df.groupby(["exit_policy", "exit_reason"])
            .size()
            .rename("rows")
            .reset_index()
            .pivot(index="exit_policy", columns="exit_reason", values="rows")
            .fillna(0)
        )
        reason_pivot.div(reason_pivot.sum(axis=1), axis=0).plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Exit Reason Mix By Policy")
        ax.set_xlabel("Exit policy")
        ax.set_ylabel("Share")
        ax.tick_params(axis="x", labelrotation=30)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        _save(fig, pdf, figures_dir / "liquid_exit_reason_mix_by_policy.png")

        fig, ax = plt.subplots(figsize=(11, 6))
        reversion.plot(x="year", y="reversion_share", marker="o", ax=ax, legend=False)
        ax.set_ylim(0, 1)
        ax.set_title("Share Of Entry Episodes With Mid-Price Reversion Before T1")
        ax.set_xlabel("Signal year")
        ax.set_ylabel("Reversion share")
        ax.grid(alpha=0.25)
        _save(fig, pdf, figures_dir / "liquid_exit_reversion_share_by_year.png")

        fig, ax = plt.subplots(figsize=(11, 6))
        valid_primary = primary[primary["is_valid_backtest"] & primary["wait_days_after_reversion"].notna()]
        for strategy, sub in valid_primary.groupby("strategy"):
            clipped = sub["wait_days_after_reversion"].clip(upper=10)
            ax.hist(clipped, bins=np.arange(-0.5, 11.5, 1), alpha=0.45, label=strategy, density=True)
        ax.set_title("Wait Days After Reversion For First-Liquid Exit")
        ax.set_xlabel("Calendar days after reversion, clipped at 10")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.25)
        _save(fig, pdf, figures_dir / "liquid_exit_wait_days_distribution.png")

        _plot_text_page(
            pdf,
            "Research Reading",
            [
                "## Main Finding",
                "The first-liquid-after-reversion rule increases executable rows compared with exact reversion-day exit.",
                "This confirms the original concern: reversion can happen before the selected legs are cleanly executable.",
                "",
                "## Trading Interpretation",
                "T1 and K1 remain the primary product-weighted strategies in aggregate.",
                "However, median returns can still be negative even when total P&L is positive, so ML selection is needed.",
                "",
                "## ML Use",
                "Each row can become an ML action label: episode, strategy, and exit policy.",
                "Features must be entry-time only; reversion date, exit date, and future liquidity stay as labels only.",
                "",
                "## Caveat",
                "Premium-resale labels and expiration-payoff labels answer different execution questions.",
                "The expiration-payoff policy is a benchmark, while first-liquid-after-reversion is the main live-style exit.",
            ],
        )

    print(f"Wrote tables to {tables_dir}", flush=True)
    print(f"Wrote figures to {figures_dir}", flush=True)
    print(f"Wrote PDF report to {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
