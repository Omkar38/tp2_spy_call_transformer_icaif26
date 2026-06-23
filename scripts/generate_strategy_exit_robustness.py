#!/usr/bin/env python3
"""Generate strategy robustness diagnostics across exit policies."""
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


STRATEGIES = ["T1", "T2", "K1", "K2"]
POLICY_ORDER = [
    "exit_at_reversion_day",
    "exit_first_liquid_on_or_after_reversion",
    "exit_reversion_plus_1_session",
    "exit_reversion_plus_3_sessions",
    "exit_reversion_plus_5_sessions",
    "exit_selected_leg_expiration_payoff",
]
POLICY_LABEL = {
    "exit_at_reversion_day": "Reversion day",
    "exit_first_liquid_on_or_after_reversion": "First liquid",
    "exit_reversion_plus_1_session": "Reversion +1",
    "exit_reversion_plus_3_sessions": "Reversion +3",
    "exit_reversion_plus_5_sessions": "Reversion +5",
    "exit_selected_leg_expiration_payoff": "Expiration",
}
COLORS = {
    "T1": "#2f6f9f",
    "T2": "#bf5b17",
    "K1": "#3f7f4f",
    "K2": "#7a4fa3",
}


def money(value: float) -> str:
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


def pct(value: float) -> str:
    return "NA" if pd.isna(value) else f"{100 * float(value):.1f}%"


def add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    fig.text(0.06, 0.93, title, fontsize=20, fontweight="bold", ha="left", va="top")
    y = 0.86
    for line in lines:
        if line.startswith("## "):
            y -= 0.025
            fig.text(0.06, y, line[3:], fontsize=13, fontweight="bold", ha="left", va="top")
            y -= 0.04
        elif line.strip() == "":
            y -= 0.025
        else:
            fig.text(0.07, y, line, fontsize=10.5, ha="left", va="top")
            y -= 0.03
        if y < 0.08:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(11, 8.5))
            fig.patch.set_facecolor("white")
            plt.axis("off")
            y = 0.93
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, table: pd.DataFrame, note: str = "") -> None:
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
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.35)
    if note:
        fig.text(0.06, 0.06, note, ha="left", va="bottom", fontsize=9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_policy_pnl(pdf: PdfPages, summary: pd.DataFrame, figures_dir: Path) -> None:
    pivot = summary.pivot(index="exit_policy", columns="strategy", values="net_pnl").reindex(POLICY_ORDER)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(pivot.index))
    width = 0.18
    for i, strategy in enumerate(STRATEGIES):
        values = pivot[strategy].to_numpy() / 1e6
        ax.bar(x + (i - 1.5) * width, values, width, label=strategy, color=COLORS[strategy])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABEL[p] for p in pivot.index], rotation=20, ha="right")
    ax.set_ylabel("Net P&L ($ millions)")
    ax.set_title("Strategy Net P&L By Exit Policy")
    ax.legend(ncol=4)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figures_dir / "strategy_robustness_policy_pnl.png"
    fig.savefig(path, dpi=180)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_policy_median_return(pdf: PdfPages, summary: pd.DataFrame, figures_dir: Path) -> None:
    pivot = summary.pivot(index="exit_policy", columns="strategy", values="median_return").reindex(POLICY_ORDER)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(pivot.index))
    for strategy in STRATEGIES:
        ax.plot(
            x,
            pivot[strategy].to_numpy() * 100,
            marker="o",
            linewidth=2,
            label=strategy,
            color=COLORS[strategy],
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABEL[p] for p in pivot.index], rotation=20, ha="right")
    ax.set_ylabel("Median realized return (%)")
    ax.set_title("Median Return By Exit Policy")
    ax.legend(ncol=4)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figures_dir / "strategy_robustness_median_return.png"
    fig.savefig(path, dpi=180)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_year_heatmap(pdf: PdfPages, year: pd.DataFrame, figures_dir: Path) -> None:
    primary = year[year["exit_policy"].eq("exit_first_liquid_on_or_after_reversion")].copy()
    pivot = primary.pivot(index="year", columns="strategy", values="net_pnl").reindex(columns=STRATEGIES)
    values = pivot.to_numpy() / 1e6
    vmax = np.nanpercentile(np.abs(values), 95)
    vmax = max(vmax, 1.0)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(STRATEGIES)))
    ax.set_xticklabels(STRATEGIES)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(int(y)) for y in pivot.index])
    ax.set_title("First-Liquid Exit: Year x Strategy Net P&L")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Net P&L ($ millions)")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    path = figures_dir / "strategy_robustness_first_liquid_year_heatmap.png"
    fig.savefig(path, dpi=180)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_policy_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["exit_policy", "strategy"], sort=False)
        .agg(
            candidates=("episode_id", "count"),
            valid=("is_valid_backtest", "sum"),
            positive=("net_pnl", lambda x: (x > 0).sum()),
            net_pnl=("net_pnl", "sum"),
            mean_pnl=("net_pnl", "mean"),
            median_pnl=("net_pnl", "median"),
            median_return=("realized_return", "median"),
            mean_holding_days=("actual_holding_days", "mean"),
        )
        .reset_index()
    )
    summary["valid_rate"] = summary["valid"] / summary["candidates"]
    summary["hit_rate"] = summary["positive"] / summary["valid"].replace(0, np.nan)
    summary["exit_policy_label"] = summary["exit_policy"].map(POLICY_LABEL)
    return summary


def build_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    year = (
        df.groupby(["year", "exit_policy", "strategy"], sort=False)
        .agg(
            candidates=("episode_id", "count"),
            valid=("is_valid_backtest", "sum"),
            positive=("net_pnl", lambda x: (x > 0).sum()),
            net_pnl=("net_pnl", "sum"),
            median_return=("realized_return", "median"),
        )
        .reset_index()
    )
    year["hit_rate"] = year["positive"] / year["valid"].replace(0, np.nan)
    return year


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate strategy robustness report across exit policies.")
    parser.add_argument("--labels", default="data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet")
    parser.add_argument("--tables-dir", default="reports/raw_archive2_eda/tables/strategy_exit_robustness")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--pdf-output", default="reports/strategy_exit_robustness_report.pdf")
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
        "is_valid_backtest",
        "net_pnl",
        "realized_return",
        "actual_holding_days",
        "entry_gross_exposure",
    ]
    df = pd.read_parquet(root / args.labels, columns=cols)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["year"] = df["signal_date"].dt.year
    df = df[df["strategy"].isin(STRATEGIES) & df["exit_policy"].isin(POLICY_ORDER)].copy()

    summary = build_policy_summary(df)
    year = build_year_summary(df)

    summary.to_csv(tables_dir / "strategy_exit_policy_summary.csv", index=False)
    year.to_csv(tables_dir / "strategy_exit_policy_year_summary.csv", index=False)

    ranking = summary.copy()
    ranking["pnl_rank_within_policy"] = ranking.groupby("exit_policy")["net_pnl"].rank(
        ascending=False,
        method="first",
    )
    ranking["median_return_rank_within_policy"] = ranking.groupby("exit_policy")["median_return"].rank(
        ascending=False,
        method="first",
    )
    ranking.to_csv(tables_dir / "strategy_exit_policy_rankings.csv", index=False)

    first_liquid = summary[summary["exit_policy"].eq("exit_first_liquid_on_or_after_reversion")]
    expiration = summary[summary["exit_policy"].eq("exit_selected_leg_expiration_payoff")]
    result = {
        "rows": int(len(df)),
        "episodes": int(df["episode_id"].nunique()),
        "policies": POLICY_ORDER,
        "first_liquid_net_pnl": dict(zip(first_liquid["strategy"], first_liquid["net_pnl"])),
        "expiration_net_pnl": dict(zip(expiration["strategy"], expiration["net_pnl"])),
        "positive_policy_strategy_cells": int((summary["net_pnl"] > 0).sum()),
        "total_policy_strategy_cells": int(len(summary)),
        "best_by_policy": {
            policy: summary[summary["exit_policy"].eq(policy)]
            .sort_values("net_pnl", ascending=False)
            .iloc[0]["strategy"]
            for policy in POLICY_ORDER
        },
    }
    (tables_dir / "strategy_exit_robustness_summary.json").write_text(json.dumps(result, indent=2, default=str))

    display = summary.copy()
    display["Exit"] = display["exit_policy"].map(POLICY_LABEL)
    display["Strategy"] = display["strategy"]
    display["Valid"] = display["valid"].map(lambda x: f"{int(x):,}")
    display["Valid rate"] = display["valid_rate"].map(pct)
    display["Hit rate"] = display["hit_rate"].map(pct)
    display["Net P&L"] = display["net_pnl"].map(money)
    display["Median return"] = display["median_return"].map(pct)
    display["Mean hold"] = display["mean_holding_days"].map(lambda x: "NA" if pd.isna(x) else f"{x:.2f}")

    rank_display = ranking.sort_values(["exit_policy", "pnl_rank_within_policy"]).copy()
    rank_display["Exit"] = rank_display["exit_policy"].map(POLICY_LABEL)
    rank_display["Strategy"] = rank_display["strategy"]
    rank_display["P&L rank"] = rank_display["pnl_rank_within_policy"].astype(int).astype(str)
    rank_display["Return rank"] = rank_display["median_return_rank_within_policy"].astype(int).astype(str)
    rank_display["Net P&L"] = rank_display["net_pnl"].map(money)
    rank_display["Median return"] = rank_display["median_return"].map(pct)

    first_liquid_year = year[year["exit_policy"].eq("exit_first_liquid_on_or_after_reversion")].copy()
    first_liquid_year["Year"] = first_liquid_year["year"].astype(str)
    first_liquid_year["Strategy"] = first_liquid_year["strategy"]
    first_liquid_year["Valid"] = first_liquid_year["valid"].map(lambda x: f"{int(x):,}")
    first_liquid_year["Hit rate"] = first_liquid_year["hit_rate"].map(pct)
    first_liquid_year["Net P&L"] = first_liquid_year["net_pnl"].map(money)
    first_liquid_year["Median return"] = first_liquid_year["median_return"].map(pct)

    with PdfPages(pdf_path) as pdf:
        add_text_page(
            pdf,
            "Strategy Exit-Policy Robustness",
            [
                "## Purpose",
                "This report checks whether T1/T2/K1/K2 conclusions depend on one exit rule.",
                "It uses the existing liquid-exit strategy labels, so no raw data or TP2 detection is recalculated.",
                "",
                "## Exit Policies",
                "Reversion day: close exactly on the first true TP2 reversion day if selected legs are liquid.",
                "First liquid: close on the first common liquid selected-leg date on or after true reversion.",
                "Reversion +1/+3/+5: fixed waiting robustness checks after true reversion.",
                "Expiration: selected-leg expiration payoff benchmark.",
                "",
                "## Main Diagnostic",
                "If a strategy only works under one exit policy, the economic claim is fragile.",
                "If a strategy remains useful across several exit policies, it is more defensible.",
                "If expiration and premium-resale exits disagree, that should be reported as a design difference, not hidden.",
            ],
        )
        add_table_page(
            pdf,
            "Exit Policy x Strategy Summary",
            display[["Exit", "Strategy", "Valid", "Valid rate", "Hit rate", "Net P&L", "Median return", "Mean hold"]],
            "All rows are unconditional strategy labels. ML selection is not applied in this table.",
        )
        add_table_page(
            pdf,
            "Strategy Ranking Within Each Exit Policy",
            rank_display[["Exit", "Strategy", "P&L rank", "Return rank", "Net P&L", "Median return"]],
            "P&L rank uses aggregate net P&L. Return rank uses median realized return.",
        )
        plot_policy_pnl(pdf, summary, figures_dir)
        plot_policy_median_return(pdf, summary, figures_dir)
        plot_year_heatmap(pdf, year, figures_dir)
        add_table_page(
            pdf,
            "First-Liquid Yearly Strategy Summary",
            first_liquid_year[["Year", "Strategy", "Valid", "Hit rate", "Net P&L", "Median return"]],
            "This is the current primary ML label exit policy before model selection.",
        )
        add_text_page(
            pdf,
            "Interpretation",
            [
                "## What The Robustness Says",
                f"Positive policy-strategy cells: {result['positive_policy_strategy_cells']} of {result['total_policy_strategy_cells']}.",
                f"Best strategy by policy: {result['best_by_policy']}.",
                "",
                "First-liquid premium resale remains difficult unconditionally: all four strategies have negative aggregate P&L.",
                "T2 is the least negative under first-liquid, which explains why the ML target often selects T2 episodes.",
                "Expiration payoff is structurally different and can rank strategies differently because it measures terminal payoff, not resale at TP2 reversion.",
                "",
                "## Research Implication",
                "The paper should not claim one universal arbitrage strategy.",
                "The stronger claim is that TP2 violation episodes have different behavior under different denomination and exit choices, and ML is used to select the tradeable cases.",
                "",
                "## Next Step",
                "Add market-quality robustness filters: positive volume, delta screen, open interest, rounded-strike distance, and static no-arbitrage sanity checks.",
            ],
        )

    print(f"Wrote {tables_dir / 'strategy_exit_policy_summary.csv'}")
    print(f"Wrote {tables_dir / 'strategy_exit_policy_year_summary.csv'}")
    print(f"Wrote {tables_dir / 'strategy_exit_policy_rankings.csv'}")
    print(f"Wrote {tables_dir / 'strategy_exit_robustness_summary.json'}")
    print(f"Wrote {pdf_path}")
    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
