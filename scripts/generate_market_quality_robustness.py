#!/usr/bin/env python3
"""Generate market-quality robustness diagnostics for TP2 ML episodes."""
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
ACTION_COL = "label_best_action"


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
    tbl.set_fontsize(8.2)
    tbl.scale(1, 1.28)
    if note:
        fig.text(0.06, 0.06, note, ha="left", va="bottom", fontsize=9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_gate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    leg_volume = [f"feature_{leg}_volume" for leg in "ABCD"]
    leg_oi = [f"feature_{leg}_open_interest" for leg in "ABCD"]
    leg_delta = [f"feature_{leg}_delta" for leg in "ABCD"]
    leg_spread = [f"feature_{leg}_spread_pct" for leg in "ABCD"]
    leg_bid_size = [f"feature_{leg}_bid_size" for leg in "ABCD"]
    leg_ask_size = [f"feature_{leg}_ask_size" for leg in "ABCD"]

    out["q_all_volume_positive"] = out[leg_volume].fillna(0).gt(0).all(axis=1)
    out["q_all_open_interest_positive"] = out[leg_oi].fillna(0).gt(0).all(axis=1)
    out["q_all_delta_available"] = out[leg_delta].notna().all(axis=1)
    out["q_delta_01_99"] = out[leg_delta].abs().ge(0.01).all(axis=1) & out[leg_delta].abs().le(0.99).all(axis=1)
    out["q_all_bidask_size_positive"] = (
        out[leg_bid_size].fillna(0).gt(0).all(axis=1)
        & out[leg_ask_size].fillna(0).gt(0).all(axis=1)
    )
    out["q_max_spread_le_25pct"] = out[leg_spread].le(0.25).all(axis=1)
    out["q_max_spread_le_10pct"] = out[leg_spread].le(0.10).all(axis=1)

    out["q_K1_round_distance"] = (out["feature_K1_forward_rounded"] - out["feature_K1_forward_adjusted"]).abs()
    out["q_K2_round_distance"] = (out["feature_K2_forward_rounded"] - out["feature_K2_forward_adjusted"]).abs()
    out["q_max_round_distance"] = out[["q_K1_round_distance", "q_K2_round_distance"]].max(axis=1)
    out["q_round_distance_le_1"] = out["q_max_round_distance"].le(1.0)
    out["q_round_distance_le_2_5"] = out["q_max_round_distance"].le(2.5)
    out["q_round_distance_le_5"] = out["q_max_round_distance"].le(5.0)

    out["q_volume_and_delta"] = out["q_all_volume_positive"] & out["q_delta_01_99"]
    out["q_strict_quality"] = (
        out["q_all_volume_positive"]
        & out["q_all_open_interest_positive"]
        & out["q_delta_01_99"]
        & out["q_max_spread_le_25pct"]
        & out["q_round_distance_le_2_5"]
    )
    return out


def gate_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "base_conservative_bidask": pd.Series(True, index=df.index),
        "all_legs_volume_positive": df["q_all_volume_positive"],
        "all_legs_open_interest_positive": df["q_all_open_interest_positive"],
        "paper_delta_01_99": df["q_delta_01_99"],
        "all_bidask_size_positive": df["q_all_bidask_size_positive"],
        "max_spread_le_25pct": df["q_max_spread_le_25pct"],
        "max_spread_le_10pct": df["q_max_spread_le_10pct"],
        "round_distance_le_1": df["q_round_distance_le_1"],
        "round_distance_le_2_5": df["q_round_distance_le_2_5"],
        "round_distance_le_5": df["q_round_distance_le_5"],
        "volume_and_delta": df["q_volume_and_delta"],
        "strict_quality_combo": df["q_strict_quality"],
    }


def summarize_gate(df: pd.DataFrame, name: str, mask: pd.Series) -> dict:
    sub = df[mask].copy()
    row = {
        "gate": name,
        "episodes": int(len(sub)),
        "kept_share": float(len(sub) / len(df)) if len(df) else np.nan,
        "positive_label_rows": int(sub["label_any_positive"].sum()) if "label_any_positive" in sub else 0,
        "positive_label_share": float(sub["label_any_positive"].mean()) if len(sub) and "label_any_positive" in sub else np.nan,
        "median_max_spread_pct": float(sub["feature_max_leg_spread_pct"].median()) if len(sub) else np.nan,
        "median_min_volume": float(sub["feature_min_leg_volume"].median()) if len(sub) else np.nan,
        "median_min_open_interest": float(sub["feature_min_leg_open_interest"].median()) if len(sub) else np.nan,
        "median_max_round_distance": float(sub["q_max_round_distance"].median()) if len(sub) else np.nan,
    }
    for action in ["skip", *STRATEGIES]:
        row[f"label_{action}_count"] = int((sub[ACTION_COL] == action).sum()) if len(sub) else 0
    return row


def strategy_summary_for_gate(df: pd.DataFrame, name: str, mask: pd.Series) -> list[dict]:
    sub = df[mask].copy()
    rows = []
    for strategy in STRATEGIES:
        valid = sub[f"label_is_valid_{strategy}"].fillna(False)
        pnl = sub[f"label_net_pnl_{strategy}"].where(valid)
        ret = sub[f"label_realized_return_{strategy}"].where(valid)
        rows.append(
            {
                "gate": name,
                "strategy": strategy,
                "episodes": int(len(sub)),
                "valid": int(valid.sum()),
                "valid_rate": float(valid.mean()) if len(sub) else np.nan,
                "positive": int((pnl > 0).sum()),
                "hit_rate": float((pnl > 0).sum() / valid.sum()) if valid.sum() else np.nan,
                "net_pnl": float(pnl.sum(skipna=True)),
                "median_return": float(ret.median()) if valid.sum() else np.nan,
            }
        )
    return rows


def plot_gate_retention(pdf: PdfPages, gate_summary: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = gate_summary.sort_values("kept_share", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(plot_df["gate"], plot_df["kept_share"] * 100, color="#2f6f9f")
    ax.set_xlabel("Kept share of conservative episodes (%)")
    ax.set_title("Market-Quality Gate Retention")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = figures_dir / "market_quality_gate_retention.png"
    fig.savefig(path, dpi=180)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_pnl(pdf: PdfPages, strategy_summary: pd.DataFrame, figures_dir: Path) -> None:
    selected = strategy_summary[
        strategy_summary["gate"].isin(
            [
                "base_conservative_bidask",
                "all_legs_volume_positive",
                "paper_delta_01_99",
                "round_distance_le_2_5",
                "strict_quality_combo",
            ]
        )
    ].copy()
    pivot = selected.pivot(index="gate", columns="strategy", values="net_pnl")
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(pivot.index))
    width = 0.18
    colors = {"T1": "#2f6f9f", "T2": "#bf5b17", "K1": "#3f7f4f", "K2": "#7a4fa3"}
    for i, strategy in enumerate(STRATEGIES):
        ax.bar(x + (i - 1.5) * width, pivot[strategy] / 1e6, width, label=strategy, color=colors[strategy])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel("First-liquid net P&L ($ millions)")
    ax.set_title("First-Liquid Strategy P&L Under Market-Quality Gates")
    ax.legend(ncol=4)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figures_dir / "market_quality_strategy_pnl.png"
    fig.savefig(path, dpi=180)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate market-quality robustness diagnostics.")
    parser.add_argument("--dataset", default="data/ml/archive2_ml_episode_dataset_2013_2026.parquet")
    parser.add_argument("--tables-dir", default="reports/raw_archive2_eda/tables/market_quality_robustness")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--pdf-output", default="reports/market_quality_robustness_report.pdf")
    args = parser.parse_args()

    root = CONFIG.paths.root
    tables_dir = root / args.tables_dir
    figures_dir = root / args.figures_dir
    pdf_path = root / args.pdf_output
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(root / args.dataset)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["year"] = df["signal_date"].dt.year
    df = add_gate_columns(df)
    masks = gate_masks(df)

    gate_summary = pd.DataFrame([summarize_gate(df, name, mask) for name, mask in masks.items()])
    strategy_summary = pd.DataFrame(
        row for name, mask in masks.items() for row in strategy_summary_for_gate(df, name, mask)
    )

    gate_year_rows = []
    for name, mask in masks.items():
        sub = df[mask].copy()
        year = (
            sub.groupby("year")
            .agg(
                episodes=("episode_id", "size"),
                positive_label_rows=("label_any_positive", "sum"),
                median_max_spread_pct=("feature_max_leg_spread_pct", "median"),
                median_min_volume=("feature_min_leg_volume", "median"),
                median_min_open_interest=("feature_min_leg_open_interest", "median"),
            )
            .reset_index()
        )
        year["gate"] = name
        gate_year_rows.append(year)
    gate_year = pd.concat(gate_year_rows, ignore_index=True)

    gate_summary.to_csv(tables_dir / "market_quality_gate_summary.csv", index=False)
    strategy_summary.to_csv(tables_dir / "market_quality_strategy_summary.csv", index=False)
    gate_year.to_csv(tables_dir / "market_quality_gate_year_summary.csv", index=False)

    result = {
        "rows": int(len(df)),
        "gates": gate_summary.set_index("gate")["episodes"].to_dict(),
        "strict_quality_rows": int(gate_summary.loc[gate_summary["gate"].eq("strict_quality_combo"), "episodes"].iloc[0]),
        "base_rows": int(len(df)),
        "strict_quality_share": float(
            gate_summary.loc[gate_summary["gate"].eq("strict_quality_combo"), "kept_share"].iloc[0]
        ),
    }
    (tables_dir / "market_quality_robustness_summary.json").write_text(json.dumps(result, indent=2, default=str))

    display_gate = gate_summary.copy()
    display_gate["Episodes"] = display_gate["episodes"].map(lambda x: f"{int(x):,}")
    display_gate["Kept"] = display_gate["kept_share"].map(pct)
    display_gate["Positive labels"] = display_gate["positive_label_share"].map(pct)
    display_gate["Med max spread"] = display_gate["median_max_spread_pct"].map(pct)
    display_gate["Med min volume"] = display_gate["median_min_volume"].map(lambda x: "NA" if pd.isna(x) else f"{x:,.0f}")
    display_gate["Med min OI"] = display_gate["median_min_open_interest"].map(lambda x: "NA" if pd.isna(x) else f"{x:,.0f}")
    display_gate["Med round dist"] = display_gate["median_max_round_distance"].map(lambda x: "NA" if pd.isna(x) else f"{x:.2f}")

    display_strategy = strategy_summary.copy()
    display_strategy["Valid"] = display_strategy["valid"].map(lambda x: f"{int(x):,}")
    display_strategy["Valid rate"] = display_strategy["valid_rate"].map(pct)
    display_strategy["Hit rate"] = display_strategy["hit_rate"].map(pct)
    display_strategy["Net P&L"] = display_strategy["net_pnl"].map(money)
    display_strategy["Median return"] = display_strategy["median_return"].map(pct)

    with PdfPages(pdf_path) as pdf:
        add_text_page(
            pdf,
            "Market-Quality Robustness",
            [
                "## Purpose",
                "This report tests whether the TP2 episode universe is dominated by weak market-quality rows.",
                "These are sensitivity gates, not permanent filters yet.",
                "",
                "## Gates",
                "Volume: all four entry legs have positive daily volume.",
                "Open interest: all four entry legs have positive open interest.",
                "Delta: all four deltas are available and satisfy 0.01 <= abs(delta) <= 0.99.",
                "Bid/ask size: all four bid and ask sizes are positive where available.",
                "Spread: stricter 25% and 10% max entry-leg spread thresholds.",
                "Rounded-strike distance: max distance between adjusted target and rounded strike <= 1, 2.5, or 5.",
                "Strict combo: volume, open interest, delta, 25% spread, and 2.5 rounded-distance gates together.",
            ],
        )
        add_table_page(
            pdf,
            "Gate Retention And Quality Summary",
            display_gate[
                [
                    "gate",
                    "Episodes",
                    "Kept",
                    "Positive labels",
                    "Med max spread",
                    "Med min volume",
                    "Med min OI",
                    "Med round dist",
                ]
            ],
            "Positive labels means at least one first-liquid strategy has positive net P&L.",
        )
        plot_gate_retention(pdf, gate_summary, figures_dir)
        plot_strategy_pnl(pdf, strategy_summary, figures_dir)
        selected = display_strategy[
            display_strategy["gate"].isin(
                [
                    "base_conservative_bidask",
                    "all_legs_volume_positive",
                    "paper_delta_01_99",
                    "round_distance_le_2_5",
                    "strict_quality_combo",
                ]
            )
        ]
        add_table_page(
            pdf,
            "First-Liquid Strategy Results Under Key Gates",
            selected[["gate", "strategy", "Valid", "Valid rate", "Hit rate", "Net P&L", "Median return"]],
            "These are unconditional strategy labels under the current first-liquid exit policy.",
        )
        add_text_page(
            pdf,
            "Interpretation",
            [
                "## How To Use This",
                "A gate that removes most episodes but keeps or improves economic behavior is a candidate final robustness filter.",
                "A gate that destroys sample size should be reported as sensitivity, not used as the main filter.",
                "",
                "The final paper should show that ML results are not only caused by zero-volume, missing-delta, wide-spread, or distant-rounded-strike episodes.",
                "The next implementation step is to rerun rolling ML on the most reasonable filtered universes after reviewing these diagnostics.",
            ],
        )

    print(f"Wrote {tables_dir / 'market_quality_gate_summary.csv'}")
    print(f"Wrote {tables_dir / 'market_quality_strategy_summary.csv'}")
    print(f"Wrote {tables_dir / 'market_quality_gate_year_summary.csv'}")
    print(f"Wrote {tables_dir / 'market_quality_robustness_summary.json'}")
    print(f"Wrote {pdf_path}")
    print(gate_summary.to_string(index=False))


if __name__ == "__main__":
    main()
