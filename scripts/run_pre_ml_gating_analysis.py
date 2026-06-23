#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TEXT = (35, 39, 47)
GRID = (220, 225, 232)
BLUE = (76, 120, 168)
GREEN = (84, 162, 75)
RED = (228, 87, 86)
ORANGE = (245, 133, 24)


FEATURE_COLUMNS = [
    "episode_id",
    "signal_date",
    "conservative_bidask_normalized_severity",
    "feature_normalized_severity",
    "feature_avg_spread_pct",
    "feature_max_spread_pct",
    "feature_min_volume",
    "feature_total_volume",
    "feature_min_open_interest",
    "feature_total_open_interest",
    "feature_T1_DTE",
    "feature_T2_DTE",
    "feature_term_gap",
    "feature_moneyness_K1",
    "feature_moneyness_K2",
    "feature_distance_K1_ATM",
    "feature_distance_K2_ATM",
    "feature_A_delta",
    "feature_B_delta",
    "feature_C_delta",
    "feature_D_delta",
]

BACKTEST_COLUMNS = [
    "episode_id",
    "signal_date",
    "strategy",
    "is_valid_backtest",
    "exit_reason",
    "net_pnl",
    "realized_return",
    "actual_holding_days",
]


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


def _load_modeling_frame(labels_path: Path, backtest_path: Path) -> pd.DataFrame:
    labels = pd.read_parquet(labels_path, columns=FEATURE_COLUMNS)
    backtest = pd.read_parquet(backtest_path, columns=BACKTEST_COLUMNS)
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    backtest["signal_date"] = pd.to_datetime(backtest["signal_date"])
    labels["year"] = labels["signal_date"].dt.year
    for col in ["feature_A_delta", "feature_B_delta", "feature_C_delta", "feature_D_delta"]:
        labels[col] = pd.to_numeric(labels[col], errors="coerce")
    delta_values = labels[["feature_A_delta", "feature_B_delta", "feature_C_delta", "feature_D_delta"]].abs()
    labels["mean_abs_delta"] = delta_values.mean(axis=1)
    labels["max_abs_delta"] = delta_values.max(axis=1)
    labels["min_abs_delta"] = delta_values.min(axis=1)
    feature_keep = [col for col in labels.columns if col not in {"signal_date"}]
    return backtest.merge(labels[feature_keep], on="episode_id", how="left", validate="many_to_one")


def _gate_stats(df: pd.DataFrame, gate_name: str, gate_description: str) -> dict:
    valid = df[df["is_valid_backtest"]].copy()
    years = sorted(df["year"].dropna().astype(int).unique())
    if not valid.empty:
        yearly = valid.groupby("year", dropna=False)["net_pnl"].sum()
        positive_years = int((yearly > 0).sum())
        min_year_net = float(yearly.min())
        worst_year = int(yearly.idxmin())
        max_year_net = float(yearly.max())
    else:
        positive_years = 0
        min_year_net = np.nan
        worst_year = np.nan
        max_year_net = np.nan

    return {
        "gate_name": gate_name,
        "gate_description": gate_description,
        "candidate_rows": int(len(df)),
        "unique_episodes": int(df["episode_id"].nunique()),
        "valid_rows": int(valid.shape[0]),
        "valid_share": float(valid.shape[0] / len(df)) if len(df) else 0.0,
        "hit_rate_valid": float((valid["realized_return"] > 0).mean()) if len(valid) else np.nan,
        "mean_return_valid": float(valid["realized_return"].mean()) if len(valid) else np.nan,
        "median_return_valid": float(valid["realized_return"].median()) if len(valid) else np.nan,
        "net_pnl_valid": float(valid["net_pnl"].sum()) if len(valid) else 0.0,
        "net_pnl_per_candidate": float(valid["net_pnl"].sum() / len(df)) if len(df) else np.nan,
        "mean_holding_days_valid": float(valid["actual_holding_days"].mean()) if len(valid) else np.nan,
        "positive_years": positive_years,
        "years_present": int(len(years)),
        "positive_year_share": float(positive_years / len(years)) if years else np.nan,
        "min_year_net_pnl": min_year_net,
        "max_year_net_pnl": max_year_net,
        "worst_year": worst_year,
    }


def _apply_gate(df: pd.DataFrame, gate: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, op, value in gate["conditions"]:
        if op == ">=":
            mask &= df[col] >= value
        elif op == "<=":
            mask &= df[col] <= value
        elif op == "between":
            lo, hi = value
            mask &= df[col].between(lo, hi, inclusive="both")
        else:
            raise ValueError(f"Unknown gate operator: {op}")
    return df[mask].copy()


def _candidate_gates(frame: pd.DataFrame) -> list[dict]:
    sev = frame["conservative_bidask_normalized_severity"].dropna()
    q = sev.quantile([0.25, 0.50, 0.75, 0.90]).to_dict()
    gates: list[dict] = [{"name": "all_conservative", "description": "No extra gate beyond conservative bid/ask TP2", "conditions": []}]

    for label, threshold in [
        ("sev_q25", q[0.25]),
        ("sev_q50", q[0.50]),
        ("sev_q75", q[0.75]),
        ("sev_q90", q[0.90]),
        ("sev_0p03", 0.03),
        ("sev_0p05", 0.05),
        ("sev_0p10", 0.10),
    ]:
        gates.append(
            {
                "name": label,
                "description": f"conservative severity >= {threshold:.4f}",
                "conditions": [("conservative_bidask_normalized_severity", ">=", float(threshold))],
            }
        )
    for threshold in [0.10, 0.15, 0.25]:
        gates.append(
            {
                "name": f"max_spread_le_{threshold:.2f}",
                "description": f"max leg spread <= {threshold:.0%}",
                "conditions": [("feature_max_spread_pct", "<=", threshold)],
            }
        )
    for threshold in [1, 10, 50, 100]:
        gates.append(
            {
                "name": f"total_volume_ge_{threshold}",
                "description": f"total four-leg volume >= {threshold}",
                "conditions": [("feature_total_volume", ">=", threshold)],
            }
        )
    for threshold in [1, 100, 1000]:
        gates.append(
            {
                "name": f"total_oi_ge_{threshold}",
                "description": f"total four-leg open interest >= {threshold}",
                "conditions": [("feature_total_open_interest", ">=", threshold)],
            }
        )
    for lo, hi in [(1, 7), (8, 30), (31, 60), (61, 90)]:
        gates.append(
            {
                "name": f"t1_dte_{lo}_{hi}",
                "description": f"T1 DTE in [{lo}, {hi}]",
                "conditions": [("feature_T1_DTE", "between", (lo, hi))],
            }
        )
    for lo, hi in [(1, 14), (15, 45), (46, 90)]:
        gates.append(
            {
                "name": f"term_gap_{lo}_{hi}",
                "description": f"term gap in [{lo}, {hi}] days",
                "conditions": [("feature_term_gap", "between", (lo, hi))],
            }
        )
    for threshold in [0.05, 0.10, 0.20]:
        gates.append(
            {
                "name": f"max_abs_delta_le_{threshold:.2f}",
                "description": f"max absolute leg delta <= {threshold:.2f}",
                "conditions": [("max_abs_delta", "<=", threshold)],
            }
        )

    for severity in [0.03, 0.05, 0.10]:
        for spread in [0.15, 0.25, 0.50]:
            for volume in [0, 10, 50]:
                for oi in [0, 100, 1000]:
                    conditions = [
                        ("conservative_bidask_normalized_severity", ">=", severity),
                        ("feature_max_spread_pct", "<=", spread),
                    ]
                    if volume > 0:
                        conditions.append(("feature_total_volume", ">=", volume))
                    if oi > 0:
                        conditions.append(("feature_total_open_interest", ">=", oi))
                    gates.append(
                        {
                            "name": f"grid_sev{severity:.2f}_spr{spread:.2f}_vol{volume}_oi{oi}",
                            "description": f"severity>={severity:.2f}, spread<={spread:.0%}, volume>={volume}, OI>={oi}",
                            "conditions": conditions,
                        }
                    )
    return gates


def run_gating(frame: pd.DataFrame, min_candidates: int) -> pd.DataFrame:
    rows = []
    gates = _candidate_gates(frame)
    for strategy, strategy_frame in frame.groupby("strategy", sort=True):
        for gate in gates:
            gated = _apply_gate(strategy_frame, gate)
            if len(gated) < min_candidates:
                continue
            row = _gate_stats(gated, gate["name"], gate["description"])
            row["strategy"] = strategy
            rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["strategy", "net_pnl_per_candidate"], ascending=[True, False]).reset_index(drop=True)


def build_dimension_tables(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    bins = {
        "severity_bin": pd.qcut(
            frame["conservative_bidask_normalized_severity"],
            q=[0, 0.25, 0.50, 0.75, 0.90, 1.0],
            duplicates="drop",
        ),
        "max_spread_bin": pd.cut(frame["feature_max_spread_pct"], bins=[0, 0.05, 0.10, 0.20, 0.50], include_lowest=True),
        "t1_dte_bin": pd.cut(frame["feature_T1_DTE"], bins=[0, 7, 30, 60, 90], include_lowest=True),
        "term_gap_bin": pd.cut(frame["feature_term_gap"], bins=[0, 14, 45, 90], include_lowest=True),
        "delta_bin": pd.cut(frame["max_abs_delta"], bins=[0, 0.05, 0.10, 0.20, 1.0], include_lowest=True),
    }
    for name, series in bins.items():
        temp = frame.copy()
        temp[name] = series.astype(str)
        table = (
            temp.groupby(["strategy", name], dropna=False)
            .apply(lambda g: pd.Series(_gate_stats(g, str(g.name), str(g.name))))
            .reset_index()
        )
        path = output_dir / f"{name}_performance.csv"
        table.to_csv(path, index=False)
        paths[name] = str(path)
    return paths


def save_top_gate_chart(top: pd.DataFrame, path: Path) -> None:
    img = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(img)
    draw.text((60, 35), "Top Pre-ML Gates By Net P&L Per Candidate", fill=TEXT, font=_font(34, True))
    draw.text((60, 82), "Conservative bid/ask subset; valid reversion-resale P&L only.", fill=(95, 103, 115), font=_font(20))
    plot = top.head(15).copy()
    plot["label"] = plot["strategy"] + " | " + plot["gate_name"]
    left, top_y, right, bottom = 500, 145, 1500, 825
    max_value = max(float(plot["net_pnl_per_candidate"].max()), 1.0)
    row_h = (bottom - top_y) / len(plot)
    for i, row in plot.reset_index(drop=True).iterrows():
        y = top_y + i * row_h + 6
        draw.text((50, y), row["label"][:55], fill=TEXT, font=_font(15))
        bar_w = float(row["net_pnl_per_candidate"]) / max_value * (right - left)
        draw.rectangle((left, y + 2, left + bar_w, y + row_h - 8), fill=GREEN if row["net_pnl_per_candidate"] >= 0 else RED)
        draw.text((left + bar_w + 8, y + 1), f"${row['net_pnl_per_candidate']:,.0f}", fill=TEXT, font=_font(14))
    img.save(path)


def save_strategy_stability_chart(by_year: pd.DataFrame, path: Path) -> None:
    img = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(img)
    draw.text((60, 35), "Conservative Subset Net P&L By Year And Strategy", fill=TEXT, font=_font(34, True))
    draw.text((60, 82), "Uneven episode arrivals make year stability important before rolling ML.", fill=(95, 103, 115), font=_font(20))
    pivot = by_year.pivot_table(index="year", columns="strategy", values="net_pnl", fill_value=0)
    strategies = [s for s in ["T1", "K1", "T2", "K2"] if s in pivot.columns]
    colors = {"T1": BLUE, "K1": GREEN, "T2": ORANGE, "K2": RED}
    left, top_y, right, bottom = 115, 160, 1480, 700
    low = float(min(pivot[strategies].min().min(), 0))
    high = float(max(pivot[strategies].max().max(), 0))
    pad = (high - low) * 0.08 if high > low else 1.0
    low -= pad
    high += pad
    zero_y = bottom - (0 - low) / (high - low) * (bottom - top_y)
    draw.line((left, zero_y, right, zero_y), fill=TEXT, width=2)
    draw.line((left, top_y, left, bottom), fill=TEXT, width=2)
    for i in range(6):
        value = low + (high - low) * i / 5
        y = bottom - (value - low) / (high - low) * (bottom - top_y)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((18, y - 10), f"{value/1_000_000:.0f}M", fill=(92, 99, 112), font=_font(14))
    years = pivot.index.astype(str).tolist()
    slot = (right - left) / max(1, len(years) - 1)
    for strategy in strategies:
        points = []
        for i, year in enumerate(pivot.index):
            value = float(pivot.loc[year, strategy])
            x = left + i * slot
            y = bottom - (value - low) / (high - low) * (bottom - top_y)
            points.append((x, y))
        draw.line(points, fill=colors[strategy], width=4)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors[strategy])
    for i, year in enumerate(years):
        x = left + i * slot
        draw.text((x - 24, bottom + 14), year, fill=TEXT, font=_font(14))
    lx, ly = 1120, 40
    for i, strategy in enumerate(strategies):
        y = ly + i * 28
        draw.line((lx, y + 8, lx + 35, y + 8), fill=colors[strategy], width=4)
        draw.text((lx + 45, y - 2), strategy, fill=TEXT, font=_font(17))
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pre-ML gating analysis on conservative bid/ask TP2 subset.")
    parser.add_argument("--labels", default="data/labels/archive2_episode_start_labels_conservative_bidask_2013_2026.parquet")
    parser.add_argument("--backtest", default="data/labels/archive2_episode_strategy_mini_backtest_conservative_bidask_2013_2026.parquet")
    parser.add_argument("--output-dir", default="reports/raw_archive2_eda/tables/pre_ml_gating")
    parser.add_argument("--figure-dir", default="reports/figures")
    parser.add_argument("--min-candidates", type=int, default=1000)
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    figure_dir = ROOT / args.figure_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    frame = _load_modeling_frame(ROOT / args.labels, ROOT / args.backtest)
    frame.to_parquet(output_dir / "pre_ml_modeling_frame_sample_columns.parquet", index=False)
    print(f"Loaded modeling frame: {len(frame):,} strategy rows")

    gate_results = run_gating(frame, min_candidates=args.min_candidates)
    gate_path = output_dir / "pre_ml_gate_grid_results.csv"
    gate_results.to_csv(gate_path, index=False)

    top_path = output_dir / "pre_ml_top_gates_by_strategy.csv"
    top = (
        gate_results.sort_values(["strategy", "net_pnl_per_candidate"], ascending=[True, False])
        .groupby("strategy", as_index=False)
        .head(15)
    )
    top.to_csv(top_path, index=False)

    by_year = (
        frame[frame["is_valid_backtest"]]
        .groupby(["year", "strategy"], as_index=False)
        .agg(
            valid_rows=("episode_id", "size"),
            net_pnl=("net_pnl", "sum"),
            mean_return=("realized_return", "mean"),
            median_return=("realized_return", "median"),
            hit_rate=("realized_return", lambda s: float((s > 0).mean())),
        )
        .sort_values(["year", "strategy"])
    )
    by_year_path = output_dir / "pre_ml_strategy_year_stability.csv"
    by_year.to_csv(by_year_path, index=False)

    dimension_paths = build_dimension_tables(frame, output_dir)
    save_top_gate_chart(
        gate_results.sort_values("net_pnl_per_candidate", ascending=False),
        figure_dir / "pre_ml_top_gates_by_net_pnl_per_candidate.png",
    )
    save_strategy_stability_chart(by_year, figure_dir / "pre_ml_strategy_year_stability.png")

    print(f"Wrote {gate_path}")
    print(f"Wrote {top_path}")
    print(f"Wrote {by_year_path}")
    print(f"Wrote dimension tables: {dimension_paths}")
    print(gate_results.sort_values("net_pnl_per_candidate", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
