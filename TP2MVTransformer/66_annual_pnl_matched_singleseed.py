"""#4 (figure part): rebuild the annual P&L bar chart at MATCHED single seed --
all three learned models at seed 42 -- so it no longer pairs single-seed
LightGBM/MLP against a five-seed transformer. Reuses 38's loaders and exact
plotting style; only the transformer series changes (single-seed ablation
per-year P&L, computed from its selected-trades parquet). Overwrites
figures/annual_pnl_bar_chart.pdf.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TP2MVTransformer"
DATA_DIR = OUT / "data"; TABLE_DIR = OUT / "tables"; FIG_DIR = OUT / "figures"

spec = importlib.util.spec_from_file_location("mk38", OUT / "38_make_annual_pnl_bar_chart.py")
m38 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m38)
YEARS, COLORS, ORDER = m38.YEARS, m38.COLORS, m38.ORDER


def transformer_singleseed_yearly() -> dict[int, float]:
    df = pd.read_parquet(DATA_DIR / "mv_no_alibi_ablation_selected_trades.parquet")
    if "selected" in df.columns:
        df = df[df["selected"].astype(bool)]
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    s = df.groupby(df["signal_date"].dt.year)["target_norm_pnl"].sum()
    return {y: float(s.get(y, 0.0)) for y in YEARS}


def main() -> None:
    series = {
        "Unconditional": m38.load_unconditional_yearly(),
        "LightGBM": m38.load_lgbm_yearly(),
        "MLP": m38.load_mlp_yearly(),
        "Transformer": transformer_singleseed_yearly(),   # single seed (42), matched
    }
    plt.rcParams.update({"font.size": 8.5})
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    n = len(ORDER); width = 0.8 / n; x = np.arange(len(YEARS))
    for i, name in enumerate(ORDER):
        vals = np.array([series[name][y] for y in YEARS])
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width=width * 0.92, label=name, color=COLORS[name], linewidth=0)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_yscale("symlog", linthresh=1_000)
    ax.set_xticks(x); ax.set_xticklabels([str(y)[-2:] for y in YEARS], fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Test-year P&L ($, symlog)", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M" if abs(v) >= 1e6 else (f"{v/1e3:.0f}K" if abs(v) != 0 else "0")))
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=7, columnspacing=1.0, handlelength=1.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "annual_pnl_bar_chart.pdf", bbox_inches="tight")
    print("Wrote matched-single-seed", FIG_DIR / "annual_pnl_bar_chart.pdf")
    for name in ORDER:
        print(f"{name:14s}", {y: round(series[name][y]) for y in YEARS})


if __name__ == "__main__":
    main()
