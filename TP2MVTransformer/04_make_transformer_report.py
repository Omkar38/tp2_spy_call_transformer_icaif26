"""
Generate the TP2-MV-ALiBi transformer implementation report.

The report is intentionally produced from saved result tables, not pasted
manual numbers, so the PDF matches the current run artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TP2MVTransformer"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MPL_DIR = OUT / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fmt_money(x: float) -> str:
    return f"\\${x:,.0f}"


def fmt_float(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.{digits}f}"


def daily_sharpe_from_predictions(path: Path) -> float:
    pred = pd.read_parquet(path)
    selected = pred[pred["selected"].astype(bool)].copy()
    if selected.empty:
        return float("nan")
    selected["signal_date"] = pd.to_datetime(selected["signal_date"])
    daily = selected.groupby("signal_date")["target_norm_pnl"].sum()
    std = daily.std()
    if std == 0 or pd.isna(std):
        return float("nan")
    return float(daily.mean() / std * np.sqrt(252))


def save_yearly_pnl_plot(compare: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    years = compare["test_year"].astype(str).to_numpy()
    x = np.arange(len(years))
    width = 0.26
    ax.bar(x - width, compare["lightgbm_pnl"] / 1_000_000, width, label="Corrected LightGBM")
    ax.bar(x, compare["mlp_pnl"] / 1_000_000, width, label="MLP on F*")
    ax.bar(x + width, compare["mv_alibi_pnl"] / 1_000_000, width, label="TP2-MV-ALiBi")
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Normalized P&L ($M)")
    ax.set_title("Out-of-sample normalized P&L by test year")
    ax.legend(frameon=False, ncols=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "yearly_pnl_comparison.pdf")
    plt.close(fig)


def save_selected_plot(compare: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    years = compare["test_year"].astype(str).to_numpy()
    x = np.arange(len(years))
    width = 0.26
    ax.bar(x - width, compare["lightgbm_selected"], width, label="Corrected LightGBM")
    ax.bar(x, compare["mlp_selected"], width, label="MLP on F*")
    ax.bar(x + width, compare["mv_alibi_selected"], width, label="TP2-MV-ALiBi")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Selected trades")
    ax.set_title("Selected episode trades by test year")
    ax.legend(frameon=False, ncols=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "selected_trades_comparison.pdf")
    plt.close(fig)


def save_strategy_plot(mv: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    years = mv["test_year"].astype(str).to_numpy()
    ax.bar(years, mv["test_t1_selected"], label="T1")
    ax.bar(years, mv["test_k2_selected"], bottom=mv["test_t1_selected"], label="K2")
    ax.set_ylabel("Selected trades")
    ax.set_title("TP2-MV-ALiBi strategy selections")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mv_alibi_strategy_mix.pdf")
    plt.close(fig)


def save_feature_block_plot() -> pd.DataFrame:
    features = pd.read_csv(TABLE_DIR / "feature_dictionary.csv")
    selected = features[features["selected_for_transformer"].astype(bool)].copy()
    counts = selected["feature_block"].value_counts().rename_axis("block").reset_index(name="n_features")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh(counts["block"], counts["n_features"], color="#52796f")
    ax.set_xlabel("Selected features")
    ax.set_title("F* feature dictionary by block")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_blocks.pdf")
    plt.close(fig)
    return counts


def make_table_rows(df: pd.DataFrame) -> str:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"{int(row['test_year'])} & "
            f"{fmt_money(row['lightgbm_pnl'])} & {int(row['lightgbm_selected']):,} & "
            f"{fmt_money(row['mlp_pnl'])} & {int(row['mlp_selected']):,} & "
            f"{fmt_money(row['mv_alibi_pnl'])} & {int(row['mv_alibi_selected']):,} \\\\"
        )
    return "\n".join(rows)


def main() -> None:
    mv = pd.read_csv(TABLE_DIR / "mv_alibi_transformer_results.csv")
    mlp = pd.read_csv(TABLE_DIR / "mlp_baseline_results.csv")
    lgbm = pd.read_csv(ROOT / "ExpiryMLCorrection/tables/meta_selector_clean_t1k2_only_results.csv")
    with open(TABLE_DIR / "mv_alibi_transformer_summary.json", "r", encoding="utf-8") as fh:
        mv_summary = json.load(fh)
    with open(TABLE_DIR / "mlp_baseline_summary.json", "r", encoding="utf-8") as fh:
        mlp_summary = json.load(fh)
    with open(TABLE_DIR / "build_summary.json", "r", encoding="utf-8") as fh:
        build_summary = json.load(fh)

    compare = lgbm[["test_year", "norm_total_pnl", "n_selected"]].rename(
        columns={"norm_total_pnl": "lightgbm_pnl", "n_selected": "lightgbm_selected"}
    )
    compare = compare.merge(
        mlp[["test_year", "test_norm_total_pnl", "test_n_selected"]].rename(
            columns={"test_norm_total_pnl": "mlp_pnl", "test_n_selected": "mlp_selected"}
        ),
        on="test_year",
        how="inner",
    )
    compare = compare.merge(
        mv[["test_year", "test_norm_total_pnl", "test_n_selected"]].rename(
            columns={"test_norm_total_pnl": "mv_alibi_pnl", "test_n_selected": "mv_alibi_selected"}
        ),
        on="test_year",
        how="inner",
    )
    compare.to_csv(TABLE_DIR / "transformer_model_comparison.csv", index=False)

    save_yearly_pnl_plot(compare)
    save_selected_plot(compare)
    save_strategy_plot(mv)
    block_counts = save_feature_block_plot()

    lgbm_total = float(compare["lightgbm_pnl"].sum())
    mlp_total = float(compare["mlp_pnl"].sum())
    mv_total = float(compare["mv_alibi_pnl"].sum())
    lgbm_selected = int(compare["lightgbm_selected"].sum())
    mlp_selected = int(compare["mlp_selected"].sum())
    mv_selected = int(compare["mv_alibi_selected"].sum())

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=0.8in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\usepackage{{float}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}

\title{{TP2-MV-ALiBi Transformer Implementation Report}}
\author{{SPY American Call TP2 Project}}
\date{{\today}}

\begin{{document}}
\maketitle

\section{{What was implemented}}
This stage implements the professor-proposed transformer path on top of the corrected TP2 expiry-label pipeline. The project now has a separate \texttt{{TP2MVTransformer/}} folder containing: feature-dictionary construction, an MLP ablation, and the first multivariate ALiBi cross-attention transformer.

The transformer does not receive LightGBM predictions, tree leaves, SHAP vectors, or booster scores. Boosting is used only once to identify a stable entry-time feature dictionary \(\mathcal{{F}}^\star\). The model input is one token per TP2 episode and viable strategy, restricted to \(T1\) and \(K2\). The label is whether the corrected expiry payoff is positive under the \(\$1{{,}}000\) normalized exposure convention.

\section{{Feature dictionary}}
The token dataset contains {build_summary['token_rows']:,} strategy tokens from {build_summary['unique_episodes']:,} unique TP2 episodes. From {build_summary['candidate_features']} candidate point-in-time features, the feature discovery stage selected {build_summary['selected_features']} features for \(\mathcal{{F}}^\star\). The feature blocks are shown in Figure~\ref{{fig:blocks}}.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.86\linewidth]{{figures/feature_blocks.pdf}}
\caption{{Selected transformer feature dictionary by semantic block.}}
\label{{fig:blocks}}
\end{{figure}}

\section{{Rolling evaluation protocol}}
All models use the same rolling protocol: train three years, tune threshold on the next validation year, and evaluate once on the next test year. The six test years are 2021--2026. The feature \texttt{{signal\_year}} is excluded. Future payoff, realized return, holding-period, and label columns are excluded from model inputs.

\section{{Model comparison}}
\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{rrrrrrr}}
\toprule
Test year & LightGBM P\&L & LGBM trades & MLP P\&L & MLP trades & MV-ALiBi P\&L & MV trades \\
\midrule
{make_table_rows(compare)}
\bottomrule
\end{{tabular}}
\caption{{Out-of-sample normalized P\&L and selected trades by test year.}}
\end{{table}}

\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrrr}}
\toprule
Model & Total P\&L & Selected trades & Notes \\
\midrule
Corrected LightGBM T1/K2 & {fmt_money(lgbm_total)} & {lgbm_selected:,} & current strong tabular benchmark \\
MLP on \(\mathcal{{F}}^\star\) & {fmt_money(mlp_total)} & {mlp_selected:,} & no attention context \\
TP2-MV-ALiBi & {fmt_money(mv_total)} & {mv_selected:,} & historical cross-attention \\
\bottomrule
\end{{tabular}}
\caption{{Aggregate normalized P\&L from the saved result tables.}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.94\linewidth]{{figures/yearly_pnl_comparison.pdf}}
\caption{{Year-wise out-of-sample normalized P\&L. The transformer has larger aggregate P\&L but a clear 2022 loss, so robustness still needs work.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.94\linewidth]{{figures/selected_trades_comparison.pdf}}
\caption{{Selected episode trades by year. The transformer currently trades more aggressively than the corrected LightGBM selector.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.86\linewidth]{{figures/mv_alibi_strategy_mix.pdf}}
\caption{{Transformer-selected strategy mix. Only \(T1\) and \(K2\) are eligible.}}
\end{{figure}}

\section{{Transformer result}}
The current TP2-MV-ALiBi run selected {mv_summary['total_selected']:,} trades with total normalized P\&L of {fmt_money(mv_summary['total_norm_pnl'])}. It produced positive P\&L in {mv_summary['positive_years']} of {mv_summary['total_years']} test years. The combined daily Sharpe from selected transformer trades is {fmt_float(mv_summary['combined_daily_sharpe'])}, and the selected-trade win rate is {fmt_float(100 * mv_summary['combined_win_rate'], 1)}\%.

This should be treated as a first implemented transformer baseline, not the final conference result. The model is intentionally compact for CPU execution: it attends to the most recent {mv_summary['max_history_tokens']} prior TP2 strategy tokens. The important scientific result is that the professor's architecture now runs end-to-end on our corrected labels and rolling protocol.

\section{{Current caveats and next work}}
\begin{{itemize}}
    \item The transformer improves aggregate normalized P\&L versus the current MLP ablation and the saved corrected LightGBM yearly CSV, but 2022 is negative and must be diagnosed before claiming robustness.
    \item The first transformer uses fixed-size recent-history retrieval. The next version should add ablations: no-bias transformer, time-only ALiBi, time+surface, time+surface+short-D, and full multivariate bias.
    \item Threshold selection is validation-only and therefore leakage-safe, but the low selected-trade win rate means the model is relying on large winners. We need tail-risk diagnostics and capped-return robustness.
    \item The old \texttt{{ExpiryMLCorrection/meta\_selector\_clean\_summary.json}} appears stale relative to its yearly CSV; this report recomputes comparison totals from the yearly result table.
\end{{itemize}}

\section{{Files produced}}
\begin{{itemize}}
    \item \texttt{{TP2MVTransformer/data/tp2\_t1k2\_token\_dataset.parquet}}
    \item \texttt{{TP2MVTransformer/tables/feature\_dictionary.csv}}
    \item \texttt{{TP2MVTransformer/tables/mlp\_baseline\_results.csv}}
    \item \texttt{{TP2MVTransformer/tables/mv\_alibi\_transformer\_results.csv}}
    \item \texttt{{TP2MVTransformer/data/mv\_alibi\_transformer\_test\_predictions.parquet}}
\end{{itemize}}

\end{{document}}
"""
    (OUT / "transformer_implementation_report.tex").write_text(tex, encoding="utf-8")
    print(f"Wrote {OUT / 'transformer_implementation_report.tex'}")
    print(f"Wrote {TABLE_DIR / 'transformer_model_comparison.csv'}")


if __name__ == "__main__":
    main()
