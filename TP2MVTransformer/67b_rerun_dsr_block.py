"""67b: rerun ONLY the DSR block of script 67 after the one-line MLP(1) file fix
(mlp_baseline -> mlp_final), reusing the already-saved MLP-5 predictions so no
30-60 min retrain is needed. Imports script 67's exact deflated_sharpe / metrics /
sel_by_thresh (module-level) and copies its CONFIG_SHARPE verbatim, so the numbers
are identical to what a full corrected run of 67 would produce.

Verifies both gates before writing anything:
  MLP(1) from mlp_final @0.5  -> N=14,258, Sharpe 2.60
  MLP(5) from saved preds @0.5 -> N=24,977, Sharpe 1.66
If either gate fails it aborts without writing.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "TP2MVTransformer" / "data"
OUT = ROOT / "TP2MVTransformer" / "tables" / "action_list"

spec = importlib.util.spec_from_file_location("s67", ROOT / "TP2MVTransformer" / "67_mlp5_selectivity_and_dsr_fix.py")
s67 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s67)  # guarded by __main__, so main() does not run on import

# CONFIG_SHARPE copied verbatim from 67.main() (identical 21-trial list as script 63)
CONFIG_SHARPE = {
    "LightGBM(1)": 1.25, "LightGBM 4-strat(1)": -1.76, "MLP(1)": 2.60, "MLP(5)": 1.66,
    "Transformer plain(1)": 1.73, "Transformer plain(5)": 2.18,
    "Transformer ALiBi(1)": 1.51, "Transformer ALiBi(5)": 2.36,
    "No-history(1)": 0.87, "Logistic-L2(1)": 2.27, "Random@N": -1.14,
    "Expanding-window TF(1)": 1.42, "Fraction-cap 0.10": 1.55, "Fraction-cap 0.20": 1.60,
    "Fraction-cap 0.30": 1.58, "ALiBi 10-seed": 2.30, "Prob-floor 0.10": 1.70,
    "Prob-floor 0.20": 1.72, "Prob-floor 0.30": 1.75, "Val-tuned-threshold TF": 1.40,
    "Top-K fraction rule": 1.35,
}
var_sr = float(np.var(list(CONFIG_SHARPE.values()), ddof=1))


def load(path):
    df = pd.read_parquet(D / path); df["signal_date"] = pd.to_datetime(df["signal_date"]); return df


def main():
    # MLP(5): reuse the saved full-population predictions from the passing run
    mlp5 = load("mlp_5seed_ensemble_test_predictions.parquet")
    sel5 = s67.sel_by_thresh(mlp5, 0.5); m5 = s67.metrics(sel5)
    # MLP(1): the CORRECTED file
    mlp1 = load("mlp_final_test_predictions.parquet")
    sel1 = s67.sel_by_thresh(mlp1, 0.5); m1 = s67.metrics(sel1)

    print(f"MLP(5) gate: N={m5['n']:,} Sharpe={m5['sharpe']:.2f}  (expect 24,977 / 1.66)")
    print(f"MLP(1) gate: N={m1['n']:,} Sharpe={m1['sharpe']:.2f}  (expect 14,258 / 2.60)")
    g5 = abs(m5['n'] - 24977) <= 250 and abs(m5['sharpe'] - 1.66) <= 0.05
    g1 = abs(m1['n'] - 14258) <= 50 and abs(m1['sharpe'] - 2.60) <= 0.02
    if not (g5 and g1):
        print("GATE FAILED -- aborting, nothing written."); return

    series = {("MLP", 1): sel1.groupby("signal_date")["target_norm_pnl"].sum(),
              ("MLP", 5): sel5.groupby("signal_date")["target_norm_pnl"].sum()}
    sr_ann = {("MLP", 1): m1["sharpe"], ("MLP", 5): m5["sharpe"]}
    results = {}
    for (name, seeds), daily in series.items():
        for n_trials in (21, 25):
            res = s67.deflated_sharpe(daily, sr_ann[(name, seeds)], n_trials, var_sr)
            res["configs_note"] = ("21 evaluated configurations; 25 adds the four other seeds "
                                   "of the architecture whose reported single-seed row is the max over five")
            path = OUT / f"dsr_{name}_{seeds}seed_ntrials{n_trials}.json"
            json.dump(res, open(path, "w"), indent=2, default=float)
            results[f"{name}{seeds}_n{n_trials}"] = res["DSR"]
            print(f"  DSR {name}({seeds}) n_trials={n_trials}: {res['DSR']:.3f} -> {path.name}")
    print("\nCorrected MLP(1) DSR (ntrials21):", round(results["MLP1_n21"], 3),
          "| ntrials25:", round(results["MLP1_n25"], 3))


if __name__ == "__main__":
    main()
