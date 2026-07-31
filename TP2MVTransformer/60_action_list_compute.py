"""
Professor action-list computations that need NO retraining -- all derived from
existing per-trade / full-population prediction parquets already in the repo.

Covers: #2 (Table 2 Sortino/win/mean-ret/skew), #1 (selectivity sweep: threshold
vs N,Sharpe,Sortino,PnL,mean-ret per model), #4 inputs (T1/K2 at matched single
seed), #10 (metrics excluding the partial 2026 year).

Conventions match the paper's code path: daily-grouped Sharpe = mean/std*sqrt(252)
over per-day summed normalized P&L; Sortino uses downside std; per-trade return %
= target_norm_pnl/10 ($1,000 exposure convention); skew = scipy.stats.skew of
per-trade normalized P&L.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "TP2MVTransformer" / "data"
T = ROOT / "TP2MVTransformer" / "tables"
OUT = ROOT / "TP2MVTransformer" / "tables" / "action_list"
OUT.mkdir(parents=True, exist_ok=True)

ANN = np.sqrt(252)

def _pnl_col(df):
    return "target_norm_pnl" if "target_norm_pnl" in df.columns else "norm_pnl"

def metrics(df, pnl=None):
    """Full metric bundle on a set of selected trades."""
    if pnl is None:
        pnl = _pnl_col(df)
    r = df[pnl].astype(float)
    daily = df.groupby("signal_date")[pnl].sum()
    sharpe = daily.mean() / daily.std() * ANN if daily.std() > 0 else float("nan")
    dn = daily[daily < 0]
    sortino = daily.mean() / dn.std() * ANN if len(dn) > 1 and dn.std() > 0 else float("nan")
    eq = daily.sort_index().cumsum()
    maxdd = float((eq - eq.cummax()).min()) if len(eq) else float("nan")
    yr = df.groupby(df["signal_date"].dt.year)[pnl].sum()
    return {
        "n": int(len(df)), "pnl": float(r.sum()), "sharpe": float(sharpe),
        "sortino": float(sortino), "maxdd": maxdd,
        "win_rate": float((r > 0).mean()), "mean_ret_pct": float(r.mean() / 10.0),
        "skew": float(stats.skew(r)) if len(r) > 2 else float("nan"),
        "pos_years": int((yr > 0).sum()), "n_years": int(len(yr)),
    }

def load(path):
    df = pd.read_parquet(D / path)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    return df

def sel_by_thresh(pred, thr, valid_filter=True):
    """Full-population predictions -> episode-level selection at predicted_prob>=thr:
    keep the single highest-prob strategy per episode (the paper's canonical
    select_episode_actions rule: trade T1, trade K2, or skip -- one action/episode)."""
    df = pred.copy()
    if "is_valid" in df.columns and valid_filter:
        df = df[df["is_valid"].astype(bool)]
    if "strategy" in df.columns:
        df = df[df["strategy"].isin(["T1", "K2"])]
    cand = df[df["predicted_prob"] >= thr]
    if cand.empty or "episode_id" not in cand.columns:
        return cand
    return (cand.sort_values(["predicted_prob", "strategy"], ascending=[False, True])
                .groupby("episode_id", as_index=False).first())

# ---------- full-population prediction sets (for selectivity + p>=0.5 selection) ----------
POP = {
    "LightGBM (1)": ("lgbm_final_test_predictions.parquet", "norm_pnl"),
    "MLP (1)":      ("mlp_final_test_predictions.parquet", "target_norm_pnl"),
    "Transformer (1)": ("mv_no_alibi_ablation_test_predictions.parquet", "target_norm_pnl"),
    "Transformer (5)": ("mv_no_alibi_5seed_test_predictions.parquet", "target_norm_pnl"),
}
pops = {}
for k, (f, pcol) in POP.items():
    df = load(f)
    if "is_valid" in df.columns:
        df = df[df["is_valid"].astype(bool)]
    df = df[df["strategy"].isin(["T1", "K2"])] if "strategy" in df.columns else df
    pops[k] = (df, pcol)

# ---------- #2: Table 2 augmented metrics (selected at p>=0.5) ----------
print("=" * 70); print("#2  TABLE 2 AUGMENTED METRICS (p>=0.5 selection)"); print("=" * 70)
table2 = {}
# selected-trade files where they exist (canonical), else threshold the population
SEL_FILES = {
    "LightGBM (1)": "lgbm_final_t1k2_selected_trades.parquet",
    "MLP (5)": "mlp_5seed_ensemble_selected_trades.parquet",
    "Transformer (1)": "mv_no_alibi_ablation_selected_trades.parquet",
    "Transformer (5)": "mv_no_alibi_5seed_selected_trades.parquet",
}
rows = {}
# MLP(1) from population threshold
rows["MLP (1)"] = metrics(sel_by_thresh(pops["MLP (1)"][0], 0.5), "target_norm_pnl")
for name, f in SEL_FILES.items():
    df = load(f)
    if "selected" in df.columns:
        df = df[df["selected"].astype(bool)]
    rows[name] = metrics(df)
# Unconditional (all valid T1/K2 candidates) from any full pop
unc = pops["MLP (1)"][0]
rows["Unconditional"] = metrics(unc, "target_norm_pnl")
order = ["Unconditional", "LightGBM (1)", "MLP (1)", "MLP (5)", "Transformer (1)", "Transformer (5)"]
hdr = f"{'model':18s} {'N':>7s} {'PnL$':>12s} {'Sharpe':>7s} {'Sortino':>8s} {'Win%':>6s} {'MeanRet%':>9s} {'Skew':>7s} {'PosY':>5s}"
print(hdr)
for k in order:
    m = rows[k]
    print(f"{k:18s} {m['n']:>7,} {m['pnl']:>12,.0f} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
          f"{m['win_rate']*100:>5.1f} {m['mean_ret_pct']:>8.1f} {m['skew']:>7.1f} {m['pos_years']:>3}/{m['n_years']}")
    table2[k] = m
json.dump(table2, open(OUT / "table2_augmented.json", "w"), indent=2, default=float)

# ---------- #1: selectivity sweep ----------
print("\n" + "=" * 70); print("#1  SELECTIVITY SWEEP (threshold -> N, Sharpe, Sortino, PnL, MeanRet%)"); print("=" * 70)
thresholds = [round(x, 2) for x in np.arange(0.30, 0.86, 0.05)]
sweep_rows = []
for name, (df, pcol) in pops.items():
    for thr in thresholds:
        s = sel_by_thresh(df, thr)
        if len(s) < 20:
            continue
        m = metrics(s, pcol)
        sweep_rows.append({"model": name, "threshold": thr, **m})
sweep = pd.DataFrame(sweep_rows)
sweep.to_csv(OUT / "selectivity_sweep.csv", index=False)
# matched-N comparison: for a grid of N, interpolate Sharpe per model
print("Matched-N Sharpe (MLP(1) vs Transformer(1) vs Transformer(5)):")
for Ntarget in [8000, 10000, 12000, 14000, 16000]:
    line = f"  N~{Ntarget:>6,}: "
    for name in ["MLP (1)", "Transformer (1)", "Transformer (5)"]:
        sub = sweep[sweep.model == name].sort_values("n")
        if sub.n.min() <= Ntarget <= sub.n.max():
            sh = np.interp(Ntarget, sub.n, sub.sharpe)
            so = np.interp(Ntarget, sub.n, sub.sortino)
            line += f"{name.split()[0][:4]}{name.split()[1]} S={sh:.2f}/So={so:.2f}  "
    print(line)

# ---------- #4: T1/K2 at matched single seed ----------
print("\n" + "=" * 70); print("#4  T1/K2 BREAKDOWN AT MATCHED SINGLE SEED"); print("=" * 70)
single_sel = {
    "LightGBM (1)": load("lgbm_final_t1k2_selected_trades.parquet"),
    "MLP (1)": sel_by_thresh(pops["MLP (1)"][0], 0.5),
    "Transformer (1)": load("mv_no_alibi_ablation_selected_trades.parquet"),
}
t1k2 = {}
for name, df in single_sel.items():
    if "selected" in df.columns:
        df = df[df["selected"].astype(bool)]
    row = {}
    for strat in ["T1", "K2"]:
        sub = df[df["strategy"] == strat]
        row[strat] = metrics(sub)
    row["Combined"] = metrics(df)
    t1k2[name] = row
    print(f"{name:16s} T1: ${row['T1']['pnl']:>10,.0f} S={row['T1']['sharpe']:.2f}  "
          f"K2: ${row['K2']['pnl']:>10,.0f} S={row['K2']['sharpe']:.2f}  "
          f"Comb: ${row['Combined']['pnl']:>10,.0f}")
json.dump({k: v for k, v in t1k2.items()}, open(OUT / "t1k2_matched_single.json", "w"), indent=2, default=float)

# ---------- #10: exclude partial 2026 ----------
print("\n" + "=" * 70); print("#10  METRICS EXCLUDING PARTIAL 2026 (2021-2025 only)"); print("=" * 70)
excl = {}
for name in order:
    if name == "Unconditional":
        df = unc.copy(); pcol = "target_norm_pnl"
        df = df[df["signal_date"].dt.year <= 2025]
    else:
        f = SEL_FILES.get(name)
        if f is None and name == "MLP (1)":
            df = sel_by_thresh(pops["MLP (1)"][0], 0.5); pcol = "target_norm_pnl"
        else:
            df = load(f)
            if "selected" in df.columns:
                df = df[df["selected"].astype(bool)]
            pcol = _pnl_col(df)
        df = df[df["signal_date"].dt.year <= 2025]
    m = metrics(df, pcol)
    excl[name] = m
    print(f"{name:18s} N={m['n']:>7,} PnL=${m['pnl']:>12,.0f} Sharpe={m['sharpe']:.2f} PosY={m['pos_years']}/{m['n_years']}")
json.dump(excl, open(OUT / "metrics_excl_2026.json", "w"), indent=2, default=float)

print("\nDONE -> outputs in", OUT)
