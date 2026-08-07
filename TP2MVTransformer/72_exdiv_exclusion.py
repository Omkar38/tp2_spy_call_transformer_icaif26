"""brief-16: primary metrics EXCLUDING trades whose short leg (D, expiry T1) spans an ex-dividend
date, with the full sample as robustness. Ex-div flag replicates 49_execution_robustness.add_exdiv_flag:
exposed = next_ex_div_date (at signal time) <= feature_D_expiration. Uses each model's OWN committed
selected-trade P&L (so the full-sample gate reproduces Table 2 exactly) and left-merges the flag.
metrics() copied VERBATIM from 60_action_list_compute.py. Artifact + CSV.
"""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent; DATA = OUT/"data"; ACTION = OUT/"tables"/"action_list"
def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m49 = _load("exec49", OUT/"49_execution_robustness.py")
LABELS_PATH, DIVDATE_PATH = m49.LABELS_PATH, m49.DIVDATE_PATH

ANN = np.sqrt(252)
def metrics(df, pnl):   # verbatim from 60
    r = df[pnl].astype(float); daily = df.groupby("signal_date")[pnl].sum()
    sharpe = daily.mean() / daily.std() * ANN if daily.std() > 0 else float("nan")
    dn = daily[daily < 0]; sortino = daily.mean() / dn.std() * ANN if len(dn) > 1 and dn.std() > 0 else float("nan")
    eq = daily.sort_index().cumsum(); maxdd = float((eq - eq.cummax()).min()) if len(eq) else float("nan")
    yr = df.groupby(df["signal_date"].dt.year)[pnl].sum()
    return {"n": int(len(df)), "pnl": float(r.sum()), "sharpe": float(sharpe), "sortino": float(sortino),
            "maxdd": maxdd, "win_rate": float((r > 0).mean()), "mean_ret_pct": float(r.mean() / 10.0),
            "skew": float(stats.skew(r)) if len(r) > 2 else float("nan"),
            "pos_years": int((yr > 0).sum()), "n_years": int(len(yr))}

# ex-div exposure flag per (episode_id, strategy)
lab = pd.read_parquet(LABELS_PATH, columns=["episode_id", "strategy", "signal_date", "exit_policy", "feature_D_expiration"])
lab = lab[(lab["exit_policy"] == "exit_selected_leg_expiration_payoff") & lab["strategy"].isin(["T1", "K2"])].copy()
lab["signal_date"] = pd.to_datetime(lab["signal_date"]); lab["feature_D_expiration"] = pd.to_datetime(lab["feature_D_expiration"])
div = pd.read_parquet(DIVDATE_PATH, columns=["date", "next_ex_div_date"]).drop_duplicates("date")
div["date"] = pd.to_datetime(div["date"]); div["next_ex_div_date"] = pd.to_datetime(div["next_ex_div_date"])
lab = lab.merge(div, left_on="signal_date", right_on="date", how="left")
lab["exdiv_exposed"] = lab["next_ex_div_date"].notna() & (lab["next_ex_div_date"] <= lab["feature_D_expiration"])
flag = lab[["episode_id", "strategy", "exdiv_exposed"]].drop_duplicates(["episode_id", "strategy"])

MODELS = {  # (file, pnl_col)
    "LightGBM": ("lgbm_final_t1k2_selected_trades.parquet", "norm_pnl"),
    "MLP-5": ("mlp_5seed_ensemble_selected_trades.parquet", "target_norm_pnl"),
    "Transformer-plain-5": ("mv_no_alibi_5seed_selected_trades.parquet", "target_norm_pnl"),
    "Transformer-ALiBi-5": ("mv_alibi_transformer_v4final_selected_trades.parquet", "target_norm_pnl"),
    "Logistic": ("logistic_l2_selected_trades.parquet", "target_norm_pnl"),
}
GATE = {"LightGBM": (19636, 682411), "MLP-5": (24977, 3433936), "Transformer-plain-5": (14338, 695173),
        "Transformer-ALiBi-5": (14648, 703705)}  # Logistic gate is the known-pending row -> committed 13,206

result = {}; csv_rows = []
print(f"{'model':22} {'full N':>8} {'full Sh':>8} {'excl N':>8} {'excl Sh':>8} {'%exposed':>9} {'exposed PnL':>12}")
for name, (f, col) in MODELS.items():
    d = pd.read_parquet(DATA/f)
    if "selected" in d.columns: d = d[d["selected"].astype(bool)]
    d = d[["episode_id", "signal_date", "strategy", col]].copy(); d["signal_date"] = pd.to_datetime(d["signal_date"])
    d = d.merge(flag, on=["episode_id", "strategy"], how="left"); d["exdiv_exposed"] = d["exdiv_exposed"].fillna(False)
    full = metrics(d, col); excl = metrics(d[~d["exdiv_exposed"]], col); exp = d[d["exdiv_exposed"]]
    if name in GATE:
        gn, gp = GATE[name]; assert full["n"] == gn and abs(full["pnl"] - gp) < 5000, f"{name} full-sample gate FAILED ({full['n']}/{full['pnl']:.0f} vs {gn}/{gp})"
    result[name] = {"full": full, "excl_exdiv": excl, "pct_exposed": float(d["exdiv_exposed"].mean()),
                    "exposed_n": int(len(exp)), "exposed_pnl": float(exp[col].sum()),
                    "exposed_win_rate": float((exp[col] > 0).mean()) if len(exp) else float("nan")}
    csv_rows.append({"model": name, "full_n": full["n"], "full_pnl": full["pnl"], "full_sharpe": full["sharpe"],
                     "excl_n": excl["n"], "excl_pnl": excl["pnl"], "excl_sharpe": excl["sharpe"],
                     "pct_exposed": result[name]["pct_exposed"], "exposed_pnl": result[name]["exposed_pnl"],
                     "exposed_win_rate": result[name]["exposed_win_rate"]})
    print(f"{name:22} {full['n']:>8,} {full['sharpe']:>8.2f} {excl['n']:>8,} {excl['sharpe']:>8.2f} "
          f"{result[name]['pct_exposed']*100:>8.1f}% ${result[name]['exposed_pnl']:>11,.0f}")

json.dump({"models": result, "flag_rule": "exposed = next_ex_div_date(at signal) <= feature_D_expiration (short leg D / T1)",
           "note": "brief-16. Full-sample gate reproduces Table 2 (Logistic uses committed 13,206). "
                   "metrics() verbatim from 60; ex-div flag replicates 49_execution_robustness."},
          open(ACTION/"exdiv_exclusion_rerun.json", "w"), indent=2, default=float)
pd.DataFrame(csv_rows).to_csv(ACTION/"exdiv_exclusion_rerun.csv", index=False)
print("\nwrote", ACTION/"exdiv_exclusion_rerun.json", "+ .csv")
