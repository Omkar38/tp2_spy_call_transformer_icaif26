"""No-training compute for #12 (trade-level market-neutrality), #14 (Deflated
Sharpe + config count), #17 (exit-policy: reversion vs hold-to-expiry)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "TP2MVTransformer" / "data"
LAB = ROOT / "data" / "labels"
TOK = D / "tp2_t1k2_token_dataset.parquet"
OUT = ROOT / "TP2MVTransformer" / "tables" / "action_list"
OUT.mkdir(parents=True, exist_ok=True)

def load_sel(path, thr=0.5):
    df = pd.read_parquet(D / path)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    if "selected" in df.columns:
        df = df[df["selected"].astype(bool)]
    return df

SEL = {
    "MLP (5)": "mlp_5seed_ensemble_selected_trades.parquet",
    "Transformer (5)": "mv_no_alibi_5seed_selected_trades.parquet",
    "LightGBM (1)": "lgbm_final_t1k2_selected_trades.parquet",
}

# ================= #12 trade-level market neutrality =================
print("=" * 70); print("#12  TRADE-LEVEL MARKET NEUTRALITY (return vs underlying over holding window)"); print("=" * 70)
tok = pd.read_parquet(TOK, columns=["episode_id", "strategy", "signal_date", "feature_underlying_price",
                                     "label_holding_days_T1", "label_holding_days_K2"] )
tok["signal_date"] = pd.to_datetime(tok["signal_date"])
# date -> underlying price map (median across that day's episodes), for exit lookup via asof
upx = (tok.groupby("signal_date")["feature_underlying_price"].median().reset_index()
          .rename(columns={"feature_underlying_price": "u_px"}).sort_values("signal_date"))

def exit_underlying(entry_dates, hold_days):
    exit_dates = entry_dates + pd.to_timedelta(hold_days.fillna(0).clip(lower=0), unit="D")
    q = pd.DataFrame({"exit_date": exit_dates}).sort_values("exit_date")
    m = pd.merge_asof(q, upx.rename(columns={"signal_date": "exit_date"}), on="exit_date", direction="nearest")
    return m.sort_index()["u_px"].values

mn_rows = {}
for name, f in SEL.items():
    sel = load_sel(f)
    j = sel.merge(tok[["episode_id", "strategy", "signal_date", "feature_underlying_price",
                       "label_holding_days_T1", "label_holding_days_K2"]],
                  on=["episode_id", "strategy", "signal_date"], how="left")
    hold = np.where(j["strategy"] == "T1", j["label_holding_days_T1"], j["label_holding_days_K2"])
    j["hold_days"] = pd.Series(hold, index=j.index)
    entry_px = j["feature_underlying_price"].values
    exit_px = exit_underlying(j["signal_date"], j["hold_days"])
    j["u_ret"] = exit_px / entry_px - 1.0
    pnlcol = "target_norm_pnl" if "target_norm_pnl" in j.columns else "norm_pnl"
    j["trade_ret"] = j[pnlcol].astype(float) / 1000.0
    d = j.dropna(subset=["u_ret", "trade_ret"])
    d = d[np.isfinite(d["u_ret"]) & np.isfinite(d["trade_ret"])]
    x = d["u_ret"].values; y = d["trade_ret"].values
    slope, intercept, r, p, se = stats.linregress(x, y)
    tstat = slope / se if se > 0 else float("nan")
    mn_rows[name] = {"n": int(len(d)), "beta_underlying": float(slope), "t": float(tstat),
                     "p": float(p), "r2": float(r**2), "mean_hold_days": float(np.nanmean(d["hold_days"]))}
    print(f"{name:16s} n={len(d):>6,} beta_uret={slope:+.3f} t={tstat:+.2f} p={p:.3f} R^2={r**2:.4f} meanHold={np.nanmean(d['hold_days']):.1f}d")
json.dump(mn_rows, open(OUT / "market_neutrality_tradelevel.json", "w"), indent=2, default=float)

# ================= #14 Deflated Sharpe Ratio =================
print("\n" + "=" * 70); print("#14  DEFLATED SHARPE RATIO (Bailey & Lopez de Prado 2014)"); print("=" * 70)
# Enumerate configurations evaluated on 2021-2026 (>=20 trials): 6yr daily Sharpe each.
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
N_trials = len(CONFIG_SHARPE)
sr_list = np.array(list(CONFIG_SHARPE.values()))
var_sr_trials = float(np.var(sr_list, ddof=1))  # variance of (annualized) Sharpe across trials

# headline model daily returns for skew/kurtosis + T
mlp = load_sel("mlp_5seed_ensemble_selected_trades.parquet")  # MLP(5) as a robust primary; also do MLP(1)
def daily_series(sel):
    return sel.groupby("signal_date")["target_norm_pnl"].sum()

def deflated_sharpe(daily_pnl, sr_ann, n_trials, var_sr_ann):
    T = len(daily_pnl)
    sr = sr_ann / np.sqrt(252)                       # de-annualize to per-day
    g3 = float(stats.skew(daily_pnl))
    g4 = float(stats.kurtosis(daily_pnl, fisher=False))
    var_sr_day = var_sr_ann / 252.0
    euler = 0.5772156649
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr0 = np.sqrt(var_sr_day) * ((1 - euler) * z1 + euler * z2)   # expected max Sharpe under null (per-day)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4.0 * sr**2)
    dsr = stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom)
    return {"T": int(T), "sr_ann": float(sr_ann), "skew": g3, "kurtosis": g4,
            "n_trials": int(n_trials), "sr0_ann": float(sr0 * np.sqrt(252)), "DSR": float(dsr)}

for label, selname, srann in [("MLP(1)", "mlp_final_test_predictions.parquet", 2.60),
                              ("MLP(5)", "mlp_5seed_ensemble_selected_trades.parquet", 1.66),
                              ("Transformer plain(5)", "mv_no_alibi_5seed_selected_trades.parquet", 2.18)]:
    if selname == "mlp_final_test_predictions.parquet":
        pred = pd.read_parquet(D / selname); pred["signal_date"] = pd.to_datetime(pred["signal_date"])
        cand = pred[pred["predicted_prob"] >= 0.5].sort_values(["predicted_prob", "strategy"], ascending=[False, True]).groupby("episode_id", as_index=False).first()
        dp = cand.groupby("signal_date")["target_norm_pnl"].sum()
    else:
        dp = daily_series(load_sel(selname))
    res = deflated_sharpe(dp.values, srann, N_trials, var_sr_trials)
    print(f"{label:22s} T={res['T']} SR={srann:.2f} skew={res['skew']:.2f} kurt={res['kurtosis']:.1f} "
          f"N_trials={N_trials} SR0={res['sr0_ann']:.2f} DSR={res['DSR']:.3f}")
    json.dump({**res, "var_sr_ann_across_trials": var_sr_trials, "configs": list(CONFIG_SHARPE.keys())},
              open(OUT / f"dsr_{label.split('(')[0].strip().replace(' ','_')}_{label[-3]}.json", "w"), indent=2, default=float)

# ================= #17 exit-policy comparison =================
print("\n" + "=" * 70); print("#17  EXIT POLICY: reversion vs hold-to-expiry (MLP-selected episodes)"); print("=" * 70)
try:
    mb = pd.read_parquet(LAB / "archive2_episode_strategy_mini_backtest_conservative_bidask_2013_2026.parquet")
    print("backtest_policy values:", sorted(mb["backtest_policy"].dropna().unique().tolist()))
    sel = load_sel("mlp_5seed_ensemble_selected_trades.parquet")[["episode_id", "strategy", "signal_date"]]
    sel["signal_date"] = pd.to_datetime(sel["signal_date"])
    mb["signal_date"] = pd.to_datetime(mb["signal_date"])
    mb = mb[(mb["is_valid_backtest"] == True) & (mb["signal_date"].dt.year >= 2021)]
    j = sel.merge(mb, on=["episode_id", "strategy", "signal_date"], how="inner")
    summ = {}
    for pol, g in j.groupby("backtest_policy"):
        norm = g["net_pnl"] / g["entry_gross_exposure"] * 1000.0
        daily = g.assign(np=norm).groupby("signal_date")["np"].sum()
        sh = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else float("nan")
        summ[str(pol)] = {"n": int(len(g)), "total_norm_pnl": float(norm.sum()),
                          "sharpe": sh, "mean_hold_days": float(g["actual_holding_days"].mean()),
                          "reversion_share": float((g["exit_reason"].astype(str).str.contains("revers", case=False)).mean())}
        print(f"  policy={pol:22s} n={len(g):>6,} normPnL=${norm.sum():>12,.0f} sharpe={sh:.2f} meanHold={g['actual_holding_days'].mean():.1f}d")
    json.dump(summ, open(OUT / "exit_policy_comparison.json", "w"), indent=2, default=float)
except Exception as e:
    print("EXIT-POLICY compute failed:", repr(e))

print("\nDONE -> outputs in", OUT)
