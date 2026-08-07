"""brief-22: date- and episode-clustered inference for the headline results.
Trade-level observations are dependent (many share dates and option legs), so we report
clustered/bootstrap 95% intervals rather than an independent-trade bootstrap:
  - date block bootstrap (moving blocks of trading days, block_len=10) on each model's daily P&L
  - episode bootstrap (resample selected episodes with replacement)
Plus the headline ALiBi-vs-plain paired daily-block-bootstrap P&L-difference interval.
Base point estimates reproduce Table 2. Beta t-stats are the market-exposure regression
(guide P3.4) and are noted, not recomputed here. metrics daily convention. Artifact + CSV.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent; DATA = OUT/"data"; ACTION = OUT/"tables"/"action_list"
ANN = np.sqrt(252); NB = 2000; BLOCK = 10; rng = np.random.default_rng(42)

def load(f, col):
    d = pd.read_parquet(DATA/f)
    if "selected" in d.columns: d = d[d["selected"].astype(bool)]
    d = d[["episode_id", "signal_date", col]].copy(); d["signal_date"] = pd.to_datetime(d["signal_date"])
    d = d.rename(columns={col: "pnl"}); return d

def daily_series(d):
    return d.groupby("signal_date")["pnl"].sum().sort_index()

def block_boot_daily(daily):
    x = daily.to_numpy(); n = len(x); nb = int(np.ceil(n / BLOCK)); maxs = n - BLOCK
    pnl = np.empty(NB); sh = np.empty(NB)
    for i in range(NB):
        starts = rng.integers(0, maxs + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        r = x[idx]; pnl[i] = r.sum(); sh[i] = r.mean() / r.std() * ANN if r.std() > 0 else np.nan
    return pnl, sh

def episode_boot(d):
    ep = d["episode_id"].to_numpy(); pnl_by = d.groupby("episode_id")["pnl"].sum(); eps = pnl_by.index.to_numpy(); vals = pnl_by.to_numpy()
    n = len(eps); out = np.empty(NB)
    for i in range(NB):
        out[i] = vals[rng.integers(0, n, size=n)].sum()
    return out

MODELS = {"LightGBM": ("lgbm_final_t1k2_selected_trades.parquet", "norm_pnl"),
          "MLP-5": ("mlp_5seed_ensemble_selected_trades.parquet", "target_norm_pnl"),
          "Transformer-plain-5": ("mv_no_alibi_5seed_selected_trades.parquet", "target_norm_pnl"),
          "Transformer-ALiBi-5": ("mv_alibi_transformer_v4final_selected_trades.parquet", "target_norm_pnl"),
          "Logistic": ("logistic_l2_selected_trades.parquet", "target_norm_pnl")}
def ci(a): return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]

result = {}; csv_rows = []; series = {}
print(f"{'model':22} {'PnL':>12} {'date-block 95% CI':>34} {'Sharpe 95% CI':>20}")
for name, (f, col) in MODELS.items():
    d = load(f, col); dl = daily_series(d); series[name] = dl
    pnl_b, sh_b = block_boot_daily(dl); ep_b = episode_boot(d)
    r = {"pnl_point": float(dl.sum()), "sharpe_point": float(dl.mean() / dl.std() * ANN),
         "pnl_ci_dateblock": ci(pnl_b), "sharpe_ci_dateblock": ci(sh_b), "pnl_ci_episode": ci(ep_b),
         "pnl_gt0_prob_dateblock": float((pnl_b > 0).mean())}
    result[name] = r
    csv_rows.append({"model": name, "pnl": r["pnl_point"], "pnl_lo": r["pnl_ci_dateblock"][0], "pnl_hi": r["pnl_ci_dateblock"][1],
                     "sharpe": r["sharpe_point"], "sharpe_lo": r["sharpe_ci_dateblock"][0], "sharpe_hi": r["sharpe_ci_dateblock"][1]})
    print(f"{name:22} ${r['pnl_point']:>11,.0f} [${r['pnl_ci_dateblock'][0]:>10,.0f},${r['pnl_ci_dateblock'][1]:>11,.0f}] "
          f"[{r['sharpe_ci_dateblock'][0]:>6.2f},{r['sharpe_ci_dateblock'][1]:>6.2f}]")

# headline: ALiBi vs plain paired daily-block-bootstrap P&L difference
a = series["Transformer-ALiBi-5"]; p = series["Transformer-plain-5"]
idxu = a.index.union(p.index); da = a.reindex(idxu, fill_value=0).to_numpy(); dp = p.reindex(idxu, fill_value=0).to_numpy()
diff = da - dp; n = len(diff); nb = int(np.ceil(n / BLOCK)); maxs = n - BLOCK; dd = np.empty(NB)
for i in range(NB):
    starts = rng.integers(0, maxs + 1, size=nb); idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]; dd[i] = diff[idx].sum()
alibi_vs_plain = {"pnl_diff_point": float(diff.sum()), "pnl_diff_ci_dateblock": ci(dd), "p_alibi_gt_plain": float((dd > 0).mean())}
print(f"\nALiBi - plain P&L diff: ${alibi_vs_plain['pnl_diff_point']:,.0f} 95% CI [{alibi_vs_plain['pnl_diff_ci_dateblock'][0]:,.0f}, {alibi_vs_plain['pnl_diff_ci_dateblock'][1]:,.0f}] | P(ALiBi>plain)={alibi_vs_plain['p_alibi_gt_plain']:.2f}")

json.dump({"models": result, "alibi_vs_plain": alibi_vs_plain, "n_resamples": NB, "block_len_days": BLOCK,
           "note": "brief-22. Date block bootstrap (moving blocks of trading days) + episode bootstrap; "
                   "trade-level dependence handled by clustering on date/episode. Beta t-stats = market-exposure "
                   "regression (guide P3.4), noted not recomputed. Point estimates reproduce Table 2."},
          open(ACTION/"clustered_inference.json", "w"), indent=2, default=float)
pd.DataFrame(csv_rows).to_csv(ACTION/"clustered_inference.csv", index=False)
print("\nwrote", ACTION/"clustered_inference.json", "+ .csv")
