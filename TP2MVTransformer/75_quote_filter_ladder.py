"""brief-27: quote-quality filter ladder on each model's selected trades.
Rungs: base -> no zero bids (already satisfied: cleaned panel) -> max relative spread (all 4 legs)
-> min open interest (all legs) -> min volume (all legs). Per-leg quote quality from the episode
dataset (rectangle-level, keyed by episode_id). Shows the results survive stricter quote quality.
Base rung reproduces Table 2. 'synchronized subset' needs per-leg quote timestamps (not in the
panel) -> noted. metrics() verbatim from 60. Artifact + CSV.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent; ROOT = OUT.parent; DATA = OUT/"data"; ACTION = OUT/"tables"/"action_list"
EPISODES = ROOT/"data/ml/archive2_ml_episode_dataset_2013_2026.parquet"
ANN = np.sqrt(252)
def metrics(df, pnl):   # verbatim from 60
    r = df[pnl].astype(float); daily = df.groupby("signal_date")[pnl].sum()
    sharpe = daily.mean() / daily.std() * ANN if daily.std() > 0 else float("nan")
    dn = daily[daily < 0]; sortino = daily.mean() / dn.std() * ANN if len(dn) > 1 and dn.std() > 0 else float("nan")
    yr = df.groupby(df["signal_date"].dt.year)[pnl].sum()
    return {"n": int(len(df)), "pnl": float(r.sum()), "sharpe": float(sharpe), "sortino": float(sortino),
            "win_rate": float((r > 0).mean()), "pos_years": int((yr > 0).sum())}

# per-episode (rectangle) quote quality across the 4 legs
legs = ["A", "B", "C", "D"]
cols = ["episode_id"] + [f"feature_{l}_{q}" for l in legs for q in ("spread_pct", "open_interest", "volume")]
ep = pd.read_parquet(EPISODES, columns=cols)
ep["max_spread_pct"] = ep[[f"feature_{l}_spread_pct" for l in legs]].max(axis=1)
ep["min_oi"] = ep[[f"feature_{l}_open_interest" for l in legs]].min(axis=1)
ep["min_vol"] = ep[[f"feature_{l}_volume" for l in legs]].min(axis=1)
q = ep[["episode_id", "max_spread_pct", "min_oi", "min_vol"]]

MODELS = {"LightGBM": ("lgbm_final_t1k2_selected_trades.parquet", "norm_pnl"),
          "MLP-5": ("mlp_5seed_ensemble_selected_trades.parquet", "target_norm_pnl"),
          "Transformer-plain-5": ("mv_no_alibi_5seed_selected_trades.parquet", "target_norm_pnl"),
          "Transformer-ALiBi-5": ("mv_alibi_transformer_v4final_selected_trades.parquet", "target_norm_pnl"),
          "Logistic": ("logistic_l2_selected_trades.parquet", "target_norm_pnl")}
GATE = {"LightGBM": 19636, "MLP-5": 24977, "Transformer-plain-5": 14338, "Transformer-ALiBi-5": 14648}

# ladder rungs: (label, predicate on merged df)
RUNGS = [
    ("base (all selected)", lambda x: x.index == x.index),
    ("max_spread<=10%", lambda x: x["max_spread_pct"] <= 0.10),
    ("max_spread<=5%", lambda x: x["max_spread_pct"] <= 0.05),
    ("min_OI>=10", lambda x: x["min_oi"] >= 10),
    ("min_volume>=1", lambda x: x["min_vol"] >= 1),
    ("all: spread<=10% & OI>=10 & vol>=1", lambda x: (x["max_spread_pct"] <= 0.10) & (x["min_oi"] >= 10) & (x["min_vol"] >= 1)),
]
result = {}; csv_rows = []
for name, (f, col) in MODELS.items():
    d = pd.read_parquet(DATA/f)
    if "selected" in d.columns: d = d[d["selected"].astype(bool)]
    d = d[["episode_id", "signal_date", "strategy", col]].copy(); d["signal_date"] = pd.to_datetime(d["signal_date"])
    d = d.merge(q, on="episode_id", how="left")
    rungs = {}
    print(f"\n{name}:")
    for lab, pred in RUNGS:
        sub = d[pred(d)]; m = metrics(sub, col); rungs[lab] = m
        if lab.startswith("base") and name in GATE: assert m["n"] == GATE[name], f"{name} base gate FAILED {m['n']}"
        print(f"  {lab:38} N={m['n']:>7,} PnL=${m['pnl']:>11,.0f} Sh={m['sharpe']:>6.2f} win={m['win_rate']*100:.1f}%")
        csv_rows.append({"model": name, "rung": lab, **m})
    result[name] = rungs

json.dump({"models": result, "note": "brief-27. Quote-quality ladder on selected trades. 'no zero bids' already "
           "satisfied (cleaned panel). 'synchronized subset' needs per-leg quote timestamps (not available). "
           "Base rung reproduces Table 2. Per-leg quality from episode dataset (rectangle-level). metrics() verbatim from 60."},
          open(ACTION/"quote_filter_ladder.json", "w"), indent=2, default=float)
pd.DataFrame(csv_rows).to_csv(ACTION/"quote_filter_ladder.csv", index=False)
print("\nwrote", ACTION/"quote_filter_ladder.json", "+ .csv")
