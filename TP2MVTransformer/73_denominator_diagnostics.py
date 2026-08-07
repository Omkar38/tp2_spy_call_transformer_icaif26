"""brief-26: yearly denominator diagnostics + 2013 diagnosis.
Per year from the raw SPY call quote panel: eligible contract-quotes (denominator), median
relative spread, zero-bid share, weekly-expiry share; plus TP2 violations (episodes) and a
violation rate per thousand eligible contract-quotes. Diagnoses why 2013 is excluded (raw
count anomaly). The exact A/B/C/D rectangle enumeration needs the detector; the contract-quote
denominator is the reproducible proxy used to normalise the violation rate across years.
Artifact + CSV.
"""
from pathlib import Path
import json
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent; ROOT = OUT.parent; ACTION = OUT/"tables"/"action_list"
PANEL = ROOT/"data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet"
EPISODES = ROOT/"data/ml/archive2_ml_episode_dataset_2013_2026.parquet"

def third_friday(dates):
    d = pd.to_datetime(dates)
    return (d.dt.weekday == 4) & (d.dt.day >= 15) & (d.dt.day <= 21)   # 3rd Friday = monthly expiry

print("loading raw panel...")
p = pd.read_parquet(PANEL, columns=["date", "expiration_date", "bid", "ask", "mid", "open_interest", "volume", "strike"])
p["date"] = pd.to_datetime(p["date"]); p["year"] = p["date"].dt.year
p["rel_spread"] = np.where(p["mid"] > 0, (p["ask"] - p["bid"]) / p["mid"], np.nan)
p["zero_bid"] = p["bid"] <= 0
p["weekly"] = ~third_friday(p["expiration_date"])   # non-3rd-Friday = weekly/other

ep = pd.read_parquet(EPISODES, columns=["signal_date"]); ep["year"] = pd.to_datetime(ep["signal_date"]).dt.year
viol = ep.groupby("year").size()

rows = []
for y, g in p.groupby("year"):
    n_q = len(g); n_days = g["date"].nunique(); v = int(viol.get(y, 0))
    rows.append({"year": int(y), "eligible_contract_quotes": n_q, "trading_days": n_days,
                 "quotes_per_day": round(n_q / n_days, 1) if n_days else 0,
                 "violations": v, "viol_per_1k_quotes": round(1000 * v / n_q, 3) if n_q else 0,
                 "median_rel_spread": round(float(g["rel_spread"].median()), 4),
                 "zero_bid_share": round(float(g["zero_bid"].mean()), 4),
                 "weekly_expiry_share": round(float(g["weekly"].mean()), 4),
                 "median_open_interest": float(g["open_interest"].median()),
                 "median_volume": float(g["volume"].median())})
df = pd.DataFrame(rows).sort_values("year")
print(df.to_string(index=False))

# ---- 2013 diagnosis ----
r13 = df[df.year == 2013].iloc[0]; rest = df[(df.year >= 2014) & (df.year <= 2020)]
diag = {
    "2013_violations": int(r13["violations"]), "2014_2020_median_violations": int(rest["violations"].median()),
    "2013_viol_per_1k": float(r13["viol_per_1k_quotes"]), "2014_2020_median_viol_per_1k": float(rest["viol_per_1k_quotes"].median()),
    "2013_median_rel_spread": float(r13["median_rel_spread"]), "2014_2020_median_rel_spread": float(rest["median_rel_spread"].median()),
    "2013_zero_bid_share": float(r13["zero_bid_share"]), "2014_2020_median_zero_bid_share": float(rest["zero_bid_share"].median()),
    "2013_weekly_share": float(r13["weekly_expiry_share"]), "2014_2020_median_weekly_share": float(rest["weekly_expiry_share"].median()),
}
print("\n=== 2013 diagnosis ===")
for k, v in diag.items(): print(f"  {k}: {v}")

json.dump({"by_year": rows, "diagnosis_2013": diag,
           "note": "brief-26. Denominator = eligible contract-quotes in the cleaned panel (reproducible proxy; "
                   "exact A/B/C/D rectangle enumeration needs the detector). weekly = expiry not on the 3rd Friday. "
                   "2013 excluded on the raw count anomaly -- see diagnosis_2013 for the quote-quality drivers."},
          open(ACTION/"denominator_diagnostics.json", "w"), indent=2, default=float)
df.to_csv(ACTION/"denominator_diagnostics.csv", index=False)
print("\nwrote", ACTION/"denominator_diagnostics.json", "+ .csv")
