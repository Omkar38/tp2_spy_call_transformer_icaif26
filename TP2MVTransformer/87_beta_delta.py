"""P3.4: beta vs net-short entry delta. The position buys A,B and sells C,D, so its net entry delta
= dA+dB-dC-dD is typically NEGATIVE (net-short). Yet the realized beta (trade return on underlying move)
is positive. Test whether short-gamma path-dependence explains it: regress trade return on the underlying
move (linear beta), then add the squared move (gamma/convexity proxy). If the linear beta is not a
directional bet but a convexity effect, gamma absorbs it. Computed on the NEW MLP-5 selected trades
(the row that flagged beta=+7.2). Underlying move = SPY(exit)/SPY(signal)-1. Artifact + CSV.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent; ROOT = OUT.parent; DATA = OUT/"data"; ACTION = OUT/"tables"/"action_list"; NESTED = ROOT/"FeatureDataDocs_2026-08-04"/"nested"
# net entry delta per episode
e = pd.read_parquet(ROOT/"data/ml/archive2_ml_episode_dataset_2013_2026.parquet",
                    columns=["episode_id"] + [f"feature_{l}_delta" for l in "ABCD"])
e["net_delta"] = e.feature_A_delta + e.feature_B_delta - e.feature_C_delta - e.feature_D_delta
netd = e.set_index("episode_id")["net_delta"]
# underlying per date, exit date + return from labels
u = pd.read_parquet(ROOT/"data/processed/archive2_spy_call_allmoneyness_lastquote_clean_2013_2026.parquet", columns=["date", "spy_close"]).drop_duplicates("date")
u["date"] = pd.to_datetime(u["date"]); spot = u.set_index("date")["spy_close"]
lab = pd.read_parquet(ROOT/"data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet",
                      columns=["episode_id", "strategy", "signal_date", "exit_date", "exit_policy", "is_valid_backtest", "realized_return"])
lab = lab[(lab.exit_policy == "exit_selected_leg_expiration_payoff") & lab.is_valid_backtest & lab.strategy.isin(["T1", "K2"])].copy()
lab["signal_date"] = pd.to_datetime(lab.signal_date); lab["exit_date"] = pd.to_datetime(lab.exit_date)
lab["s0"] = lab.signal_date.map(spot); lab["s1"] = lab.exit_date.map(spot)
lab["move"] = lab.s1/lab.s0 - 1.0; lab["net_delta"] = lab.episode_id.map(netd)

def ols(y, X):   # X includes intercept column
    b, *_ = np.linalg.lstsq(X, y, rcond=None); yh = X @ b; ss = ((y - yh)**2).sum(); tss = ((y - y.mean())**2).sum()
    return b, 1 - ss/tss

def analyze(df, name):
    df = df.dropna(subset=["move", "net_delta", "realized_return"]); y = df.realized_return.to_numpy()
    mv = df.move.to_numpy(); n = len(df)
    b1, r2_1 = ols(y, np.column_stack([np.ones(n), mv]))                    # return ~ move (linear beta)
    b2, r2_2 = ols(y, np.column_stack([np.ones(n), mv, mv**2]))             # + gamma proxy (move^2)
    out = {"n": n, "mean_net_delta": float(df.net_delta.mean()), "pct_net_short": float((df.net_delta < 0).mean()),
           "beta_linear": float(b1[1]), "r2_linear": float(r2_1),
           "beta_with_gamma": float(b2[1]), "gamma_coef": float(b2[2]), "r2_with_gamma": float(r2_2)}
    print(f"{name}: n={n:,} net-short={out['pct_net_short']*100:.0f}% mean_delta={out['mean_net_delta']:.2f} | "
          f"beta_linear={out['beta_linear']:+.2f} (R2={r2_1:.3f}) -> +gamma: beta={out['beta_with_gamma']:+.2f}, "
          f"gamma_coef={out['gamma_coef']:+.1f} (R2={r2_2:.3f})")
    return out

result = {}
for name, f, col in [("MLP-5(new)", NESTED/"trades_mlp5.parquet", "target_norm_pnl"),
                     ("Transformer-ALiBi-5(new)", NESTED/"trades_transformer_alibi.parquet", "target_norm_pnl")]:
    sel = pd.read_parquet(f)[["episode_id", "strategy"]].drop_duplicates()
    df = sel.merge(lab, on=["episode_id", "strategy"], how="inner")
    result[name] = analyze(df, name)
result["all_valid_T1K2"] = analyze(lab, "all valid T1/K2")

json.dump({"models": result, "net_delta_def": "dA+dB-dC-dD (buy A,B; sell C,D)",
           "note": "P3.4. Position is net-short delta but realized linear beta is positive; test whether the "
                   "squared-move (gamma/convexity) term absorbs it -> short-gamma path-dependence, not a directional bet. "
                   "move = SPY(exit)/SPY(signal)-1."},
          open(ACTION/"beta_delta.json", "w"), indent=2, default=float)
pd.DataFrame([{"model": k, **v} for k, v in result.items()]).to_csv(ACTION/"beta_delta.csv", index=False)
print("wrote", ACTION/"beta_delta.json", "+ .csv")
