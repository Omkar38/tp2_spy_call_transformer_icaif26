"""Entry-only commission sensitivity (rebuttal-prep P0 item 1).

The label code (scripts/build_liquid_exit_labels.py, _build_expiration_payoff_rows)
charges transaction_cost = 2.0 * 0.65 * |qty| per traded leg -- a round-trip
convention -- while the expiration-payoff policy has no closing execution.
This script recomputes the unconditional book and each model's selected trades
under entry-only commissions (1.0 * 0.65 * |qty| per leg), i.e.
    net_entry = price_pnl - transaction_cost / 2,
    norm_entry = net_entry / entry_gross_exposure * 1000.
Gate: the 2x numbers must reproduce -$12.52M unconditional and each model's
committed Table 1 total before the 1x variant is trusted.

Run on the compute machine (needs data/labels parquet + nested/trades_*).
Artifact: tables/action_list/entry_only_commissions.{json,csv}
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
ACTION = OUT / "tables" / "action_list"
LABELS = ROOT / "data/labels/archive2_liquid_exit_strategy_labels_2013_2026.parquet"
NORM = 1000.0
ANN = np.sqrt(252)

d = pd.read_parquet(LABELS, columns=[
    "episode_id", "strategy", "signal_date", "exit_policy", "is_valid_backtest",
    "price_pnl", "transaction_cost", "net_pnl", "entry_gross_exposure"])
d = d[(d.exit_policy == "exit_selected_leg_expiration_payoff")
      & d.is_valid_backtest & d.strategy.isin(["T1", "K2"])].copy()
d["signal_date"] = pd.to_datetime(d["signal_date"])
d = d[d.signal_date.dt.year >= 2021]

d["net_norm"] = d.net_pnl / d.entry_gross_exposure * NORM
d["entry_only_norm"] = (d.price_pnl - d.transaction_cost / 2.0) / d.entry_gross_exposure * NORM


def daily_sharpe(frame: pd.DataFrame, col: str) -> float:
    daily = frame.groupby("signal_date")[col].sum()
    return float(daily.mean() / daily.std() * ANN) if daily.std() > 0 else float("nan")


def block(frame: pd.DataFrame, name: str) -> dict:
    return {
        "name": name,
        "n": int(len(frame)),
        "pnl_2x": float(frame.net_norm.sum()),
        "pnl_entry_only": float(frame.entry_only_norm.sum()),
        "win_2x": float((frame.net_norm > 0).mean()),
        "win_entry_only": float((frame.entry_only_norm > 0).mean()),
        "sharpe_2x": daily_sharpe(frame, "net_norm"),
        "sharpe_entry_only": daily_sharpe(frame, "entry_only_norm"),
    }


rows = [block(d, "unconditional")]
assert abs(rows[0]["pnl_2x"] - (-12.52e6)) < 0.05e6, f"gate failed: {rows[0]['pnl_2x']:.0f}"

TRADES = {
    "lightgbm": OUT / "nested" / "trades_lgbm.parquet",
    "logistic": OUT / "nested" / "trades_logistic.parquet",
    "mlp5": OUT / "nested" / "trades_mlp5.parquet",
    "transformer_plain": OUT / "nested" / "trades_transformer_plain.parquet",
    "transformer_alibi": OUT / "nested" / "trades_transformer_alibi.parquet",
}
for model, path in TRADES.items():
    if not path.exists():
        print(f"[skip] {model}: {path} missing")
        continue
    t = pd.read_parquet(path)
    keys = [k for k in ("episode_id", "strategy") if k in t.columns]
    sel = d.merge(t[keys].drop_duplicates(), on=keys, how="inner")
    rows.append(block(sel, model))

ACTION.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(ACTION / "entry_only_commissions.csv", index=False)
(ACTION / "entry_only_commissions.json").write_text(json.dumps({
    "convention": "labels charge 2.0*0.65*|qty| per leg (round trip); entry-only = half",
    "rows": rows}, indent=2))
for r in rows:
    print(f"{r['name']:>18}: 2x ${r['pnl_2x']:>13,.0f} (Sh {r['sharpe_2x']:5.2f})"
          f" | entry-only ${r['pnl_entry_only']:>13,.0f} (Sh {r['sharpe_entry_only']:5.2f})")
