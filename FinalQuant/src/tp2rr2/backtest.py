from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .config import BacktestConfig


def apply_trade_gate(df: pd.DataFrame, config: BacktestConfig) -> pd.Series:
    mask = (
        (df["predicted_return"] > config.prediction_threshold)
        & (df["normalized_severity"] >= config.min_normalized_severity)
        & (df["max_spread_pct"] <= config.max_spread_pct)
        & (df["avg_spread_pct"] <= config.max_avg_spread_pct)
        & (df["min_volume"] >= config.min_volume)
        & (df["min_open_interest"] >= config.min_open_interest)
    )
    size_cols = [c for c in df.columns if c.endswith("_bid_size") or c.endswith("_ask_size")]
    if size_cols:
        present = df[size_cols].notna().any(axis=1)
        positive_if_present = df[size_cols].fillna(1).gt(0).all(axis=1)
        mask &= (~present) | positive_if_present
    return mask


def select_trades(scored_strategy_rows: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    valid = scored_strategy_rows[scored_strategy_rows["is_valid_label"]].copy()
    if valid.empty:
        return valid
    valid = valid[apply_trade_gate(valid, config)].copy()
    if valid.empty:
        return valid

    # At most one strategy per violation.
    valid = valid.sort_values(["violation_id", "predicted_return"], ascending=[True, False])
    best = valid.groupby("violation_id", as_index=False).head(1)

    accepted = []
    for date, day in best.sort_values(["date", "predicted_return"], ascending=[True, False]).groupby("date"):
        contract_counts: Counter[str] = Counter()
        n = 0
        for _, row in day.iterrows():
            if n >= config.max_trades_per_day:
                break
            contracts = [row.get(f"{leg}_contract_id") for leg in ["A", "B", "C", "D"]]
            contracts = [c for c in contracts if isinstance(c, str)]
            if any(contract_counts[c] >= config.max_contract_overlap_per_day for c in contracts):
                continue
            accepted.append(row)
            contract_counts.update(contracts)
            n += 1
    out = pd.DataFrame(accepted)
    if out.empty or not config.target_gross_exposure:
        return out

    original_gross = out["entry_gross_exposure"].astype(float)
    scale = config.target_gross_exposure / original_gross
    out["unscaled_net_pnl"] = out["net_pnl"].astype(float)
    out["unscaled_entry_gross_exposure"] = original_gross
    out["scale_factor"] = scale
    out["net_pnl"] = out["unscaled_net_pnl"] * scale
    out["entry_gross_exposure"] = config.target_gross_exposure
    out["realized_return"] = out["net_pnl"] / out["entry_gross_exposure"]
    return out


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "net_pnl": 0.0,
            "avg_return": np.nan,
            "median_return": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown": np.nan,
        }
    gross_profit = trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum()
    gross_loss = -trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum()
    trades = trades.copy()
    if "date" in trades.columns and "year" not in trades.columns:
        trades["year"] = pd.to_datetime(trades["date"]).dt.year
    daily = trades.groupby("date")["net_pnl"].sum().sort_index()
    equity = daily.cumsum()
    drawdown = equity - equity.cummax()
    daily_sharpe_like = np.nan
    if len(daily) > 1 and daily.std(ddof=1) > 0:
        daily_sharpe_like = float(daily.mean() / daily.std(ddof=1) * np.sqrt(252))

    def grouped_summary(group_cols: list[str]) -> dict:
        present = [col for col in group_cols if col in trades.columns]
        if not present:
            return {}
        out = {}
        for key, group in trades.groupby(present):
            key_text = "_".join(str(part) for part in (key if isinstance(key, tuple) else (key,)))
            wins = group["net_pnl"] > 0
            out[key_text] = {
                "trades": int(len(group)),
                "net_pnl": float(group["net_pnl"].sum()),
                "avg_return": float(group["realized_return"].mean()),
                "win_rate": float(wins.mean()),
            }
        return out

    return {
        "trades": int(len(trades)),
        "net_pnl": float(trades["net_pnl"].sum()),
        "avg_return": float(trades["realized_return"].mean()),
        "median_return": float(trades["realized_return"].median()),
        "win_rate": float((trades["net_pnl"] > 0).mean()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.inf,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
        "daily_sharpe_like": daily_sharpe_like,
        "avg_actual_holding_days": float(trades["actual_holding_days"].mean()) if "actual_holding_days" in trades else np.nan,
        "exit_reasons": trades["exit_reason"].value_counts(dropna=False).to_dict() if "exit_reason" in trades else {},
        "by_option_type": grouped_summary(["option_type"]),
        "by_strategy": grouped_summary(["strategy"]),
        "by_year": grouped_summary(["year"]),
    }
