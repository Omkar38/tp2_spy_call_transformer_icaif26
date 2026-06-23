from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LabelConfig


STRATEGIES = ["T1", "T2", "K1", "K2"]
SELECTED_LEGS = {
    "T1": ["A", "D"],
    "T2": ["B", "C"],
    "K1": ["A", "C"],
    "K2": ["B", "D"],
}


@dataclass(frozen=True)
class EpisodeMiniBacktestResult:
    rows: pd.DataFrame
    summary: dict


def required_episode_label_columns() -> list[str]:
    cols = [
        "episode_id",
        "start_violation_id",
        "signal_date",
        "split_by_signal_date",
        "duration_sessions",
        "episode_row_count",
        "label_end_state",
        "label_end_state_detail",
        "label_next_trading_date",
        "label_true_reversion_next_session",
        "feature_option_type",
        "feature_normalized_severity",
        "feature_avg_spread_pct",
        "feature_max_spread_pct",
        "feature_min_volume",
        "feature_total_volume",
        "feature_min_open_interest",
        "feature_total_open_interest",
    ]
    for leg in ["A", "B", "C", "D"]:
        cols.extend(
            [
                f"feature_{leg}_contract_id",
                f"feature_{leg}_option_symbol",
                f"feature_{leg}_expiration",
                f"feature_{leg}_strike",
                f"feature_{leg}_bid",
                f"feature_{leg}_ask",
                f"feature_{leg}_mid",
                f"feature_{leg}_spread_pct",
            ]
        )
    return cols


def prepare_episode_quote_frame(clean_options: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "contractID", "option_symbol", "bid", "ask", "mid", "spread_pct"]
    missing = sorted(set(required).difference(clean_options.columns))
    if missing:
        raise ValueError(f"Clean option frame is missing columns: {missing}")
    out = clean_options[required].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["quote_key"] = out["option_symbol"].where(out["option_symbol"].notna(), out["contractID"]).astype(str)
    out = out.rename(
        columns={
            "date": "exit_date",
            "bid": "exit_bid",
            "ask": "exit_ask",
            "mid": "exit_mid",
            "spread_pct": "exit_spread_pct",
        }
    )
    return out[["exit_date", "quote_key", "exit_bid", "exit_ask", "exit_mid", "exit_spread_pct"]]


def _leg_quantity_and_side(rows: pd.DataFrame, leg: str) -> tuple[np.ndarray, np.ndarray]:
    is_call = rows["feature_option_type"].eq("call").to_numpy()
    if leg == "A":
        return rows["feature_B_mid"].to_numpy(dtype=float), np.where(is_call, 1.0, -1.0)
    if leg == "B":
        return rows["feature_A_mid"].to_numpy(dtype=float), np.where(is_call, 1.0, -1.0)
    if leg == "C":
        return rows["feature_D_mid"].to_numpy(dtype=float), np.where(is_call, -1.0, 1.0)
    if leg == "D":
        return rows["feature_C_mid"].to_numpy(dtype=float), np.where(is_call, -1.0, 1.0)
    raise ValueError(f"Unknown leg: {leg}")


def _entry_valid(rows: pd.DataFrame, leg: str, max_spread_pct: float) -> np.ndarray:
    bid = rows[f"feature_{leg}_bid"].to_numpy(dtype=float)
    ask = rows[f"feature_{leg}_ask"].to_numpy(dtype=float)
    mid = rows[f"feature_{leg}_mid"].to_numpy(dtype=float)
    spread = rows[f"feature_{leg}_spread_pct"].to_numpy(dtype=float)
    return (bid > 0) & (ask > bid) & (mid > 0) & np.isfinite(spread) & (spread <= max_spread_pct)


def _exit_valid(rows: pd.DataFrame, max_spread_pct: float) -> np.ndarray:
    bid = rows["exit_bid"].to_numpy(dtype=float)
    ask = rows["exit_ask"].to_numpy(dtype=float)
    mid = rows["exit_mid"].to_numpy(dtype=float)
    spread = rows["exit_spread_pct"].to_numpy(dtype=float)
    return (bid > 0) & (ask > bid) & (mid > 0) & np.isfinite(spread) & (spread <= max_spread_pct)


def build_episode_reversion_mini_backtest_rows(
    episode_labels: pd.DataFrame,
    quote_frame: pd.DataFrame,
    config: LabelConfig,
) -> EpisodeMiniBacktestResult:
    labels = episode_labels.copy()
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    labels["label_next_trading_date"] = pd.to_datetime(labels["label_next_trading_date"])
    for leg in ["A", "B", "C", "D"]:
        labels[f"feature_{leg}_expiration"] = pd.to_datetime(labels[f"feature_{leg}_expiration"])

    chunks = []
    for strategy in STRATEGIES:
        chunk = labels.copy()
        chunk["strategy"] = strategy
        chunk["selected_legs"] = ",".join(SELECTED_LEGS[strategy])
        chunks.append(chunk)
    rows = pd.concat(chunks, ignore_index=True)
    n = len(rows)

    rows["backtest_policy"] = "episode_true_reversion_exit"
    rows["exit_date"] = pd.to_datetime(rows["label_next_trading_date"])
    rows["exit_reason"] = np.where(
        rows["label_end_state"].eq("true_reversion"),
        "true_reversion_exit",
        rows["label_end_state"].astype(str),
    )
    rows["is_valid_backtest"] = False
    rows["entry_gross_exposure"] = np.nan
    rows["price_pnl"] = np.nan
    rows["transaction_cost"] = np.nan
    rows["net_pnl"] = np.nan
    rows["realized_return"] = np.nan
    rows["actual_holding_days"] = np.nan

    valid = rows["label_end_state"].eq("true_reversion").to_numpy().copy()
    valid &= rows["exit_date"].notna().to_numpy()
    rows.loc[rows["label_end_state"].eq("true_reversion") & rows["exit_date"].isna(), "exit_reason"] = "missing_exit_date"

    gross = np.zeros(n, dtype=float)
    price_pnl = np.zeros(n, dtype=float)
    transaction_cost = np.zeros(n, dtype=float)

    quote_lookup = quote_frame.copy()
    quote_lookup["exit_date"] = pd.to_datetime(quote_lookup["exit_date"])

    selected_exp_min = np.full(n, np.datetime64("2262-04-11"), dtype="datetime64[ns]")
    for leg in ["A", "B", "C", "D"]:
        selected = rows["strategy"].map(lambda name: leg in SELECTED_LEGS[name]).to_numpy()
        qty, side = _leg_quantity_and_side(rows, leg)
        entry_bid = rows[f"feature_{leg}_bid"].to_numpy(dtype=float)
        entry_ask = rows[f"feature_{leg}_ask"].to_numpy(dtype=float)
        entry_mid = rows[f"feature_{leg}_mid"].to_numpy(dtype=float)
        leg_exp = pd.to_datetime(rows[f"feature_{leg}_expiration"]).to_numpy(dtype="datetime64[ns]")
        selected_exp_min = np.where(selected & (leg_exp < selected_exp_min), leg_exp, selected_exp_min)

        valid &= ~selected | _entry_valid(rows, leg, config.max_label_spread_pct)
        gross += np.where(selected, np.abs(entry_mid * qty * config.option_multiplier), 0.0)

        exit_bid = np.full(n, np.nan, dtype=float)
        exit_ask = np.full(n, np.nan, dtype=float)
        exit_mid = np.full(n, np.nan, dtype=float)
        exit_spread = np.full(n, np.nan, dtype=float)
        key_mask = selected & rows["exit_date"].notna().to_numpy()
        if key_mask.any():
            keys = pd.DataFrame(
                {
                    "_row": np.flatnonzero(key_mask),
                    "exit_date": rows.loc[key_mask, "exit_date"].to_numpy(),
                    "quote_key": rows.loc[key_mask, f"feature_{leg}_option_symbol"]
                    .where(
                        rows.loc[key_mask, f"feature_{leg}_option_symbol"].notna(),
                        rows.loc[key_mask, f"feature_{leg}_contract_id"],
                    )
                    .astype(str)
                    .to_numpy(),
                }
            )
            merged = keys.merge(quote_lookup, on=["exit_date", "quote_key"], how="left")
            row_idx = merged["_row"].to_numpy(dtype=int)
            exit_bid[row_idx] = merged["exit_bid"].to_numpy(dtype=float)
            exit_ask[row_idx] = merged["exit_ask"].to_numpy(dtype=float)
            exit_mid[row_idx] = merged["exit_mid"].to_numpy(dtype=float)
            exit_spread[row_idx] = merged["exit_spread_pct"].to_numpy(dtype=float)

        rows[f"{leg}_exit_bid"] = exit_bid
        rows[f"{leg}_exit_ask"] = exit_ask
        rows[f"{leg}_exit_mid"] = exit_mid
        rows[f"{leg}_exit_spread_pct"] = exit_spread

        leg_exit_valid = _exit_valid(
            pd.DataFrame(
                {
                    "exit_bid": exit_bid,
                    "exit_ask": exit_ask,
                    "exit_mid": exit_mid,
                    "exit_spread_pct": exit_spread,
                }
            ),
            config.max_label_spread_pct,
        )
        valid &= ~selected | leg_exit_valid

        long_pnl = (exit_bid - entry_ask) * qty * config.option_multiplier
        short_pnl = (entry_bid - exit_ask) * qty * config.option_multiplier
        leg_price_pnl = np.where(side == 1, long_pnl, short_pnl)
        leg_fees = 2.0 * config.commission_per_contract * np.abs(qty)
        price_pnl += np.where(selected, leg_price_pnl, 0.0)
        transaction_cost += np.where(selected, leg_fees, 0.0)

    exit_values = pd.to_datetime(rows["exit_date"]).to_numpy(dtype="datetime64[ns]")
    pre_exp_exit = exit_values < selected_exp_min
    valid &= pre_exp_exit
    rows.loc[rows["label_end_state"].eq("true_reversion") & ~pre_exp_exit, "exit_reason"] = "exit_on_or_after_selected_expiration"

    gross_valid = np.isfinite(gross) & (gross > 0)
    valid &= gross_valid
    rows.loc[~gross_valid, "exit_reason"] = "bad_entry_gross"
    invalid_quote = rows["label_end_state"].eq("true_reversion").to_numpy() & gross_valid & pre_exp_exit & ~valid
    rows.loc[invalid_quote, "exit_reason"] = "invalid_entry_or_exit_quote"

    rows.loc[gross_valid, "entry_gross_exposure"] = gross[gross_valid]
    rows.loc[valid, "is_valid_backtest"] = True
    rows.loc[valid, "price_pnl"] = price_pnl[valid]
    rows.loc[valid, "transaction_cost"] = transaction_cost[valid]
    rows.loc[valid, "net_pnl"] = price_pnl[valid] - transaction_cost[valid]
    rows.loc[valid, "realized_return"] = rows.loc[valid, "net_pnl"] / rows.loc[valid, "entry_gross_exposure"]
    rows.loc[valid, "actual_holding_days"] = (
        pd.to_datetime(rows.loc[valid, "exit_date"]) - pd.to_datetime(rows.loc[valid, "signal_date"])
    ).dt.days

    keep = [
        "episode_id",
        "start_violation_id",
        "signal_date",
        "split_by_signal_date",
        "strategy",
        "selected_legs",
        "backtest_policy",
        "label_end_state",
        "label_end_state_detail",
        "duration_sessions",
        "episode_row_count",
        "feature_option_type",
        "feature_normalized_severity",
        "feature_avg_spread_pct",
        "feature_max_spread_pct",
        "feature_min_volume",
        "feature_total_volume",
        "feature_min_open_interest",
        "feature_total_open_interest",
        "exit_date",
        "exit_reason",
        "is_valid_backtest",
        "entry_gross_exposure",
        "price_pnl",
        "transaction_cost",
        "net_pnl",
        "realized_return",
        "actual_holding_days",
    ]
    for leg in ["A", "B", "C", "D"]:
        keep.extend(
            [
                f"feature_{leg}_contract_id",
                f"feature_{leg}_option_symbol",
                f"feature_{leg}_expiration",
                f"feature_{leg}_strike",
                f"feature_{leg}_bid",
                f"feature_{leg}_ask",
                f"feature_{leg}_mid",
                f"{leg}_exit_bid",
                f"{leg}_exit_ask",
                f"{leg}_exit_mid",
                f"{leg}_exit_spread_pct",
            ]
        )

    rows = rows[keep].copy()
    counts = rows["exit_reason"].value_counts(dropna=False)
    by_strategy = (
        rows.groupby("strategy", dropna=False)
        .agg(
            candidate_rows=("episode_id", "size"),
            valid_rows=("is_valid_backtest", "sum"),
            mean_return=("realized_return", "mean"),
            median_return=("realized_return", "median"),
            net_pnl=("net_pnl", "sum"),
            transaction_cost=("transaction_cost", "sum"),
        )
        .reset_index()
    )
    summary = {
        "strategy_rows": int(len(rows)),
        "unique_episodes": int(rows["episode_id"].nunique()),
        "strategies": STRATEGIES,
        "valid_backtest_rows": int(rows["is_valid_backtest"].sum()),
        "valid_backtest_share": float(rows["is_valid_backtest"].mean()) if len(rows) else 0.0,
        "exit_reason_counts": {str(k): int(v) for k, v in counts.items()},
        "net_pnl_valid": float(rows.loc[rows["is_valid_backtest"], "net_pnl"].sum()),
        "transaction_cost_valid": float(rows.loc[rows["is_valid_backtest"], "transaction_cost"].sum()),
        "mean_realized_return_valid": float(rows.loc[rows["is_valid_backtest"], "realized_return"].mean()),
        "median_realized_return_valid": float(rows.loc[rows["is_valid_backtest"], "realized_return"].median()),
        "by_strategy": {
            str(row.strategy): {
                "candidate_rows": int(row.candidate_rows),
                "valid_rows": int(row.valid_rows),
                "mean_return": None if pd.isna(row.mean_return) else float(row.mean_return),
                "median_return": None if pd.isna(row.median_return) else float(row.median_return),
                "net_pnl": float(row.net_pnl),
                "transaction_cost": float(row.transaction_cost),
            }
            for row in by_strategy.itertuples(index=False)
        },
        "policy_note": "Conservative first baseline: valid only when episode end-state is true_reversion and selected entry/exit legs are tradable before expiration.",
    }
    return EpisodeMiniBacktestResult(rows=rows, summary=summary)
