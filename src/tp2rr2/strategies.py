from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


STRATEGIES = ["full4", "T1", "T2", "K1", "K2"]


@dataclass(frozen=True)
class StrategyLeg:
    leg: str
    side: int
    quantity: float
    contract_id: str
    option_symbol: str | None
    expiration: pd.Timestamp
    strike: float


def _full4_directions(row: pd.Series) -> dict[str, tuple[int, float]]:
    """Return side and product quantity for A/B/C/D.

    side: +1 long, -1 short. Quantities are inherited from product replication.
    """
    a_mid = float(row["A_mid"])
    b_mid = float(row["B_mid"])
    c_mid = float(row["C_mid"])
    d_mid = float(row["D_mid"])

    if row["option_type"] == "call":
        # TP2 violation: LHS A*B cheap, RHS C*D rich.
        return {
            "A": (+1, b_mid),
            "B": (+1, a_mid),
            "C": (-1, d_mid),
            "D": (-1, c_mid),
        }
    # RR2 violation: LHS A*B rich, RHS C*D cheap.
    return {
        "C": (+1, d_mid),
        "D": (+1, c_mid),
        "A": (-1, b_mid),
        "B": (-1, a_mid),
    }


def selected_leg_names(strategy: str) -> list[str]:
    if strategy == "full4":
        return ["A", "B", "C", "D"]
    if strategy == "T1":
        return ["A", "D"]
    if strategy == "T2":
        return ["B", "C"]
    if strategy == "K1":
        return ["A", "C"]
    if strategy == "K2":
        return ["B", "D"]
    raise ValueError(f"Unknown strategy: {strategy}")


def strategy_legs(row: pd.Series, strategy: str) -> list[StrategyLeg]:
    directions = _full4_directions(row)
    legs = []
    for name in selected_leg_names(strategy):
        side, qty = directions[name]
        legs.append(
            StrategyLeg(
                leg=name,
                side=side,
                quantity=float(qty),
                contract_id=str(row[f"{name}_contract_id"]),
                option_symbol=str(row[f"{name}_option_symbol"]) if f"{name}_option_symbol" in row else None,
                expiration=row[f"{name}_expiration"],
                strike=float(row[f"{name}_strike"]),
            )
        )
    return legs


def expand_strategy_rows(features: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for strategy in STRATEGIES:
        chunk = features.copy()
        chunk["strategy"] = strategy
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True)
