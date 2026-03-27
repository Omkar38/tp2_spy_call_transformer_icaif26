"""
ml/backtest.py — Backtest DS-CAT predictions against actual reversion outcomes.

Simulates trading TP2 violations flagged by the model:
  - For each violation the model predicts as "fast reversion" (label=1),
    enter the 4-leg arbitrage trade at ask/bid prices.
  - Track whether the violation actually reverted within 2 hours.
  - Compute net PnL after Deribit fees.

Deribit fee structure (as of 2024-2025):
  - Maker: 0.02% of underlying
  - Taker: 0.05% of underlying
  - We assume TAKER on entry (market orders to capture fleeting arb)
  - We assume MAKER on exit (limit orders at fair value once reverted)
  - Each trade = 4 legs = 4 × fee per leg
  - Settlement fee: 0.015% (if held to expiry, but we exit early)

PnL per trade:
  Entry: buy K1T1 ask, buy K2T2 ask, sell K1T2 bid, sell K2T1 bid
  Exit (if reverted): unwind at mid (approximation)
  Gross PnL = executable_profit at entry
  Net PnL   = gross - (4 × taker_fee) - (4 × maker_fee)
"""

import logging
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger("ml.backtest")


# ── Fee constants (Deribit) ──────────────────────────────────

TAKER_FEE_PCT = 0.0005   # 0.05% of underlying per leg
MAKER_FEE_PCT = 0.0002   # 0.02% of underlying per leg
N_LEGS        = 4
ENTRY_FEE_TOTAL = N_LEGS * TAKER_FEE_PCT   # taker on entry
EXIT_FEE_TOTAL  = N_LEGS * MAKER_FEE_PCT   # maker on exit


@dataclass
class BacktestTrade:
    """One simulated arbitrage trade."""
    violation_id:     int
    snapshot_hour:    str
    entry_cost:       float    # ask-side entry cost (sum of 4 legs)
    entry_revenue:    float    # bid-side entry revenue (sum of 4 legs)
    gross_edge:       float    # entry_revenue - entry_cost
    fees:             float    # total round-trip fees
    net_pnl:          float    # gross_edge - fees
    btc_spot:         float
    model_prob:       float    # model's P(fast reversion)
    actual_label:     int      # 1=fast, 0=slow/never
    actually_reverted: bool
    revert_speed:     str


@dataclass
class BacktestResult:
    """Aggregated backtest results."""
    trades:           list = field(default_factory=list)
    n_signals:        int = 0      # violations model flagged as "fast"
    n_traded:         int = 0      # trades actually entered
    n_profitable:     int = 0
    n_reverted:       int = 0
    total_gross_pnl:  float = 0.0
    total_fees:       float = 0.0
    total_net_pnl:    float = 0.0
    avg_net_pnl:      float = 0.0
    win_rate:         float = 0.0
    sharpe_proxy:     float = 0.0  # annualized Sharpe approximation
    max_drawdown:     float = 0.0


def backtest(
    model: torch.nn.Module,
    test_loader,
    test_metadata: list,
    conn: sqlite3.Connection,
    device: str = "cpu",
    threshold: float = 0.5,
) -> BacktestResult:
    """
    Run backtest on the test set.

    Args:
        model:         trained DS-CAT model
        test_loader:   DataLoader for test set
        test_metadata: list of violation dicts (from features.py) for test set
        conn:          DB connection for looking up trade details
        device:        cuda or cpu
        threshold:     probability threshold for trading (default 0.5)

    Returns:
        BacktestResult with trade-level and aggregate stats
    """
    model.eval()
    result = BacktestResult()

    # ── Collect all model predictions ──
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for temporal, surface, surface_mask, labels in test_loader:
            temporal     = temporal.to(device)
            surface      = surface.to(device)
            surface_mask = surface_mask.to(device)

            logits = model(temporal, surface, surface_mask)
            probs = F.softmax(logits, dim=1)[:, 1]  # P(fast reversion)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    # ── Simulate trades for predicted-positive violations ──
    for i, (prob, actual_label) in enumerate(zip(all_probs, all_labels)):
        if i >= len(test_metadata):
            break

        meta = test_metadata[i]

        if prob < threshold:
            continue  # model says don't trade this one

        result.n_signals += 1

        # ── Reconstruct entry prices from violation data ──
        v = conn.execute("""
            SELECT
                C_K1_T1, C_K2_T1, C_K1_T2, C_K2_T2,
                avg_spread_pct, net_edge, btc_spot, violation_pct
            FROM violations WHERE id = ?
        """, (meta["id"],)).fetchone()

        if not v:
            continue

        spot = v["btc_spot"] or 1.0
        spread_half = (v["avg_spread_pct"] or 0) / 100.0 / 2.0

        # Entry: buy C_K1_T1 and C_K2_T2 at ask, sell C_K1_T2 and C_K2_T1 at bid
        c11, c21, c12, c22 = v["C_K1_T1"], v["C_K2_T1"], v["C_K1_T2"], v["C_K2_T2"]

        ask_k1t1 = c11 * (1 + spread_half)
        ask_k2t2 = c22 * (1 + spread_half)
        bid_k1t2 = c12 * (1 - spread_half)
        bid_k2t1 = c21 * (1 - spread_half)

        entry_cost    = ask_k1t1 + ask_k2t2                     # what we pay
        entry_revenue = bid_k1t2 + bid_k2t1                      # what we receive
        gross_edge    = (bid_k1t2 * bid_k2t1) - (ask_k1t1 * ask_k2t2)

        # Fees in BTC terms (fee_pct × spot equivalent × 4 legs)
        notional_btc = (c11 + c21 + c12 + c22) / 4.0            # avg leg size
        fees = notional_btc * (ENTRY_FEE_TOTAL + EXIT_FEE_TOTAL)

        net_pnl = gross_edge - fees

        # ── Check actual outcome ──
        revert_row = conn.execute("""
            SELECT revert_speed, reverted
            FROM reversions
            WHERE violation_id = ?
            ORDER BY hours_elapsed ASC
            LIMIT 1
        """, (meta["id"],)).fetchone()

        actually_reverted = False
        revert_speed = "unknown"
        if revert_row:
            revert_speed = revert_row["revert_speed"]
            actually_reverted = revert_speed in ("superfast", "fast")

        trade = BacktestTrade(
            violation_id=meta["id"],
            snapshot_hour=meta["snapshot_hour"],
            entry_cost=round(entry_cost, 8),
            entry_revenue=round(entry_revenue, 8),
            gross_edge=round(gross_edge, 8),
            fees=round(fees, 8),
            net_pnl=round(net_pnl, 8),
            btc_spot=spot,
            model_prob=round(float(prob), 4),
            actual_label=int(actual_label),
            actually_reverted=actually_reverted,
            revert_speed=revert_speed,
        )
        result.trades.append(trade)

    # ── Aggregate statistics ──
    result.n_traded = len(result.trades)

    if result.n_traded > 0:
        pnls = np.array([t.net_pnl for t in result.trades])
        result.n_profitable  = int((pnls > 0).sum())
        result.n_reverted    = sum(1 for t in result.trades if t.actually_reverted)
        result.total_gross_pnl = sum(t.gross_edge for t in result.trades)
        result.total_fees    = sum(t.fees for t in result.trades)
        result.total_net_pnl = float(pnls.sum())
        result.avg_net_pnl   = float(pnls.mean())
        result.win_rate      = result.n_profitable / result.n_traded

        # Sharpe proxy: annualized (assuming ~1 trade per day avg)
        if pnls.std() > 0:
            daily_sharpe = pnls.mean() / pnls.std()
            result.sharpe_proxy = round(float(daily_sharpe * np.sqrt(365)), 4)

        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        result.max_drawdown = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

    return result


def print_backtest_report(result: BacktestResult):
    """Pretty-print the backtest results."""
    print("\n" + "=" * 60)
    print("  DS-CAT BACKTEST REPORT")
    print("=" * 60)
    print(f"  Signals generated:  {result.n_signals:>8}")
    print(f"  Trades entered:     {result.n_traded:>8}")
    print(f"  Profitable trades:  {result.n_profitable:>8}")
    print(f"  Win rate:           {result.win_rate:>8.1%}")
    print(f"  Actually reverted:  {result.n_reverted:>8}")
    print(f"  ─────────────────────────────────")
    print(f"  Total gross PnL:    {result.total_gross_pnl:>11.6f} BTC")
    print(f"  Total fees:         {result.total_fees:>11.6f} BTC")
    print(f"  Total net PnL:      {result.total_net_pnl:>11.6f} BTC")
    print(f"  Avg net PnL/trade:  {result.avg_net_pnl:>11.6f} BTC")
    print(f"  ─────────────────────────────────")
    print(f"  Sharpe (annualized):{result.sharpe_proxy:>8.2f}")
    print(f"  Max drawdown:       {result.max_drawdown:>11.6f} BTC")
    print("=" * 60)

    if result.trades:
        # Top 5 trades by net PnL
        sorted_trades = sorted(result.trades, key=lambda t: t.net_pnl, reverse=True)
        print("\n  Top 5 Trades:")
        for t in sorted_trades[:5]:
            print(
                f"    {t.snapshot_hour[:16]} | "
                f"net={t.net_pnl:+.6f} BTC | "
                f"prob={t.model_prob:.3f} | "
                f"reverted={'Y' if t.actually_reverted else 'N'}"
            )
    print()
