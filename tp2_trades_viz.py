"""
tp2_trades_viz.py — Visualize the trades table
Run from your tp2_project folder:
    python tp2_trades_viz.py

Reads from data/db/tp2.db  →  trades table
Saves tp2_trades_dashboard.png
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timezone

# ── paths ─────────────────────────────────────────────────────────────────────
DB_PATH  = Path("data/db/tp2.db")
OUT_PATH = Path("tp2_trades_dashboard.png")

# ── colours ───────────────────────────────────────────────────────────────────
BG    = "#FFFFFF"
NAVY  = "#1B2A4A"
BLUE  = "#2E6DB4"
LBLUE = "#EBF2FB"
MUTED = "#6B7C93"
RULE  = "#D0DAE8"
RED   = "#C0392B"
GREEN = "#27AE60"
AMBER = "#E67E22"

plt.rcParams.update({
    "figure.facecolor": BG,    "axes.facecolor":  LBLUE,
    "axes.edgecolor":   RULE,  "axes.labelcolor": NAVY,
    "xtick.color":      MUTED, "ytick.color":     MUTED,
    "text.color":       NAVY,  "grid.color":      RULE,
    "grid.linewidth":   0.7,   "font.family":     "DejaVu Sans",
    "axes.spines.top":  False, "axes.spines.right": False,
})

# ── load trades ───────────────────────────────────────────────────────────────
print(f"Reading: {DB_PATH.resolve()}")
conn = sqlite3.connect(DB_PATH)

df = pd.read_sql("""
    SELECT timestamp_ms, instrument_name, expiry, strike,
           option_type, price, amount, iv, mark_price, index_price, direction
    FROM trades
    ORDER BY timestamp_ms
""", conn)
conn.close()

print(f"Trades loaded: {len(df):,}")

# ── derived columns ───────────────────────────────────────────────────────────
df["dt"]      = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
df["date"]    = df["dt"].dt.date
df["hour"]    = df["dt"].dt.hour
df["weekday"] = df["dt"].dt.day_name()
df["notional"]= df["price"] * df["amount"]   # in BTC

# separate calls and puts
calls = df[df["option_type"] == "C"]
puts  = df[df["option_type"] == "P"]

print(f"  Calls: {len(calls):,}  |  Puts: {len(puts):,}")
print(f"  Date range: {df['dt'].min().date()} → {df['dt'].max().date()}")
print(f"  Unique instruments: {df['instrument_name'].nunique()}")
print(f"  IV range: {df['iv'].min():.1f}% – {df['iv'].max():.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE  —  2 rows × 3 cols
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.suptitle("AMS 520 · BTC Options Trade Activity · Deribit",
             fontsize=15, fontweight="bold", color=NAVY, y=0.98)
fig.text(0.5, 0.955,
         f"{df['dt'].min().strftime('%d %b %Y')}  →  "
         f"{df['dt'].max().strftime('%d %b %Y')}  ·  "
         f"{len(df):,} trades  ·  {df['instrument_name'].nunique()} instruments",
         ha="center", fontsize=10, color=MUTED)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.35,
                       left=0.07, right=0.97, top=0.92, bottom=0.10)

# ── Chart 1: Daily trade count (calls vs puts) ────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])

daily_calls = calls.groupby("date").size()
daily_puts  = puts.groupby("date").size()
all_dates   = sorted(set(daily_calls.index) | set(daily_puts.index))

dc = [daily_calls.get(d, 0) for d in all_dates]
dp = [daily_puts.get(d, 0)  for d in all_dates]
x  = range(len(all_dates))

ax1.bar(x, dc, label="Calls", color=BLUE,  width=0.6, zorder=2)
ax1.bar(x, dp, label="Puts",  color=AMBER, width=0.6,
        bottom=dc, zorder=2)

# x-axis: show every ~5th date to avoid crowding
step = max(1, len(all_dates) // 8)
ax1.set_xticks(range(0, len(all_dates), step))
ax1.set_xticklabels([str(all_dates[i])[5:] for i in range(0, len(all_dates), step)],
                    rotation=30, ha="right", fontsize=8)
ax1.set_ylabel("Number of trades", fontsize=9)
ax1.set_title("Daily Trade Count\n(Calls vs Puts)", fontsize=11,
              fontweight="bold", color=NAVY, pad=8)
ax1.legend(fontsize=9)
ax1.grid(axis="y", zorder=1)

# ── Chart 2: Implied volatility over time (scatter) ──────────────────────────
ax2 = fig.add_subplot(gs[0, 1])

# sample to keep chart readable (max 3000 points)
sample = df.sample(min(3000, len(df)), random_state=42).sort_values("dt")
ax2.scatter(sample["dt"], sample["iv"],
            c=[BLUE if o == "C" else AMBER for o in sample["option_type"]],
            s=6, alpha=0.4, zorder=2)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
ax2.set_ylabel("Implied Volatility (%)", fontsize=9)
ax2.set_title("Implied Volatility Over Time", fontsize=11,
              fontweight="bold", color=NAVY, pad=8)

import matplotlib.patches as mpatches
ax2.legend(handles=[
    mpatches.Patch(color=BLUE,  label="Calls"),
    mpatches.Patch(color=AMBER, label="Puts"),
], fontsize=9)
ax2.grid(zorder=1)

# ── Chart 3: IV distribution calls vs puts ───────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])

iv_max = df["iv"].quantile(0.99)   # clip extreme outliers for display
ax3.hist(calls["iv"].clip(upper=iv_max), bins=40,
         color=BLUE,  alpha=0.6, label=f"Calls ({len(calls):,})",
         edgecolor=BG, linewidth=0.3, density=True, zorder=2)
ax3.hist(puts["iv"].clip(upper=iv_max),  bins=40,
         color=AMBER, alpha=0.6, label=f"Puts ({len(puts):,})",
         edgecolor=BG, linewidth=0.3, density=True, zorder=2)

ax3.axvline(calls["iv"].median(), color=BLUE,  lw=1.5, ls="--",
            label=f"Call median {calls['iv'].median():.0f}%")
ax3.axvline(puts["iv"].median(),  color=AMBER, lw=1.5, ls="--",
            label=f"Put median  {puts['iv'].median():.0f}%")
ax3.set_xlabel("Implied Volatility (%)", fontsize=9)
ax3.set_ylabel("Density", fontsize=9)
ax3.set_title("IV Distribution\n(Calls vs Puts)", fontsize=11,
              fontweight="bold", color=NAVY, pad=8)
ax3.legend(fontsize=8)
ax3.grid(axis="y", zorder=1)

# ── Chart 4: Trade size distribution ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])

# amount in contracts; log scale to show full range
ax4.hist(df["amount"].clip(upper=df["amount"].quantile(0.99)),
         bins=40, color=BLUE, edgecolor=BG, linewidth=0.3, zorder=2)
ax4.set_xlabel("Trade size (contracts)", fontsize=9)
ax4.set_ylabel("Count", fontsize=9)
ax4.set_title("Trade Size Distribution", fontsize=11,
              fontweight="bold", color=NAVY, pad=8)
ax4.grid(axis="y", zorder=1)

median_sz = df["amount"].median()
ax4.axvline(median_sz, color=AMBER, lw=1.5, ls="--",
            label=f"Median = {median_sz:.1f}")
ax4.legend(fontsize=9)

# small stats box
stats_txt = (f"Total contracts: {df['amount'].sum():,.0f}\n"
             f"Avg per trade:   {df['amount'].mean():.2f}\n"
             f"Max single trade:{df['amount'].max():.0f}")
ax4.text(0.97, 0.97, stats_txt, transform=ax4.transAxes,
         ha="right", va="top", fontsize=8.5, color=NAVY,
         bbox=dict(boxstyle="round,pad=0.4", facecolor=BG, edgecolor=RULE))

# ── Chart 5: Buy vs Sell volume by day ───────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])

buy_vol  = df[df["direction"]=="buy"].groupby("date")["amount"].sum()
sell_vol = df[df["direction"]=="sell"].groupby("date")["amount"].sum()

bv = [buy_vol.get(d, 0)  for d in all_dates]
sv = [sell_vol.get(d, 0) for d in all_dates]

ax5.bar(x, bv,  label="Buy",  color=GREEN, width=0.6, zorder=2)
ax5.bar(x, [-v for v in sv], label="Sell", color=RED, width=0.6, zorder=2)
ax5.axhline(0, color=NAVY, lw=0.8)

ax5.set_xticks(range(0, len(all_dates), step))
ax5.set_xticklabels([str(all_dates[i])[5:] for i in range(0, len(all_dates), step)],
                    rotation=30, ha="right", fontsize=8)
ax5.set_ylabel("Contracts (buy up / sell down)", fontsize=9)
ax5.set_title("Buy vs Sell Volume by Day", fontsize=11,
              fontweight="bold", color=NAVY, pad=8)
ax5.legend(fontsize=9)
ax5.grid(axis="y", zorder=1)

# ── Chart 6: Trade activity by hour of day ───────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])

hourly = df.groupby("hour").size()
hours  = range(24)
counts = [hourly.get(h, 0) for h in hours]

bar_colors = [BLUE if c < np.percentile(counts, 75) else AMBER for c in counts]
ax6.bar(hours, counts, color=bar_colors, width=0.7, zorder=2)
ax6.set_xticks(range(0, 24, 2))
ax6.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)],
                    rotation=30, ha="right", fontsize=8)
ax6.set_xlabel("Hour of day (UTC)", fontsize=9)
ax6.set_ylabel("Number of trades", fontsize=9)
ax6.set_title("Trade Activity by Hour (UTC)\n(orange = peak hours)",
              fontsize=11, fontweight="bold", color=NAVY, pad=8)
ax6.grid(axis="y", zorder=1)

peak_hour = max(hours, key=lambda h: hourly.get(h, 0))
ax6.text(0.97, 0.97, f"Peak hour: {peak_hour:02d}:00 UTC",
         transform=ax6.transAxes, ha="right", va="top", fontsize=9,
         color=AMBER, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=RULE))

# ── save ──────────────────────────────────────────────────────────────────────
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=BG)

print(f"\nDashboard saved -> {OUT_PATH.resolve()}")
print(f"\n{'='*50}")
print(f"  Total trades        : {len(df):,}")
print(f"  Calls / Puts        : {len(calls):,} / {len(puts):,}")
print(f"  Date range          : {df['dt'].min().date()} -> {df['dt'].max().date()}")
print(f"  Unique instruments  : {df['instrument_name'].nunique()}")
print(f"  Avg IV (calls)      : {calls['iv'].mean():.1f}%")
print(f"  Avg IV (puts)       : {puts['iv'].mean():.1f}%")
print(f"  Avg trade size      : {df['amount'].mean():.2f} contracts")
print(f"  Total volume        : {df['amount'].sum():,.0f} contracts")
print(f"  Peak trading hour   : {peak_hour:02d}:00 UTC")
print(f"{'='*50}")
