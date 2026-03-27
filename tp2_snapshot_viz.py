"""
tp2_snapshot_viz.py — Snapshot & Reversion Deep-Dive Dashboard
Run from tp2_project folder:
    python tp2_snapshot_viz.py

Generates tp2_snapshots_dashboard.png with 6 charts:
  1. BTC spot price over time with violation count overlay
  2. Violation rate (% of clean options) per snapshot hour
  3. Reversion survival curve — how fast violations disappear
  4. Violation decay: avg residual violation % vs hours elapsed
  5. Fast vs medium vs slow reversion by original violation size
  6. Snapshot coverage: instruments & clean options over time
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime

# ── paths ─────────────────────────────────────────────────────────────────────
DB_PATH  = Path("data/db/tp2.db")
OUT_PATH = Path("tp2_snapshots_dashboard.png")

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
PURPLE = "#8E44AD"

plt.rcParams.update({
    "figure.facecolor": BG,   "axes.facecolor":  LBLUE,
    "axes.edgecolor":   RULE,  "axes.labelcolor": NAVY,
    "xtick.color":      MUTED, "ytick.color":     MUTED,
    "text.color":       NAVY,  "grid.color":      RULE,
    "grid.linewidth":   0.7,   "font.family":     "DejaVu Sans",
    "axes.spines.top":  False, "axes.spines.right": False,
})

# ── connect & load ────────────────────────────────────────────────────────────
print(f"Reading database: {DB_PATH.resolve()}")
conn = sqlite3.connect(DB_PATH)

# Snapshot-level aggregates
df_snap = pd.read_sql("""
    SELECT snapshot_hour,
           COUNT(*) as total,
           SUM(is_clean) as clean,
           AVG(underlying_price) as avg_spot
    FROM snapshots
    GROUP BY snapshot_hour
    ORDER BY snapshot_hour
""", conn)
df_snap["snapshot_dt"] = pd.to_datetime(df_snap["snapshot_hour"])

# Violations per snapshot
df_viol_snap = pd.read_sql("""
    SELECT snapshot_hour, COUNT(*) as n_viols,
           AVG(violation_pct) as avg_viol_pct,
           MAX(violation_pct) as max_viol_pct,
           SUM(CASE WHEN net_edge > 0 THEN 1 ELSE 0 END) as n_profitable
    FROM violations
    GROUP BY snapshot_hour
    ORDER BY snapshot_hour
""", conn)
df_viol_snap["snapshot_dt"] = pd.to_datetime(df_viol_snap["snapshot_hour"])

# Merge snap + viol
df_ts = df_snap.merge(df_viol_snap, on=["snapshot_hour", "snapshot_dt"], how="left")
df_ts["n_viols"] = df_ts["n_viols"].fillna(0).astype(int)
df_ts["viol_rate"] = df_ts["n_viols"] / df_ts["clean"] * 100

# Reversions with timing detail
df_rev = pd.read_sql("""
    SELECT r.violation_id, r.hours_elapsed, r.current_viol,
           r.reverted, r.revert_speed, v.violation_pct as original_pct
    FROM reversions r
    JOIN violations v ON r.violation_id = v.id
""", conn)

# All violations for reversion analysis
df_all_viols = pd.read_sql("""
    SELECT id, violation_pct, snapshot_hour FROM violations
""", conn)

conn.close()

# ── reversion survival data ──────────────────────────────────────────────────
# For each violation, find the earliest reversion time (if any)
df_first_rev = df_rev[df_rev["reverted"] == 1].groupby("violation_id")["hours_elapsed"].min().reset_index()
df_first_rev.columns = ["violation_id", "revert_hours"]

total_viols = len(df_all_viols)
max_h = 48
hours_grid = np.arange(0, max_h + 1, 1)
survival = []
for h in hours_grid:
    n_survived = total_viols - (df_first_rev["revert_hours"] <= h).sum()
    survival.append(n_survived / total_viols * 100)

# ── violation decay curve ────────────────────────────────────────────────────
df_decay = df_rev[df_rev["current_viol"].notna()].copy()
df_decay["h_bucket"] = df_decay["hours_elapsed"].round().astype(int)
decay_agg = df_decay.groupby("h_bucket").agg(
    avg_viol=("current_viol", "mean"),
    n=("current_viol", "count")
).reset_index()
decay_agg = decay_agg[decay_agg["h_bucket"] <= 48]

# ── reversion speed by original size ─────────────────────────────────────────
df_reverted = df_rev[df_rev["reverted"] == 1].copy()
speed_order = ["fast", "medium", "slow"]
df_reverted["revert_speed"] = pd.Categorical(df_reverted["revert_speed"],
                                              categories=speed_order, ordered=True)
# Bin original violation size
bins_pct = [0, 2, 5, 10, 20, 50, 200]
labels_pct = ["0-2%", "2-5%", "5-10%", "10-20%", "20-50%", "50%+"]
df_reverted["size_bin"] = pd.cut(df_reverted["original_pct"], bins=bins_pct, labels=labels_pct)

speed_by_size = df_reverted.groupby(["size_bin", "revert_speed"], observed=True).size().unstack(fill_value=0)

print(f"Snapshots: {len(df_snap)}")
print(f"Violations: {total_viols}")
print(f"Reverted: {len(df_first_rev)} ({len(df_first_rev)/total_viols*100:.1f}%)")
print(f"Median reversion time: {df_first_rev['revert_hours'].median():.1f}h")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 13), facecolor=BG)
fig.suptitle("AMS 520 · TP2 Snapshot & Reversion Analysis · BTC Options on Deribit",
             fontsize=15, fontweight="bold", color=NAVY, y=0.98)
fig.text(0.5, 0.955,
         f"{len(df_snap)} snapshots  ·  {df_snap['snapshot_dt'].min():%b %d} – "
         f"{df_snap['snapshot_dt'].max():%b %d %Y}  ·  "
         f"{total_viols:,} violations  ·  {len(df_first_rev):,} reverted "
         f"(median {df_first_rev['revert_hours'].median():.0f}h)",
         ha="center", fontsize=10, color=MUTED)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.06, right=0.97, top=0.92, bottom=0.08)

# ── Chart 1: BTC Spot + Violation Count ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df_ts["snapshot_dt"], df_ts["avg_spot"], color=BLUE, lw=1.5, label="BTC spot")
ax1.set_ylabel("BTC Price ($)", fontsize=9, color=BLUE)
ax1.set_title("BTC Price & Violation Count", fontsize=11, fontweight="bold", color=NAVY, pad=8)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
ax1.grid(axis="y", zorder=1)

ax1b = ax1.twinx()
ax1b.bar(df_ts["snapshot_dt"], df_ts["n_viols"], width=0.04, color=RED, alpha=0.4, label="Violations")
ax1b.set_ylabel("Violations", fontsize=9, color=RED)
ax1b.tick_params(axis="y", colors=RED)
ax1b.spines["right"].set_visible(True)
ax1b.spines["right"].set_color(RED)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

# ── Chart 2: Violation Rate Time Series ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.fill_between(df_ts["snapshot_dt"], df_ts["viol_rate"], color=RED, alpha=0.3)
ax2.plot(df_ts["snapshot_dt"], df_ts["viol_rate"], color=RED, lw=1.2)
mean_rate = df_ts["viol_rate"].mean()
ax2.axhline(mean_rate, color=AMBER, lw=1.5, ls="--", label=f"Mean = {mean_rate:.2f}%")
ax2.set_ylabel("Violation rate (%)", fontsize=9)
ax2.set_title("Violation Rate per Snapshot", fontsize=11, fontweight="bold", color=NAVY, pad=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
ax2.legend(fontsize=9)
ax2.grid(axis="y", zorder=1)

# ── Chart 3: Reversion Survival Curve ────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(hours_grid, survival, color=BLUE, lw=2.5, zorder=3)
ax3.fill_between(hours_grid, survival, color=BLUE, alpha=0.15, zorder=2)
# Mark key thresholds
for h_mark, label, color in [(4, "4h (fast)", GREEN), (24, "24h (medium)", AMBER)]:
    idx = np.searchsorted(hours_grid, h_mark)
    pct = survival[idx]
    ax3.axvline(h_mark, color=color, lw=1.5, ls="--", alpha=0.7)
    ax3.annotate(f"{label}\n{pct:.1f}% survive",
                 xy=(h_mark, pct), xytext=(h_mark + 3, pct + 5),
                 fontsize=8, color=color, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
ax3.set_xlabel("Hours since detection", fontsize=9)
ax3.set_ylabel("% violations still open", fontsize=9)
ax3.set_title("Reversion Survival Curve", fontsize=11, fontweight="bold", color=NAVY, pad=8)
ax3.set_xlim(0, max_h)
ax3.set_ylim(0, 105)
ax3.grid(zorder=1)

# ── Chart 4: Violation Decay Curve ───────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.bar(decay_agg["h_bucket"], decay_agg["avg_viol"], color=PURPLE, alpha=0.6, width=0.8, zorder=2)
ax4.plot(decay_agg["h_bucket"], decay_agg["avg_viol"], color=PURPLE, lw=1.5, marker="o",
         markersize=3, zorder=3)
ax4.set_xlabel("Hours since detection", fontsize=9)
ax4.set_ylabel("Avg residual violation (%)", fontsize=9)
ax4.set_title("Violation Decay Over Time", fontsize=11, fontweight="bold", color=NAVY, pad=8)
ax4.grid(axis="y", zorder=1)

# Add sample size on secondary axis
ax4b = ax4.twinx()
ax4b.plot(decay_agg["h_bucket"], decay_agg["n"], color=MUTED, lw=1, ls=":", alpha=0.7)
ax4b.set_ylabel("Sample size", fontsize=8, color=MUTED)
ax4b.tick_params(axis="y", colors=MUTED, labelsize=7)
ax4b.spines["right"].set_visible(True)
ax4b.spines["right"].set_color(MUTED)

# ── Chart 5: Reversion Speed by Violation Size ──────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
if len(speed_by_size) > 0:
    colors_speed = {
        "fast": GREEN,
        "medium": AMBER,
        "slow": MUTED,
    }
    bottom = np.zeros(len(speed_by_size))
    for speed in speed_order:
        if speed in speed_by_size.columns:
            vals = speed_by_size[speed].values
            ax5.bar(range(len(speed_by_size)), vals, bottom=bottom,
                    color=colors_speed[speed], label=speed, width=0.6, zorder=2)
            bottom += vals
    ax5.set_xticks(range(len(speed_by_size)))
    ax5.set_xticklabels(speed_by_size.index, fontsize=9)
    ax5.legend(fontsize=9, title="Reversion speed", title_fontsize=9)
ax5.set_xlabel("Original violation size", fontsize=9)
ax5.set_ylabel("Count of reverted violations", fontsize=9)
ax5.set_title("Reversion Speed by Violation Size", fontsize=11, fontweight="bold", color=NAVY, pad=8)
ax5.grid(axis="y", zorder=1)

# ── Chart 6: Snapshot Coverage ───────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.fill_between(df_snap["snapshot_dt"], df_snap["total"], color=BLUE, alpha=0.2, label="Total instruments")
ax6.plot(df_snap["snapshot_dt"], df_snap["total"], color=BLUE, lw=1.5)
ax6.fill_between(df_snap["snapshot_dt"], df_snap["clean"], color=GREEN, alpha=0.3, label="Clean (pass filters)")
ax6.plot(df_snap["snapshot_dt"], df_snap["clean"], color=GREEN, lw=1.5)
ax6.set_ylabel("Count", fontsize=9)
ax6.set_title("Snapshot Coverage Over Time", fontsize=11, fontweight="bold", color=NAVY, pad=8)
ax6.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax6.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
ax6.legend(fontsize=9)
ax6.grid(axis="y", zorder=1)

clean_pct = df_snap["clean"].mean() / df_snap["total"].mean() * 100
ax6.text(0.97, 0.97, f"Avg clean: {clean_pct:.1f}%",
         transform=ax6.transAxes, ha="right", va="top", fontsize=9,
         color=GREEN, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=RULE))

# ── save ─────────────────────────────────────────────────────────────────────
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nDashboard saved -> {OUT_PATH.resolve()}")
