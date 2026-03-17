"""
tp2_visualize.py — TP2 Dashboard
Run from your tp2_project folder:
    python tp2_visualize.py

Reads directly from data/db/tp2.db
violation_pct and avg_spread_pct are stored as percentages (e.g. 4.0 = 4%)
"""

import sqlite3
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime, timezone

# ── paths ─────────────────────────────────────────────────────────────────────
DB_PATH  = Path("data/db/tp2.db")
OUT_PATH = Path("tp2_dashboard.png")

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
    "figure.facecolor": BG,   "axes.facecolor":  LBLUE,
    "axes.edgecolor":   RULE,  "axes.labelcolor": NAVY,
    "xtick.color":      MUTED, "ytick.color":     MUTED,
    "text.color":       NAVY,  "grid.color":      RULE,
    "grid.linewidth":   0.7,   "font.family":     "DejaVu Sans",
    "axes.spines.top":  False, "axes.spines.right": False,
})

# ── connect ───────────────────────────────────────────────────────────────────
print(f"Reading database: {DB_PATH.resolve()}")
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

# ── latest snapshot hour ──────────────────────────────────────────────────────
snap_hour = conn.execute(
    "SELECT snapshot_hour FROM snapshots GROUP BY snapshot_hour "
    "ORDER BY snapshot_hour DESC LIMIT 1"
).fetchone()
if snap_hour is None:
    raise RuntimeError("No snapshots found.")
snap_hour = snap_hour["snapshot_hour"]
print(f"Latest snapshot: {snap_hour}")

# ── load config ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
try:
    import config as cfg
    CALLS_ONLY = cfg.CALLS_ONLY
    MIN_BID    = cfg.MIN_BID
    MAX_SPR    = cfg.MAX_SPREAD_PCT
    MIN_OI     = cfg.MIN_OPEN_INTEREST
    MIN_MON    = cfg.MIN_MONEYNESS
    MAX_MON    = cfg.MAX_MONEYNESS
    MIN_TTM    = cfg.MIN_DAYS_TO_EXPIRY
    MIN_IV     = cfg.MIN_IV_PCT
    MAX_IV     = cfg.MAX_IV_PCT
except Exception:
    CALLS_ONLY = True
    MIN_BID, MAX_SPR, MIN_OI = 0.0, 0.40, 1
    MIN_MON, MAX_MON = 0.50, 3.00
    MIN_TTM, MIN_IV, MAX_IV = 1, 0.5, 500.0

MONTH_MAP = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
             "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

def parse_expiry(s):
    try:
        m = re.match(r"(\d{1,2})([A-Z]{3})(\d{2})", s)
        if not m: return None
        return datetime(2000+int(m.group(3)), MONTH_MAP[m.group(2)],
                        int(m.group(1)), 8, 0, tzinfo=timezone.utc)
    except Exception:
        return None

snap_dt = datetime.fromisoformat(snap_hour)

# ── cleaning funnel ───────────────────────────────────────────────────────────
rows_all = conn.execute(
    "SELECT * FROM snapshots WHERE snapshot_hour = ?", (snap_hour,)
).fetchall()
total = len(rows_all)

after_calls = after_bid = after_ask = after_mid = 0
after_oi = after_spr = after_mon = after_ttm = after_iv = 0

for r in rows_all:
    bid  = r["bid_price"]; ask = r["ask_price"]
    mid  = ((bid or 0) + (ask or 0)) / 2 if bid and ask else None
    oi   = r["open_interest"]; iv = r["mark_iv"]
    spot = r["underlying_price"]; strike = r["strike"]

    if CALLS_ONLY and r["option_type"] != "C": continue
    after_calls += 1
    if bid is None or bid <= MIN_BID: continue
    after_bid += 1
    if ask is None or ask <= bid: continue
    after_ask += 1
    if mid is None or mid <= 0: continue
    after_mid += 1
    if oi is None or oi < MIN_OI: continue
    after_oi += 1
    if (ask - bid) / mid > MAX_SPR: continue
    after_spr += 1
    if spot is None or spot <= 0: continue
    mon = strike / spot
    if mon < MIN_MON or mon > MAX_MON: continue
    after_mon += 1
    exp_dt = parse_expiry(r["expiry"])
    if exp_dt:
        tte = (exp_dt - snap_dt).total_seconds() / 86400
        if tte < MIN_TTM: continue
    after_ttm += 1
    if iv is not None and (iv < MIN_IV or iv > MAX_IV): continue
    after_iv += 1

n_clean = after_iv

FUNNEL = [
    ("Raw instruments",   total),
    ("Calls only",        after_calls),
    ("bid > 0",           after_bid),
    ("ask > bid",         after_ask),
    ("mid > 0",           after_mid),
    ("Open interest >= 1",after_oi),
    ("Spread < 40%",      after_spr),
    ("Moneyness 0.5-3x",  after_mon),
    ("TTM >= 1 day",      after_ttm),
    ("IV sanity check",   after_iv),
]
print(f"Funnel: {total} -> {n_clean} clean ({n_clean/total*100:.1f}%)")

# ── violations per snapshot ───────────────────────────────────────────────────
viol_by_snap = conn.execute("""
    SELECT snapshot_hour, COUNT(*) as n
    FROM violations
    GROUP BY snapshot_hour
    ORDER BY snapshot_hour
""").fetchall()
snap_labels = [r["snapshot_hour"][11:16] for r in viol_by_snap]
viol_counts = [r["n"] for r in viol_by_snap]

clean_by_snap = {r["snapshot_hour"]: r["clean"] for r in conn.execute("""
    SELECT snapshot_hour, SUM(is_clean) as clean
    FROM snapshots GROUP BY snapshot_hour
""").fetchall()}

# ── violations — already stored as % ─────────────────────────────────────────
df_viol = pd.read_sql("""
    SELECT violation_pct, avg_spread_pct, net_edge, K1, K2, T1, T2, snapshot_hour
    FROM violations
""", conn)
print(f"Violations: {len(df_viol)} | "
      f"avg={df_viol['violation_pct'].mean():.2f}% | "
      f"max={df_viol['violation_pct'].max():.2f}%")

# ── reversions ────────────────────────────────────────────────────────────────
df_rev = pd.read_sql("""
    SELECT r.violation_id, r.hours_elapsed, r.current_viol,
           r.reverted, r.revert_speed, v.violation_pct as original_pct
    FROM reversions r
    JOIN violations v ON r.violation_id = v.id
""", conn)
print(f"Reversion checks: {len(df_rev)} | reverted={df_rev['reverted'].sum()}")
conn.close()

# ── expiry sort for heatmap ───────────────────────────────────────────────────
def expiry_key(e):
    m = re.match(r"(\d{1,2})([A-Z]{3})(\d{2})", e)
    if not m: return (99, 99, 0)
    return (int(m.group(3)), MONTH_MAP.get(m.group(2), 99), int(m.group(1)))

top_expiries = sorted(df_viol["T1"].unique(), key=expiry_key)[:6]

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.suptitle("AMS 520 · TP2 Violation Detection · BTC Options on Deribit",
             fontsize=15, fontweight="bold", color=NAVY, y=0.98)
fig.text(0.5, 0.955,
         f"Latest snapshot: {snap_hour}  ·  "
         f"{total} instruments  ·  {n_clean} clean ({n_clean/total*100:.1f}%)",
         ha="center", fontsize=10, color=MUTED)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.35,
                       left=0.07, right=0.97, top=0.92, bottom=0.10)

# ── Chart 1: Cleaning funnel ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
f_labels = [f[0] for f in FUNNEL]
f_values = [f[1] for f in FUNNEL]
f_colors = [BLUE if i == 0 else (GREEN if i == len(FUNNEL)-1 else MUTED)
            for i in range(len(FUNNEL))]
bars1 = ax1.barh(range(len(FUNNEL)), f_values, color=f_colors, height=0.65, zorder=2)
ax1.set_yticks(range(len(FUNNEL)))
ax1.set_yticklabels(f_labels, fontsize=8.5)
ax1.invert_yaxis()
ax1.set_xlabel("Instruments remaining", fontsize=9)
ax1.set_title("Data Cleaning Funnel", fontsize=11, fontweight="bold", color=NAVY, pad=8)
ax1.grid(axis="x", zorder=1)
ax1.axvline(n_clean, color=GREEN, lw=1.5, ls="--", zorder=3)
step = max(f_values) * 0.01
for bar, val in zip(bars1, f_values):
    ax1.text(val + step, bar.get_y() + bar.get_height()/2,
             str(val), va="center", fontsize=8.5, color=NAVY)
ax1.text(n_clean + step, len(FUNNEL) - 0.6,
         f"{n_clean} clean", fontsize=8, color=GREEN, fontweight="bold")

# ── Chart 2: Violations per snapshot ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
if viol_counts:
    bars2 = ax2.bar(range(len(snap_labels)), viol_counts,
                    color=BLUE, width=0.55, zorder=2)
    mean_v = np.mean(viol_counts)
    ax2.axhline(mean_v, color=AMBER, lw=1.5, ls="--", label=f"Mean = {mean_v:.1f}")
    for bar, val in zip(bars2, viol_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 str(val), ha="center", fontsize=9, fontweight="bold", color=NAVY)
    ax2.set_xticks(range(len(snap_labels)))
    ax2.set_xticklabels(snap_labels, rotation=30, ha="right", fontsize=8)
    ax2.set_ylim(0, max(viol_counts) * 1.35)
    ax2.legend(fontsize=9)
    rates = [v / clean_by_snap.get(r["snapshot_hour"], n_clean) * 100
             for r, v in zip(viol_by_snap, viol_counts)]
    ax2.text(0.97, 0.97, f"Avg rate: {np.mean(rates):.2f}%",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9,
             color=BLUE, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=RULE))
else:
    ax2.text(0.5, 0.5, "No violations yet", ha="center", va="center",
             fontsize=12, color=MUTED, transform=ax2.transAxes)
ax2.set_xlabel("Snapshot time (UTC)", fontsize=9)
ax2.set_ylabel("Violations detected", fontsize=9)
ax2.set_title("Violations per Snapshot", fontsize=11, fontweight="bold", color=NAVY, pad=8)
ax2.grid(axis="y", zorder=1)

# ── Chart 3: Violation size distribution ─────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
if len(df_viol) > 0:
    vp   = df_viol["violation_pct"]          # already %
    bins = np.linspace(0, vp.max() + 0.5, 20)
    _, bin_edges, patches = ax3.hist(vp, bins=bins,
                                     color=RED, edgecolor=BG, linewidth=0.5, zorder=2)
    if (df_viol["net_edge"] > 0).any():
        threshold = df_viol[df_viol["net_edge"] > 0]["violation_pct"].min()
        for patch, left in zip(patches, bin_edges[:-1]):
            if left >= threshold:
                patch.set_facecolor(GREEN)
    ax3.axvline(vp.mean(), color=AMBER, lw=1.5, ls="--",
                label=f"Mean = {vp.mean():.2f}%")
    ax3.legend(handles=[
        mpatches.Patch(color=RED,   label="Below tradeable threshold"),
        mpatches.Patch(color=GREEN, label="Potentially profitable"),
    ], fontsize=8, loc="upper right")
    ax3.set_xlabel("Violation size (%)", fontsize=9)
    ax3.set_ylabel("Count", fontsize=9)
else:
    ax3.text(0.5, 0.5, "No violations yet", ha="center", va="center",
             fontsize=12, color=MUTED, transform=ax3.transAxes)
ax3.set_title("Violation Size Distribution", fontsize=11, fontweight="bold", color=NAVY, pad=8)
ax3.grid(axis="y", zorder=1)

# ── Chart 4: Violation vs spread cost ────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
if len(df_viol) > 0:
    x = df_viol["avg_spread_pct"]            # already %
    y = df_viol["violation_pct"]             # already %
    dot_colors = [GREEN if ne > 0 else RED for ne in df_viol["net_edge"]]
    ax4.scatter(x, y, c=dot_colors, s=28, alpha=0.7, zorder=3)
    lim = max(x.max(), y.max()) * 1.08
    xs  = np.linspace(0, lim, 100)
    ax4.plot(xs, xs, color=AMBER, lw=1.5, ls="--", label="Break-even", zorder=2)
    ax4.fill_between(xs, 0,   xs,  alpha=0.07, color=RED,   zorder=1)
    ax4.fill_between(xs, xs, lim,  alpha=0.07, color=GREEN, zorder=1)
    ax4.set_xlim(0, lim); ax4.set_ylim(0, lim)
    n_profit = (df_viol["net_edge"] > 0).sum()
    ax4.text(0.98, 0.08, f"Profitable: {n_profit}/{len(df_viol)}",
             transform=ax4.transAxes, ha="right", va="bottom", fontsize=9,
             color=GREEN if n_profit > 0 else RED, fontweight="bold")
    ax4.text(0.98, 0.02, "Above line = edge > cost",
             transform=ax4.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)
    ax4.legend(fontsize=9)
ax4.set_xlabel("Avg spread cost per leg (%)", fontsize=9)
ax4.set_ylabel("Violation size (%)", fontsize=9)
ax4.set_title("Violation vs Spread Cost", fontsize=11, fontweight="bold", color=NAVY, pad=8)
ax4.grid(zorder=1)

# ── Chart 5: Reversion pie ────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
if len(df_rev) > 0:
    n_reverted = int(df_rev["reverted"].sum())
    n_open     = max(len(df_viol) - n_reverted, 0)
    wedges, texts, autotexts = ax5.pie(
        [n_reverted, n_open],
        labels=[f"Reverted\n({n_reverted})", f"Still open\n({n_open})"],
        autopct="%1.0f%%",
        colors=[GREEN, BLUE],
        explode=(0.05, 0),
        startangle=90,
        textprops={"fontsize": 10},
        wedgeprops={"edgecolor": BG, "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(10); at.set_fontweight("bold"); at.set_color(BG)
    speed_counts = df_rev[df_rev["reverted"] == 1]["revert_speed"].value_counts()
    for j, (k, lbl, c) in enumerate([("fast","<4h",GREEN),
                                      ("medium","4-24h",AMBER),
                                      ("slow",">24h",MUTED)]):
        ax5.annotate(f"{lbl}: {speed_counts.get(k, 0)}",
                     xy=(0.16 + j*0.34, 0.04), xycoords="axes fraction",
                     ha="center", fontsize=9, color=c, fontweight="bold")
    ax5.annotate("Reversion speed:", xy=(0.5, 0.10), xycoords="axes fraction",
                 ha="center", fontsize=8.5, color=MUTED)
else:
    ax5.text(0.5, 0.5, "No reversion data yet", ha="center", va="center",
             fontsize=12, color=MUTED, transform=ax5.transAxes)
ax5.set_title("Violation Reversion Status", fontsize=11, fontweight="bold", color=NAVY, pad=8)

# ── Chart 6: Heatmap strike x expiry ─────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
if len(df_viol) > 0 and len(df_viol["T1"].unique()) >= 2:
    all_strikes = sorted(df_viol["K1"].unique())
    n_buckets   = min(6, len(all_strikes))
    edges       = np.percentile(all_strikes, np.linspace(0, 100, n_buckets + 1))
    blabels     = [f"{int((edges[i]+edges[i+1])/2/1000)}k" for i in range(n_buckets)]

    heat = np.zeros((len(top_expiries), n_buckets))
    for _, row in df_viol.iterrows():
        if row["T1"] not in top_expiries: continue
        ti = top_expiries.index(row["T1"])
        bi = min(range(n_buckets),
                 key=lambda i: abs((edges[i]+edges[i+1])/2 - row["K1"]))
        heat[ti, bi] += row["violation_pct"]   # already %

    im = ax6.imshow(heat, cmap="Blues", aspect="auto", vmin=0)
    ax6.set_xticks(range(n_buckets))
    ax6.set_xticklabels(blabels, fontsize=9)
    ax6.set_yticks(range(len(top_expiries)))
    ax6.set_yticklabels(top_expiries, fontsize=9)
    for i in range(len(top_expiries)):
        for j in range(n_buckets):
            if heat[i, j] > 0:
                ax6.text(j, i, f"{heat[i,j]:.1f}", ha="center", va="center",
                         fontsize=8,
                         color=BG if heat[i, j] > heat.max()*0.6 else NAVY)
    plt.colorbar(im, ax=ax6, shrink=0.8, label="Cumulative violation %")
else:
    ax6.text(0.5, 0.5, "Need more snapshots\nfor heatmap",
             ha="center", va="center", fontsize=11, color=MUTED,
             transform=ax6.transAxes, linespacing=1.8)
ax6.set_xlabel("Strike K1 bucket (near leg)", fontsize=9)
ax6.set_ylabel("Expiry T1 (near leg)", fontsize=9)
ax6.set_title("Violation Intensity\nby Strike & Expiry",
              fontsize=11, fontweight="bold", color=NAVY, pad=6)

# ── save + print summary ──────────────────────────────────────────────────────
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=BG)

print(f"\nDashboard saved -> {OUT_PATH.resolve()}")
print(f"\n{'='*50}")
print(f"  Snapshots collected : {len(viol_by_snap)}")
print(f"  Clean options       : {n_clean}/{total} ({n_clean/total*100:.1f}%)")
print(f"  Total violations    : {len(df_viol)}")
print(f"  Avg violation size  : {df_viol['violation_pct'].mean():.2f}%")
print(f"  Max violation size  : {df_viol['violation_pct'].max():.2f}%")
print(f"  Reversion checks    : {len(df_rev)}")
print(f"  Reverted            : {df_rev['reverted'].sum()}")
print(f"  Tradeable (net>0)   : {(df_viol['net_edge']>0).sum()}")
print(f"{'='*50}")