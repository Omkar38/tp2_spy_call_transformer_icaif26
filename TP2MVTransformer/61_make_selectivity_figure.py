"""#1 selectivity curves: Sharpe & Sortino vs number of trades N, as the
calibrated selection threshold sweeps. Adjudicates whether the model ranking
is an operating-point artifact. Writes icaif26/figures/selectivity_curves.pdf."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "TP2MVTransformer" / ".mplconfig"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "TP2MVTransformer" / "tables" / "action_list" / "selectivity_sweep.csv"
FIGDIR = ROOT / "AICIF" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

sweep = pd.read_csv(SWEEP)
# operating point N at p>=0.5 (from paper): the threshold==0.5 row
style = {
    "MLP (1)": ("#1f77b4", "o", "-"),
    "Transformer (1)": ("#d62728", "s", "--"),
    "Transformer (5)": ("#2ca02c", "^", "-"),
    "LightGBM (1)": ("#7f7f7f", "D", ":"),
}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for metric, ax, ylab in [("sharpe", axes[0], "Annualized Sharpe"),
                         ("sortino", axes[1], "Annualized Sortino")]:
    for name, (c, mk, ls) in style.items():
        sub = sweep[sweep.model == name].sort_values("n")
        if sub.empty:
            continue
        ax.plot(sub["n"] / 1000, sub[metric], ls, color=c, marker=mk, ms=3.2, lw=1.3, label=name)
        # mark the p>=0.5 operating point
        op = sub[np.isclose(sub["threshold"], 0.5)]
        if not op.empty:
            ax.scatter(op["n"] / 1000, op[metric], s=55, facecolors="none",
                       edgecolors=c, linewidths=1.6, zorder=5)
    ax.set_xlabel("Trades selected $N$ (thousands)")
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.25, lw=0.5)
axes[0].legend(fontsize=7, loc="lower left", framealpha=0.9)
axes[0].set_title("Sharpe vs.\\ selectivity", fontsize=9)
axes[1].set_title("Sortino vs.\\ selectivity", fontsize=9)
fig.tight_layout()
fig.savefig(FIGDIR / "selectivity_curves.pdf", bbox_inches="tight")
print("wrote", FIGDIR / "selectivity_curves.pdf")

# print matched-N read-off for the caption
print("\nMatched-N read-off (interp):")
for N in [10000, 12000, 14000]:
    parts = []
    for name in ["MLP (1)", "Transformer (1)", "Transformer (5)"]:
        sub = sweep[sweep.model == name].sort_values("n")
        if sub.n.min() <= N <= sub.n.max():
            parts.append(f"{name}={np.interp(N, sub.n, sub.sharpe):.2f}")
    print(f"  N={N}: " + ", ".join(parts))
