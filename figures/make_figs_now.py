"""Figures that need no raw data: Fig. 2 (votes, numbers from results_votes.txt),
Fig. 5 (balance-capacity plane of exponents). Output: figures/*.pdf, *.eps, *.png."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({
    "font.family": "Liberation Sans", "mathtext.fontset": "cm", "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "legend.frameon": False,
})
BLUE, ORANGE, GREEN, INK, INK2, GRID, GREY = "#2a78d6", "#eb6834", "#2e9e5b", "#0b0b0b", "#52514e", "#e4e3df", "#b8b6b0"
os.makedirs("figures", exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "eps", "png"):
        fig.savefig(f"figures/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def arrow(ax, p, q, lw=1.2, color=INK2, rad=0.0, ms=8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=6, shrinkB=6))


def node(ax, xy, r, fc, label=None, ec=INK, fs=8, tc="white"):
    ax.add_patch(Circle(xy, r, fc=fc, ec=ec, lw=0.8, zorder=3))
    if label:
        ax.text(*xy, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


# ------------------------------------------------------------------ Fig. 2 (votes, pilot)
k = np.array([1.4, 2.9, 5.0, 7.6, 10.9, 14.8, 23.3, 37.0, 61.0, 102.4])
e = np.array([0.039, 0.0, 0.055, -0.130, -0.096, -0.291, -0.608, -1.419, -2.752, -3.012])
s = np.array([0.026, 0.0, 0.022, 0.094, 0.112, 0.113, 0.122, 0.174, 0.544, 0.423])
alpha, kref, g = 0.069, 2.9, 0.117
fig, ax = plt.subplots(figsize=(3.4, 2.6))
kk = np.geomspace(1.2, 130, 200)
club = lambda x: np.log(-np.expm1(-alpha * x) / x)
ax.plot(kk, np.exp(club(kk) - club(kref)), color=ORANGE, lw=1.6, label="club, $k_c=1/\\alpha=14.5$")
ax.plot(kk, (kk / kref) ** (-g), color=BLUE, lw=1.4, ls="--", label=f"power law, $\\gamma={g:.2f}$")
ax.plot(kk, np.minimum(1, 10 / kk) / np.minimum(1, 10 / kref), color=INK2, lw=1.0, ls=":",
        label="protocol, break at 10")
ax.errorbar(k, np.exp(e), yerr=[np.exp(e) - np.exp(e - 1.96 * s), np.exp(e + 1.96 * s) - np.exp(e)],
            fmt="o", ms=4, color=INK, ecolor=INK2, elinewidth=0.9, capsize=0, zorder=5, label="bins (95% CI)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1])
ax.set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1"])
ax.set_ylim(0.015, 1.5)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("voting load $k$ (full votes per day, preceding 5 days)")
ax.set_ylabel("value of a full vote (relative)")
ax.grid(True, color=GRID, lw=0.5)
ax.legend(fontsize=6.5, loc="lower left")
ax.text(0.98, 0.98, "20 000 posts, 218 742 votes", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK2)
save(fig, "fig2_votes")

# ------------------------------------------------------------------ Fig. 5 (exponent plane)
fig, ax = plt.subplots(figsize=(3.4, 3.1))
ax.plot([0, 1], [1, 0], ls="--", color=INK2, lw=0.9)
ax.text(0.37, 0.76, "$\\gamma+\\theta=1$ (same spectrum)", fontsize=6.5, color=INK2, rotation=-43, ha="center", va="center")
corners = [((0, 0), "copying $A$", (0.03, 0.04), "left"), ((1, 0), "random walk $AD^{-1}$\n(material)", (-0.03, 0.05), "right"),
           ((0, 1), "consensus $D^{-1}A$ (informational)", (0.05, 0.03), "left"),
           ((0.5, 0.5), "$D^{-1/2}AD^{-1/2}$", (0.05, -0.02), "left")]
for (x, y), lab, (dx, dy), ha in corners:
    ax.plot(x, y, "s", ms=6, mfc="white", mec=INK, mew=1, zorder=4, clip_on=False)
    ax.text(x + dx, y + dy, lab, fontsize=6.8, ha=ha, va="bottom")
# citations: source exponent measured, recipient unknown
ax.axvspan(0.35, 0.51, color="#fbe0d3", lw=0, zorder=0)  # solid light colour: EPS has no transparency
ax.text(0.43, 0.14, "citations\n$\\theta$ not measured", fontsize=6.8, ha="center", color="#c0461a", zorder=5)
# Steem feed
ax.errorbar([0.0], [0.99], xerr=[[0.03], [0.03]], yerr=[[0.02], [0.02]], fmt="o", ms=6, color=BLUE, zorder=6,
            clip_on=False)
ax.plot([0.0], [0.82], marker="o", ms=4.5, mfc="white", mec=BLUE, ls="none", clip_on=False, zorder=6)
ax.plot([0.0], [0.51], marker="v", ms=4.5, mfc="white", mec=BLUE, ls="none", clip_on=False, zorder=6)
ax.text(0.05, 0.64, "Steem feed\n(within readers)", fontsize=6.8, color=BLUE, va="center")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.12)
ax.set_xlabel("source exponent $\\gamma$")
ax.set_ylabel("recipient exponent $\\theta$")
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
save(fig, "fig5_plane")
print("saved figures/fig2_votes, fig5_plane")
