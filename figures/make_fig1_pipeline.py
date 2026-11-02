"""Fig. 1 of the merged paper (two layers, what is measured where, and the source/recipient/reblog
schemes) and the pipeline figure for the appendix. Output: figures/fig1_scheme.*, figures/figA_pipeline.*"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

plt.rcParams.update({
    "font.family": "Liberation Sans", "mathtext.fontset": "cm", "font.size": 8.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})
BLUE, ORANGE, GREEN, INK, INK2, GREY = "#2a78d6", "#eb6834", "#2e9e5b", "#0b0b0b", "#52514e", "#b8b6b0"
PALE_B, PALE_O = "#e6f0fb", "#fdebe2"
os.makedirs("figures", exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "eps", "png"):
        fig.savefig(f"figures/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def arrow(ax, p, q, lw=1.2, color=INK2, rad=0.0, ms=8, ls="-", shrink=6, z=2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=shrink, shrinkB=shrink, zorder=z))


def node(ax, xy, r, fc, label=None, ec=INK, fs=8, tc="white", z=3):
    ax.add_patch(Circle(xy, r, fc=fc, ec=ec, lw=0.8, zorder=z))
    if label:
        ax.text(*xy, label, ha="center", va="center", fontsize=fs, color=tc, zorder=z + 1)


def tag(ax, xy, s, color=INK2, ha="left", fs=6.8):
    ax.text(*xy, s, fontsize=fs, color=color, ha=ha, va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=0.6), zorder=6)


# ====================================================================== Fig. 1
fig = plt.figure(figsize=(7.0, 5.0))
axA = fig.add_axes([0.0, 0.43, 1.0, 0.57])
axA.set_xlim(0, 10)
axA.set_ylim(-0.45, 5.2)
axA.axis("off")
axA.text(0.05, 5.1, "(a)", fontsize=9, fontweight="bold", va="top")


def plane(ax, y0, fc, ec, label):
    pts = [(0.9, y0), (7.3, y0), (8.1, y0 + 1.25), (1.7, y0 + 1.25)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=0.9, zorder=0))
    ax.text(1.55, y0 + 0.12, label, fontsize=8, color=ec, va="bottom", fontweight="bold")


yI, yM = 3.45, 0.75
plane(axA, yI, PALE_B, BLUE, "information layer")
plane(axA, yM, PALE_O, ORANGE, "material layer")
acc = np.array([(2.3, 0.55), (3.5, 0.95), (4.6, 0.4), (5.7, 0.85), (6.8, 0.5)])
names = ["$a$", "$v$", "$x$", "", "$b$"]
for i, (x, dy) in enumerate(acc):
    axA.plot([x + 0.6 * dy, x + 0.6 * dy], [yM + dy, yI + dy], color=GREY, lw=0.6, ls=":", zorder=1)
PI = [(x + 0.6 * dy, yI + dy) for x, dy in acc]
PM = [(x + 0.6 * dy, yM + dy) for x, dy in acc]
for i, p in enumerate(PI):
    node(axA, p, 0.17, BLUE, names[i], fs=7.5)
for i, p in enumerate(PM):
    node(axA, p, 0.17, ORANGE if i != 4 else INK2, names[i], fs=7.5)
# information edges: post, reblog, follow
arrow(axA, PI[0], PI[1], color=BLUE, lw=1.1)
arrow(axA, PI[1], PI[2], color=BLUE, lw=1.1)
arrow(axA, PI[3], PI[2], color=BLUE, lw=1.1)
arrow(axA, PI[4], PI[3], color=BLUE, lw=0.9, rad=0.25)
# material edges: transfers, payment to bid bot b, bot vote back
arrow(axA, PM[1], PM[2], color=ORANGE, lw=1.1)
arrow(axA, PM[3], PM[2], color=ORANGE, lw=1.1, rad=-0.2)
arrow(axA, PM[0], PM[4], color=ORANGE, lw=1.0, rad=-0.32)
arrow(axA, PM[4], PM[0], color=INK2, lw=1.0, rad=-0.32, ls="--")
# inter-layer channels
lam_x = 8.65
arrow(axA, (lam_x, yI + 0.25), (lam_x, yM + 1.05), color=INK, lw=1.4, ms=10, shrink=0)
arrow(axA, (lam_x + 0.45, yM + 1.05), (lam_x + 0.45, yI + 0.25), color=INK, lw=1.4, ms=10, shrink=0)
axA.text(lam_x - 0.08, (yI + yM + 1.3) / 2, "$\\lambda_{IM}$", ha="right", va="center", fontsize=9)
axA.text(lam_x + 0.53, (yI + yM + 1.3) / 2, "$\\lambda_{MI}$", ha="left", va="center", fontsize=9)
# tags: what is measured where
tag(axA, (2.0, 5.05), "copying at the source, $\\gamma\\approx0$ (Fig. 3)", BLUE)
tag(axA, (5.15, 5.05), "capacity at the recipient, $\\theta\\approx1$ (Fig. 4)", BLUE)
tag(axA, (7.25, 4.45), "$W^I=D^{-1}A$", BLUE)
tag(axA, (8.25, 3.05), "attention $\\to$ reward: weak, +15% (Fig. 6)", INK, ha="right")
tag(axA, (9.3, 1.25), "value $\\to$ readers:\nnot significant (Fig. 7)", INK, ha="left")
tag(axA, (1.2, -0.25), "market for votes: payment to a bid bot, bought vote (dashed) attracts votes (Fig. 7)", INK2)
tag(axA, (3.55, 2.45), "transfers: P1 holds, row sums $R_v$ do not freeze (Fig. 8)", ORANGE)
tag(axA, (0.95, 2.62), "votes divided by protocol (Fig. 2)", INK2)
axA.text(PM[4][0] + 0.25, PM[4][1] + 0.18, "bid bot", fontsize=6.5, color=INK2)

# ---------------- lower row: (b) source, (c) recipient, (d) reblog design
axs = [fig.add_axes([0.02, 0.0, 0.27, 0.40]), fig.add_axes([0.31, 0.0, 0.27, 0.40]),
       fig.add_axes([0.60, 0.0, 0.40, 0.40])]
for ax, w in zip(axs, (1, 1, 1.35)):
    ax.set_xlim(0, w)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
ax = axs[0]
ax.text(0.0, 1.0, "(b) source: $h(k)\\propto k^{-\\gamma}$", fontsize=8.5, va="top")
src = (0.2, 0.5)
node(ax, src, 0.07, BLUE, "$v$")
ys = np.linspace(0.18, 0.82, 5)
for y in ys:
    node(ax, (0.78, y), 0.045, ORANGE)
    arrow(ax, src, (0.78, y), lw=2.2 * 5 ** -0.5)
ax.text(0.78, 0.0, "$k$ items\nshared by $v$", ha="center", fontsize=7, color=INK2)
ax.text(0.2, 0.33, "audience\n$n_v$", ha="center", va="top", fontsize=7.5, color=INK2)
ax = axs[1]
ax.text(0.0, 1.0, "(c) recipient: $\\propto n^{-\\theta}$", fontsize=8.5, va="top")
rec = (0.78, 0.5)
node(ax, rec, 0.07, GREEN, "$x$")
for y in ys:
    node(ax, (0.2, y), 0.045, BLUE)
    arrow(ax, (0.2, y), rec, lw=2.2 * 5 ** -0.5)
ax.text(0.2, 0.0, "$n$ sources\nfollowed by $x$", ha="center", fontsize=7, color=INK2)
ax = axs[2]
ax.text(0.0, 1.0, "(d) reblog design", fontsize=8.5, va="top")
a, p, v1, v2 = (0.1, 0.62), (0.4, 0.62), (0.75, 0.82), (0.75, 0.42)
node(ax, a, 0.045, GREY, "$a$", tc=INK)
ax.add_patch(Rectangle((p[0] - 0.04, p[1] - 0.04), 0.08, 0.08, fc=ORANGE, ec=INK, lw=0.8, zorder=3))
ax.text(p[0], p[1] - 0.09, "post $p$", ha="center", fontsize=7, color=INK2)
arrow(ax, a, p)
for v, lab in ((v1, "$v_1$"), (v2, "$v_2$")):
    node(ax, v, 0.05, BLUE, lab)
    arrow(ax, p, v)
for v, ks in ((v1, (0.93, 0.85, 0.77)), (v2, (0.5, 0.42, 0.34))):
    for j, y in enumerate(ks):
        node(ax, (1.16, y), 0.03, GREEN if j == 0 else "white", ec=INK2)
        arrow(ax, v, (1.16, y), lw=0.7, ms=6)
ax.text(1.22, 0.93, "$Y_{pv}$", fontsize=7.5, va="center")
ax.text(1.10, 0.20, "readers (not following $a$);\nfilled: engaged within 7 d", ha="center", fontsize=6.5, color=INK2)
ax.add_patch(Rectangle((0.36, 0.02), 0.08, 0.08, fc="white", ec=ORANGE, lw=1.0, hatch="////", zorder=3))
ax.text(0.49, 0.06, "placebo $p''$: same category,\nnot shared by $v$", fontsize=6.5, va="center", color=INK2)
save(fig, "fig1_scheme")

# ====================================================================== pipeline (appendix)
fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.set_xlim(-0.12, 10.12)
ax.set_ylim(-0.15, 6.3)
ax.axis("off")
cols = [0.0, 2.45, 5.35, 7.85]
widths = [2.05, 2.5, 2.1, 2.15]
heads = ["data", "constructed objects", "estimators", "results"]
for x, w, h in zip(cols, widths, heads):
    ax.text(x + w / 2, 6.18, h, ha="center", fontsize=8.5, fontweight="bold")


def box(x, y, w, h, text, ec=INK2, fc="white", fs=6.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15)
    return (x, y, w, h)


def link(b1, b2, color=GREY, dy1=0.0):
    p = (b1[0] + b1[2], b1[1] + b1[3] / 2 + dy1)
    q = (b2[0], b2[1] + b2[3] / 2)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.6, color=color, shrinkA=3, shrinkB=3))


W0, W1, W2, W3 = widths
d1 = box(cols[0], 4.55, W0, 1.3, "SteemOps archives,\nMar 2016-Nov 2019:\ncomment, vote,\ncustom_json, transfer")
d2 = box(cols[0], 2.75, W0, 1.3, "public API:\nget_active_votes\n(rshares of votes),\nget_content (rewards)")
d3 = box(cols[0], 1.0, W0, 1.25, "rules for bots: bid bots,\nvote bots, trails, pools,\ncomment bots, services")
o1 = box(cols[1], 5.05, W1, 0.8, "follow graph since Aug 2016,\nactive followers", BLUE, PALE_B)
o2 = box(cols[1], 4.0, W1, 0.8, "reblog pairs and placebos,\ncase-control sample, reader panel", BLUE, PALE_B)
o3 = box(cols[1], 2.95, W1, 0.8, "author x week panel: posts,\nfollowers, shocks, organic rshares")
o4 = box(cols[1], 1.9, W1, 0.8, "purchases x 12-h bins: bot vote,\nplacebo, services, voter types")
o5 = box(cols[1], 0.85, W1, 0.8, "transfer operator $W'$, row sums $R$,\ninflow $I$, stake (value of a vote)", ORANGE, PALE_O)
e1 = box(cols[2], 4.3, W2, 1.3, "conditional Poisson\nwith fixed effects (post,\npair, reader, voter),\nclustered errors")
e2 = box(cols[2], 2.95, W2, 0.8, "stacked event study,\ndistributed lags, permutations")
e3 = box(cols[2], 1.9, W2, 0.8, "within-post Poisson,\nplacebo, dose")
e4 = box(cols[2], 0.85, W2, 0.8, "cross-section, within\naccount, timing")
r1 = box(cols[3], 4.3, W3, 1.3, "$\\gamma\\approx0,\\ \\theta\\approx1$: $W^I=D^{-1}A$;\nvotes: protocol rule\n(Figs. 2-4)", BLUE, PALE_B)
r2 = box(cols[3], 2.95, W3, 0.8, "$\\lambda_{IM}$: attention $\\to$ reward\n(Fig. 6)")
r3 = box(cols[3], 1.9, W3, 0.8, "$\\lambda_{MI}$, market for votes\n(Fig. 7)")
r4 = box(cols[3], 0.85, W3, 0.8, "P1 and result (i)\n(Fig. 8)", ORANGE, PALE_O)
link(d1, o1, dy1=0.3)
link(d1, o2, dy1=0.1)
link(d1, o3, dy1=-0.1)
link(d1, o4, dy1=-0.25)
link(d1, o5, dy1=-0.4)
link(d2, o3, dy1=0.2)
link(d2, o4)
link(d2, o5, dy1=-0.2)
link(d3, o4, dy1=0.2)
link(d3, o5, dy1=-0.1)
link(o1, e1)
link(o2, e1)
for o, e in ((o3, e2), (o4, e3), (o5, e4)):
    link(o, e, INK2)
for e, r in ((e1, r1), (e2, r2), (e3, r3), (e4, r4)):
    link(e, r, INK2)
sv = box(cols[1], -0.12, cols[2] + W2 - cols[1], 0.6,
         "synthetic data with known parameters in the same formats (App. A):\n"
         "each estimator recovers the truth and returns zero without an effect", GREEN, "#e7f5ec", fs=6.0)
ax.add_patch(FancyArrowPatch((cols[2] + W2 + 0.18, sv[1] + sv[3] / 2), (cols[2] + W2 + 0.18, e1[1] + 0.12),
                             arrowstyle="-|>", mutation_scale=6, lw=0.8, color=GREEN, linestyle="--",
                             connectionstyle="arc3,rad=0", shrinkA=0, shrinkB=0))
ax.plot([cols[2] + W2, cols[2] + W2 + 0.18], [sv[1] + sv[3] / 2] * 2, color=GREEN, lw=0.8, ls="--")
for e in (e1, e2, e3, e4):
    ax.plot([cols[2] + W2 + 0.18, cols[2] + W2 + 0.04], [e[1] + 0.12] * 2, color=GREEN, lw=0.8, ls="--")
save(fig, "figA_pipeline")
print("saved figures/fig1_scheme, figures/figA_pipeline")
