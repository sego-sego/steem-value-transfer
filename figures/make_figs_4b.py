"""Figures 6-8 of the merged paper (inter-layer channels and material layer), from the numbers in
results_model_c_fresh.txt, results_mi.txt (version 5) and results_material.txt (version 2).
Output: figures/fig6_im.*, fig7_mi.*, fig8_material.*"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "Liberation Sans", "mathtext.fontset": "cm", "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "legend.frameon": False,
})
BLUE, ORANGE, GREEN, INK, INK2, GRID = "#2a78d6", "#eb6834", "#2e9e5b", "#0b0b0b", "#52514e", "#e4e3df"
os.makedirs("figures", exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "eps", "png"):
        fig.savefig(f"figures/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def label(ax, s):
    ax.text(-0.18, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


# ------------------------------------------------------------------ Fig. 6: I -> M
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
ax = axs[0]
lags = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
b = np.array([-0.007, -0.119, 0.065, 0.061, 0.038, 0.155, 0.165, 0.166, 0.163, 0.153, 0.161])
se = np.array([0.070, 0.065, 0.061, np.nan, 0.072, 0.084, 0.091, 0.094, 0.099, 0.099, 0.121])
ax.axhline(0, color=INK2, lw=0.6)
ax.axvline(-0.5, color="#b8b6b0", lw=0.6, ls=":")
ax.errorbar(lags, b, yerr=1.96 * np.nan_to_num(se), fmt="o-", ms=3.5, lw=1, color=BLUE, capsize=0)
ax.set_xlabel("weeks since the reblog by a big account")
ax.set_ylabel("log reward per later post\n(rel. to pre-shock weeks)")
ax.text(0.98, 0.03, "2643 authors, audience held fixed\nmean, weeks 0-6: $+0.14$; permutation $p<0.04$", transform=ax.transAxes,
        fontsize=6.8, va="bottom", ha="right", color=INK2)
ax.set_ylim(-0.45, 0.42)
label(ax, "(a)")
ax = axs[1]
bl = np.array([0.0024, 0.0068, 0.0080, 0.0075, 0.0053, 0.0059, 0.0089, 0.0099, 0.0038, 0.0104, 0.0083])
sl = np.array([0.0047, 0.0044, 0.0046, 0.0054, 0.0052, 0.0050, 0.0047, 0.0048, 0.0047, 0.0045, 0.0056])
ax.axhline(0, color=INK2, lw=0.6)
ax.axvline(-0.5, color="#b8b6b0", lw=0.6, ls=":")
ax.errorbar(lags, bl, yerr=1.96 * sl, fmt="o-", ms=3.5, lw=1, color=BLUE, capsize=0)
pre, post = bl[:4].mean(), bl[4:].mean()
ax.plot([-4, -1], [pre, pre], color=ORANGE, lw=1.2)
ax.plot([0, 6], [post, post], color=ORANGE, lw=1.2)
ax.set_xlabel("weeks since the reblogs (lead < 0 < lag)")
ax.set_ylabel("effect of $\\ln(1+$reach$)$\non log reward per post")
ax.text(0.02, 0.03, "7507 authors; lines: mean of leads and of lags", transform=ax.transAxes, fontsize=6.8,
        va="bottom", color=INK2)
ax.set_ylim(-0.012, 0.022)
label(ax, "(b)")
fig.tight_layout(w_pad=2.0)
save(fig, "fig6_im")

# ------------------------------------------------------------------ Fig. 7: M -> I and M -> M
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axs[0]
types = ["new readers\n(write posts)", "new voters\n(no posts)", "loyal\nreaders", "very active\nvoters", "replies"]
voted = [0.910, 1.509, 0.386, 0.603, 0.517]
voted_se = [0.068, 0.154, 0.075, 0.138, 0.054]
plac = [0.584, 0.290, 0.157, 0.195, 0.394]
plac_se = [0.219, 0.301, 0.100, 0.246, 0.159]
serv = [1.740, 1.126, 0.838, 0.519, 0.416]
serv_se = [0.099, 0.166, 0.134, 0.176, 0.076]
x = np.arange(len(types))
for vals, ses, col, dx, lab in ((voted, voted_se, BLUE, -0.2, "bid bot voted"), (plac, plac_se, ORANGE, 0.0,
                                "paid, bot did not vote"), (serv, serv_se, GREEN, 0.2, "paid vote-selling service")):
    ax.errorbar(x + dx, vals, yerr=1.96 * np.array(ses), fmt="o", ms=4, color=col, capsize=0, label=lab)
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels(types, fontsize=7)
ax.set_ylabel("log change after the event\n(mean of 12-h bins, 0-7 days)")
ax.legend(fontsize=6.5, loc="upper right")
label(ax, "(a)")
ax = axs[1]
contr = [0.326, 1.219, 0.229, 0.408, 0.123]
contr_se = [0.227, 0.321, 0.116, 0.250, 0.176]
ax.errorbar(contr, x, xerr=1.96 * np.array(contr_se), fmt="s", ms=4, color=INK, capsize=0, label="bot vote $-$ placebo")
dose = {0: (0.009, 0.023), 3: (-0.108, 0.033)}
for i, (d, s) in dose.items():
    ax.errorbar(d, i - 0.25, xerr=1.96 * s, fmt="^", ms=4, color=ORANGE, capsize=0,
                label="dose: per unit of ln rshares" if i == 0 else None)
ax.axvline(0, color=INK2, lw=0.6)
ax.set_yticks(x)
ax.set_yticklabels([t.replace("\n", " ") for t in types], fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("contribution of the bought vote (log)")
ax.legend(fontsize=6.5, loc="lower right")
label(ax, "(b)")
fig.tight_layout(w_pad=1.5)
save(fig, "fig7_mi")

# ------------------------------------------------------------------ Fig. 8: material layer
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
ax = axs[0]
months = np.arange(22, 29)
names = ["Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
bR = [1.2361, 1.1116, 0.7217, 0.7806, 0.8478, 0.6636, 0.8929]
sR = [0.0561, 0.0452, 0.0352, 0.0365, 0.0342, 0.0348, 0.0422]
bRI = [-0.3950, -0.4352, -0.4700, -0.4666, -0.5973, -0.7243, -0.5398]
sRI = [0.0523, 0.0405, 0.0329, 0.0334, 0.0304, 0.0307, 0.0376]
bI = [0.5769, 0.5439, 0.4864, 0.5145, 0.6203, 0.5964, 0.6063]
sI = [0.0090, 0.0071, 0.0065, 0.0062, 0.0070, 0.0068, 0.0071]
ax.errorbar(months - 0.15, bR, yerr=1.96 * np.array(sR), fmt="o", ms=4, color=BLUE, capsize=0,
            label="$\\ln(1+R)$ alone")
ax.errorbar(months, bRI, yerr=1.96 * np.array(sRI), fmt="o", ms=4, mfc="white", color=BLUE, capsize=0,
            label="$\\ln(1+R)$, inflow held fixed")
ax.errorbar(months + 0.15, bI, yerr=1.96 * np.array(sI), fmt="s", ms=3.5, color=ORANGE, capsize=0,
            label="$\\ln(1+I)$, with $R$")
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xticks(months)
ax.set_xticklabels(names)
ax.set_ylabel("elasticity of the stake")
ax.set_ylim(-1.0, 2.05)
ax.set_yticks([-0.5, 0, 0.5, 1.0, 1.5])
ax.legend(fontsize=6.3, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=0.8)
label(ax, "(a)")
ax = axs[1]
labs = ["$R_{m-1}\\to$ stake$_m$", "$R_{m-1}\\to$ stake$_m$\n| $I_{m-1}$", "$I_{m-1}\\to$ stake$_m$\n| $R_{m-1}$",
        "stake$_{m-1}\\to R_m$"]
vals = [0.0910, -0.1066, 0.0925, 0.0039]
ses = [0.0060, 0.0069, 0.0017, 0.0005]
cols = [BLUE, BLUE, ORANGE, INK2]
y = np.arange(len(labs))
for i in range(len(labs)):
    ax.errorbar(vals[i], y[i], xerr=1.96 * ses[i], fmt="o", ms=4, color=cols[i], capsize=0,
                mfc="white" if i == 1 else cols[i])
ax.axvline(0, color=INK2, lw=0.6)
ax.set_yticks(y)
ax.set_yticklabels(labs, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("coefficient (log); users, 137 445 pairs of consecutive months")
ax.set_xlim(-0.14, 0.14)
label(ax, "(b)")
fig.tight_layout(w_pad=2.0)
save(fig, "fig8_material")
print("saved figures/fig6_im, fig7_mi, fig8_material")
