"""Figures 3 and 4 from the pair files (run on the machine with the data).

Fig. 3  source side: transfer per reblog versus the reblogger's daily output k (bins, post fixed
        effects), for the reblogged post and for the category placebo; March and April.
Fig. 4  recipient side: engagement per exposed reader versus (a) number of followed sources,
        (b) output per source, (c) deviation of the day's feed from the reader's mean (bins, pair
        fixed effects, case-control offsets, all three components and activity jointly);
        (d) within-reader panel coefficients (numbers from fit_follower_panel.py).

Usage (from the data folder):
  python $REPO/figures/make_figs_data.py --mar data25/pairs_month25.csv --apr data26/pairs_month26.csv
Uses src/cpois.py. Writes figures/fig3_source.* and figures/fig4_recipient.*
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cpois import cpois  # noqa: E402

plt.rcParams.update({
    "font.family": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"], "mathtext.fontset": "cm",
    "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "legend.frameon": False,
})
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
BPD = 28800
K_EDGES = [1, 2, 3, 5, 9, 17, 33, 51]
# panel numbers (fit_follower_panel.py, February-April 2018, with ln activity)
PANEL = [("all readers", -0.995, 0.017), ("small changes", -0.820, 0.088),
         ("sources grew", -1.025, 0.020), ("sources fell", -0.505, 0.086)]
PANEL_POOLED = (-0.574, 0.021)


def save(fig, name):
    os.makedirs("figures", exist_ok=True)
    for ext in ("pdf", "eps", "png"):
        fig.savefig(f"figures/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def bin_effects(y, b, unit, extra, small, ref, nb, offset=None, cluster=None):
    cols = [(b == q).astype(float) for q in range(nb) if q != ref]
    X = np.column_stack(cols + extra)
    r = cpois(y, X, unit, small, cluster=cluster if cluster is not None else unit, offset=offset)
    eff, se, j = np.zeros(nb), np.zeros(nb), 0
    for q in range(nb):
        if q != ref:
            eff[q], se[q] = r["beta"][j], r["se"][j]
            j += 1
    return eff, se


def source_bins(path, maxday=50, min_pairs=200, min_pos=20):
    d = pd.read_csv(path, dtype={"permlink": str, "cat_permlink": str, "cat_author": str})
    d = d[(d["v_bot"] == 0) & (d["a_bot"] == 0) & (d["v_maxday"] <= maxday) & (d["n_v"] > 0)
          & (d["k_items_day"] >= 1) & (d["k_items_day"] < K_EDGES[-1])].copy()
    d["bin"] = np.searchsorted(K_EDGES, d["k_items_day"].values, side="right") - 1
    nb = len(K_EDGES) - 1
    out = {}
    for lab, y, ua, up in (("reblogged post", "y_human", "author", "permlink"),
                           ("category placebo", "yc_human", "cat_author", "cat_permlink")):
        dd = d[d[up].notna()] if lab != "reblogged post" else d
        # keep only bins with enough pairs and non-zero counts (sparse bins give unstable estimates)
        good = [q for q in range(nb) if (dd["bin"] == q).sum() >= min_pairs and dd.loc[dd["bin"] == q, y].gt(0).sum() >= min_pos]
        dd = dd[dd["bin"].isin(good)]
        unit = (dd[ua].astype(str) + "/" + dd[up].astype(str)).values
        day = ((dd["reblog_block"] - dd["reblog_block"].min()) // BPD).astype(int).values
        cols = [(dd["bin"].values == q).astype(float) for q in good[1:]]
        r = cpois(dd[y].values.astype(float), np.column_stack(cols + [np.log(dd["n_v"].values)]), unit, [day],
                  cluster=unit)
        eff, se = np.full(nb, np.nan), np.full(nb, np.nan)
        eff[good[0]], se[good[0]] = 0.0, 0.0
        for j, q in enumerate(good[1:]):
            eff[q], se[q] = r["beta"][j], r["se"][j]
        kmed = np.array([dd.loc[dd["bin"] == q, "k_items_day"].median() if q in good else np.nan for q in range(nb)])
        out[lab] = (kmed, eff, se)
    # within pair: ratio reblogged / placebo engagement per bin of k (pair fixed effects)
    q = d[d["cat_permlink"].notna()].reset_index(drop=True)
    good = [b for b in range(nb) if (q["bin"] == b).sum() >= min_pairs and q.loc[q["bin"] == b, "yc_human"].gt(0).sum() >= min_pos]
    q = q[q["bin"].isin(good)].reset_index(drop=True)
    n = len(q)
    yy = np.r_[q["y_human"].values, q["yc_human"].values].astype(float)
    isr = np.r_[np.ones(n), np.zeros(n)]
    bb = np.r_[q["bin"].values, q["bin"].values]
    X = np.column_stack([isr * (bb == b) for b in good])
    pair = np.r_[np.arange(n), np.arange(n)]
    cl = np.r_[(q["author"] + "/" + q["permlink"]).values, (q["author"] + "/" + q["permlink"]).values]
    r = cpois(yy, X, pair, [], cluster=cl)
    ratio, rse = np.full(nb, np.nan), np.full(nb, np.nan)
    for j, b in enumerate(good):
        ratio[b], rse[b] = r["beta"][j], r["se"][j]
    kmed = np.array([q.loc[q["bin"] == b, "k_items_day"].median() if b in good else np.nan for b in range(nb)])
    out["ratio"] = (kmed, ratio, rse)
    return out


def recipient_bins(path, maxday=50):
    pairs = pd.read_csv(path, usecols=["pair", "author", "permlink", "v_maxday"], dtype={"permlink": str})
    cc = pd.read_csv(path.replace(".csv", "_cc.csv")).merge(pairs, on="pair")
    cc = cc[cc["v_maxday"] <= maxday]
    off = np.where(cc["y"] == 1, 0.0, np.log(cc["w"].values))
    edges = {"ln_nfol": np.log([1, 30, 100, 300, 1000, 3000, 1e9]),
             "ln_items_per_followee": np.log([1e-9, 0.5, 1, 2, 4, 1e9]),
             "ln_dev_day": np.log([1e-9, 0.5, 0.8, 1.25, 2, 1e9])}
    bins, nbs, refs = {}, {}, {}
    for c, e in edges.items():
        bins[c] = np.clip(np.searchsorted(e, cc[c].values, side="right") - 1, 0, len(e) - 2)
        nbs[c] = len(e) - 1
        refs[c] = int(np.bincount(bins[c]).argmax())
    cols, index = [], []
    yv = cc["y"].values
    ok = {}
    for c in edges:
        for q in range(nbs[c]):
            sel = bins[c] == q
            ok[(c, q)] = sel.sum() >= 200 and yv[sel].sum() >= 20       # skip sparse bins
            if q != refs[c] and ok[(c, q)]:
                cols.append(sel.astype(float))
                index.append((c, q))
    act = [(cc["act_bin"].values == b).astype(float) for b in range(1, cc["act_bin"].max() + 1)]
    r = cpois(cc["y"].values.astype(float), np.column_stack(cols + act), cc["pair"].values, [],
              cluster=(cc["author"] + "/" + cc["permlink"]).values, offset=off)
    out = {}
    for c in edges:
        x = np.array([np.exp(np.median(cc[c].values[bins[c] == q])) if ok[(c, q)] else np.nan
                      for q in range(nbs[c])])
        eff, se = np.full(nbs[c], np.nan), np.full(nbs[c], np.nan)
        eff[refs[c]], se[refs[c]] = 0.0, 0.0
        for j, (cc_, q) in enumerate(index):
            if cc_ == c:
                eff[q], se[q] = r["beta"][j], r["se"][j]
        # normalise to the first bin for plotting
        first = np.nanmin(np.where(np.isfinite(eff), np.arange(nbs[c]), np.nan))
        out[c] = (x, eff - eff[int(first)], se)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mar", required=True)
    ap.add_argument("--apr", default=None)
    a = ap.parse_args()
    months = [("March", a.mar, "o")] + ([("April", a.apr, "s")] if a.apr else [])
    # ---------------- Fig. 3
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for mname, path, mk in months:
        res = source_bins(path)
        for lab, col in (("reblogged post", BLUE), ("category placebo", ORANGE)):
            k, e, s = res[lab]
            ok = np.isfinite(e)
            ax.errorbar(k[ok] * (1.04 if mname == "April" else 1), np.exp(e[ok]),
                        yerr=[np.exp(e[ok]) - np.exp(e[ok] - 1.96 * s[ok]), np.exp(e[ok] + 1.96 * s[ok]) - np.exp(e[ok])],
                        fmt=mk, ms=3.8, color=col, mfc=col if mname == "March" else "white", elinewidth=0.8,
                        capsize=0, label=f"{lab}, {mname}")
        k, e, s = res["ratio"]
        ok = np.isfinite(e)
        ax2.errorbar(k[ok] * (1.04 if mname == "April" else 1), np.exp(e[ok]),
                     yerr=[np.exp(e[ok]) - np.exp(e[ok] - 1.96 * s[ok]), np.exp(e[ok] + 1.96 * s[ok]) - np.exp(e[ok])],
                     fmt=mk, ms=3.8, color=INK, mfc=INK if mname == "March" else "white", elinewidth=0.8, capsize=0,
                     label=mname)
    kk = np.geomspace(1, 50, 50)
    ax.plot(kk, kk ** -0.5, ls=":", color=INK2, lw=1, label="$k^{-1/2}$ (citations)")
    for a_ in (ax, ax2):
        a_.set_xscale("log")
        a_.set_yscale("log")
        a_.set_xticks([1, 2, 5, 10, 20, 40])
        for axis in (a_.xaxis, a_.yaxis):
            axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a_.grid(True, color=GRID, lw=0.5)
        a_.set_xlabel("items shared by the reblogger that day, $k$")
    ax.set_ylim(0.1, 2.0)
    ax.set_ylabel("engagement per reblog (relative to $k=1$)")
    ax.set_title("(a) slope", fontsize=8, loc="left")
    ax.legend(fontsize=6, loc="lower left")
    ax2.set_ylim(1, 50)
    ax2.set_ylabel("reblogged / placebo, same pair")
    ax2.set_title("(b) level", fontsize=8, loc="left")
    ax2.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig3_source")
    # ---------------- Fig. 4
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.9]))
    titles = {"ln_nfol": "(a) followed sources $n^{\\rm fol}$",
              "ln_items_per_followee": "(b) output per source $\\bar q$",
              "ln_dev_day": "(c) day load / mean"}
    for mname, path, mk in months:
        res = recipient_bins(path)
        for ax, c in zip(axs[:3], titles):
            x, e, s = res[c]
            ax.errorbar(x, np.exp(e), yerr=[np.exp(e) - np.exp(e - 1.96 * s), np.exp(e + 1.96 * s) - np.exp(e)],
                        fmt=mk, ms=3.5, color=BLUE, mfc=BLUE if mname == "March" else "white", elinewidth=0.8,
                        capsize=0, label=mname)
    for ax, c in zip(axs[:3], titles):
        x0 = ax.get_xlim()
        xx = np.geomspace(max(x0[0], 1e-3), x0[1], 50)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(titles[c], fontsize=8, loc="left")
        ax.grid(True, color=GRID, lw=0.5)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    lo, hi = axs[0].get_xlim()
    yl = axs[0].get_ylim()
    xx = np.geomspace(lo, hi, 50)
    axs[0].plot(xx, (xx / lo) ** -1.0, ls=":", color=INK2, lw=1, label="$\\propto 1/n^{\\rm fol}$")
    axs[0].set_xlim(lo, hi)
    axs[0].set_ylim(yl)
    axs[0].set_ylabel("engagement per exposed reader\n(relative to first bin)")
    axs[0].legend(fontsize=6.5, loc="lower left")
    ax = axs[3]
    ys = np.arange(len(PANEL))[::-1]
    for (lab, b, s), yv in zip(PANEL, ys):
        ax.errorbar(b, yv, xerr=1.96 * s, fmt="o", color=BLUE, ms=4, elinewidth=0.9, capsize=0)
    ax.errorbar(PANEL_POOLED[0], -1, xerr=1.96 * PANEL_POOLED[1], fmt="D", color=INK2, ms=3.5, elinewidth=0.9)
    ax.set_yticks(list(ys) + [-1])
    ax.set_yticklabels([p[0] for p in PANEL] + ["pooled"], fontsize=6.8)
    ax.axvline(-1, color=INK2, lw=0.8, ls=":")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-1.3, 0.1)
    ax.set_xlabel("within-reader $b_1$", fontsize=7.5)
    ax.set_title("(d) panel", fontsize=8, loc="left")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig4_recipient")
    print("saved figures/fig3_source.* and figures/fig4_recipient.*")


if __name__ == "__main__":
    main()
