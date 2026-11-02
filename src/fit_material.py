"""Material layer: Result 1 of the PRL in the Steem transfer network.

Result 1: conservative coupling with unequal row sums freezes into a pattern
    x_v / x* - 1 = c s_v,   c = eps / [1 - (1 - eps) f'*],   s_v = R_v - mean(R)
where R_v is the row sum of the column-stochastic transfer operator (share of other accounts'
outflows collected by v). With the material state measured by the stake proxy (value of a full
vote), per month and over accounts of one kind (default: users):

  1. cross-section: Spearman rho(stake, R); OLS slope c of stake/mean(stake) - 1 on s (the PRL form);
     the same in logs (robust to heavy tails); binned profile
  2. within accounts over months (account and month fixed effects): does a change of R go with a
     change of stake?
  3. timing: R_{m-1} -> stake_m given stake_{m-1}, and stake_{m-1} -> R_m given R_{m-1}

Usage:
  python src/fit_material.py --panel material_panel.csv [--kind user] [--fig fig_material]
  python src/fit_material.py --selftest
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    XtXi = np.linalg.pinv(X.T @ X)
    V = XtXi @ (X.T * e ** 2) @ X @ XtXi                       # HC0
    return b[1:], np.sqrt(np.diag(V))[1:]


def within(d, ycol, xcols):
    z = d[[ycol] + xcols].copy()
    for c in [ycol] + xcols:
        z[c] = z[c] - d.groupby("account")[c].transform("mean") - d.groupby("month")[c].transform("mean") \
               + z[c].mean()
    return ols(z[ycol].values, z[xcols].values)


def analyse(p, kind="user", fig=None, quiet=False):
    d = p[(p["kind"] == kind)].copy()
    d["stake"] = pd.to_numeric(d["stake"], errors="coerce")
    out = {}
    rows = []
    for m, g in p.groupby("month"):
        # s is defined on all accounts of the month (the operator), the fit uses accounts of the kind with a stake
        Rbar = g["R"].mean()
        gg = d[(d["month"] == m) & d["stake"].notna() & (d["stake"] > 0)].copy()
        if len(gg) < 50:
            continue
        gg["s"] = gg["R"] - Rbar
        y = gg["stake"] / gg["stake"].mean() - 1
        mod = (gg["R"] <= 3).values                          # the PRL linear form holds for moderate row sums
        c, se = ols(y.values[mod], gg["s"].values[mod][:, None])
        cl, sel = ols(np.log(gg["stake"].values), np.log1p(gg["R"].values)[:, None])
        ci, sei = ols(np.log(gg["stake"].values),
                      np.column_stack([np.log1p(gg["R"].values), np.log1p(gg["I"].astype(float).values)]))
        rho = gg[["stake", "R"]].rank().corr().iloc[0, 1]
        rows.append((m, len(gg), rho, c[0], se[0], cl[0], sel[0], ci[0], sei[0], ci[1], sei[1],
                     float((gg["R"] > 0).mean())))
    t = pd.DataFrame(rows, columns=["month", "n", "spearman", "c_level(R<=3)", "se", "b_log", "se_log",
                                    "b_log|I", "se", "b_lnI", "se", "share_R>0"])
    t["c_level"] = t["c_level(R<=3)"]
    out["cross"] = t
    if not quiet:
        print(f"--- kind = {kind}: cross-section per month")
        print(t.round(4).to_string(index=False))
    d = d[d["stake"].notna() & (d["stake"] > 0)].copy()
    d["lns"] = np.log(d["stake"])
    d["lnR"] = np.log1p(d["R"])
    d["Rbar"] = d["month"].map(p.groupby("month")["R"].mean())
    d["s"] = d["R"] - d["Rbar"]
    multi = d.groupby("account")["month"].transform("count") > 1
    d["lnI"] = np.log1p(d["I"].astype(float))
    b, se = within(d[multi], "lns", ["lnR"])
    bI, seI = within(d[multi], "lns", ["lnR", "lnI"])
    out["within"] = (b[0], se[0], int(multi.sum()), d[multi]["account"].nunique())
    out["within_I"] = (bI[0], seI[0], bI[1], seI[1])
    if not quiet:
        print(f"--- within accounts (account + month effects): d ln stake / d ln(1+R) = {b[0]:+.4f} ± {se[0]:.4f} "
              f"({multi.sum()} account-months, {d[multi]['account'].nunique()} accounts)")
        print(f"    with the inflow amount held fixed: ln(1+R) {bI[0]:+.4f} ± {seI[0]:.4f}, ln(1+I) {bI[1]:+.4f} ± {seI[1]:.4f}")
    d = d.sort_values(["account", "month"])
    g = d.groupby("account")
    d["lns_prev"], d["lnR_prev"], d["m_prev"] = g["lns"].shift(), g["lnR"].shift(), g["month"].shift()
    d["lnI_prev"] = g["lnI"].shift()
    q = d[d["m_prev"] == d["month"] - 1]
    if len(q) > 50:
        b1, s1 = ols(q["lns"].values, q[["lns_prev", "lnR_prev"]].values)
        b2, s2 = ols(q["lnR"].values, q[["lnR_prev", "lns_prev"]].values)
        b3, s3 = ols(q["lns"].values, q[["lns_prev", "lnR_prev", "lnI_prev"]].values)
        out["timing"] = (b1[1], s1[1], b2[1], s2[1], b3[1], s3[1], b3[2], s3[2])
        if not quiet:
            print(f"--- timing ({len(q)} consecutive account-months): R_(m-1) -> stake_m given stake_(m-1): "
                  f"{b1[1]:+.4f} ± {s1[1]:.4f};  stake_(m-1) -> R_m given R_(m-1): {b2[1]:+.4f} ± {s2[1]:.4f}")
            print(f"    with the inflow amount I_(m-1) held fixed: R_(m-1) -> stake_m {b3[1]:+.4f} ± {s3[1]:.4f}, "
                  f"I_(m-1) -> stake_m {b3[2]:+.4f} ± {s3[2]:.4f}")
    if fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f_, ax = plt.subplots(figsize=(3.4, 2.6))
        e = d[d["R"] > 0].copy()
        e["bin"] = pd.qcut(e["R"], 12, duplicates="drop")
        e["rel"] = e["stake"] / e.groupby("month")["stake"].transform("mean")
        bb = e.groupby("bin", observed=True).agg(R=("R", "median"), rel=("rel", "mean"), se=("rel", "sem"))
        ax.errorbar(bb["R"], bb["rel"], yerr=1.96 * bb["se"], fmt="o", ms=4, color="#2a78d6", capsize=0)
        z = d[d["R"] == 0]
        if len(z):
            ax.axhline((z["stake"] / z.groupby("month")["stake"].transform("mean")).mean(), color="#eb6834", lw=0.8,
                       ls="--", label="R = 0 (no inflow)")
            ax.legend(fontsize=6.5, frameon=False)
        ax.set_xscale("log")
        ax.set_xlabel("row sum R (share of others' outflows collected)")
        ax.set_ylabel("stake / monthly mean")
        for ext in ("pdf", "png"):
            f_.savefig(f"{fig}.{ext}", bbox_inches="tight", dpi=300)
        print(f"saved {fig}.pdf/.png")
    return out


def selftest():
    import json
    import subprocess
    import tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    synth = os.path.join(here, "..", "validation", "synth_material.py")
    for case in ("prl", "null"):
        tmp = tempfile.mkdtemp(prefix=f"mat_{case}_")
        subprocess.run([sys.executable, synth, "--out", tmp, "--case", case], check=True, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, os.path.join(here, "build_material.py"), "--data-root", tmp, "--rewards",
                        os.path.join(tmp, "votes.jsonl"), "--bots", os.path.join(tmp, "none.csv"), "--out",
                        os.path.join(tmp, "material_panel.csv")], check=True, stdout=subprocess.DEVNULL)
        truth = json.load(open(os.path.join(tmp, "truth.json")))
        r = analyse(pd.read_csv(os.path.join(tmp, "material_panel.csv")), quiet=True)
        t = r["cross"]
        print(f"=== case {case}: slope in the simulated states {truth['c_sim']:.3f} (PRL Eq. floor {truth['c']:.3f}, "
              f"full linear response {truth['c_linear_full']:.3f}); estimated {t['c_level'].mean():.3f} "
              f"(months {', '.join(f'{x:.3f}' for x in t['c_level'])}); spearman {t['spearman'].mean():.3f}; "
              f"within {r['within'][0]:+.3f} ± {r['within'][1]:.3f}; timing R->stake {r['timing'][0]:+.3f}, "
              f"stake->R {r['timing'][2]:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="material_panel.csv")
    ap.add_argument("--kind", default="user")
    ap.add_argument("--fig", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    p = pd.read_csv(a.panel)
    print("accounts by kind (account-months with a stake / all):")
    print(p.assign(has=pd.to_numeric(p["stake"], errors="coerce").notna()).groupby("kind")["has"].agg(["sum", "count"]))
    analyse(p, a.kind, a.fig or None)


if __name__ == "__main__":
    main()
