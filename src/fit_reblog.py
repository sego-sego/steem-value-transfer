"""Model A fits on the version-2 reblog pairs and Model B (recipient load) on the cells file.

Pair level (as in the citation paper):
  E[Y_pv] = exp(alpha_p - gamma ln k_v + delta ln n_v + day FE)   for the cited post and both placebos
  within-pair difference real vs category placebo (pair fixed effects)
  shape: power vs club (quasi-likelihood ratio)
Cell level (recipient side):
  E[Y_pv,l,a] = N_pv,l,a * exp(alpha_p + beta_l + eta_a - gamma ln k_v)
  beta_l: engagement per exposed follower in load bin l (items per day in the feed); a: activity bin.
  theta = -slope of beta_l on ln(median load of the bin), and a continuous version with ln L_bin.

Usage: python fit_reblog.py pairs_month25.csv [--k k_items_day] [--max-k 200] [--max-maxday 1000]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpois import cpois  # noqa: E402

BPD = 28800
LOAD_MID = [5, 17, 55, 170, 550, 1700]   # representative items/day of the load bins [0,10,30,100,300,1000,inf]


def fit(d, y, k, label, unit_cols=("author", "permlink")):
    d = d[(d[k] > 0) & (d["n_v"] > 0)]
    unit = (d[unit_cols[0]].astype(str) + "/" + d[unit_cols[1]].astype(str)).values
    day = ((d["reblog_block"] - d["reblog_block"].min()) // BPD).astype(int).values
    X = np.column_stack([np.log(d[k].values), np.log(d["n_v"].values)])
    r = cpois(d[y].values.astype(float), X, unit, [day], cluster=unit)
    print(f"{label:42s} gamma = {-r['beta'][0]:.3f} ± {r['se'][0]:.3f}; delta = {r['beta'][1]:.3f} ± {r['se'][1]:.3f}; "
          f"n = {r['n']}, units {r['nunits']}; mean y {d[y].mean():.3f}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--k", default="k_items_day")
    ap.add_argument("--max-k", type=float, default=200)
    ap.add_argument("--max-maxday", type=float, default=1e9, help="drop rebloggers with more reblogs on one day")
    a = ap.parse_args()
    d = pd.read_csv(a.pairs, dtype={"permlink": str, "placebo_permlink": str, "cat_permlink": str,
                                     "cat_author": str, "category": str})
    print(f"{len(d)} pairs")
    d = d[(d["v_bot"] == 0) & (d["a_bot"] == 0) & (d[a.k] <= a.max_k) & (d["v_maxday"] <= a.max_maxday)]
    print(f"after bots, k <= {a.max_k}, maxday <= {a.max_maxday}: {len(d)} pairs\n")
    print("--- pair level")
    fit(d, "y_human", a.k, f"cited post ({a.k})")
    fit(d, "y_human", "k_day", "cited post (k_day, reblogs only)")
    dp = d[d["placebo_permlink"].notna()].copy()
    fit(dp, "yp_human", a.k, "same-author placebo", ("author", "placebo_permlink"))
    dc = d[d["cat_permlink"].notna()].copy()
    if (dc["yc_human"] > 0).sum() >= 200:
        fit(dc, "yc_human", a.k, "category placebo", ("cat_author", "cat_permlink"))
    else:
        print(f"category placebo: mean y {dc['yc_human'].mean():.4f}, share>0 {(dc['yc_human'] > 0).mean():.4f} "
              f"(too few non-zero counts for a slope)")
    q = dc.reset_index(drop=True)
    n = len(q)
    if n:
        yy = np.r_[q["y_human"], q["yc_human"]].astype(float)
        isr = np.r_[np.ones(n), np.zeros(n)]
        lk = np.r_[np.log(q[a.k]), np.log(q[a.k])]
        pair = np.r_[np.arange(n), np.arange(n)]
        cl = np.r_[(q["author"] + "/" + q["permlink"]).values] * 1
        cl = np.r_[cl, cl]
        r = cpois(yy, np.column_stack([isr, isr * lk]), pair, [], cluster=cl)
        print(f"{'within pair, cited vs category placebo':42s} gamma-gamma'' = {-r['beta'][1]:.3f} ± {r['se'][1]:.3f}; "
              f"ratio at median k = {np.exp(r['beta'][0] + r['beta'][1] * np.log(q[a.k].median())):.1f}; n = {r['n'] // 2}")
    # shape test
    dd = d[(d["n_v"] > 0)]
    u = (dd["author"] + "/" + dd["permlink"]).values
    day = ((dd["reblog_block"] - dd["reblog_block"].min()) // BPD).astype(int).values
    y, k, ln = dd["y_human"].values.astype(float), dd[a.k].values.astype(float), np.log(dd["n_v"].values)
    rp = cpois(y, np.column_stack([np.log(k), ln]), u, [day])
    best = None
    for al in np.geomspace(0.002, 2, 50):
        rc = cpois(y, ln[:, None], u, [day], offset=np.log(-np.expm1(-al * k) / k))
        if best is None or rc["loglik"] > best[1]:
            best = (al, rc["loglik"])
    print(f"{'shape':42s} power gamma {-rp['beta'][0]:.3f}; club k_c {1 / best[0]:.0f}; "
          f"quasi-LR club - power {2 * (best[1] - rp['loglik']) / rp['dispersion']:.1f}")
    # cells: recipient load
    cfile = a.pairs.replace(".csv", "_cells.csv")
    if not os.path.exists(cfile):
        print("\nno cells file: recipient-load analysis skipped")
        return
    c = pd.read_csv(cfile)
    c = c.merge(d[["pair", "author", "permlink", a.k, "reblog_block"]], on="pair")
    c = c[c["N"] > 0].copy()
    c["Y"] = np.minimum(c["Y"], c["N"])
    print(f"\n--- recipient side: {len(c)} cells, {c['N'].sum()} exposed followers, {c['Y'].sum()} engagements")
    print(c.groupby("load_bin").agg(N=("N", "sum"), Y=("Y", "sum")).assign(rate=lambda t: t.Y / t.N).to_string())
    u = (c["author"] + "/" + c["permlink"]).values
    lb, ab = c["load_bin"].values, c["act_bin"].values
    ref = int(np.bincount(lb, weights=c["N"].values).argmax())
    Xl = [(lb == b).astype(float) for b in range(lb.max() + 1) if b != ref]
    Xa = [(ab == b).astype(float) for b in range(1, ab.max() + 1)]
    X = np.column_stack(Xl + Xa + [np.log(c[a.k].values)])
    r = cpois(c["Y"].values.astype(float), X, u, [], cluster=u, offset=np.log(c["N"].values))
    print(f"engagement per exposed follower by feed load (ref bin {ref}), activity bins controlled:")
    j = 0
    for b in range(lb.max() + 1):
        if b == ref:
            print(f"  load bin {b} (~{LOAD_MID[b]}/day): 0 (ref.)")
        else:
            print(f"  load bin {b} (~{LOAD_MID[b]}/day): {r['beta'][j]:+.3f} ({r['se'][j]:.3f})")
            j += 1
    print(f"  source exponent at cell level: gamma = {-r['beta'][-1]:.3f} ± {r['se'][-1]:.3f}")
    lL = c["mean_lnL"].values if "mean_lnL" in c else np.log(np.array(LOAD_MID, dtype=float)[lb])
    X2 = np.column_stack(Xa + [np.log(c[a.k].values), lL])
    r2 = cpois(c["Y"].values.astype(float), X2, u, [], cluster=u, offset=np.log(c["N"].values))
    print(f"  continuous: theta = {-r2['beta'][-1]:.3f} ± {r2['se'][-1]:.3f} (recipient load), "
          f"gamma = {-r2['beta'][-2]:.3f} ± {r2['se'][-2]:.3f} (source fan-out)")


if __name__ == "__main__":
    main()
