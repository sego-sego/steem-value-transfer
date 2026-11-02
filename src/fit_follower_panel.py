"""Within-follower panel: capacity versus selection on the number of followed sources.

Input: the *_followers.csv files written by build_reblog_pairs.py for several months.
For follower x in month m: exposures X (reblogged target posts that reached x through a followee),
engagements E (credited human engagements), number of followed active sources n_fol, items per
followee, activity (votes + comments).

  E[E_xm] = X_xm * exp(alpha_x + tau_m + b1 ln n_fol_xm + b2 ln items_per_followee_xm [+ c ln activity])

alpha_x removes everything stable about the follower, including how selectively she follows.
Capacity (theta by sources) predicts b1 ~ -1 within followers as well; pure selection between
followers predicts b1 ~ 0 within followers while the pooled (between) estimate stays near -1.

Usage:
  python fit_follower_panel.py data24/pairs_month24_followers.csv pairs_month25_followers.csv data26/pairs_month26_followers.csv
  python fit_follower_panel.py --selftest
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpois import cpois  # noqa: E402


def estimate(d, label, with_activity):
    d = d[(d["exposures"] > 0) & (d["n_followees"] > 0) & (d["load_per_day"] > 0)].copy()
    d["ln_nfol"] = np.log(d["n_followees"])
    d["ln_ipf"] = np.log(d["load_per_day"] / d["n_followees"])
    d["ln_act"] = np.log1p(d["activity"])
    cols = ["ln_nfol", "ln_ipf"] + (["ln_act"] if with_activity else [])
    X = d[cols].values
    y = d["engagements"].values.astype(float)
    off = np.log(d["exposures"].values)
    month = d["month"].astype("category").cat.codes.values
    # pooled (between + within): a single unit, month effects as dummies
    Xm = np.column_stack([X] + [(month == k).astype(float) for k in range(1, month.max() + 1)])
    rp = cpois(y, Xm, np.zeros(len(d), dtype=int), [], cluster=d["account"].values, offset=off)
    # within follower: follower fixed effects (conditional likelihood), month effects
    rw = cpois(y, X, d["account"].values, [month], cluster=d["account"].values, offset=off)
    fmt = lambda r: ", ".join(f"{c} {r['beta'][i]:+.3f} ± {r['se'][i]:.3f}" for i, c in enumerate(cols))
    print(f"{label}")
    print(f"  pooled         : {fmt(rp)}")
    print(f"  within follower: {fmt(rw)}  (followers with >= 2 months and engagements: {rw['nunits']})")
    return rp, rw


def selftest():
    """Follower-month data with selection between followers and a known within-follower theta."""
    R = np.random.default_rng(5)
    rows = []
    for theta_w in (0.0, 1.0):
        rows = []
        for i in range(20000):
            base = R.normal(4.5, 1.2)                       # typical log number of followees
            interest = -1.0 * base + R.normal(0, 0.5)       # selection: indiscriminate followers care less
            for m in range(3):
                lnf = base + R.normal(0, 0.3)               # within-person changes
                ipf = R.normal(0.4, 0.3)
                X = R.integers(20, 400)
                rate = np.exp(interest + 1.5 - theta_w * (lnf - base) - 0.3 * ipf)
                rows.append((f"x{i}", m, X, R.poisson(X * rate), np.exp(lnf), np.exp(lnf + ipf), 50))
        d = pd.DataFrame(rows, columns=["account", "month", "exposures", "engagements", "n_followees",
                                        "load_per_day", "activity"])
        print(f"truth: between-follower slope on ln n_fol = -1 (selection), within-follower = {-theta_w}")
        estimate(d, "selftest", False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    parts = []
    for i, f in enumerate(a.files):
        d = pd.read_csv(f)
        d["month"] = i
        parts.append(d)
        print(f"{f}: {len(d)} followers")
    d = pd.concat(parts, ignore_index=True)
    n_months = d.groupby("account")["month"].nunique()
    print(f"followers present in >= 2 months: {(n_months >= 2).sum()}")
    estimate(d, "without activity control", False)
    estimate(d, "with ln activity", True)
    # robustness: size and direction of the within-follower change in the number of followed sources
    d = d[(d["exposures"] > 0) & (d["n_followees"] > 0) & (d["load_per_day"] > 0)].copy()
    d["ln_nfol"] = np.log(d["n_followees"])
    g = d.sort_values("month").groupby("account")["ln_nfol"]
    rng = g.max() - g.min()
    chg = g.last() - g.first()
    d["range"] = d["account"].map(rng)
    d["chg"] = d["account"].map(chg)
    print("\n--- robustness (with ln activity)")
    estimate(d[d["range"] < 0.3], "small changes only (range of ln n_fol < 0.3, i.e. < 35%)", True)
    estimate(d[d["chg"] > 0], "followers whose number of sources grew", True)
    estimate(d[d["chg"] < 0], "followers whose number of sources fell", True)


if __name__ == "__main__":
    main()
