"""Within-reader panel with the number of active sources split into the reader's follow decisions
(S, followed accounts in the universe) and the sources' activity (A = active share), and the
asymmetry by direction of change of each part.

  E[E_xm] = X_xm exp(alpha_x + tau_m + bS ln S_xm + bA ln A_xm + b2 ln q_xm + c ln a_xm)

Readings of the asymmetry found with ln n_fol (gaining sources -1.03, losing -0.51):
  silence     -> bS ~ -1 in both directions; the weak response comes from A (sources falling silent)
  hysteresis  -> the weak response appears for S falling (unfollowing) as well

Usage (from the data folder):
  python fit_asymmetry.py --decomp follow_decomposition.csv --months 24 25 26 \
      --followers data24/pairs_v2_month24_followers.csv pairs_v2_month25_followers.csv data26/pairs_v2_month26_followers.csv
  python fit_asymmetry.py --selftest
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpois import cpois  # noqa: E402


def fit(d, cols, label):
    y = d["engagements"].values.astype(float)
    off = np.log(d["exposures"].values)
    month = d["month"].astype("category").cat.codes.values
    r = cpois(y, d[cols].values, d["account"].values, [month], cluster=d["account"].values, offset=off)
    txt = ", ".join(f"{c} {r['beta'][i]:+.3f} ± {r['se'][i]:.3f}" for i, c in enumerate(cols))
    print(f"  {label:44s} {txt}  (readers {r['nunits']})")
    return r


def analyse(d):
    d = d[(d["exposures"] > 0) & (d["S"] > 0) & (d["n_active"] > 0) & (d["load_per_day"] > 0)].copy()
    d["ln_S"] = np.log(d["S"])
    d["ln_A"] = np.log(d["n_active"] / d["S"])
    d["ln_n"] = np.log(d["n_active"])
    d["ln_ipf"] = np.log(d["load_per_day"] / d["n_followees"].clip(lower=1))
    d["ln_act"] = np.log1p(d["activity"])
    g = d.sort_values("month").groupby("account")
    dS = (g["ln_S"].last() - g["ln_S"].first())
    dA = (g["ln_A"].last() - g["ln_A"].first())
    d["dS"], d["dA"] = d["account"].map(dS), d["account"].map(dA)
    print(f"{d['account'].nunique()} readers; within-reader s.d. of ln S {d.groupby('account')['ln_S'].std().mean():.3f}, "
          f"of ln A {d.groupby('account')['ln_A'].std().mean():.3f}")
    print("all readers")
    fit(d, ["ln_n", "ln_ipf", "ln_act"], "active sources n (as before)")
    fit(d, ["ln_S", "ln_A", "ln_ipf", "ln_act"], "decisions S and activity share A")
    print("by direction of change (decomposition, with ln activity)")
    for lab, sel in (("S grew (followed more)", d["dS"] > 0), ("S fell (unfollowed)", d["dS"] < 0),
                     ("A grew (sources woke up)", d["dA"] > 0), ("A fell (sources fell silent)", d["dA"] < 0)):
        fit(d[sel], ["ln_S", "ln_A", "ln_ipf", "ln_act"], lab)
    print("readers whose S did not change (only activity of sources moved n)")
    fit(d[d["dS"] == 0], ["ln_A", "ln_ipf", "ln_act"], "S constant")


def selftest():
    R = np.random.default_rng(7)
    for bS, bA, lab in ((-1.0, -1.0, "capacity by active sources (bS = bA = -1)"),
                        (-1.0, -0.5, "silence responds by half (bS = -1, bA = -0.5)")):
        rows = []
        for i in range(20000):
            S0 = R.normal(5.5, 1.0)
            A0 = R.normal(-0.5, 0.3)
            alpha = R.normal(-2.5, 0.5) - 0.8 * S0
            for m in range(3):
                lS = S0 + R.normal(0, 0.15)
                lA = min(0.0, A0 + R.normal(0, 0.15))
                X = R.integers(30, 400)
                rate = np.exp(alpha + bS * (lS - S0) + bA * (lA - A0) + 0.8 * S0)
                S = int(np.exp(lS))
                n = max(1, int(S * np.exp(lA)))
                rows.append((f"x{i}", m, X, R.poisson(X * min(rate, 1)), S, n, n, n * 1.5, 50))
        d = pd.DataFrame(rows, columns=["account", "month", "exposures", "engagements", "S", "n_active",
                                        "n_followees", "load_per_day", "activity"])
        print(f"=== selftest: {lab}")
        analyse(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp")
    ap.add_argument("--months", type=int, nargs="+")
    ap.add_argument("--followers", nargs="+")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    dec = pd.read_csv(a.decomp)
    parts = []
    for m, f in zip(a.months, a.followers):
        x = pd.read_csv(f)
        x["month"] = m
        parts.append(x)
    d = pd.concat(parts).merge(dec, on=["account", "month"], how="inner")
    print(f"{len(d)} reader-months after merge")
    analyse(d)


if __name__ == "__main__":
    main()
