"""Recipient-side decomposition (capacity theta) on the case-control sample of build_reblog_pairs.py.

Rows: for each reblog pair, all credited engagers (cases, y = 1) and up to 10 randomly sampled exposed
followers who did not engage (controls, y = 0, weight w = exposed / sampled). With pair fixed effects
the Poisson likelihood of the full exposed population is recovered by the offset ln w on controls
(rare outcome), and everything about the post and the reblogger (k_v, n_v, quality) is absorbed.

Feed load of follower x on the reblog day: L = (number of followees) x (items per followee per day)
x (deviation of that day from x's monthly mean). Pure capacity predicts the same exponent -theta for
all three components; indiscriminate following (selection) predicts a large coefficient on the number
of followees and small ones on items per followee and on the daily deviation.

Usage: python fit_recipient_cc.py pairs_month25.csv [--max-maxday 50]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpois import cpois  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--max-maxday", type=float, default=1e9)
    a = ap.parse_args()
    pairs = pd.read_csv(a.pairs, usecols=["pair", "author", "permlink", "v_maxday"], dtype={"permlink": str})
    cc = pd.read_csv(a.pairs.replace(".csv", "_cc.csv"))
    cc = cc.merge(pairs, on="pair")
    cc = cc[cc["v_maxday"] <= a.max_maxday]
    print(f"{len(cc)} rows: {int(cc['y'].sum())} cases, {int((cc['y'] == 0).sum())} controls, "
          f"{cc['pair'].nunique()} pairs")
    unit = cc["pair"].values
    clus = (cc["author"] + "/" + cc["permlink"]).values
    off = np.where(cc["y"] == 1, 0.0, np.log(cc["w"].values))
    y = cc["y"].values.astype(float)
    act = [(cc["act_bin"].values == b).astype(float) for b in range(1, cc["act_bin"].max() + 1)]

    def run(cols, label):
        X = np.column_stack([cc[c].values for c in cols] + act)
        r = cpois(y, X, unit, [], cluster=clus, offset=off)
        parts = ", ".join(f"{c}: {r['beta'][i]:+.3f} ± {r['se'][i]:.3f}" for i, c in enumerate(cols))
        print(f"{label}: {parts}  (n informative {r['n']})")
        return r

    print("coefficients are elasticities of the engagement rate per exposed follower; theta = -coefficient")
    run(["ln_load_day"], "total daily load       ")
    run(["ln_nfol", "ln_items_per_followee", "ln_dev_day"], "decomposition          ")
    # sanity: distribution of the components
    print(cc[["ln_nfol", "ln_items_per_followee", "ln_dev_day"]].describe().loc[["mean", "std", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
