"""Model 0 (positive control): value of a vote versus the voter's recent voting load.

Protocol rule (to be checked against the documentation for March 2018): a vote uses a share of
the voter's current voting power proportional to its percent, and power regenerates at a fixed
rate per day. In steady state the value of one full vote is flat up to about 10 full votes per
day and falls as 1/k above: a club transfer h(k) = min(1, k_c/k), k_c ~ 10, set by the protocol.

y   = rshares per full vote = rshares * 10000 / percent (value passed by one unit of voting effort)
k   = 1 + full-vote equivalents per day (sum of percent/10000) cast by the voter over the preceding
      --window-days days (default 5, the regeneration memory), from the complete SteemOps vote file
E[y] = exp(alpha_voter + day FE) * h(k);  h = k^-gamma (power) or (1-exp(-a k))/k (club)

Usage:
  python fit_votes_model0.py --rshares votes_rshares_month25.csv --month-dir data25 --month 25
"""
from __future__ import annotations

import argparse
import bisect
import collections
import io
import os
import sys
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpois import cpois  # noqa: E402

BPD = 28800


def zlines(path):
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            it = io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline="")
            next(it)
            for line in it:
                yield line.rstrip("\r\n")


def to_unix(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rshares", required=True)
    ap.add_argument("--month-dir", required=True)
    ap.add_argument("--month", default="25")
    ap.add_argument("--max-k", type=float, default=200)
    ap.add_argument("--window-days", type=float, default=5.0,
                    help="load window: full-vote equivalents per day averaged over this many preceding days "
                         "(power regenerates over ~5 days, so 5 matches the protocol memory)")
    a = ap.parse_args()
    r = pd.read_csv(a.rshares, dtype={"permlink": str, "voter": str})
    r = r[(r["percent"] > 0) & (r["rshares"].astype(float) > 0)].copy()
    r["t"] = r["time"].map(to_unix)
    voters = set(r["voter"])
    print(f"{len(r)} positive votes with rshares, {len(voters)} voters")
    # complete vote histories of these voters from SteemOps (block, percent), and block->time map
    hist = collections.defaultdict(list)
    keyblock = {}
    want = set(zip(r["voter"], r["author"], r["permlink"]))
    for i, line in enumerate(zlines(os.path.join(a.month_dir, f"vote-month-{a.month}.zip"))):
        f = line.split(",")
        if len(f) != 5 or f[1] not in voters:
            continue
        b, w = int(f[0]), int(f[4])
        if w > 0:
            hist[f[1]].append((b, w))
        k = (f[1], f[2], f[3])
        if k in want:
            keyblock[k] = b
        if i % 5_000_000 == 0 and i:
            print(f"  votes scanned: {i / 1e6:.0f} M", flush=True)
    r["block"] = [keyblock.get(k) for k in zip(r["voter"], r["author"], r["permlink"])]
    m = r.dropna(subset=["block"])
    A = np.vstack([np.ones(len(m)), m["block"].values.astype(float)]).T
    c0, c1 = np.linalg.lstsq(A, m["t"].values, rcond=None)[0]
    print(f"block->time: {c1:.4f} s per block (3 s expected); matched {len(m)}/{len(r)} votes")
    r = m.copy()
    r["block"] = r["block"].astype(int)
    for v in hist:
        hist[v].sort()
    cum = {v: (np.array([b for b, _ in l]), np.cumsum([w for _, w in l]) / 10000.0) for v, l in hist.items()}
    k = []
    for v, b in zip(r["voter"], r["block"]):
        bs, cs = cum[v]
        hi = bisect.bisect_left(bs, b)            # votes strictly before this one
        lo = bisect.bisect_left(bs, b - int(a.window_days * BPD))
        k.append(1.0 + ((cs[hi - 1] if hi > 0 else 0.0) - (cs[lo - 1] if lo > 0 else 0.0)) / a.window_days)
    r["k"] = k
    # require a full load window inside the month
    r = r[(r["block"] >= r["block"].min() + int(a.window_days * BPD)) & (r["k"] <= a.max_k)]
    r["y"] = r["rshares"].astype(float) * 10000.0 / r["percent"] / 1e9
    r["day"] = ((r["block"] - r["block"].min()) // BPD).astype(int)
    y, unit, day = r["y"].values, r["voter"].values, r["day"].values
    lk = np.log(r["k"].values)
    rp = cpois(y, lk[:, None], unit, [day], cluster=unit)
    print(f"power: gamma = {-rp['beta'][0]:.3f} ± {rp['se'][0]:.3f}  (n = {rp['n']}, voters {rp['nunits']})")
    best = None
    for al in np.geomspace(0.005, 2.0, 70):
        off = np.log(-np.expm1(-al * r["k"].values) / r["k"].values)
        rc = cpois(y, np.zeros((len(y), 0)), unit, [day], offset=off)
        if best is None or rc["loglik"] > best[1]:
            best = (al, rc["loglik"])
    # refine alpha on a fine grid around the coarse optimum
    for al in np.geomspace(best[0] / 1.1, best[0] * 1.1, 21):
        off = np.log(-np.expm1(-al * r["k"].values) / r["k"].values)
        rc = cpois(y, np.zeros((len(y), 0)), unit, [day], offset=off)
        if rc["loglik"] > best[1]:
            best = (al, rc["loglik"])
    qlr = 2 * (best[1] - rp["loglik"]) / rp["dispersion"]
    # voter-clustered s.e. of alpha: one-step linearization, regressor d ln h / d alpha at the optimum
    al, kk = best[0], r["k"].values
    off = np.log(-np.expm1(-al * kk) / kk)
    z = kk / np.expm1(al * kk)
    rs = cpois(y, z[:, None], unit, [day], cluster=unit, offset=off)
    se_a = rs["se"][0]
    lo_a, hi_a = max(al - 1.96 * se_a, 1e-6), al + 1.96 * se_a
    print(f"club: alpha = {al:.4f} ± {se_a:.4f} (k_c = {1 / al:.1f}, 95% CI {1 / hi_a:.1f}-{1 / lo_a:.1f}); "
          f"quasi-LR club - power = {qlr:.1f} (positive favours club)")
    edges = [1, 2, 4, 6, 9, 13, 20, 30, 50, 80, 10 ** 9]
    r["bin"] = np.searchsorted(edges, r["k"].values, side="right") - 1
    nb = r["bin"].max() + 1
    ref = 1
    X = np.column_stack([(r["bin"].values == q).astype(float) for q in range(nb) if q != ref])
    rb = cpois(y, X, unit, [day], cluster=unit)
    rows, j = [], 0
    for q in range(nb):
        sel = r["bin"] == q
        if q == ref:
            eff, se = 0.0, 0.0
        else:
            eff, se = rb["beta"][j], rb["se"][j]
            j += 1
        rows.append((q, float(r.loc[sel, "k"].median()), int(sel.sum()), eff, se))
    print("bin  k_med   votes  ln[h/h_ref]  se   (ref: k in [2,4))")
    for q, km, n, e, s in rows:
        print(f"{q:3d} {km:6.1f} {n:7d} {e:+9.3f} {s:6.3f}")
    print("club prediction with k_c = 10: flat up to ~10, then slope -1 on a log-log scale")


if __name__ == "__main__":
    main()
