"""Synthetic material layer: the PRL model itself.

N accounts, logistic production f(x) = r x (1 - x) (r = 2.8, stable fixed point, f'* = -0.8), eps = 0.2,
conservative coupling x <- (1 - eps) f(x) + eps W' f(x) with a column-stochastic W': each account
sends to 20-40 targets chosen in proportion to their attractiveness a_v (lognormal), so the row sums
R_v differ. Each month 15% of the links are redrawn and the dynamics relaxes to its frozen pattern.
Written out: SteemOps transfer archives (T_{u->v} = eps W'_vu f(x_u) x 1000, in a few transfers), and
votes with rshares = stake x voting power (0.8-1.0) x 10^12, five per account and month.
Case "prl": as above, true slope c = eps / [1 - (1 - eps) f'*]. Case "null": stakes drawn
independently of the network (c = 0).
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile

import numpy as np

BPD = 28800


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", choices=["prl", "null"], default="prl")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    R = np.random.default_rng(a.seed)
    n, r, eps = a.n, 2.8, 0.2
    xs = 1 - 1 / r
    fp = 2 - r
    c = eps / (1 - (1 - eps) * fp)
    attr = R.lognormal(0, 0.3, n)
    names = [f"u{i}" for i in range(n)]
    links = {}
    for u in range(n):
        k = R.integers(20, 41)
        tg = R.choice(np.delete(np.arange(n), u), k, replace=False, p=np.delete(attr, u) / np.delete(attr, u).sum())
        links[u] = (tg, R.dirichlet(5 * np.ones(k)))
    os.makedirs(a.out, exist_ok=True)
    votes = open(os.path.join(a.out, "votes.jsonl"), "w")
    x = np.full(n, xs)
    c_sim, c_lin = [], []
    for m in range(22, 29):
        for u in R.choice(n, int(0.15 * n), replace=False):        # redraw 15% of the links
            k = R.integers(20, 41)
            tg = R.choice(np.delete(np.arange(n), u), k, replace=False, p=np.delete(attr, u) / np.delete(attr, u).sum())
            links[u] = (tg, R.dirichlet(5 * np.ones(k)))
        rows, cols, vals = [], [], []
        for u, (tg, w) in links.items():
            rows += list(tg)
            cols += [u] * len(tg)
            vals += list(w)
        W = np.zeros((n, n))
        W[rows, cols] = vals
        for _ in range(400):
            fx = r * x * (1 - x)
            x = (1 - eps) * fx + eps * W @ fx
        fx = r * x * (1 - x)
        b0 = 18_500_000 + (m - 22) * 30 * BPD
        lines = []
        for u, (tg, w) in links.items():
            for v, wv in zip(tg, w):
                amt = eps * wv * fx[u] * 1000
                parts = R.integers(1, 4)
                for p in range(parts):
                    lines.append((b0 + R.integers(30 * BPD), f"{names[u]},{names[v]},{amt / parts:.3f} STEEM,thanks"))
        lines.sort()
        d = os.path.join(a.out, f"data{m}")
        os.makedirs(d, exist_ok=True)
        with zipfile.ZipFile(os.path.join(d, f"transfer-month-{m}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"transfer-month-{m}.csv", "block_no,from,to,amount,memo\n" +
                       "\n".join(f"{b},{l}" for b, l in lines) + "\n")
        stake = x if a.case == "prl" else R.permutation(x)
        Rv = W.sum(axis=1)
        sv = Rv - Rv.mean()
        c_sim.append(float(np.polyfit(sv, x / x.mean() - 1, 1)[0]))              # slope in the simulated states
        M = (1 - eps) * np.eye(n) + eps * W                                      # full linear response
        dx = np.linalg.solve(np.eye(n) - fp * M, eps * xs * sv)
        c_lin.append(float(np.polyfit(sv, dx / xs, 1)[0]))
        yr, mo = 2016 + (m + 1) // 12, (m + 1) % 12 + 1
        for i in range(n):
            vl = [[names[i], str(int(stake[i] * 1e12 * R.uniform(0.8, 1.0))), 10000, f"{yr}-{mo:02d}-15T00:00:00"]
                  for _ in range(5)]
            votes.write(json.dumps({"author": "p", "permlink": f"{m}-{i}", "votes": vl}) + "\n")
    votes.close()
    with open(os.path.join(a.out, "truth.json"), "w") as fh:
        json.dump(dict(case=a.case, c=c if a.case == "prl" else 0.0, eps=eps, r=r,
                       c_sim=float(np.mean(c_sim)) if a.case == "prl" else 0.0,
                       c_linear_full=float(np.mean(c_lin)) if a.case == "prl" else 0.0), fh)
    print(f"synthetic PRL material layer in {a.out}: n {n}, eps {eps}, true c {c:.3f}, case {a.case}")


if __name__ == "__main__":
    main()
