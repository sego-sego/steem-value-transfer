"""Permutation tests for model C (channel I->M): how often would the estimates seen
in the data arise if the information shocks were unrelated to the authors who received them?

stacked: the first-shock weeks are permuted among the shocked authors (the set of authors and the
         distribution of shock weeks are kept, the link between them is broken); statistic: mean
         lambda over lags 0..6 (audience held fixed) and the pre-trend Wald statistic
dl:      the weekly reach series are permuted among authors (the time pattern of reblogs is kept,
         its link to the author is broken); statistic: cumulative effect of lags 0..6 and the
         leads Wald statistic

Reports the observed values, the permutation distribution (mean, 2.5 and 97.5 percentiles) and
the permutation p-values; the 97.5 percentile of |mean lambda| under the permutation gives the
precision against which an effect could have been detected.

Usage (from the data folder):
  python src/perm_model_c.py --panel-dir panel_fresh05 --n-stacked 40 --n-dl 100
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_model_c as fc          # noqa: E402
import fit_model_c_dl as fdl      # noqa: E402


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def stacked_stat(weeks, posts, seed=1, controls=1500):
    d, nc, nt = fc.stack(weeks, posts, -4, 6, controls, seed)
    r = fc.fit(d, -4, 6, "r_org", True, "")
    return r["mean"], r["wald_pre"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-dir", default="panel_fresh05")
    ap.add_argument("--n-stacked", type=int, default=40)
    ap.add_argument("--n-dl", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--controls", type=int, default=1500, help="clean controls per stack")
    ap.add_argument("--control-seeds", type=int, default=1,
                    help="draws of the control sample for the observed statistic (its spread is reported)")
    a = ap.parse_args()
    rnd = np.random.default_rng(a.seed)
    weeks = pd.read_csv(os.path.join(a.panel_dir, "panel_weeks.csv"))
    posts = quiet(fc.drop_reblogged, pd.read_csv(os.path.join(a.panel_dir, "panel_posts.csv")), a.panel_dir)
    # ---------------- stacked
    if a.n_stacked:
        draws = [quiet(stacked_stat, weeks, posts, seed=1 + j, controls=a.controls) for j in range(a.control_seeds)]
        obs = (float(np.mean([x[0] for x in draws])), float(np.mean([x[1] for x in draws])))
        if a.control_seeds > 1:
            print(f"observed mean lambda over {a.control_seeds} control draws: "
                  + ", ".join(f"{x[0]:+.3f}" for x in draws) + f"; average {obs[0]:+.3f}")
        first = weeks[weeks["shocks"] > 0].groupby("author")["week"].min()
        perm = []
        for i in range(a.n_stacked):
            w = weeks.copy()
            w["shocks"] = 0
            pf = pd.Series(rnd.permutation(first.values), index=first.index)
            key = pd.MultiIndex.from_arrays([w["author"], w["week"]])
            hit = set(zip(pf.index, pf.values))
            w.loc[[k in hit for k in zip(w["author"], w["week"])], "shocks"] = 1
            perm.append(quiet(stacked_stat, w, posts, seed=1 + i % a.control_seeds, controls=a.controls))
            print(f"\r  stacked permutation {i + 1}/{a.n_stacked}", end="", flush=True)
        pm = np.array(perm)
        p_mean = float(np.mean(np.abs(pm[:, 0]) >= abs(obs[0])))
        print(f"\n=== stacked event study ({first.size} shocked authors), {a.n_stacked} permutations of the shock weeks")
        print(f"    mean lambda (lags 0..6): observed {obs[0]:+.3f}; permutation mean {pm[:, 0].mean():+.3f}, "
              f"95% band [{np.percentile(pm[:, 0], 2.5):+.3f}, {np.percentile(pm[:, 0], 97.5):+.3f}]; two-sided p = {p_mean:.2f}")
        print(f"    pre-trend Wald: observed {obs[1]:.1f}; permutation 95th percentile {np.percentile(pm[:, 1], 95):.1f}; "
              f"p = {float(np.mean(pm[:, 1] >= obs[1])):.2f}")
    # ---------------- distributed lags
    if a.n_dl:
        r0 = quiet(fdl.fit, a.panel_dir, weeks=weeks, posts=posts)
        authors = weeks["author"].unique()
        perm = []
        for i in range(a.n_dl):
            mp = dict(zip(authors, rnd.permutation(authors)))
            src = weeks.set_index(["author", "week"])[["reach", "shocks_any"]]
            w = weeks.copy()
            key = pd.MultiIndex.from_arrays([w["author"].map(mp), w["week"]])
            w[["reach", "shocks_any"]] = src.reindex(key).fillna(0).values
            r = quiet(fdl.fit, a.panel_dir, weeks=w, posts=posts)
            perm.append((r["cum"], r["wald_lead"]))
            print(f"\r  distributed-lag permutation {i + 1}/{a.n_dl}", end="", flush=True)
        pm = np.array(perm)
        print(f"\n=== distributed lags, {a.n_dl} permutations of the reach series among authors")
        print(f"    cumulative effect l = 0..6: observed {r0['cum']:+.4f}; permutation mean {pm[:, 0].mean():+.4f}, "
              f"95% band [{np.percentile(pm[:, 0], 2.5):+.4f}, {np.percentile(pm[:, 0], 97.5):+.4f}]; "
              f"two-sided p = {float(np.mean(np.abs(pm[:, 0] - pm[:, 0].mean()) >= abs(r0['cum'] - pm[:, 0].mean()))):.2f}")
        print(f"    leads Wald: observed {r0['wald_lead']:.1f}; permutation 95th percentile {np.percentile(pm[:, 1], 95):.1f}")


if __name__ == "__main__":
    main()
