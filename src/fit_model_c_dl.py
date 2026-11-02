"""Model C, distributed-lag version: all authors, all reblogs of past posts.

  E[r_p] = exp( alpha_a + tau_w + sum_{l=-L1..L2} beta_l x_{a, w-l} + mu ln(1+F_aw) + kappa ln(1+P_aw) )

x_{a,w} = ln(1 + reach_{a,w}), reach = total followers of the accounts that reblogged a PAST post
of author a in week w (--x count: ln(1 + number of such reblogs)). Post-level Poisson
pseudo-likelihood, author and week fixed effects, errors clustered by author. Posts reblogged
after payout are dropped from the outcome. Only weeks with all leads and lags inside the
panel are used.

beta_l for l < 0 (leads) test for anticipation / reverse timing: they should be 0. The sum of
beta_0..beta_L2 is the cumulative elasticity of the reward per post with respect to the reach
of the information shock, with the audience F held fixed.

Usage:
  python src/fit_model_c_dl.py --panel-dir . [--x reach|count] [--outcome r_org] [--fig fig_model_c_dl]
  python src/fit_model_c_dl.py --selftest
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpois import cpois  # noqa: E402
from fit_model_c import drop_reblogged  # noqa: E402


def fit(panel_dir, x="reach", outcome="r_org", L1=4, L2=6, use_F=True, fig=None, quiet=False, weeks=None, posts=None):
    if weeks is None:
        weeks = pd.read_csv(os.path.join(panel_dir, "panel_weeks.csv"))
    weeks = weeks.dropna(subset=["F"]).copy()
    if posts is None:
        posts = drop_reblogged(pd.read_csv(os.path.join(panel_dir, "panel_posts.csv")), panel_dir)
    posts = posts[posts["bought"].astype(int) == 0]
    col = "reach" if x == "reach" else "shocks_any"
    weeks["x"] = np.log1p(weeks[col].astype(float))
    W = int(weeks["week"].max())
    xw = weeks.set_index(["author", "week"])["x"]
    d = posts.merge(weeks[["author", "week", "F", "P"]], on=["author", "week"], how="inner")
    d = d[(d["week"] >= L2) & (d["week"] <= W - L1)].copy()
    lags = list(range(-L1, L2 + 1))
    cols = []
    for l in lags:
        key = pd.MultiIndex.from_arrays([d["author"], d["week"] - l])
        d[f"x{l:+d}"] = xw.reindex(key).fillna(0.0).values
        cols.append(f"x{l:+d}")
    names = list(cols)
    X = [d[c].values for c in cols]
    if use_F:
        X.append(np.log1p(d["F"].astype(float).values))
        names.append("ln F")
    X.append(np.log1p(d["P"].astype(float).values))
    names.append("ln P")
    r = cpois(d[outcome].astype(float).values, np.column_stack(X), d["author"].values, [d["week"].values],
              cluster=d["author"].values)
    b, se, V = r["beta"], r["se"], r["vcov"]
    lead = [i for i, l in enumerate(lags) if l < 0]
    post = [i for i, l in enumerate(lags) if l >= 0]
    wl = float(b[lead] @ np.linalg.pinv(V[np.ix_(lead, lead)]) @ b[lead])
    c = np.zeros(len(b))
    c[post] = 1
    cum, cum_se = float(c @ b), float(np.sqrt(c @ V @ c))
    share = float((weeks[col] > 0).mean())
    if not quiet:
        print(f"--- distributed lags, x = ln(1 + {col}), outcome {outcome}, {'with' if use_F else 'without'} ln F: "
              f"{r['n']} posts, {r['nunits']} authors; author-weeks with x > 0: {share:.2%}")
        print("    " + "  ".join(f"{names[i]} {b[i]:+.4f}({se[i]:.4f})" for i in range(len(lags))))
        print("    " + "  ".join(f"{names[i]} {b[i]:+.3f}({se[i]:.3f})" for i in range(len(lags), len(b))))
        print(f"    leads Wald chi2({len(lead)}) = {wl:.1f}; cumulative effect l = 0..{L2}: {cum:+.4f} ± {cum_se:.4f}")
    if fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f_, ax = plt.subplots(figsize=(3.4, 2.5))
        ax.errorbar(lags, b[:len(lags)], yerr=1.96 * se[:len(lags)], fmt="o-", ms=3.5, lw=1, color="#2a78d6", capsize=0)
        ax.axhline(0, color="#52514e", lw=0.6)
        ax.axvline(-0.5, color="#b8b6b0", lw=0.6, ls=":")
        ax.set_xlabel("weeks since the reblogs of past posts")
        ax.set_ylabel(f"effect of ln(1+{col})\non log reward per post")
        for ext in ("pdf", "png"):
            f_.savefig(f"{fig}.{ext}", bbox_inches="tight", dpi=300)
        print(f"saved {fig}.pdf/.png")
    return dict(lags=lags, b=b[:len(lags)], se=se[:len(lags)], cum=cum, cum_se=cum_se, wald_lead=wl,
                mu=float(b[names.index("ln F")]) if use_F else None)


def selftest(seeds=(11, 12, 13)):
    import json
    import subprocess
    import tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    synth = os.path.join(here, "..", "validation", "synth_twolayer.py")
    out = {}
    for case in ("zero", "lambda"):
        for seed in seeds:
            tmp = tempfile.mkdtemp(prefix=f"modelcdl_{case}_{seed}_")
            subprocess.run([sys.executable, synth, "--out", tmp, "--case", case, "--seed", str(seed)], check=True,
                           stdout=subprocess.DEVNULL)
            subprocess.run([sys.executable, os.path.join(here, "build_panel_4b.py"), "--data-root", tmp,
                            "--hist-dir", os.path.join(tmp, "custom_json_hist"), "--authors",
                            os.path.join(tmp, "authors_sample.csv"), "--posts", os.path.join(tmp, "posts_fetch.csv"),
                            "--rewards", os.path.join(tmp, "post_rewards.csv"), "--bots", os.path.join(tmp, "bots.csv"),
                            "--out-dir", tmp], check=True, stdout=subprocess.DEVNULL)
            print(f"=== case {case}, seed {seed}")
            out.setdefault(case, []).append(fit(tmp))
    print("\n=== summary (mean over seeds); the synthetic effect comes from big-account reblogs only")
    for case, rs in out.items():
        prof = np.mean([r["b"] for r in rs], axis=0)
        cum = np.array([r["cum"] for r in rs])
        print(f"case {case}: profile " + "  ".join(f"{l:+d}:{v:+.4f}" for l, v in zip(rs[0]["lags"], prof)))
        print(f"   cumulative {cum.mean():+.4f} (s.d. {cum.std(ddof=1):.4f}); leads chi2 "
              f"{', '.join(f'{r['wald_lead']:.1f}' for r in rs)}; mu {np.mean([r['mu'] for r in rs]):.3f} (true 0.8)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-dir", default=".")
    ap.add_argument("--x", choices=["reach", "count"], default="reach")
    ap.add_argument("--outcome", choices=["r_org", "r_all"], default="r_org")
    ap.add_argument("--no-F", action="store_true")
    ap.add_argument("--fig", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    fit(a.panel_dir, a.x, a.outcome, use_F=not a.no_F, fig=a.fig or None)


if __name__ == "__main__":
    main()
