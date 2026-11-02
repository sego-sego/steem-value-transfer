"""Model C: does an information shock raise the rewards of the author's LATER posts?

Stacked event study (one stack per cohort of first shock week c), post-level Poisson
pseudo-likelihood with fixed effects:

  E[r_p] = exp( alpha_{s,a} + tau_{s,w} + sum_{l != -1} lambda_l 1[treated, w - c = l]
                + mu ln(1 + F_aw) + kappa ln(1 + P_aw) )

r_p: organic rshares of post p (bots and self-votes excluded; --outcome r_all for all votes),
s: stack, a: author, w: week of the post. Stack c contains the authors first shocked in week c
and up to --controls clean controls: authors never shocked or first shocked after c + L_max.
Weeks c - L_min ... c + L_max. Errors clustered by author.

lambda_l for l < -1 test for pre-trends (should be 0); lambda_l, l >= 0, give the delay profile
of the I->M channel. With mu ln F the audience growth caused by the shock is held fixed, so
lambda measures the change of production per unit of audience; --no-F gives the total effect.

Usage:
  python src/fit_model_c.py --panel-dir . [--outcome r_org] [--no-F] [--fig fig_model_c]
  python src/fit_model_c.py --selftest
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpois import cpois  # noqa: E402


def drop_reblogged(posts, panel_dir):
    """Posts reblogged after their payout were chosen for their quality: keeping them puts a
    selected high-reward post into the pre-shock weeks (a mechanical peak before the shock)."""
    f = os.path.join(panel_dir, "reblogs_past.csv")
    if not os.path.exists(f):
        print("WARNING: reblogs_past.csv not found (old panel): reblogged posts kept")
        return posts
    rb = pd.read_csv(f, usecols=["author", "permlink"]).drop_duplicates()
    m = posts.merge(rb.assign(_rb=1), on=["author", "permlink"], how="left")
    print(f"dropped {int(m['_rb'].notna().sum())} posts reblogged after payout (selected on quality)")
    return m[m["_rb"].isna()].drop(columns="_rb")


def stack(weeks, posts, lmin, lmax, n_controls, seed, exclude_bought=True):
    weeks = weeks.copy()
    first = weeks[weeks["shocks"] > 0].groupby("author")["week"].min()
    weeks["first"] = weeks["author"].map(first)
    wmax = int(weeks["week"].max())
    # F at the start of the week; weeks without follow data are dropped
    weeks = weeks.dropna(subset=["F"])
    posts = posts.merge(weeks[["author", "week", "F", "P", "first"]], on=["author", "week"], how="inner")
    if exclude_bought:
        posts = posts[posts["bought"].astype(int) == 0]
    rnd = np.random.default_rng(seed)
    never = first.reindex(weeks["author"].unique()).isna()
    authors_all = np.array(sorted(weeks["author"].unique()))
    out = []
    cohorts = sorted(int(c) for c in first.unique() if c - lmin >= 0 and c + lmax <= wmax)
    for c in cohorts:
        treated = first[first == c].index
        clean = [x for x in authors_all if (never.get(x, False) or first.get(x, np.inf) > c + lmax)]
        if len(clean) > n_controls:
            clean = list(rnd.choice(clean, n_controls, replace=False))
        sel = posts[(posts["week"] >= c + lmin) & (posts["week"] <= c + lmax)
                    & (posts["author"].isin(treated) | posts["author"].isin(clean))].copy()
        sel["stack"] = c
        sel["treated"] = sel["author"].isin(treated).astype(int)
        sel["rel"] = sel["week"] - c
        out.append(sel)
    if not out:
        sys.exit("no cohort with a full event window")
    d = pd.concat(out, ignore_index=True)
    return d, len(cohorts), int(first.size)


def fit(d, lmin, lmax, outcome, use_F, label):
    rels = [l for l in range(lmin, lmax + 1) if l != -1]
    X = [((d["treated"] == 1) & (d["rel"] == l)).astype(float).values for l in rels]
    names = [f"l{l:+d}" for l in rels]
    if use_F:
        X.append(np.log1p(d["F"].astype(float).values))
        names.append("ln F")
    X.append(np.log1p(d["P"].astype(float).values))
    names.append("ln P")
    X = np.column_stack(X)
    unit = (d["stack"].astype(str) + ":" + d["author"]).values
    sw = (d["stack"].astype(str) + ":" + d["week"].astype(str)).values
    r = cpois(d[outcome].astype(float).values, X, unit, [sw], cluster=d["author"].values)
    b, se, V = r["beta"], r["se"], r["vcov"]
    nl = len(rels)
    pre = [i for i, l in enumerate(rels) if l < -1]
    post = [i for i, l in enumerate(rels) if l >= 0]
    wpre = float(b[pre] @ np.linalg.pinv(V[np.ix_(pre, pre)]) @ b[pre]) if pre else float("nan")
    # reference: mean of all pre-shock weeks (week -1 is 0 by construction), not week -1 alone
    ref = np.zeros(len(b))
    ref[pre] = 1 / (len(pre) + 1)
    prof, prof_se = [], []
    for i in range(nl):
        c = -ref.copy()
        c[i] += 1
        prof.append(float(c @ b))
        prof_se.append(float(np.sqrt(c @ V @ c)))
    prof_m1 = float(-ref @ b)                       # week -1 relative to the pre-period mean
    avg = -ref.copy()
    avg[post] += 1 / len(post)
    m, s = float(avg @ b), float(np.sqrt(avg @ V @ avg))
    print(f"--- {label}: outcome {outcome}, {'with' if use_F else 'without'} ln F; {r['n']} posts, "
          f"{r['nunits']} stack-authors, {r['nclusters']} authors")
    print("    rel. to pre-period mean: " + "  ".join(f"{names[i]} {prof[i]:+.3f}({prof_se[i]:.3f})" for i in range(nl)))
    print("    " + "  ".join(f"{names[i]} {b[i]:+.3f}({se[i]:.3f})" for i in range(nl, len(b))))
    print(f"    pre-trend Wald chi2({len(pre)}) = {wpre:.1f}; mean lambda over l = 0..{lmax} "
          f"(rel. to pre-period mean): {m:+.3f} ± {s:.3f}")
    return dict(rels=rels + [-1], b=np.r_[prof, prof_m1], se=np.r_[prof_se, np.nan], mean=m, mean_se=s,
                wald_pre=wpre, names=names, all_b=b, all_se=se)


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for k, (lab, r, col, dx) in enumerate(res):
        x = np.array(r["rels"]) + dx
        y = np.asarray(r["b"])
        e = np.nan_to_num(1.96 * np.asarray(r["se"]))
        o = np.argsort(x)
        ax.errorbar(x[o], y[o], yerr=e[o], fmt="o-", ms=3.5, lw=1, color=col, capsize=0, label=lab)
    ax.axhline(0, color="#52514e", lw=0.6)
    ax.axvline(-0.5, color="#b8b6b0", lw=0.6, ls=":")
    ax.set_xlabel("weeks since the information shock")
    ax.set_ylabel("log reward per post,\nrel. to pre-shock weeks")
    ax.legend(fontsize=6.5, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight", dpi=300)
    print(f"saved {path}.pdf/.png")


def run(panel_dir, outcome, lmin, lmax, n_controls, seed, fig=None, no_F=False):
    weeks = pd.read_csv(os.path.join(panel_dir, "panel_weeks.csv"))
    posts = drop_reblogged(pd.read_csv(os.path.join(panel_dir, "panel_posts.csv")), panel_dir)
    d, nc, nt = stack(weeks, posts, lmin, lmax, n_controls, seed)
    print(f"{nt} shocked authors, {nc} cohorts with a full window, {len(d)} stacked posts")
    res = []
    rF = fit(d, lmin, lmax, outcome, True, "audience held fixed")
    res.append(("with ln F", rF, "#2a78d6", -0.1))
    if not no_F:
        r0 = fit(d, lmin, lmax, outcome, False, "total effect")
        res.append(("without ln F", r0, "#eb6834", 0.1))
    if fig:
        figure(res, fig)
    return res


def selftest(lmin, lmax, seeds=(11, 12, 13, 14, 15, 16)):
    """Synthetic data with lambda = 0 and lambda > 0, several seeds; table truth / estimate."""
    import json
    import subprocess
    import tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    synth = os.path.join(here, "..", "validation", "synth_twolayer.py")
    summary = {}
    for case in ("zero", "lambda"):
        for seed in seeds:
            tmp = tempfile.mkdtemp(prefix=f"modelc_{case}_{seed}_")
            subprocess.run([sys.executable, synth, "--out", tmp, "--case", case, "--seed", str(seed)], check=True,
                           stdout=subprocess.DEVNULL)
            subprocess.run([sys.executable, os.path.join(here, "build_panel_4b.py"), "--data-root", tmp,
                            "--hist-dir", os.path.join(tmp, "custom_json_hist"), "--authors",
                            os.path.join(tmp, "authors_sample.csv"), "--posts", os.path.join(tmp, "posts_fetch.csv"),
                            "--rewards", os.path.join(tmp, "post_rewards.csv"), "--bots", os.path.join(tmp, "bots.csv"),
                            "--out-dir", tmp], check=True, stdout=subprocess.DEVNULL)
            truth = json.load(open(os.path.join(tmp, "truth.json")))
            fw = pd.read_csv(os.path.join(tmp, "panel_weeks.csv")).merge(
                pd.read_csv(os.path.join(tmp, "F_true.csv")), on=["author", "week"])
            print(f"    builder check: F equals the true active followers in {(fw.F == fw.F_true).mean():.1%} "
                  f"of {len(fw)} author-weeks")
            print(f"=== case {case}, seed {seed}: true lambda {truth['lambda']}, mu {truth['mu']}")
            res = run(tmp, "r_org", lmin, lmax, 400, 1)
            summary.setdefault(case, []).append((truth, res))
    print("\n=== summary over seeds (audience held fixed): truth / mean estimate (s.d. over seeds)")
    for case, lst in summary.items():
        truth = lst[0][0]
        lam = truth["lambda"]
        rF = [r[0][1] for _, r in lst]
        rels = rF[0]["rels"]
        print(f"case {case}:")
        for l in range(0, lmax + 1):
            k = rels.index(l)
            est = np.array([r["b"][k] for r in rF])
            print(f"   lag {l}: true {lam[l] if l < len(lam) else 0:+.3f}  estimate {est.mean():+.3f} ({est.std(ddof=1):.3f})")
        mt = np.mean(lam[:lmax + 1])
        me = np.array([r["mean"] for r in rF])
        mu = np.array([r["all_b"][r["names"].index("ln F")] for r in rF])
        pre = np.array([r["wald_pre"] for r in rF])
        tot = np.array([r[1][1]["mean"] for _, r in lst])
        print(f"   mean lambda: true {mt:+.3f}  estimate {me.mean():+.3f} ({me.std(ddof=1):.3f}); "
              f"mu: true {truth['mu']}  estimate {mu.mean():.3f} ({mu.std(ddof=1):.3f}); "
              f"pre-trend chi2(3) {', '.join(f'{x:.1f}' for x in pre)}; total effect without F {tot.mean():+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-dir", default=".")
    ap.add_argument("--outcome", default="r_org", choices=["r_org", "r_all"])
    ap.add_argument("--lmin", type=int, default=-4)
    ap.add_argument("--lmax", type=int, default=6)
    ap.add_argument("--controls", type=int, default=1500, help="clean control authors per stack")
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--no-F", action="store_true", help="skip the total-effect fit")
    ap.add_argument("--fig", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.lmin, a.lmax)
        return
    run(a.panel_dir, a.outcome, a.lmin, a.lmax, a.controls, a.seed, a.fig or None, a.no_F)


if __name__ == "__main__":
    main()
