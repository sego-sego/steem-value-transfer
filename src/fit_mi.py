"""Channel M->I: does a bought bot vote bring organic attention to the post?

Post x age-bin Poisson pseudo-likelihood with post fixed effects:

  E[y_ph] = exp( alpha_p + delta_{h, month} + sum_k beta_k 1[voted purchase, h - e_p = k]
                 + sum_k pi_k 1[purchase without bot vote, h - e_p = k] )

y: organic upvotes (or replies) on post p in 12-hour bin h of its age; e_p: bin of the bot vote
(or of the purchase when the bot did not vote). The post effect removes the quality of the post
(buyers choose their better posts); delta gives the natural decay of attention with age, by month,
from the posts without purchase. Reference: bins k = -1 and k <= -4; k = -3, -2 test for
anticipation. pi_k (purchases without bot vote: refunds, failed bids) is the placebo: paying
without the vote should not attract attention.

--dose adds ln(bot rshares) x 1[voted, k >= 0] (centred): elasticity of the response with
respect to the size of the bought vote.

Usage:
  python src/fit_mi.py --dir . [--outcome votes|replies] [--dose] [--fig fig_mi]
  python src/fit_mi.py --selftest
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpois import cpois  # noqa: E402

KS = [-3, -2, 0, 1, 2, 3, 4, 5]          # 5 means k >= 5


def kbin(k):
    return np.where(k >= 5, 5, k)


def fit(d_dir, outcome="votes", dose=False, fig=None, quiet=False, strict=False):
    ev = pd.read_csv(os.path.join(d_dir, "mi_events.csv"), dtype={"buyers": str})
    for c in ("service", "strict_bot", "refunded"):
        if c not in ev:
            ev[c] = 0
    bins = pd.read_csv(os.path.join(d_dir, "mi_bins.csv"))
    d = bins.merge(ev[["author", "permlink", "month", "bought", "voted", "event_bin", "bot_rshares", "service", "strict_bot", "refunded"]],
                   on=["author", "permlink"], how="inner")
    d = d[(d["bought"] == 0) | (d["event_bin"].between(0, 13))].copy()
    if strict:      # purchases without vote that are not in the strict placebo leave the sample (not controls)
        nv_ = (d["bought"] == 1) & (d["voted"] == 0) & (d["service"].fillna(0).astype(int) == 0)
        ok_ = (d["strict_bot"].fillna(0).astype(int) == 1) & (d["refunded"].fillna(0).astype(int) == 1)
        d = d[~(nv_ & ~ok_)].copy()
    k = kbin(d["bin"].values - d["event_bin"].fillna(-99).values)
    svc = d["service"].fillna(0).astype(int).values == 1
    tv = ((d["bought"] == 1) & (d["voted"] == 1)).values & ~svc
    tn = ((d["bought"] == 1) & (d["voted"] == 0)).values & ~svc
    ts = (d["bought"] == 1).values & svc
    X, names = [], []
    for grp, lab in ((tv, "voted"), (tn, "no-vote"), (ts, "service")):
        for kk in KS:
            X.append((grp & (k == kk)).astype(float))
            names.append(f"{lab} k{kk:+d}")
    if dose:
        r = pd.to_numeric(d["bot_rshares"], errors="coerce").values
        ok = tv & (r > 0)
        lr = np.where(ok, np.log(np.where(ok, r, 1.0)), 0.0)
        lr = np.where(ok, lr - lr[ok].mean(), 0.0)
        X.append(lr * (k >= 0) * ok)
        names.append("dose ln rshares x after")
    X = np.column_stack(X)
    keepcol = X.any(axis=0)
    X, names = X[:, keepcol], [n for n, c in zip(names, keepcol) if c]
    unit = (d["author"] + "/" + d["permlink"]).values
    age_month = (d["bin"].astype(str) + ":" + d["month"].astype(str)).values
    res = cpois(d[outcome].astype(float).values, X, unit, [age_month], cluster=d["author"].values)
    b, se, V = res["beta"], res["se"], res["vcov"]
    iv = [i for i, n in enumerate(names) if n.startswith("voted k") and int(n.split("k")[1]) >= 0]
    inn = [i for i, n in enumerate(names) if n.startswith("no-vote k") and int(n.split("k")[1]) >= 0]
    lv = [i for i, n in enumerate(names) if n.startswith("voted k") and int(n.split("k")[1]) < 0]

    def avg(idx):
        c = np.zeros(len(b))
        c[idx] = 1 / len(idx)
        return float(c @ b), float(np.sqrt(c @ V @ c))

    mv, sv = avg(iv)
    mn, sn = avg(inn) if inn else (np.nan, np.nan)
    if iv and inn:
        c = np.zeros(len(b))
        c[iv] = 1 / len(iv)
        c[inn] = -1 / len(inn)
        dvn, dvn_se = float(c @ b), float(np.sqrt(c @ V @ c))
    else:
        dvn, dvn_se = np.nan, np.nan
    isv = [i for i, n in enumerate(names) if n.startswith("service k") and int(n.split("k")[1]) >= 0]
    ms, ss = avg(isv) if isv else (np.nan, np.nan)
    wl = float(b[lv] @ np.linalg.pinv(V[np.ix_(lv, lv)]) @ b[lv]) if lv else np.nan
    n_v = int(ev.query("bought == 1 and voted == 1 and service == 0").shape[0])
    n_n = int(ev.query("bought == 1 and voted == 0 and service == 0").shape[0])
    n_s = int(ev.query("bought == 1 and service == 1").shape[0])
    if not quiet:
        print(f"--- M->I, outcome organic {outcome}: {res['n']} post-bins, {res['nunits']} posts "
              f"({n_v} with bid-bot vote, {n_n} paid bid bot without vote, {n_s} paid services), {res['nclusters']} authors")
        for i, n in enumerate(names):
            print(f"    {n:26s} {b[i]:+.3f} ({se[i]:.3f})")
        print(f"    leads (voted, k = -3, -2) Wald chi2({len(lv)}) = {wl:.1f}")
        print(f"    mean effect after the bot vote (k >= 0): {mv:+.3f} ± {sv:.3f}  -> x{np.exp(mv):.2f} organic {outcome}")
        print(f"    placebo, paid without vote (k >= 0):     {mn:+.3f} ± {sn:.3f}")
        print(f"    paid services, no vote (k >= 0):         {ms:+.3f} ± {ss:.3f}")
        print(f"    contribution of the bot vote itself (voted - placebo): {dvn:+.3f} ± {dvn_se:.3f}"
              f"{'  [strict placebo]' if strict else ''}")
    if fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f_, ax = plt.subplots(figsize=(3.4, 2.5))
        for lab, pre, col, dx in (("bid bot voted", "voted", "#2a78d6", -0.1), ("paid, no vote", "no-vote", "#eb6834", 0.0),
                                  ("paid service", "service", "#2e9e5b", 0.1)):
            idx = [i for i, n in enumerate(names) if n.startswith(pre + " k")]
            if not idx:
                continue
            kk = np.array([int(names[i].split("k")[1]) for i in idx] + [-1]) + dx
            yy = np.r_[b[idx], 0.0]
            ee = np.r_[1.96 * se[idx], 0.0]
            o = np.argsort(kk)
            ax.errorbar(kk[o], yy[o], yerr=ee[o], fmt="o-", ms=3.5, lw=1, color=col, capsize=0, label=lab)
        ax.axhline(0, color="#52514e", lw=0.6)
        ax.axvline(-0.5, color="#b8b6b0", lw=0.6, ls=":")
        ax.set_xlabel("12-h bins since the bot vote (5 = 5 and later)")
        ax.set_ylabel(f"log organic {outcome} per bin")
        ax.legend(fontsize=6.5, frameon=False)
        for ext in ("pdf", "png"):
            f_.savefig(f"{fig}.{ext}", bbox_inches="tight", dpi=300)
        print(f"saved {fig}.pdf/.png")
    return dict(names=names, b=b, se=se, mean_voted=mv, mean_voted_se=sv, mean_novote=mn, mean_service=ms, diff=dvn,
                wald_leads=wl)


def selftest(seeds=(1, 2, 3)):
    import json
    import subprocess
    import tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    synth = os.path.join(here, "..", "validation", "synth_mi.py")
    for case in ("zero", "effect"):
        mv, mn = [], []
        for seed in seeds:
            tmp = tempfile.mkdtemp(prefix=f"mi_{case}_{seed}_")
            subprocess.run([sys.executable, synth, "--out", tmp, "--case", case, "--seed", str(seed)], check=True,
                           stdout=subprocess.DEVNULL)
            subprocess.run([sys.executable, os.path.join(here, "build_mi_events.py"), "--data-root", tmp,
                            "--posts", os.path.join(tmp, "posts_fetch.csv"), "--bots", os.path.join(tmp, "bots.csv"),
                            "--out-dir", tmp], check=True, stdout=subprocess.DEVNULL)
            truth = json.load(open(os.path.join(tmp, "truth.json")))
            r = fit(tmp, quiet=True)
            mv.append(r["mean_voted"])
            mn.append(r["mean_novote"])
            print(f"case {case}, seed {seed}: mean effect after the vote {r['mean_voted']:+.3f} ± {r['mean_voted_se']:.3f} "
                  f"(true {np.mean(truth['beta']):+.3f}); placebo {r['mean_novote']:+.3f} (true 0); service {r['mean_service']:+.3f} (true 0); "
                  f"leads chi2 {r['wald_leads']:.1f}")
        print(f"=== case {case}: mean {np.mean(mv):+.3f} (s.d. {np.std(mv, ddof=1):.3f}), "
              f"placebo {np.mean(mn):+.3f}; true {np.mean(truth['beta']):+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--outcome", choices=["votes", "replies", "votes_new", "votes_loyal", "votes_active", "votes_new_poster", "votes_new_other"], default="votes")
    ap.add_argument("--dose", action="store_true")
    ap.add_argument("--strict", action="store_true", help="placebo = refunded purchases from bots voting >= 90%% themselves")
    ap.add_argument("--fig", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    fit(a.dir, a.outcome, a.dose, a.fig or None, strict=a.strict)


if __name__ == "__main__":
    main()
