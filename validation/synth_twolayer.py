"""Synthetic two-layer Steem data for model C (channel I->M), in the SteemOps format
plus the reward tables produced by stage 2.

Accounts: readers, sampled authors, big accounts (1500-3000 followers), small rebloggers.
Every month each reader is active (one vote) with a reader-specific probability; authors post
at their own rates. Half of the authors receive, in week s_a in [5, 22], a reblog of a PAST post
(created more than 8 days earlier) by a big account: the information shock. After it they gain
10-40% more followers over ten days (audience channel). Noise: small accounts reblog past posts
of anyone (not shocks: small audience), big accounts reblog CURRENT posts of anyone (not shocks:
the reblogged post is not past).

Organic reward of a post of author a in week w:
  r = exp(alpha_a + tau_w + mu ln(1 + F_aw) + kappa ln(1 + P_aw) + lambda_{w - s_a}) * Gamma(1.5)/1.5
with F_aw the active followers at the start of week w (same definition as build_panel_4b.py).
Cases: --case zero (lambda = 0) and --case lambda (0, .25, .25, .15, .10, .05, 0 for lags 0..6).
Writes OUT/dataNN/{comment,vote,custom_json}-month-NN.zip, OUT/custom_json_hist/custom_json-month-21.zip,
authors_sample.csv, posts_fetch.csv, post_rewards.csv, bots.csv, truth.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import random
import zipfile

BPD = 28800
WEEK = 7 * BPD
B0 = 18_500_000
MONTHS = list(range(22, 29))
MSTART = {m: B0 + (m - 22) * 30 * BPD for m in MONTHS}
NWEEKS = (7 * 30 * BPD) // WEEK          # 30 weeks
LAMBDA = {"zero": [0] * 7, "lambda": [0.0, 0.25, 0.25, 0.15, 0.10, 0.05, 0.0]}


def month_of(b):
    return max([m for m in MONTHS if MSTART[m] <= b], default=MONTHS[0])


def write_zip(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(os.path.basename(path).replace(".zip", ".csv"), header + "\n" + "\n".join(rows) + "\n")


def fol_line(b, x, t, what="blog"):
    return f'{b},["{x}"],follow,["follow",{{"follower":"{x}","following":"{t}","what":["{what}"]}}]'


def reb_line(b, x, au, pl):
    return f'{b},["{x}"],follow,["reblog",{{"account":"{x}","author":"{au}","permlink":"{pl}"}}]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", choices=list(LAMBDA), default="lambda")
    ap.add_argument("--n-authors", type=int, default=1500)
    ap.add_argument("--n-readers", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    R = random.Random(a.seed)
    lam = LAMBDA[a.case]
    mu, kappa = 0.8, -0.1
    readers = [f"r{i}" for i in range(a.n_readers)]
    authors = [f"a{i}" for i in range(a.n_authors)]
    bigs = [f"big{i}" for i in range(21)]   # exactly the top 5% of 421 rebloggers
    smalls = [f"s{i}" for i in range(400)]
    p_act = {r: R.uniform(0.3, 0.9) for r in readers}
    # initial graph (history month 21)
    fol = {x: set(R.sample(readers, int(min(400, max(20, R.lognormvariate(4.3, 0.7)))))) for x in authors}
    hist = []
    for t in authors:
        for x in fol[t]:
            hist.append(fol_line(B0 - 10 * BPD + R.randrange(5 * BPD), x, t))
    for t in bigs:
        for x in R.sample(readers, R.randrange(1500, 3000)):
            hist.append(fol_line(B0 - 10 * BPD + R.randrange(5 * BPD), x, t))
    for t in smalls:
        for x in R.sample(readers, R.randrange(5, 40)):
            hist.append(fol_line(B0 - 10 * BPD + R.randrange(5 * BPD), x, t))
    hist.sort(key=lambda s: int(s.split(",", 1)[0]))
    write_zip(os.path.join(a.out, "custom_json_hist", "custom_json-month-21.zip"),
              "block_no,required_posting_auths,id,json", hist)
    # activity of readers per month
    active = {m: {r for r in readers if R.random() < p_act[r]} for m in MONTHS}
    # posts
    rate = {x: R.uniform(0.5, 4.0) for x in authors}
    posts = []                                       # (block, author, permlink)
    for x in authors:
        for w in range(NWEEKS):
            k = sum(R.random() < rate[x] / 8 for _ in range(8))
            for j in range(k):
                b = B0 + w * WEEK + R.randrange(1, WEEK - 1)
                posts.append((b, x, f"{x}-w{w}-{j}"))
    posts.sort()
    by_author = collections.defaultdict(list)
    for b, x, pl in posts:
        by_author[x].append((b, pl))
    # shocks and audience growth
    events = []                                      # custom_json lines in the window
    shock_week = {}
    for x in R.sample(authors, len(authors) // 2):
        s = R.randrange(5, 23)
        bs = B0 + s * WEEK + R.randrange(int(0.2 * WEEK), int(0.8 * WEEK))
        old = [pl for b, pl in by_author[x] if b < bs - 8 * BPD]
        pl = R.choice(old) if old else f"{x}-old-post"
        events.append((bs, reb_line(bs, R.choice(bigs), x, pl)))
        shock_week[x] = s
        g = R.uniform(0.1, 0.4)
        cand = [r for r in R.sample(readers, 3000) if r not in fol[x]]
        for r in cand[:int(g * len(fol[x]))]:
            events.append((bs + R.randrange(10 * BPD), fol_line(bs + R.randrange(10 * BPD), r, x)))
    for _ in range(3000):                            # noise: small accounts reblog past posts
        b, x, pl = R.choice(posts)
        bb = b + R.randrange(9 * BPD, 20 * BPD)
        if bb < B0 + NWEEKS * WEEK:
            events.append((bb, reb_line(bb, R.choice(smalls), x, pl)))
    for _ in range(1500):                            # noise: big accounts reblog current posts
        b, x, pl = R.choice(posts)
        bb = b + R.randrange(BPD // 2)
        events.append((bb, reb_line(bb, R.choice(bigs), x, pl)))
    for _ in range(3000):                            # ordinary follows and unfollows among readers
        bb = B0 + R.randrange(NWEEKS * WEEK)
        x, t = R.sample(readers, 2)
        events.append((bb, fol_line(bb, x, t)))
    # fix the lines' block to the event block (follow lines above carry their own)
    events = sorted(((int(l.split(",", 1)[0]), l) for _, l in events))
    # true F_aw at the start of each week: replay the follow events
    folw = {x: set(s) for x, s in fol.items()}
    F = {}
    i = 0
    for w in range(NWEEKS):
        start = B0 + w * WEEK
        while i < len(events) and events[i][0] < start:
            l = events[i][1]
            if '"follow",{"follower"' in l:
                x = l.split('"follower":"')[1].split('"')[0]
                t = l.split('"following":"')[1].split('"')[0]
                if t in folw:
                    folw[t].add(x)
            i += 1
        act = active[month_of(start)]
        for x in authors:
            F[(x, w)] = len(folw[x] & act)
    with open(os.path.join(a.out, "F_true.csv"), "w") as fh:
        fh.write("author,week,F_true\n" + "".join(f"{x},{w},{F[(x, w)]}\n" for (x, w) in F))
    P = collections.Counter((x, (b - B0) // WEEK) for b, x, pl in posts)
    alpha = {x: R.gauss(0, 0.7) for x in authors}
    tau = [R.gauss(0, 0.2) for _ in range(NWEEKS)]
    rows_rew, rows_fetch = [], []
    for b, x, pl in posts:
        w = (b - B0) // WEEK
        eta = alpha[x] + tau[w] + mu * math.log1p(F[(x, w)]) + kappa * math.log1p(P[(x, w)])
        if x in shock_week and 0 <= w - shock_week[x] < len(lam):
            eta += lam[w - shock_week[x]]
        r = math.exp(eta - 4.0) * R.gammavariate(1.5, 1 / 1.5)
        org = int(r * 1e9)
        bot = int(R.expovariate(1.0) * 1e9) if R.random() < 0.3 else 0
        rows_rew.append([x, pl, org, org + bot, 0, f"{r / 10:.3f}"])
        rows_fetch.append([x, pl, month_of(b), b, "c", 0, w, P[(x, w)], 1])
    # SteemOps archives of the window
    for m in MONTHS:
        lo, hi = MSTART[m], MSTART[m] + 30 * BPD
        com = [f"{lo},,c,filler,month-start-{m}"] + \
              [f"{b},,c,{x},{pl}" for b, x, pl in posts if lo <= b < hi]
        vot = []
        for r_ in sorted(active[m]):
            b, x, pl = R.choice(posts)
            vot.append((lo + R.randrange(30 * BPD), r_, x, pl))
        vot = [f"{b},{v},{x},{pl},10000" for b, v, x, pl in sorted(vot)]
        cj = [l for b, l in events if lo <= b < hi]
        d = os.path.join(a.out, f"data{m}")
        write_zip(os.path.join(d, f"comment-month-{m}.zip"), "block_no,parent_author,parent_permlink,author,permlink", com)
        write_zip(os.path.join(d, f"vote-month-{m}.zip"), "block_no,voter,author,permlink,weight", vot)
        write_zip(os.path.join(d, f"custom_json-month-{m}.zip"), "block_no,required_posting_auths,id,json", cj)
    with open(os.path.join(a.out, "authors_sample.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["account", "stratum"])
        w.writerows([x, "s"] for x in authors)
    with open(os.path.join(a.out, "post_rewards.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["author", "permlink", "rshares_organic", "rshares_pos", "missing", "payout_author_sbd"])
        w.writerows(rows_rew)
    with open(os.path.join(a.out, "posts_fetch.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["author", "permlink", "month", "block", "category", "bought", "week", "n_week", "weight"])
        w.writerows(rows_fetch)
    with open(os.path.join(a.out, "bots.csv"), "w") as fh:
        fh.write("account\nsomebot\n")
    with open(os.path.join(a.out, "truth.json"), "w") as fh:
        json.dump(dict(case=a.case, **{"lambda": lam}, mu=mu, kappa=kappa, n_shocked=len(shock_week)), fh)
    print(f"synthetic two-layer data in {a.out}: {len(authors)} authors, {len(posts)} posts, "
          f"{len(shock_week)} shocked, case {a.case}")


if __name__ == "__main__":
    main()
