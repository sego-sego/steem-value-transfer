"""Synthetic test for build_reblog_pairs.py with known source exponent gamma and recipient
capacity exponent theta.

Truth: an active follower x of reblogger v (not following the author) engages with the reblogged
post p with probability  q_p * c * k_v^(-gamma) * (L_x / 30)^(-theta),
where k_v = reblogs + own posts of v on that day and L_x = items per day produced by everyone x
follows (the definition used by the builder). Background engagement with random posts by random
users makes the placebos non-trivial but independent of k and L.

Usage: python synth_reblogs.py --gamma 0.5 --theta 0.5 --out synth_reblogs
"""
from __future__ import annotations

import argparse
import collections
import os
import random
import zipfile

BPD = 28800


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--out", default="synth_reblogs")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    R = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    users = [f"u{i}" for i in range(8000)]
    authors, rebloggers, readers = users[:500], users[500:1300], users[1300:]
    producers = authors + rebloggers
    cats = [f"tag{i}" for i in range(15)]
    b0 = 20_279_029
    bmax = b0 + 31 * BPD
    events_c, events_v, events_j = [], [], []
    # posts
    posts = []
    for au in authors:
        for j in range(5):
            b = b0 + R.randrange(0, 22 * BPD)
            pl, cat = f"post-{au}-{j}", R.choice(cats)
            posts.append((au, pl, b, R.lognormvariate(0, 0.7)))
            events_c.append((b, f"{b},,{cat},{au},{pl}"))
    # reblogs: target reblogs plus fillers; rebloggers also post a little (own posts count in k_items)
    rate = {v: R.choice([1, 2, 3, 5, 8, 12, 20]) for v in rebloggers}
    planned = []
    for au, pl, b, q in posts:
        for v in R.sample(rebloggers, R.randint(2, 6)):
            rb = b + R.randrange(BPD // 10, 2 * BPD)
            events_j.append((rb, f'{rb},["{v}"],follow,["reblog",{{"account":"{v}","author":"{au}","permlink":"{pl}"}}]'))
            planned.append((au, pl, q, v, rb))
            day0 = b0 + (rb - b0) // BPD * BPD
            for f in range(max(0, int(round(rate[v] * R.uniform(0.5, 1.5))) - 1)):
                fb = day0 + R.randrange(BPD)
                events_j.append((fb, f'{fb},["{v}"],follow,["reblog",{{"account":"{v}","author":"old","permlink":"old-{v}-{rb}-{f}"}}]'))
    for v in rebloggers:
        for j in range(R.randint(0, 6)):
            b = b0 + R.randrange(0, 30 * BPD)
            events_c.append((b, f"{b},,{R.choice(cats)},{v},own-{v}-{j}"))
    # items per producer per day (month average) and per (producer, day)
    items_month = collections.Counter()
    kday = collections.Counter()
    for b, s in events_j:
        v = s.split('"')[1]
        items_month[v] += 1
        kday[(v, (b - b0) // BPD)] += 1
    for b, s in events_c:
        f = s.split(",")
        if f[1] == "":
            items_month[f[3]] += 1
            kday[(f[3], (b - b0) // BPD)] += 1
    # follow graph: readers follow some rebloggers (audiences) and a heterogeneous number of producers (load)
    fol = collections.defaultdict(set)
    followees = collections.defaultdict(set)
    for v in rebloggers:
        for x in R.sample(readers, int(min(500, 5 + R.paretovariate(1.2) * 12))):
            fol[v].add(x)
            followees[x].add(v)
    for x in readers:
        m = int(min(600, R.lognormvariate(3.0, 1.2)))
        for t in R.sample(producers, m):
            followees[x].add(t)
            if t in rebloggers:
                fol[t].add(x)
    load = {x: sum(items_month[t] for t in followees[x]) / 31.0 for x in readers}
    hist = ["block_no,required_posting_auths,id,json"]
    i = 0
    for x, ts in followees.items():
        for t in ts:
            hist.append(f'{1000 + i},["{x}"],follow,["follow",{{"follower":"{x}","following":"{t}","what":["blog"]}}]')
            i += 1
    c = 0.10
    for au, pl, q, v, rb in planned:
        k = kday[(v, (rb - b0) // BPD)]
        for x in fol[v]:
            if au in followees[x]:
                continue
            p = min(1.0, q * c * k ** (-a.gamma) * (max(load[x], 0.5) / 30.0) ** (-a.theta))
            if R.random() < p:
                eb = rb + R.randrange(1, 3 * BPD)
                events_v.append((eb, f"{eb},{x},{au},{pl},10000"))
    # background: random users vote on random posts (placebo level), everyone active once
    allposts = [(au, pl) for au, pl, _, _ in posts]
    for _ in range(60000):
        x = R.choice(readers)
        au, pl = R.choice(allposts)
        eb = b0 + R.randrange(31 * BPD)
        events_v.append((eb, f"{eb},{x},{au},{pl},5000"))
    for x in users:
        eb = b0 + R.randrange(31 * BPD)
        events_v.append((eb, f"{eb},{x},old,bg,5000"))
    events_c.append((bmax, f"{bmax},old,bg,closer,end"))
    out = {}
    for name, ev, head in (("comment", events_c, "block_no,parent_author,parent_permlink,author,permlink"),
                           ("vote", events_v, "block_no,voter,author,permlink,weight"),
                           ("custom_json", events_j, "block_no,required_posting_auths,id,json")):
        ev.sort()
        out[name] = [head] + [s for _, s in ev]
    out["transfer"] = ["block_no,from,to,amount,memo", f"{b0},a,b,1.000 SBD,hello"]
    for name, lines in out.items():
        with zipfile.ZipFile(os.path.join(a.out, f"{name}-month-25.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"{name}-month-25.csv", "\n".join(lines) + "\n")
    for m in range(1, 25):
        lines = hist if m == 1 else hist[:1]
        with zipfile.ZipFile(os.path.join(a.out, f"custom_json-month-{m:02d}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"custom_json-month-{m:02d}.csv", "\n".join(lines) + "\n")
    print(f"synthetic v2 in {a.out}: {len(posts)} posts, gamma = {a.gamma}, theta = {a.theta}")


if __name__ == "__main__":
    main()
