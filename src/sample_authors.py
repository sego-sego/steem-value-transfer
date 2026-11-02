"""Reward data: sample of authors and their root posts for the reward panel.

Reads SteemOps comment, vote and transfer archives of the window months (default 22-28,
December 2017 - June 2018; months whose archives are missing are skipped with a warning) and
writes:
  bots.csv           accounts flagged as bots in any month (rules of the information-layer part: > 400 votes on a
                     day, or recipient of >= 20 transfers with a post link in the memo)
  authors_sample.csv account, stratum, posts, months_active, votes_per_post, buyer
  posts_sample.csv   author, permlink, month, block, category: every root post of the sample

Human authors: not bots, >= --min-posts root posts in the window, posts in >= --min-months
months. Strata: terciles of posts in the window x terciles of votes received per post, equal
allocation (--n-authors / 9 per stratum). Buyers: authors of posts named in memos of transfers
to bid bots in --buyer-months (default 24-26), not bots, up to --n-buyers sampled (stratum
"buyer"; a buyer drawn also into a human stratum keeps that stratum and buyer = 1).

Usage (from the data folder, archives in data22/ ... data28/):
  python sample_authors.py --data-root . --months 22 23 24 25 26 27 28
Standard library only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import os
import random
import re
import sys
import time
import zipfile

BPD = 28800
BOT_URL_MEMOS = 20
BOT_DAILY_VOTES = 400
RE_POST = re.compile(r"/@([a-z0-9][a-z0-9.\-]{1,15})/([a-z0-9][a-z0-9\-]*)")


def zlines(path):
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            it = io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline="")
            next(it)
            for line in it:
                yield line.rstrip("\r\n")


def archive(root, op, m):
    for d in (os.path.join(root, f"data{m}"), root):
        for name in (f"{op}-month-{m:02d}.zip", f"{op}-month-{m}.zip"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None


class Prog:
    def __init__(self, label):
        self.label, self.n, self.t0 = label, 0, time.time()

    def tick(self):
        self.n += 1
        if self.n % 5_000_000 == 0:
            print(f"  {self.label}: {self.n / 1e6:.0f} M lines, {(time.time() - self.t0) / 60:.1f} min", flush=True)


def terciles(values):
    s = sorted(values)
    return s[len(s) // 3], s[2 * len(s) // 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--months", type=int, nargs="+", default=list(range(22, 29)))
    ap.add_argument("--buyer-months", type=int, nargs="+", default=[24, 25, 26])
    ap.add_argument("--n-authors", type=int, default=5000)
    ap.add_argument("--n-buyers", type=int, default=3000)
    ap.add_argument("--min-posts", type=int, default=4)
    ap.add_argument("--min-months", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    t0 = time.time()
    months = [m for m in a.months if archive(a.data_root, "comment", m)]
    for m in a.months:
        if m not in months:
            print(f"WARNING: comment-month-{m} not found, month skipped", flush=True)
    if not months:
        sys.exit("no comment archives found")
    posts = collections.Counter()              # root posts per author in the window
    pmonths = collections.defaultdict(set)
    votes_recv = collections.Counter()        # votes received on root posts (approx.: all votes on author's items)
    bots = set()
    buyer_posts = collections.Counter()       # (author, permlink) named in bid-bot memos
    prev = set()                              # posts of the previous month: an edit can fall into the next month
    for m in months:
        seen = set()
        p = Prog(f"comment {m}")
        bmin = None
        for line in zlines(archive(a.data_root, "comment", m)):
            p.tick()
            f = line.split(",")
            if len(f) != 5:
                continue
            if bmin is None:
                bmin = int(f[0])
            if f[1] == "" and (f[3], f[4]) not in seen and (f[3], f[4]) not in prev:
                seen.add((f[3], f[4]))
                posts[f[3]] += 1
                pmonths[f[3]].add(m)
        prev = seen
        vp = archive(a.data_root, "vote", m)
        if vp:
            daily = collections.Counter()
            p = Prog(f"vote {m}")
            for line in zlines(vp):
                p.tick()
                f = line.split(",")
                if len(f) != 5:
                    continue
                daily[(f[1], (int(f[0]) - bmin) // BPD)] += 1
                try:
                    if int(f[4]) > 0:
                        votes_recv[f[2]] += 1
                except ValueError:
                    pass
            bots |= {v for (v, d), c in daily.items() if c > BOT_DAILY_VOTES}
            del daily
        else:
            print(f"WARNING: vote-month-{m} not found: vote bots and votes received not counted for it", flush=True)
        tp = archive(a.data_root, "transfer", m)
        if tp:
            url_to = collections.Counter()
            named = []
            for line in zlines(tp):
                f = line.split(",", 4)
                if len(f) == 5 and "/@" in f[4]:
                    mm = RE_POST.search(f[4])
                    if mm:
                        url_to[f[2]] += 1
                        if m in a.buyer_months:
                            named.append((f[2], mm.group(1), mm.group(2)))
            mbots = {x for x, c in url_to.items() if c >= BOT_URL_MEMOS}
            bots |= mbots
            for to, au, pl in named:
                if to in mbots:
                    buyer_posts[(au, pl)] += 1
        else:
            print(f"WARNING: transfer-month-{m} not found: bid bots not flagged for it", flush=True)
        print(f"month {m}: {len(posts)} authors so far, {len(bots)} bots, {len(buyer_posts)} bought posts, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "bots.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["account"])
        w.writerows([b] for b in sorted(bots))
    # human authors and strata
    elig = [x for x, n in posts.items() if x not in bots and n >= a.min_posts and len(pmonths[x]) >= a.min_months]
    vpp = {x: votes_recv[x] / posts[x] for x in elig}
    c1, c2 = terciles([posts[x] for x in elig])
    v1, v2 = terciles(list(vpp.values()))
    strata = collections.defaultdict(list)
    for x in sorted(elig):
        sp = 0 if posts[x] < c1 else (1 if posts[x] < c2 else 2)
        sv = 0 if vpp[x] < v1 else (1 if vpp[x] < v2 else 2)
        strata[f"P{sp}V{sv}"].append(x)
    rnd = random.Random(a.seed)
    per = a.n_authors // 9
    chosen = {}
    for s in sorted(strata):
        pool = strata[s]
        rnd.shuffle(pool)
        for x in pool[:per]:
            chosen[x] = s
        print(f"  stratum {s}: {len(pool)} eligible, {min(per, len(pool))} sampled", flush=True)
    buyers = sorted({au for (au, pl) in buyer_posts if au not in bots and au in posts})
    rnd.shuffle(buyers)
    buyers = set(buyers[:a.n_buyers])
    for x in buyers:
        chosen.setdefault(x, "buyer")
    print(f"{len(elig)} eligible human authors (posts >= {a.min_posts}, months >= {a.min_months}); "
          f"sample {len(chosen)} incl. {len(buyers)} buyers ({sum(1 for s in chosen.values() if s == 'buyer')} buyer-only)",
          flush=True)
    with open(os.path.join(a.out_dir, "authors_sample.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["account", "stratum", "posts", "months_active", "votes_per_post", "buyer"])
        for x in sorted(chosen):
            w.writerow([x, chosen[x], posts[x], len(pmonths[x]), f"{votes_recv[x] / max(posts[x], 1):.2f}",
                        int(x in buyers)])
    # second pass: root posts of the sample
    n = 0
    with open(os.path.join(a.out_dir, "posts_sample.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["author", "permlink", "month", "block", "category", "bought"])
        prev = set()
        for m in months:
            seen = set()
            for line in zlines(archive(a.data_root, "comment", m)):
                f = line.split(",")
                if len(f) == 5 and f[1] == "" and f[3] in chosen and (f[3], f[4]) not in seen \
                        and (f[3], f[4]) not in prev:
                    seen.add((f[3], f[4]))
                    w.writerow([f[3], f[4], m, f[0], f[2], buyer_posts.get((f[3], f[4]), 0)])
                    n += 1
            prev = seen
    print(f"saved bots.csv ({len(bots)}), authors_sample.csv ({len(chosen)}), posts_sample.csv ({n} posts); "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
