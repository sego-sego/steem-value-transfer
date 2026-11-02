"""Thin the post list for the reward download (stage 2 of the reward data).

The panel of model C needs, per author and week, the number of new posts P (from SteemOps, all
posts) and the reward per new post R/P. The mean reward per post is estimated without bias from
a random subsample of the week's posts, so for every author-week we keep all posts named in
bid-bot memos (needed for M->I) and at most --per-week of the others, chosen at random. Each kept
post carries its weight: posts of its author-week and class / kept posts of that author-week and
class.

Output: posts_fetch.csv (author, permlink, month, block, category, bought, week, n_week, weight)
and the projected download time.

Usage:
  python thin_posts.py --posts posts_sample.csv --per-week 3 --out posts_fetch.csv
Standard library only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import random

BPD = 28800


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", default="posts_sample.csv")
    ap.add_argument("--per-week", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--out", default="posts_fetch.csv")
    ap.add_argument("--rate", type=float, default=10.0, help="total requests per second, for the time estimate")
    a = ap.parse_args()
    with open(a.posts) as fh:
        rows = list(csv.DictReader(fh))
    b0 = min(int(r["block"]) for r in rows)
    groups = collections.defaultdict(list)
    for r in rows:
        r["week"] = (int(r["block"]) - b0) // (7 * BPD)
        groups[(r["author"], r["week"], int(r["bought"]) > 0)].append(r)
    rnd = random.Random(a.seed)
    out = []
    nweek = collections.Counter((r["author"], r["week"]) for r in rows)
    for (au, wk, bought), g in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        keep = g if bought else rnd.sample(g, min(len(g), a.per_week))
        for r in keep:
            r["n_week"] = nweek[(au, wk)]
            r["weight"] = f"{len(g) / len(keep):.4f}"
            out.append(r)
    cols = ["author", "permlink", "month", "block", "category", "bought", "week", "n_week", "weight"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(out, key=lambda r: int(r["block"])))
    nb = sum(int(r["bought"]) > 0 for r in out)
    print(f"{len(rows)} posts of {len({r['author'] for r in rows})} authors in {len(nweek)} author-weeks -> "
          f"{len(out)} to fetch ({nb} bought, all kept); at {a.rate:g} requests/s about {len(out) / a.rate / 3600:.1f} h")


if __name__ == "__main__":
    main()
