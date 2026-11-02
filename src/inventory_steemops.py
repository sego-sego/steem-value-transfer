"""Inventory of one SteemOps month (comment, vote, custom_json, transfer).

Reads the zip archives directly (no unpacking) and prints / saves a summary:
operation counts, malformed rows, posts vs replies, follow / unfollow / reblog
counts, reblogs per post (Model A needs posts reblogged by >= 2 accounts),
votes per voter-day (Model 0: k = daily votes), vote weights, transfers with
post-URL memos (bid bots).

Usage:
    python inventory_steemops.py data25 25
Output: inventory_month25.json and inventory_month25.txt in the data folder.
Standard library only. Runtime: ~10-30 min (the vote and custom_json files are 1.7-1.9 GB).
"""
from __future__ import annotations

import collections
import csv
import io
import json
import os
import re
import sys
import time
import zipfile

BLOCKS_PER_DAY = 28800  # 3-s blocks; approximate day index for the inventory only


def lines(path):
    """Yield text lines of the single CSV inside a zip archive."""
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            for line in io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline=""):
                yield line.rstrip("\r\n")


class Progress:
    def __init__(self, label):
        self.label, self.t0, self.n = label, time.time(), 0

    def tick(self):
        self.n += 1
        if self.n % 2_000_000 == 0:
            dt = time.time() - self.t0
            print(f"  {self.label}: {self.n / 1e6:.0f} M lines, {dt / 60:.1f} min", flush=True)


def quantiles(counter, qs=(0.5, 0.9, 0.99, 0.999)):
    """Quantiles of the values of a Counter (value -> count) given as a histogram."""
    tot = sum(counter.values())
    out, acc, keys = {}, 0, sorted(counter)
    qi = 0
    for k in keys:
        acc += counter[k]
        while qi < len(qs) and acc >= qs[qi] * tot:
            out[str(qs[qi])] = k
            qi += 1
    out["max"] = keys[-1] if keys else None
    out["mean"] = sum(k * c for k, c in counter.items()) / tot if tot else None
    return out


def do_comment(path):
    p = Progress("comment")
    it = lines(path)
    header = next(it)
    n = bad = posts = replies = 0
    authors, post_keys = set(), set()
    bmin = bmax = None
    for line in it:
        p.tick()
        n += 1
        f = line.split(",")
        if len(f) != 5:
            bad += 1
            continue
        b = int(f[0])
        bmin = b if bmin is None else min(bmin, b)
        bmax = b if bmax is None else max(bmax, b)
        authors.add(f[3])
        if f[1] == "":
            posts += 1
            post_keys.add((f[3], f[4]))
        else:
            replies += 1
    return dict(header=header, rows=n, malformed=bad, posts_ops=posts, replies_ops=replies,
                distinct_root_posts=len(post_keys), distinct_authors=len(authors),
                block_min=bmin, block_max=bmax), post_keys


def do_vote(path, b0):
    p = Progress("vote")
    it = lines(path)
    header = next(it)
    n = bad = 0
    per_vd = collections.Counter()
    weights = collections.Counter()
    voters = set()
    for line in it:
        p.tick()
        n += 1
        f = line.split(",")
        if len(f) != 5:
            bad += 1
            continue
        try:
            b, w = int(f[0]), int(f[4])
        except ValueError:
            bad += 1
            continue
        d = (b - b0) // BLOCKS_PER_DAY
        per_vd[(f[1], d)] += 1
        voters.add(f[1])
        weights[w // 1000 * 1000 if w >= 0 else -1000] += 1  # 10% bins, negative = flags
    hist = collections.Counter(per_vd.values())
    nvd = len(per_vd)
    above10 = sum(c for k, c in hist.items() if k > 10)
    return dict(header=header, rows=n, malformed=bad, distinct_voters=len(voters), voter_days=nvd,
                votes_per_voter_day=quantiles(hist),
                share_voter_days_above_10=above10 / nvd if nvd else None,
                votes_in_voter_days_above_10=sum(k * c for k, c in hist.items() if k > 10) / max(n - bad, 1),
                weight_hist_10pct_bins=dict(sorted(weights.items())))


CJ = re.compile(r'^(\d+),(\[.*?\]),([^,]*),(.*)$')


def do_custom_json(path, post_keys):
    p = Progress("custom_json")
    it = lines(path)
    header = next(it)
    n = bad = 0
    ids = collections.Counter()
    kinds = collections.Counter()
    reblogs_per_post = collections.Counter()
    rebloggers = set()
    follow_edges = 0
    for line in it:
        p.tick()
        n += 1
        m = CJ.match(line)
        if not m:
            bad += 1
            continue
        ids[m.group(3)] += 1
        if m.group(3) != "follow":
            continue
        try:
            j = json.loads(m.group(4))
        except json.JSONDecodeError:
            kinds["unparsed"] += 1
            continue
        if not (isinstance(j, list) and len(j) == 2 and isinstance(j[1], dict)):
            kinds["other_shape"] += 1
            continue
        op, d = j[0], j[1]
        if op == "follow":
            what = d.get("what") or []
            k = "follow" if "blog" in what else ("ignore" if "ignore" in what else "unfollow")
            kinds[k] += 1
            follow_edges += k == "follow"
        elif op == "reblog":
            kinds["reblog"] += 1
            key = (d.get("author"), d.get("permlink"))
            reblogs_per_post[key] += 1
            rebloggers.add(d.get("account"))
        else:
            kinds[str(op)] += 1
    hist = collections.Counter(reblogs_per_post.values())
    in_month = sum(1 for k in reblogs_per_post if k in post_keys)
    ge2 = [k for k, c in reblogs_per_post.items() if c >= 2]
    return dict(header=header, rows=n, malformed=bad, ids_top=ids.most_common(15),
                follow_kinds=dict(kinds), distinct_reblogged_posts=len(reblogs_per_post),
                reblogged_posts_created_this_month=in_month,
                posts_with_ge2_reblogs=len(ge2),
                posts_with_ge2_reblogs_created_this_month=sum(1 for k in ge2 if k in post_keys),
                reblog_pairs_on_ge2_posts=sum(reblogs_per_post[k] for k in ge2),
                reblogs_per_post=quantiles(hist), distinct_rebloggers=len(rebloggers))


def do_transfer(path):
    p = Progress("transfer")
    it = lines(path)
    header = next(it)
    n = bad = 0
    cur = collections.Counter()
    url_to = collections.Counter()
    url = 0
    for line in it:
        p.tick()
        n += 1
        f = line.split(",", 4)  # memo may contain commas
        if len(f) < 4:
            bad += 1
            continue
        parts = f[3].split()
        if len(parts) == 2:
            cur[parts[1]] += 1
        memo = f[4] if len(f) == 5 else ""
        if "steemit.com/" in memo and "/@" in memo:
            url += 1
            url_to[f[2]] += 1
    return dict(header=header, rows=n, malformed=bad, currency=dict(cur), url_memo_transfers=url,
                url_memo_share=url / max(n, 1), top_url_memo_recipients=url_to.most_common(25))


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    m = sys.argv[2] if len(sys.argv) > 2 else "25"
    zp = lambda op: os.path.join(folder, f"{op}-month-{m}.zip")
    res = {}
    t0 = time.time()
    print("comment ...", flush=True)
    res["comment"], post_keys = do_comment(zp("comment"))
    b0 = res["comment"]["block_min"]
    print("custom_json ...", flush=True)
    res["custom_json"] = do_custom_json(zp("custom_json"), post_keys)
    del post_keys
    print("vote ...", flush=True)
    res["vote"] = do_vote(zp("vote"), b0)
    print("transfer ...", flush=True)
    res["transfer"] = do_transfer(zp("transfer"))
    res["approx_days"] = (res["comment"]["block_max"] - b0) / BLOCKS_PER_DAY
    res["runtime_min"] = (time.time() - t0) / 60
    out = os.path.join(folder, f"inventory_month{m}")
    json.dump(res, open(out + ".json", "w"), indent=1, default=str)
    txt = json.dumps(res, indent=1, default=str)
    open(out + ".txt", "w").write(txt)
    print(txt)
    print(f"\nsaved {out}.json  ({res['runtime_min']:.1f} min)")


if __name__ == "__main__":
    main()
