"""Fetch per-vote rshares for a random sample of posts (Model 0).

SteemOps votes have no rshares (the vote's contribution to the payout). The API method
condenser_api.get_active_votes returns, for every vote on a post: voter, time, percent, weight,
rshares. Hive nodes serve the shared pre-2020 history.

Usage:
  python fetch_vote_rshares.py --month-dir data25 --month 25 --n-posts 20000 --out votes_rshares_month25.csv
Resumable: rerun the same command, posts already fetched are skipped.
Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import http.client
import io
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

NODES = ["https://api.hive.blog", "https://api.steemit.com"]


def zlines(path):
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            it = io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline="")
            next(it)
            for line in it:
                yield line.rstrip("\r\n")


class Rate:
    def __init__(self, per_sec):
        self.dt, self.next, self.lock = 1.0 / per_sec, time.time(), threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            if now < self.next:
                time.sleep(self.next - now)
            self.next = max(now, self.next) + self.dt


def call(node, author, permlink, rate, tries=6):
    body = json.dumps({"jsonrpc": "2.0", "method": "condenser_api.get_active_votes",
                       "params": [author, permlink], "id": 1}).encode()
    for t in range(tries):
        rate.wait()
        try:
            req = urllib.request.Request(node, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                js = json.loads(r.read().decode())
            if "result" in js:
                return js["result"]
            err = js.get("error", {})
            if "deleted" in str(err.get("data", "")) or "does not exist" in str(err):
                return []                      # deleted post: no votes, do not retry
            if t == tries - 1:
                print(f"\n  API error for {author}/{permlink}: {str(err)[:200]}", flush=True)
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError,
                http.client.HTTPException, ValueError) as e:
            if t == tries - 1:
                print(f"\n  failed {author}/{permlink}: {e}", flush=True)
        time.sleep(min(60, 2 ** t))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month-dir", required=True)
    ap.add_argument("--month", default="25")
    ap.add_argument("--n-posts", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--node", default=NODES[0])
    ap.add_argument("--rate", type=float, default=5.0, help="requests per second")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="votes_rshares_month25.csv")
    a = ap.parse_args()
    seen, posts = set(), []
    for line in zlines(os.path.join(a.month_dir, f"comment-month-{a.month}.zip")):
        f = line.split(",")
        if len(f) == 5 and f[1] == "" and (f[3], f[4]) not in seen:
            seen.add((f[3], f[4]))
            posts.append((f[3], f[4]))
    del seen
    random.Random(a.seed).shuffle(posts)
    sample = posts[:a.n_posts]
    done = set()
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        with open(a.out) as f:
            for r in csv.DictReader(f):
                done.add((r["author"], r["permlink"]))
    donef = a.out + ".done"
    if os.path.exists(donef):
        done |= {tuple(l.rstrip("\n").split("\t")) for l in open(donef) if "\t" in l}
    todo = [p for p in sample if p not in done]
    print(f"{len(posts)} posts in month; sample {len(sample)}; already done {len(sample) - len(todo)}; to fetch {len(todo)}",
          flush=True)
    new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    fout = open(a.out, "a", newline="")
    w = csv.writer(fout)
    if new:
        w.writerow(["author", "permlink", "voter", "time", "percent", "weight", "rshares"])
    fdone = open(donef, "a")
    rate, lock = Rate(a.rate), threading.Lock()
    t0, n, nv = time.time(), 0, 0

    def work(p):
        try:
            return p, call(a.node, p[0], p[1], rate)
        except Exception as e:  # never stop the whole run on one post; it is retried on the next run
            print(f"\n  skipped {p[0]}/{p[1]}: {e}", flush=True)
            return p, None

    with ThreadPoolExecutor(a.threads) as ex:
        for p, res in ex.map(work, todo):
            with lock:
                n += 1
                if res is not None:
                    for v in res:
                        w.writerow([p[0], p[1], v.get("voter"), v.get("time"), v.get("percent"), v.get("weight"),
                                    v.get("rshares")])
                        nv += 1
                    fdone.write(f"{p[0]}\t{p[1]}\n")
                if n % 200 == 0 or n == len(todo):
                    fout.flush()
                    fdone.flush()
                    el = time.time() - t0
                    eta = el / n * (len(todo) - n)
                    print(f"\r  {n}/{len(todo)} posts, {nv} votes, {el / 60:.1f} min, ETA {eta / 60:.0f} min",
                          end="", flush=True)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
