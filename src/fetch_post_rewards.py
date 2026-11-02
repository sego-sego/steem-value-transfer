"""Reward data: rewards of the sampled posts through condenser_api.get_content.

For every post in posts_sample.csv writes one JSON line to the output: all fields of the post
except body and json_metadata (tags are kept), with the votes reduced to
[voter, rshares, percent, time]. A post that no longer exists is written as
{"author", "permlink", "missing": true} and not asked again. Resumable: rerun the same command;
posts already in the output are skipped (a line cut by an interruption is ignored and refetched).

Usage:
  python fetch_post_rewards.py --posts posts_sample.csv --out post_rewards.jsonl
  python fetch_post_rewards.py --posts posts_sample.csv --out post_rewards.jsonl --limit 200   # trial
Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import http.client
import json
import os
import threading
import time
import zlib
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

NODES = ["https://api.hive.blog", "https://api.steemit.com"]
DROP = {"body", "json_metadata", "active_votes", "replies", "body_length", "url", "root_title"}


class Rate:
    def __init__(self, per_sec):
        self.dt, self.next, self.lock = 1.0 / per_sec, time.time(), threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            if now < self.next:
                time.sleep(self.next - now)
            self.next = max(now, self.next) + self.dt


def compact(c, author, permlink):
    if not c or not c.get("author"):
        return {"author": author, "permlink": permlink, "missing": True}
    r = {k: v for k, v in c.items() if k not in DROP}
    try:
        r["tags"] = json.loads(c.get("json_metadata") or "{}").get("tags", [])
    except (ValueError, AttributeError):
        r["tags"] = []
    av = c.get("active_votes")
    r["votes"] = None if av is None else [[v.get("voter"), str(v.get("rshares")), v.get("percent"), v.get("time")]
                                          for v in av]
    return r


def opener_for(node):
    """Local test nodes are reached directly; a system proxy (VPN) would otherwise intercept them."""
    host = urllib.parse.urlparse(node).hostname or ""
    if host in ("127.0.0.1", "localhost", "::1"):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def call(node, author, permlink, rate, tries=6, opener=None, timeout=60, verbose=False, light=False):
    opener = opener or opener_for(node)
    method = "condenser_api.get_active_votes" if light else "condenser_api.get_content"
    body = json.dumps({"jsonrpc": "2.0", "method": method,
                       "params": [author, permlink], "id": 1}).encode()
    for t in range(tries):
        rate.wait()
        try:
            req = urllib.request.Request(node, data=body, headers={"Content-Type": "application/json"})
            with opener.open(req, timeout=timeout) as r:
                js = json.loads(r.read().decode())
            if "result" in js:
                if light:                      # votes only: no payout fields, a few kB per post
                    return {"author": author, "permlink": permlink, "light": True,
                            "votes": [[v.get("voter"), str(v.get("rshares")), v.get("percent"), v.get("time")]
                                      for v in js["result"]]}
                return compact(js["result"], author, permlink)
            err = js.get("error", {})
            if "deleted" in str(err.get("data", "")) or "does not exist" in str(err) or "unknown key" in str(err):
                return {"author": author, "permlink": permlink, "missing": True}
            if t == tries - 1:
                print(f"\n  API error for {author}/{permlink}: {str(err)[:200]}", flush=True)
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError,
                http.client.HTTPException, ValueError) as e:
            if t == tries - 1 or verbose:
                print(f"\n  failed {author}/{permlink} (try {t + 1}): {e}", flush=True)
        time.sleep(min(60, 2 ** t))
    return None


def already(out):
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r["author"], r["permlink"]))
                except (ValueError, KeyError):
                    pass                         # line cut by an interruption: refetched
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", default="posts_sample.csv")
    ap.add_argument("--out", default="post_rewards.jsonl")
    ap.add_argument("--node", default=NODES[0])
    ap.add_argument("--rate", type=float, default=5.0, help="requests per second")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="fetch at most this many posts (trial run)")
    ap.add_argument("--timeout", type=float, default=60, help="seconds to wait for one answer")
    ap.add_argument("--verbose", action="store_true", help="print every failed try")
    ap.add_argument("--shard", default="0/1", help="i/n: fetch only part i of n (run n copies on different nodes)")
    ap.add_argument("--light", action="store_true",
                    help="get_active_votes instead of get_content: rshares of the votes only, no payouts (small answers)")
    a = ap.parse_args()
    with open(a.posts) as fh:
        posts = [(r["author"], r["permlink"]) for r in csv.DictReader(fh)]
    si, sn = (int(x) for x in a.shard.split("/"))
    posts = [p for p in posts if zlib.crc32(f"{p[0]}/{p[1]}".encode()) % sn == si]
    done = already(a.out)
    todo = [p for p in posts if p not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(posts)} posts; already done {len(posts) - len([p for p in posts if p not in done])}; "
          f"to fetch now {len(todo)} from {a.node}", flush=True)
    # a cut last line has no newline: start the appended output on a new line
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        with open(a.out, "rb") as fh:
            fh.seek(-1, 2)
            cut = fh.read(1) != b"\n"
    else:
        cut = False
    fout = open(a.out, "a", encoding="utf-8")
    if cut:
        fout.write("\n")
    rate, lock = Rate(a.rate), threading.Lock()
    opener = opener_for(a.node)
    t0, n, nmiss, nfail = time.time(), 0, 0, 0
    last = [t0]

    def work(p):
        try:
            return p, call(a.node, p[0], p[1], rate, opener=opener, timeout=a.timeout, verbose=a.verbose,
                            light=a.light)
        except Exception as e:  # never stop the whole run on one post; it is retried on the next run
            print(f"\n  skipped {p[0]}/{p[1]}: {e}", flush=True)
            return p, None

    with ThreadPoolExecutor(a.threads) as ex:
        for p, res in ex.map(work, todo):
            with lock:
                n += 1
                if res is None:
                    nfail += 1
                else:
                    nmiss += bool(res.get("missing"))
                    fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                if n % 200 == 0 or n == len(todo) or time.time() - last[0] > 10:
                    last[0] = time.time()
                    fout.flush()
                    el = time.time() - t0
                    print(f"\r  {n}/{len(todo)} posts, {nmiss} missing, {nfail} failed (retry by rerunning), "
                          f"{el / 60:.1f} min, ETA {el / n * (len(todo) - n) / 60:.0f} min", end="", flush=True)
    fout.close()
    print(f"\nsaved {a.out}" + (f"; {nfail} posts failed: rerun the same command" if nfail else ""))


if __name__ == "__main__":
    main()
