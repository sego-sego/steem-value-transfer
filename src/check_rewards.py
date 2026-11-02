"""Progress and sanity check of the reward download (stage 2 of the reward data). Safe to run at any time
while the download is going: it only reads the output files.

Prints, for each output file: posts fetched out of its half of posts_fetch.csv, speed over the
last check interval and time left, share of deleted posts, share of posts returned without votes
(should be 0), share with zero rshares, median and 90th percentile of the author payout, share of
positive rshares from accounts in bots.csv, and the number of lines cut by an interruption.

Usage (from the data folder):
  python src/check_rewards.py
Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
import zlib


def amount(s):
    try:
        return float(str(s).split()[0])
    except (ValueError, IndexError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", default="posts_fetch.csv")
    ap.add_argument("--files", nargs="+", default=["rewards_hive.jsonl", "rewards_steemit.jsonl"])
    ap.add_argument("--bots", default="bots.csv")
    a = ap.parse_args()
    with open(a.posts) as fh:
        keys = [(r["author"], r["permlink"]) for r in csv.DictReader(fh)]
    n = len(a.files)
    half = [sum(1 for k in keys if zlib.crc32(f"{k[0]}/{k[1]}".encode()) % n == i) for i in range(n)]
    bots = set()
    if os.path.exists(a.bots):
        with open(a.bots) as fh:
            bots = {r["account"] for r in csv.DictReader(fh)}
    state_f = ".check_rewards_state.json"
    state = json.load(open(state_f)) if os.path.exists(state_f) else {}
    now = time.time()
    total_done = 0
    for i, f in enumerate(a.files):
        if not os.path.exists(f):
            print(f"{f}: not started")
            continue
        seen, miss, novotes, zero, cut = set(), 0, 0, 0, 0
        pay, pos_all, pos_bot = [], 0, 0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    cut += 1
                    continue
                k = (r.get("author"), r.get("permlink"))
                if k in seen:
                    continue
                seen.add(k)
                if r.get("missing"):
                    miss += 1
                    continue
                v = r.get("votes")
                if v is None:
                    novotes += 1
                    v = []
                rs = [(x[0], int(x[1])) for x in v]
                p = sum(x for _, x in rs if x > 0)
                zero += p == 0
                pos_all += p
                pos_bot += sum(x for w, x in rs if x > 0 and w in bots)
                q = amount(r.get("total_payout_value"))
                if q is not None:
                    pay.append(q)
        d = len(seen)
        total_done += d
        prev = state.get(f)
        speed = (d - prev[0]) / (now - prev[1]) if prev and now > prev[1] else None
        state[f] = [d, now]
        left = half[i] - d
        sp = f"{speed:.1f} posts/s, about {left / speed / 3600:.1f} h left" if speed and speed > 0 else \
            "speed: run again in a few minutes"
        ok = len(pay) > 0
        print(f"{f}: {d}/{half[i]} ({100 * d / max(half[i], 1):.1f}%); {sp}")
        print(f"   deleted {100 * miss / max(d, 1):.1f}%, without votes {novotes}, zero rshares "
              f"{100 * zero / max(d - miss, 1):.1f}%, cut lines {cut}")
        if ok:
            pay.sort()
            print(f"   author payout median {statistics.median(pay):.3f}, 90th pct {pay[int(0.9 * len(pay))]:.3f}; "
                  f"bot share of rshares {100 * pos_bot / max(pos_all, 1):.1f}%")
    json.dump(state, open(state_f, "w"))
    print(f"total {total_done}/{len(keys)} ({100 * total_done / max(len(keys), 1):.1f}%)")
    if os.name == "posix":
        alive = os.popen("ps aux | grep fetch_post_rewards | grep -v grep | wc -l").read().strip()
        print(f"download processes running: {alive} (should be {n} until done)")


if __name__ == "__main__":
    main()
