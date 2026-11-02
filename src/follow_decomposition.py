"""Decompose a reader's number of active sources into her own follow decisions and the activity
of the sources (test of the asymmetry in the within-reader panel).

For reader x at the start of month m:
  S_xm   = number of followed accounts in the universe U (accounts that publish or reblog in any
           of the months analysed); changes only when x follows or unfollows
  n_xm   = number of those that publish or reblog in month m (active sources)
  A_xm   = n_xm / S_xm, the active share; changes when followed accounts fall silent or wake up
so that ln n = ln S + ln A. The reader's own decisions move S, the sources' activity moves A.

Output: follow_decomposition.csv with account, month, S, n_active, follows_in_month,
unfollows_in_month, for the readers listed in the *_followers.csv files.

Usage (from the data folder):
  python follow_decomposition.py --months 24 25 26 --month-dirs data24 data25 data26 \
      --hist-dir custom_json_hist \
      --followers data24/pairs_v2_month24_followers.csv pairs_v2_month25_followers.csv data26/pairs_v2_month26_followers.csv
Standard library only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import os
import re
import sys
import time
import zipfile

RE_CJ = re.compile(r'^(\d+),(\[.*?\]),([^,]*),(.*)$')
RE_FOLLOWER = re.compile(r'"follower"\s*:\s*"([^"]*)"')
RE_FOLLOWING = re.compile(r'"following"\s*:\s*"([^"]*)"')
RE_WHAT = re.compile(r'"what"\s*:\s*\[([^\]]*)\]')
RE_ACCOUNT = re.compile(r'"account"\s*:\s*"([^"]*)"')


def zlines(path):
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            it = io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline="")
            next(it)
            for line in it:
                yield line.rstrip("\r\n")


def find(dirs, op, m):
    for d in dirs:
        for name in (f"{op}-month-{m:02d}.zip", f"{op}-month-{m}.zip"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, nargs="+", required=True)
    ap.add_argument("--month-dirs", nargs="+", required=True)
    ap.add_argument("--hist-dir", required=True)
    ap.add_argument("--followers", nargs="+", required=True)
    ap.add_argument("--out", default="follow_decomposition.csv")
    a = ap.parse_args()
    t0 = time.time()
    readers = set()
    for f in a.followers:
        with open(f) as fh:
            readers |= {r["account"] for r in csv.DictReader(fh)}
    print(f"{len(readers)} readers", flush=True)
    # active producers per month: authors of root posts and rebloggers
    active = {}
    for m, d in zip(a.months, a.month_dirs):
        act = set()
        for line in zlines(find([d], "comment", m)):
            f = line.split(",")
            if len(f) == 5 and f[1] == "":
                act.add(f[3])
        for line in zlines(find([d], "custom_json", m)):
            if '"reblog"' in line:
                mm = RE_ACCOUNT.search(line)
                if mm:
                    act.add(mm.group(1))
        active[m] = act
        print(f"month {m}: {len(act)} active producers", flush=True)
    U = set().union(*active.values())
    uid = {u: i for i, u in enumerate(sorted(U))}
    rid = {r: i for i, r in enumerate(sorted(readers))}
    rname = sorted(readers)
    act_ids = {m: {uid[u] for u in s} for m, s in active.items()}
    edges = set()
    follows = collections.Counter()
    unfollows = collections.Counter()
    rows = []
    last = max(a.months)
    for m in range(1, last):
        path = find([a.hist_dir] + a.month_dirs, "custom_json", m)
        if path is None:
            sys.exit(f"missing custom_json-month-{m:02d}.zip")
        if m + 1 in a.months:
            follows.clear()
            unfollows.clear()
        n = 0
        for line in zlines(path):
            n += 1
            if '"follower"' not in line:
                continue
            mm = RE_CJ.match(line)
            if not mm or mm.group(3) != "follow":
                continue
            js = mm.group(4)
            fr, fg, wh = RE_FOLLOWER.search(js), RE_FOLLOWING.search(js), RE_WHAT.search(js)
            if not (fr and fg):
                continue
            x, t = rid.get(fr.group(1)), uid.get(fg.group(1))
            if x is None or t is None:
                continue
            e = (x << 22) | t
            if wh and '"blog"' in wh.group(1):
                if e not in edges:
                    edges.add(e)
                    if m + 1 in a.months:
                        follows[x] += 1
            elif e in edges:
                edges.discard(e)
                if m + 1 in a.months:
                    unfollows[x] += 1
        print(f"  custom_json month {m}: {n / 1e6:.1f} M lines, {len(edges) / 1e6:.2f} M edges, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if m + 1 in a.months:
            # snapshot at the start of month m+1
            mm_ = m + 1
            S = collections.Counter()
            nA = collections.Counter()
            act_set = act_ids[mm_]
            mask = (1 << 22) - 1
            for e in edges:
                x, t = e >> 22, e & mask
                S[x] += 1
                if t in act_set:
                    nA[x] += 1
            for x, s in S.items():
                rows.append((rname[x], mm_, s, nA.get(x, 0), follows.get(x, 0), unfollows.get(x, 0)))
            print(f"snapshot start of month {mm_}: {len(S)} readers with followees", flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["account", "month", "S", "n_active", "follows_prev_month", "unfollows_prev_month"])
        w.writerows(rows)
    print(f"saved {a.out}: {len(rows)} rows, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
