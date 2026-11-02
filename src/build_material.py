"""Material layer: material layer and Result 1 of the PRL (frozen pattern set by unequal row sums).

Per month and account, from the SteemOps transfer archives and the vote rshares of the reward files:
  O_u    outflow of u: STEEM + --sbd-rate x SBD sent to others (self-transfers excluded)
  I_v    inflow of v
  R_v    row sum of the column-stochastic transfer operator, R_v = sum_u T_{u->v} / O_u
         (the share of other accounts' outflows that v collects; s_v = R_v - 1 in the PRL)
  nin    number of distinct senders
  stake  median value of a full vote, rshares x 10^4 / percent, over the account's positive votes
         in the month (proportional to its effective Steem Power, own plus delegated, times its
         voting power, 0.8-1.0)
  kind   exchange (>= --exch-deg distinct senders and receivers in the month, or in --exchanges),
         bidbot (recipient of >= 20 transfers with a post link in the month), bot (bots.csv),
         service (>= --service-nin distinct senders: games, shops, exchangers), user

Output: material_panel.csv (account, month, kind, O, I, R, nin, nout, stake, nvotes).

Usage (from the data folder):
  python src/build_material.py --data-root . --rewards rewards_hive.jsonl rewards_steemit.jsonl --bots bots.csv
Standard library only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import os
import re
import statistics
import time
import zipfile

RE_POST = re.compile(r"/@([a-z0-9][a-z0-9.\-]{1,15})/([a-z0-9][a-z0-9\-]*)")
KNOWN_EXCHANGES = {"bittrex", "poloniex", "blocktrades", "upbit", "binance-hot", "huobi-pro", "gopax", "openledger",
                   "changelly", "coinpayments", "freewallet.org", "hitbtc-exchange", "deepcrypto8", "steemexchanger",
                   "rudex", "bitsharesdex", "gopax-deposit", "upbitsteem", "korbit3", "probitsteem", "orinoco",
                   "user.dunamu", "coinbase", "kraken", "bithumb.sunshine", "mxchive"}


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


def month_of_time(t):
    """'2018-03-05T...' -> SteemOps month index (March 2016 = 1)."""
    return (int(t[:4]) - 2016) * 12 + int(t[5:7]) - 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--months", type=int, nargs="+", default=list(range(22, 29)))
    ap.add_argument("--rewards", nargs="*", default=[])
    ap.add_argument("--bots", default="bots.csv")
    ap.add_argument("--sbd-rate", type=float, default=1.0, help="STEEM per SBD (assumption; default 1)")
    ap.add_argument("--exch-deg", type=int, default=1000)
    ap.add_argument("--service-nin", type=int, default=200, help="accounts paid by this many senders in a month are services")
    ap.add_argument("--min-votes", type=int, default=3, help="votes in the month needed for a stake value")
    ap.add_argument("--out", default="material_panel.csv")
    a = ap.parse_args()
    t0 = time.time()
    bots = set()
    if os.path.exists(a.bots):
        with open(a.bots) as fh:
            bots = {r["account"] for r in csv.DictReader(fh)}
    # ---- stake from the vote rshares
    full = collections.defaultdict(list)            # (voter, month) -> [rshares per full vote]
    for f in a.rewards:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                for v in r.get("votes") or []:
                    try:
                        rs, pct, tm = int(v[1]), int(v[2]), v[3]
                    except (TypeError, ValueError, IndexError):
                        continue
                    if rs > 0 and pct > 0 and tm:
                        full[(v[0], month_of_time(tm))].append(rs * 1e4 / pct)
    stake = {k: (statistics.median(x), len(x)) for k, x in full.items() if len(x) >= a.min_votes}
    del full
    print(f"stake values: {len(stake)} account-months, {(time.time() - t0) / 60:.1f} min", flush=True)
    rows = []
    for m in a.months:
        p = archive(a.data_root, "transfer", m)
        if not p:
            print(f"WARNING: transfer-month-{m} missing", flush=True)
            continue
        T = collections.Counter()
        links = collections.Counter()
        for line in zlines(p):
            f = line.split(",", 4)
            if len(f) != 5 or not f[0].isdigit() or f[1] == f[2]:
                continue
            try:
                v, cur = f[3].split()
                v = float(v)
            except ValueError:
                continue
            amt = v * (a.sbd_rate if cur == "SBD" else 1.0 if cur == "STEEM" else 0.0)
            if amt <= 0:
                continue
            T[(f[1], f[2])] += amt
            if RE_POST.search(f[4]):
                links[f[2]] += 1
        O, I, R = collections.Counter(), collections.Counter(), collections.Counter()
        nin, nout = collections.Counter(), collections.Counter()
        for (u, v), x in T.items():
            O[u] += x
            I[v] += x
            nin[v] += 1
            nout[u] += 1
        for (u, v), x in T.items():
            R[v] += x / O[u]
        exch = {x for x in set(nin) | set(nout) if (nin[x] >= a.exch_deg and nout[x] >= a.exch_deg)} | KNOWN_EXCHANGES
        bidbots = {x for x, c in links.items() if c >= 20}
        accts = set(O) | set(I) | {v for (v, mm) in stake if mm == m}
        for x in accts:
            kind = ("exchange" if x in exch else "bidbot" if x in bidbots else "bot" if x in bots
                    else "service" if nin[x] >= a.service_nin else "user")
            st = stake.get((x, m))
            rows.append([x, m, kind, f"{O[x]:.3f}", f"{I[x]:.3f}", f"{R[x]:.5f}", nin[x], nout[x],
                         f"{st[0]:.4g}" if st else "", st[1] if st else 0])
        print(f"month {m}: {len(T)} sender-recipient pairs, {len(O)} senders, {len(I)} recipients, "
              f"{len(exch & set(nin))} exchanges, {len(bidbots)} bid bots; {(time.time() - t0) / 60:.1f} min", flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["account", "month", "kind", "O", "I", "R", "nin", "nout", "stake", "nvotes"])
        w.writerows(rows)
    print(f"saved {a.out}: {len(rows)} account-months")


if __name__ == "__main__":
    main()
