"""Channel M->I (bought attention): purchases, bot votes and organic response by post age.

From the SteemOps archives of the window months (transfer, vote, comment) and the post sample
(posts_fetch.csv), for every sampled root post:
  purchases  transfers to bid bots with a link to the post in the memo (bid bots: recipients of
             >= 20 such transfers in a month); buyer, bot, amount, block
  bot vote   first upvote of the paid bot on the post within --vote-days of the purchase
  refund     a transfer from the paid bot back to the buyer within --vote-days
  outcome    organic upvotes (voters not in bots.csv, not bid bots, not the author or the buyer)
             and replies (by others than the author), counted per 12-hour bin of post age, 0-7 days

Outputs:
  mi_events.csv     post-level: author, permlink, month, created_block, bought, voted, refunded,
                    event_bin (bin of the first bot vote, or of the first purchase if no vote),
                    amount_sbd, amount_steem, n_bots, bot_rshares (from the reward files if given)
  mi_bins.csv       author, permlink, bin, votes, replies, votes_new (first vote of the voter for the
                    author in the window, voter not very active), votes_loyal (voted for the author
                    before), votes_active (voters with > --active-per-day votes per active day)
                    (all sampled posts with bought = 1 and
                    up to --controls posts without purchase, random)

Usage (from the data folder):
  python src/build_mi_events.py --data-root . --posts posts_fetch.csv --bots bots.csv \
      --rewards rewards_hive.jsonl rewards_steemit.jsonl
Standard library only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import os
import random
import re
import time
import zipfile

BPD = 28800
BIN = BPD // 2                       # 12 hours
NBINS = 14                           # 7 days
RE_POST = re.compile(r"/@([a-z0-9][a-z0-9.\-]{1,15})/([a-z0-9][a-z0-9\-]*)")
BID_MEMOS = 20


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


def amount(s):
    try:
        v, cur = s.strip().split()
        return float(v), cur
    except ValueError:
        return 0.0, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--months", type=int, nargs="+", default=list(range(22, 29)))
    ap.add_argument("--posts", default="posts_fetch.csv")
    ap.add_argument("--bots", default="bots.csv")
    ap.add_argument("--rewards", nargs="*", default=[])
    ap.add_argument("--vote-days", type=float, default=5.0)
    ap.add_argument("--controls", type=int, default=60000, help="posts without purchase kept in mi_bins.csv")
    ap.add_argument("--max-comments-day", type=int, default=50, help="commenters above this on any day are bots")
    ap.add_argument("--active-per-day", type=float, default=20.0,
                    help="voters with more votes per active day are 'active' (curation-seeking type)")
    ap.add_argument("--trail-blocks", type=int, default=200, help="a vote this soon after a bid-bot vote follows it (200 = 10 min)")
    ap.add_argument("--trail-min", type=int, default=10, help="voters following bid bots this often are vote trails")
    ap.add_argument("--pool-blocks", type=int, default=2400, help="window after the purchase for sold-vote pools (2 h)")
    ap.add_argument("--trail-authors", type=int, default=3, help="trails/pools must follow purchases of this many authors")
    ap.add_argument("--pool-share", type=float, default=0.5,
                    help="trails/pools: at least this share of their votes on sampled posts follow purchases")
    ap.add_argument("--strict-share", type=float, default=0.9,
                    help="strict placebo: bots that vote with their own account on at least this share of paid posts")
    ap.add_argument("--service-share", type=float, default=0.3,
                    help="recipients that vote on less than this share of the posts they are paid for are services")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    posts = {}
    with open(a.posts) as fh:
        for r in csv.DictReader(fh):
            posts[(r["author"], r["permlink"])] = (int(r["block"]), int(r["month"]))
    with open(a.bots) as fh:
        bots = {r["account"] for r in csv.DictReader(fh)}
    months = [m for m in a.months if archive(a.data_root, "transfer", m)]
    # ---- purchases and refunds
    bidbots = set()
    purchases = collections.defaultdict(list)      # post -> [(block, buyer, bot, amount, cur)]
    back = collections.defaultdict(list)           # (bot, buyer) -> [block]
    for m in months:
        url_to = collections.Counter()
        named, out_tr = [], []
        for line in zlines(archive(a.data_root, "transfer", m)):
            f = line.split(",", 4)
            if len(f) != 5 or not f[0].isdigit():      # continuation of a multi-line memo
                continue
            b = int(f[0])
            out_tr.append((b, f[1], f[2]))
            if "/@" in f[4]:
                mm = RE_POST.search(f[4])
                if mm:
                    url_to[f[2]] += 1
                    if (mm.group(1), mm.group(2)) in posts:
                        v, cur = amount(f[3])
                        named.append((b, f[1], f[2], mm.group(1), mm.group(2), v, cur))
        mb = {x for x, c in url_to.items() if c >= BID_MEMOS}
        bidbots |= mb
        for b, frm, to, au, pl, v, cur in named:
            if to in mb:
                purchases[(au, pl)].append((b, frm, to, v, cur))
        for b, frm, to in out_tr:
            if frm in bidbots:
                back[(frm, to)].append(b)
        print(f"month {m}: {len(mb)} bid bots, {len(purchases)} sampled posts bought so far, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    excluded = bots | bidbots
    # ---- votes and replies on the sampled posts
    votes = collections.defaultdict(list)          # post -> [(block, voter)] first upvote per voter
    seenv = set()
    replies = collections.defaultdict(list)
    heavy = set()                                  # comment bots: > max-comments-day on some day
    sample_authors = {au for au, pl in posts}
    posters = set()
    first_av = {}                                  # (voter, author) -> block of the first vote in the window
    vday, vtot, vdays = collections.Counter(), collections.Counter(), collections.Counter()
    for m in months:
        vp = archive(a.data_root, "vote", m)
        if vp:
            for line in zlines(vp):
                f = line.split(",")
                if len(f) != 5 or not f[0].isdigit():
                    continue
                try:
                    w = int(f[4])
                except ValueError:
                    continue
                if w <= 0:
                    continue
                b = int(f[0])
                vday[(f[1], b // BPD)] += 1                 # activity of every voter (votes per active day)
                if f[2] in sample_authors:
                    pa = (f[1], f[2])
                    if pa not in first_av:
                        first_av[pa] = b                     # first vote of this voter for this author
                k = (f[2], f[3])
                if k not in posts:
                    continue
                if (f[1], k) not in seenv:
                    seenv.add((f[1], k))
                    votes[k].append((b, f[1]))
            for (v, d), c in vday.items():
                vtot[v] += c
                vdays[v] += 1
            vday.clear()
        cp = archive(a.data_root, "comment", m)
        if cp:
            seenc = set()
            per_day = collections.Counter()
            for line in zlines(cp):
                f = line.split(",")
                if len(f) != 5 or not f[0].isdigit():
                    continue
                if f[1] == "":
                    posters.add(f[3])                         # wrote a root post in the window: a live user
                    continue
                if (f[3], f[4]) in seenc:
                    continue
                seenc.add((f[3], f[4]))
                per_day[(f[3], int(f[0]) // BPD)] += 1
                if (f[1], f[2]) in posts:
                    replies[(f[1], f[2])].append((int(f[0]), f[3]))
            heavy |= {x for (x, d), c in per_day.items() if c > a.max_comments_day}
            del per_day
        print(f"month {m}: votes on {len(votes)} posts, replies on {len(replies)} posts, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    # ---- bot vote rshares from the reward files (dose)
    rsh = {}
    for fpath in a.rewards:
        with open(fpath, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                k = (r.get("author"), r.get("permlink"))
                if k in purchases and r.get("votes"):
                    rsh[k] = {v[0]: int(v[1]) for v in r["votes"]}
    # ---- recipients: do they vote for what they are paid? (bid bots vs other paid services)
    win = int(a.vote_days * BPD)
    paid, paid_voted = collections.Counter(), collections.Counter()
    for k, pl in purchases.items():
        vk = {v for b, v in votes.get(k, [])}
        for bot in {x[2] for x in pl}:
            paid[bot] += 1
            paid_voted[bot] += bot in vk
    service = {x for x in paid if paid_voted[x] < a.service_share * paid[x]}
    with open(os.path.join(a.out_dir, "mi_recipients.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["recipient", "paid_posts", "voted_share", "service"])
        for x, c in paid.most_common():
            w.writerow([x, c, f"{paid_voted[x] / c:.3f}", int(x in service)])
    # ---- vote trails and sold-vote pools: accounts that vote soon after the bid-bot vote, or soon
    # after the purchase itself (pools of vote sellers vote from other accounts), repeatedly and for
    # several authors (a loyal reader of one author is not a pool)
    co = collections.Counter()
    co_auth = collections.defaultdict(set)
    for k, pl in purchases.items():
        bots_paid = {x[2] for x in pl}
        buyers = {x[1] for x in pl}
        tb = [b for b, v in votes.get(k, []) if v in bots_paid]
        anchors = [(min(tb), a.trail_blocks)] if tb else []
        anchors.append((pl[0][0], a.pool_blocks))
        for b, v in votes.get(k, []):
            if v in bots_paid or v == k[0] or v in buyers:
                continue
            if any(t <= b <= t + wdt for t, wdt in anchors):
                co[v] += 1
                co_auth[v].add(k[0])
    nsv = collections.Counter(v for k in votes for b, v in votes[k])   # votes of each voter on sampled posts
    trails = {v for v, c in co.items() if c >= a.trail_min and len(co_auth[v]) >= a.trail_authors
              and c >= a.pool_share * nsv[v]}
    print(f"{len(service)} paid services that rarely vote (of {len(paid)} recipients); {len(trails)} trail and pool voters; "
          f"{len(heavy)} comment bots", flush=True)
    # ---- events
    ev = []
    nv = nr = 0
    for k, pl in purchases.items():
        created, month = posts[k]
        pl.sort()
        bots_paid = {x[2] for x in pl}
        buyers = {x[1] for x in pl}
        first = pl[0][0]
        bv = [b for b, v in votes.get(k, []) if v in bots_paid and first <= b <= first + win]
        voted = int(bool(bv))
        refunded = int(any(any(first <= bb <= first + win for bb in back.get((bot, buyer), []))
                           for bot in bots_paid for buyer in buyers))
        t_ev = min(bv) if bv else first
        e_bin = (t_ev - created) // BIN
        sbd = sum(x[3] for x in pl if x[4] == "SBD")
        stm = sum(x[3] for x in pl if x[4] == "STEEM")
        brs = sum(rsh.get(k, {}).get(bot, 0) for bot in bots_paid) if k in rsh else ""
        svc = int(bots_paid <= service)
        strict = int(min(paid_voted[x] / paid[x] for x in bots_paid) >= a.strict_share)
        ev.append([k[0], k[1], month, created, 1, voted, refunded, e_bin, f"{sbd:.3f}", f"{stm:.3f}", len(bots_paid),
                   brs, ";".join(sorted(buyers)), svc, strict])
        nv += voted
        nr += refunded
    bought = {(e[0], e[1]) for e in ev}
    rnd = random.Random(a.seed)
    ctrl = [k for k in posts if k not in bought]
    rnd.shuffle(ctrl)
    ctrl = ctrl[:a.controls]
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "mi_events.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["author", "permlink", "month", "created_block", "bought", "voted", "refunded", "event_bin",
                    "amount_sbd", "amount_steem", "n_bots", "bot_rshares", "buyers", "service", "strict_bot"])
        w.writerows(ev)
        for k in ctrl:
            w.writerow([k[0], k[1], posts[k][1], posts[k][0], 0, 0, 0, "", 0, 0, 0, "", "", 0, 0])
    buyers_of = {(e[0], e[1]): set(e[12].split(";")) for e in ev}
    n = 0
    with open(os.path.join(a.out_dir, "mi_bins.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["author", "permlink", "bin", "votes", "replies", "votes_new", "votes_loyal", "votes_active",
                    "votes_trail", "votes_new_poster", "votes_new_other"])
        for k in list(bought) + ctrl:
            created = posts[k][0]
            own = {k[0]} | buyers_of.get(k, set())
            cv, cr = [0] * NBINS, [0] * NBINS
            cn, cl, ca, ct = [0] * NBINS, [0] * NBINS, [0] * NBINS, [0] * NBINS
            cnp, cno = [0] * NBINS, [0] * NBINS
            for b, v in votes.get(k, []):
                h = (b - created) // BIN
                if 0 <= h < NBINS and v in trails and v not in excluded and v not in own:
                    ct[h] += 1                                # trail / sold-vote pool (mechanical)
                if 0 <= h < NBINS and v not in excluded and v not in trails and v not in own:
                    cv[h] += 1
                    if vtot[v] > a.active_per_day * vdays[v]:
                        ca[h] += 1                            # very active voter (curation-seeking type)
                    elif first_av.get((v, k[0])) == b:
                        cn[h] += 1                            # new reader: first vote for this author
                        if v in posters:
                            cnp[h] += 1                       # ... who also writes posts (a live user)
                        else:
                            cno[h] += 1
                    else:
                        cl[h] += 1                            # loyal reader: voted for this author before
            for b, x in replies.get(k, []):
                h = (b - created) // BIN
                if 0 <= h < NBINS and x != k[0] and x not in excluded and x not in heavy:
                    cr[h] += 1
            for h in range(NBINS):
                w.writerow([k[0], k[1], h, cv[h], cr[h], cn[h], cl[h], ca[h], ct[h], cnp[h], cno[h]])
            n += 1
    print(f"{len(bidbots)} bid bots; {len(ev)} bought posts: {nv} with the bot vote, {len(ev) - nv} without "
          f"({nr} refunded); {len(ctrl)} control posts; mi_bins.csv {n} posts x {NBINS} bins; "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
