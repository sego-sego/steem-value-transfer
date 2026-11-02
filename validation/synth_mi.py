"""Synthetic data for channel M->I (bought attention), SteemOps format.

Posts of 2000 authors over months 22-28. Organic upvotes and replies arrive by post age with a
decaying profile and a post-specific quality q_p. 25% of posts are bought, preferentially the
better ones (selection on q: the post fixed effect must remove it): a transfer to a bid bot with
a link in the memo at a random age of 0-48 h; in 85% the bot votes 0.1-3 h later, otherwise it
refunds. After the bot vote the organic rate is multiplied by exp(beta_k), k = 12-h bins since
the vote (case "effect": beta = .4, .3, .2, .15, .1, .05 for k = 0..5+; case "zero": 0). Paying
without the vote has no effect. Also: vote bots voting everywhere, ordinary transfers.
Writes OUT/dataNN/{comment,vote,transfer}-month-NN.zip, posts_fetch.csv, bots.csv, truth.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import zipfile

BPD = 28800
BIN = BPD // 2
B0 = 18_500_000
MONTHS = list(range(22, 29))
MSTART = {m: B0 + (m - 22) * 30 * BPD for m in MONTHS}
BETA = {"zero": [0.0] * 6, "effect": [0.4, 0.3, 0.2, 0.15, 0.1, 0.05], "curation": [0.4, 0.3, 0.2, 0.15, 0.1, 0.05]}
# case "curation": only the very active curation-seeking voters respond to the bot vote, readers do not


def month_of(b):
    return max([m for m in MONTHS if MSTART[m] <= b], default=MONTHS[0])


def write_zip(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(os.path.basename(path).replace(".zip", ".csv"), header + "\n" + "\n".join(rows) + "\n")


def poisson(R, lam):
    if lam > 30:
        return max(0, int(round(R.gauss(lam, math.sqrt(lam)))))
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= R.random()
        if p <= L:
            return k
        k += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", choices=list(BETA), default="effect")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-posts", type=int, default=40000)
    a = ap.parse_args()
    R = random.Random(a.seed)
    beta = BETA[a.case]
    authors = [f"a{i}" for i in range(2000)]
    humans = [f"h{i}" for i in range(20000)]
    bidbots = [f"bid{i}" for i in range(8)]
    vbots = [f"vbot{i}" for i in range(10)]
    trails = [f"trail{i}" for i in range(40)]           # follow bid-bot votes within minutes (mechanical)
    svc = "resteem.svc"                                  # paid service: link memos, never votes itself
    pool = [f"pool{i}" for i in range(3000)]   # large: each pool account votes rarely, escapes the pool filter
    curators = [f"cur{i}" for i in range(30)]           # very active voters (> 20 votes a day)                # accounts voting for the service's clients
    prof = [math.exp(-0.45 * h) for h in range(14)]              # attention decays with age
    rows = {m: dict(c=[], v=[], t=[]) for m in MONTHS}
    fetch = []
    nb = nv = 0
    end = B0 + 7 * 30 * BPD - 8 * BPD
    for i in range(a.n_posts):
        au = R.choice(authors)
        created = B0 + R.randrange(end - B0)
        pl = f"p{i}"
        q = math.exp(R.gauss(0.5, 0.8))
        m = month_of(created)
        rows[m]["c"].append((created, f"{created},,c,{au},{pl}"))
        bought = R.random() < 0.15 + 0.25 * (q > 2.0)                 # selection on quality
        tv = None
        if bought:
            nb += 1
            tb = created + R.randrange(2 * BPD)
            bot = R.choice(bidbots)
            rows[month_of(tb)]["t"].append((tb, f"{tb},{au},{bot},2.000 SBD,https://steemit.com/c/@{au}/{pl}"))
            if R.random() < 0.85:
                tv = tb + R.randrange(BPD // 240, BPD // 8)
                rows[month_of(tv)]["v"].append((tv, f"{tv},{bot},{au},{pl},10000"))
                nv += 1
                for tr in R.sample(trails, R.randrange(3, 9)):          # trail votes right after the bot
                    bt = tv + R.randrange(1, 150)
                    rows[month_of(bt)]["v"].append((bt, f"{bt},{tr},{au},{pl},10000"))
                bc = tv + R.randrange(1, 100)                           # the bot's own comment
                rows[month_of(bc)]["c"].append((bc, f"{bc},{au},{pl},{bot},re-{bot}-{pl}"))
                if R.random() < 0.8:                                    # comment bot reacts to the bot vote
                    bc = tv + R.randrange(1, 300)
                    rows[month_of(bc)]["c"].append((bc, f"{bc},{au},{pl},commentbot,re-cb-{pl}"))
            else:
                rb = tb + R.randrange(BPD // 8, BPD)
                rows[month_of(rb)]["t"].append((rb, f"{rb},{bot},{au},2.000 SBD,refund: no suitable voter"))
        for h in range(14):
            mult = 1.0
            if tv is not None:
                k = h - (tv - created) // BIN
                if k >= 0:
                    mult = math.exp(beta[min(k, 5)])
            for kind, base in (("v", 2.0), ("r", 0.3), ("cur", 1.0)):
                mk = mult if (a.case != "curation" or kind == "cur") else 1.0
                n = poisson(R, q * base * prof[h] * mk)
                for _ in range(n):
                    bb = created + h * BIN + R.randrange(BIN)
                    if tv is not None and h == (tv - created) // BIN:   # within the bin of the vote: only after it
                        if bb < tv and mk > 1.0:
                            bb = tv + R.randrange(max(1, created + (h + 1) * BIN - tv))
                    if kind == "v":
                        rows[month_of(bb)]["v"].append((bb, f"{bb},{R.choice(humans)},{au},{pl},10000"))
                    elif kind == "cur":
                        rows[month_of(bb)]["v"].append((bb, f"{bb},{R.choice(curators)},{au},{pl},10000"))
                    else:
                        rows[month_of(bb)]["c"].append((bb, f"{bb},{au},{pl},{R.choice(humans)},re-{pl}-{R.randrange(10**9)}"))
        if not bought and R.random() < 0.05:                            # paid service: no vote, no effect
            ts = created + R.randrange(2 * BPD)
            rows[month_of(ts)]["t"].append((ts, f"{ts},{au},{svc},1.000 SBD,https://steemit.com/c/@{au}/{pl}"))
            for tr in R.sample(pool, R.randrange(4, 10)):               # the service sells votes from a pool
                bt = ts + R.randrange(1, 2000)
                rows[month_of(bt)]["v"].append((bt, f"{bt},{tr},{au},{pl},10000"))
        if R.random() < 0.5:                                            # comment bot everywhere (> 50 a day)
            bc = created + R.randrange(BPD)
            rows[month_of(bc)]["c"].append((bc, f"{bc},{au},{pl},commentbot,re-cb0-{pl}"))
        if R.random() < 0.3:                                             # vote bots: no organic meaning
            vb = created + R.randrange(BPD)
            rows[month_of(vb)]["v"].append((vb, f"{vb},{R.choice(vbots)},{au},{pl},10000"))
        fetch.append([au, pl, m, created, "c", 0, 0, 1, 1])
    for m in MONTHS:
        for _ in range(30):                                              # many link memos -> bid bots detected
            for bot in bidbots:
                bb = MSTART[m] + R.randrange(30 * BPD)
                rows[m]["t"].append((bb, f"{bb},{R.choice(humans)},{bot},1.000 SBD,https://steemit.com/c/@x/y{R.randrange(10**6)}"))
        d = os.path.join(a.out, f"data{m}")
        rows[m]["c"].insert(0, (MSTART[m], f"{MSTART[m]},,c,filler,start-{m}"))
        for i in range(0, len(humans), 2):                               # half of the readers also write posts
            bh = MSTART[m] + 1 + R.randrange(30 * BPD - 2)
            rows[m]["c"].append((bh, f"{bh},,c,{humans[i]},own-{m}-{i}"))
        write_zip(os.path.join(d, f"comment-month-{m}.zip"), "block_no,parent_author,parent_permlink,author,permlink",
                  [r for _, r in sorted(rows[m]["c"])])
        write_zip(os.path.join(d, f"vote-month-{m}.zip"), "block_no,voter,author,permlink,weight",
                  [r for _, r in sorted(rows[m]["v"])])
        write_zip(os.path.join(d, f"transfer-month-{m}.zip"), "block_no,from,to,amount,memo",
                  [r for _, r in sorted(rows[m]["t"])])
    with open(os.path.join(a.out, "posts_fetch.csv"), "w") as fh:
        fh.write("author,permlink,month,block,category,bought,week,n_week,weight\n")
        fh.write("".join(",".join(map(str, r)) + "\n" for r in fetch))
    with open(os.path.join(a.out, "bots.csv"), "w") as fh:
        fh.write("account\n" + "".join(f"{v}\n" for v in vbots))
    with open(os.path.join(a.out, "truth.json"), "w") as fh:
        json.dump(dict(case=a.case, beta=beta, bought=nb, voted=nv), fh)
    print(f"synthetic M->I data in {a.out}: {a.n_posts} posts, {nb} bought, {nv} with bot vote, case {a.case}")


if __name__ == "__main__":
    main()
