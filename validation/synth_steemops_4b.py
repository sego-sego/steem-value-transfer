"""Synthetic SteemOps archives (comment, vote, transfer) for months 22-28 with known bots and
buyers, to test sample_authors.py. Writes OUT/dataNN/*-month-NN.zip and OUT/truth.json.

Truth: 30 vote bots (one day with 450-600 votes), 6 bid bots (>= 25 memo transfers each in every
month), 2400 human authors with known number of posts and active months, buyers = humans whose
posts are named in memos to bid bots in months 24-26.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import zipfile

BPD = 28800
MONTH0 = {m: 18_500_000 + (m - 22) * 30 * BPD for m in range(22, 29)}


def write_zip(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(os.path.basename(path).replace(".zip", ".csv"), header + "\n" + "\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = random.Random(5)
    humans = [f"h{i:04d}" for i in range(2400)]
    vbots = [f"vbot{i}" for i in range(30)]
    bidbots = [f"bid{i}" for i in range(6)]
    rate = {h: R.choice([0.3, 1, 3, 8]) for h in humans}          # posts per month
    posts_n, months_act, buyer_true = {h: 0 for h in humans}, {h: set() for h in humans}, set()
    bought = {}
    late_edits = []
    for m in range(22, 29):
        b0 = MONTH0[m]
        comments, votes, transfers, mposts = [], [], [], []
        for (mm, h, pl) in [e for e in late_edits if e[0] == m]:
            comments.append((b0 + 10, f"{b0 + 10},,cat0,{h},{pl}"))
        for h in humans:
            k = sum(R.random() < rate[h] / 10 for _ in range(10))
            for j in range(k):
                b = b0 + R.randrange(30 * BPD)
                pl = f"p{m}-{j}-{R.randrange(10 ** 6)}"
                comments.append((b, f"{b},,cat{R.randrange(10)},{h},{pl}"))
                comments.append((b + 50, f"{b + 50},,cat0,{h},{pl}"))        # an edit: same post again
                mposts.append((h, pl, b))
                if R.random() < 0.1 and m < 28:                                  # edit falling into next month
                    late_edits.append((m + 1, h, pl))
            posts_n[h] += k
            if k:
                months_act[h].add(m)
        for (h, pl, b) in mposts:
            for _ in range(R.randrange(1, 12)):
                v = R.choice(humans)
                votes.append((b + 100, f"{b + 100},{v},{h},{pl},{R.choice([10000, 5000, 2500])}"))
        for vb in vbots:                     # one day with > 400 votes
            d = R.randrange(25)
            for _ in range(R.randrange(450, 600)):
                h, pl, b = R.choice(mposts)
                bb = b0 + d * BPD + R.randrange(BPD)
                votes.append((bb, f"{bb},{vb},{h},{pl},10000"))
        for bb_ in bidbots:
            for _ in range(R.randrange(25, 40)):
                h, pl, b = R.choice(mposts)
                bb = b + 200
                transfers.append((bb, f"{bb},{h},{bb_},2.000 SBD,https://steemit.com/cat/@{h}/{pl}"))
                if m in (24, 25, 26):
                    buyer_true.add(h)
                    bought[f"{h}/{pl}"] = bought.get(f"{h}/{pl}", 0) + 1
        for _ in range(500):                 # ordinary transfers, some with a link to fewer than 20 recipients
            x, y = R.sample(humans, 2)
            bb = b0 + R.randrange(30 * BPD)
            memo = f"thanks see https://steemit.com/c/@{x}/whatever" if R.random() < 0.02 else "gift, with comma"
            transfers.append((bb, f"{bb},{x},{y},1.000 STEEM,{memo}"))
        d = os.path.join(a.out, f"data{m}")
        write_zip(os.path.join(d, f"comment-month-{m}.zip"), "block_no,parent_author,parent_permlink,author,permlink",
                  [r for _, r in sorted(comments)])
        write_zip(os.path.join(d, f"vote-month-{m}.zip"), "block_no,voter,author,permlink,weight",
                  [r for _, r in sorted(votes)])
        write_zip(os.path.join(d, f"transfer-month-{m}.zip"), "block_no,from,to,amount,memo",
                  [r for _, r in sorted(transfers)])
    truth = dict(bots=sorted(vbots + bidbots), posts=posts_n, months={h: len(s) for h, s in months_act.items()},
                 buyers=sorted(buyer_true), bought=bought)
    with open(os.path.join(a.out, "truth.json"), "w") as fh:
        json.dump(truth, fh)
    print(f"synthetic archives in {a.out}: {len(humans)} humans, {len(vbots)} vote bots, {len(bidbots)} bid bots")


if __name__ == "__main__":
    main()
