"""Synthetic test for fit_votes_model0.py: voters with protocol-like voting power.

Power regenerates 20%/day (capped at 1); a vote with percent w uses 2% * P * w/10000 of power;
rshares = SP * P * w / 10000 * 1e9. Writes vote-month-25.zip (SteemOps format) and a rshares CSV
(API format) for a random subset of votes.
"""
from __future__ import annotations

import argparse
import os
import random
import zipfile
from datetime import datetime, timedelta, timezone

BPD = 28800


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synth_votes")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    R = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    b0 = 20_279_029
    t0 = datetime(2018, 3, 1, tzinfo=timezone.utc)
    lines = ["block_no,voter,author,permlink,weight"]
    rs = ["author,permlink,voter,time,percent,weight,rshares"]
    events = []
    for i in range(1500):
        v = f"voter{i}"
        sp = R.lognormvariate(0, 1.5)
        base = R.choice([2, 4, 6, 10, 15, 25, 40, 70])
        P, last = 1.0, 0.0
        t = R.uniform(0, 0.5)
        while t < 31:
            # day-to-day variation of the voting rate
            rate = base * R.uniform(0.3, 1.7)
            t += R.expovariate(rate)
            if t >= 31:
                break
            P = min(1.0, P + 0.2 * (t - last))
            last = t
            w = R.choice([10000, 10000, 10000, 5000, 2000, 1000])
            rsh = sp * P * w / 10000 * 1e9 * R.lognormvariate(0, 0.05)
            P -= 0.02 * P * w / 10000
            b = b0 + int(t * BPD)
            pl = f"p{i}-{b}"
            events.append((b, f"{b},{v},auth,{pl},{w}"))
            if R.random() < 0.3:
                ts = (t0 + timedelta(seconds=(b - b0) * 3)).strftime("%Y-%m-%dT%H:%M:%S")
                rs.append(f"auth,{pl},{v},{ts},{w},0,{int(rsh)}")
    events.sort()
    lines += [s for _, s in events]
    with zipfile.ZipFile(os.path.join(a.out, "vote-month-25.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("vote-month-25.csv", "\n".join(lines) + "\n")
    open(os.path.join(a.out, "votes_rshares_month25.csv"), "w").write("\n".join(rs) + "\n")
    print(f"{len(events)} votes, {len(rs) - 1} with rshares")


if __name__ == "__main__":
    main()
