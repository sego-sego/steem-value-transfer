"""Flatten post_rewards.jsonl into post_rewards.csv (one row per post) for the panels.

Columns: author, permlink, created, last_payout, payout_author_sbd (total_payout_value),
payout_curator_sbd (curator_payout_value), payout_total_sbd, net_rshares (as returned; may be
reset to 0 after payout), rshares_pos, rshares_neg (sums over the votes), n_votes, n_votes_pos,
rshares_bot, n_bot_votes (votes of accounts in bots.csv), rshares_organic (positive rshares of
non-bot voters other than the author), self_rshares, children, max_accepted_sbd,
percent_sbd, beneficiaries_pct, declined (max accepted payout 0), missing.

Usage:
  python parse_post_rewards.py --in post_rewards.jsonl --bots bots.csv --out post_rewards.csv
  python parse_post_rewards.py --in rewards_hive.jsonl rewards_steemit.jsonl --bots bots.csv --out post_rewards.csv
Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json


def amount(s):
    """'1.234 SBD' -> 1.234; numbers and missing values -> float or ''."""
    if s is None:
        return ""
    if isinstance(s, (int, float)):
        return float(s)
    if isinstance(s, dict):                      # {"amount": "1234", "precision": 3, ...}
        return int(s.get("amount", 0)) / 10 ** int(s.get("precision", 3))
    try:
        return float(str(s).split()[0])
    except (ValueError, IndexError):
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", default=["post_rewards.jsonl"])
    ap.add_argument("--bots", default="bots.csv")
    ap.add_argument("--out", default="post_rewards.csv")
    a = ap.parse_args()
    with open(a.bots) as fh:
        bots = {r["account"] for r in csv.DictReader(fh)}
    cols = ["author", "permlink", "created", "last_payout", "payout_author_sbd", "payout_curator_sbd",
            "payout_total_sbd", "net_rshares", "rshares_pos", "rshares_neg", "n_votes", "n_votes_pos",
            "rshares_bot", "n_bot_votes", "rshares_organic", "self_rshares", "children", "max_accepted_sbd",
            "percent_sbd", "beneficiaries_pct", "declined", "votes_returned", "missing"]
    seen, n, nmiss, nnov = set(), 0, 0, 0
    def lines():
        for f in a.inp:
            with open(f, encoding="utf-8") as fin:
                yield from fin

    with open(a.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, cols)
        w.writeheader()
        for line in lines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            key = (r.get("author"), r.get("permlink"))
            if key in seen:
                continue
            seen.add(key)
            n += 1
            if r.get("missing"):
                nmiss += 1
                w.writerow({"author": key[0], "permlink": key[1], "missing": 1})
                continue
            pos = neg = bot = org = selfr = 0
            npos = nbot = 0
            votes = r.get("votes")
            if votes is None:
                nnov += 1
                votes = []
            for voter, rs, pct, t in votes:
                rs = int(rs)
                if rs > 0:
                    pos += rs
                    npos += 1
                    if voter in bots:
                        bot += rs
                        nbot += 1
                    elif voter == key[0]:
                        selfr += rs
                    else:
                        org += rs
                elif rs < 0:
                    neg += rs
            pa, pc = amount(r.get("total_payout_value")), amount(r.get("curator_payout_value"))
            mx = amount(r.get("max_accepted_payout"))
            ben = sum(int(b.get("weight", 0)) for b in r.get("beneficiaries") or []) / 100.0
            w.writerow({
                "author": key[0], "permlink": key[1], "created": r.get("created", ""),
                "last_payout": r.get("last_payout", ""), "payout_author_sbd": pa, "payout_curator_sbd": pc,
                "payout_total_sbd": (pa + pc) if pa != "" and pc != "" else "",
                "net_rshares": r.get("net_rshares", ""), "rshares_pos": pos, "rshares_neg": neg,
                "n_votes": len(votes), "n_votes_pos": npos, "rshares_bot": bot, "n_bot_votes": nbot,
                "rshares_organic": org, "self_rshares": selfr, "children": r.get("children", ""),
                "max_accepted_sbd": mx, "percent_sbd": r.get("percent_steem_dollars", r.get("percent_hbd", "")),
                "beneficiaries_pct": ben, "declined": int(mx == 0.0), "votes_returned": int(r.get("votes") is not None),
                "missing": 0})
    print(f"saved {a.out}: {n} posts, {nmiss} missing, {nnov} without votes in the response")


if __name__ == "__main__":
    main()
