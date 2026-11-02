"""Reward data: check API access and list the fields we need for rewards.

Calls condenser_api.get_content for one post of March 2018 and condenser_api.get_account_history
for its author (10 operations), on each node, and prints the fields with short values: payout
fields, rshares, whether the votes (active_votes) come with the post, and whether net_rshares is
reset to zero after payout. Send the printed output back; the reward parser is set by it.

Usage:
  python probe_api.py
  python probe_api.py --author fotosedjarah --permlink infrastruktur-pembangkit-listrik-di-masa-kolonial-1930
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import urllib.request

NODES = ["https://api.hive.blog", "https://api.steemit.com"]
WANT = ["author", "permlink", "category", "created", "last_payout", "cashout_time", "net_rshares",
        "abs_rshares", "vote_rshares", "total_payout_value", "curator_payout_value", "author_rewards",
        "pending_payout_value", "total_pending_payout_value", "max_accepted_payout",
        "percent_steem_dollars", "percent_hbd", "net_votes", "children", "beneficiaries", "allow_votes"]


def rpc(node, method, params):
    body = json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": 1}).encode()
    req = urllib.request.Request(node, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def short(v, n=90):
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + f"... ({len(s)} chars)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="fotosedjarah")
    ap.add_argument("--permlink", default="infrastruktur-pembangkit-listrik-di-masa-kolonial-1930")
    a = ap.parse_args()
    for node in NODES:
        print(f"\n===== {node}")
        try:
            js = rpc(node, "condenser_api.get_content", [a.author, a.permlink])
        except Exception as e:  # report and go on to the next node
            print(f"get_content failed: {e}")
            continue
        if "result" not in js:
            print(f"get_content error: {short(js.get('error'), 300)}")
            continue
        c = js["result"]
        print(f"get_content: {len(c)} fields: {', '.join(sorted(c))}")
        print("-- fields for rewards (MISSING = not returned):")
        for k in WANT:
            print(f"  {k:28s} {short(c[k]) if k in c else 'MISSING'}")
        av = c.get("active_votes")
        if isinstance(av, list):
            rs = [int(v.get("rshares", 0)) for v in av]
            print(f"-- active_votes: {len(av)} votes; keys {sorted(av[0]) if av else []}; "
                  f"sum of positive rshares {sum(r for r in rs if r > 0)}, negative {sum(r for r in rs if r < 0)}; "
                  f"rshares type {type(av[0].get('rshares')).__name__ if av else '-'}")
        else:
            print("-- active_votes: MISSING")
        print(f"-- net_rshares after payout: {c.get('net_rshares')!r} (0 means reset at payout)")
        try:
            h = rpc(node, "condenser_api.get_account_history", [a.author, -1, 10])
            if "result" in h:
                ops = h["result"]
                print(f"get_account_history: {len(ops)} entries; first: {short(ops[0], 300) if ops else '-'}")
                print(f"  operation types: {sorted({o[1]['op'][0] if isinstance(o[1]['op'], list) else o[1]['op'].get('type') for o in ops})}")
            else:
                print(f"get_account_history error: {short(h.get('error'), 300)}")
        except Exception as e:
            print(f"get_account_history failed: {e}")


if __name__ == "__main__":
    main()
