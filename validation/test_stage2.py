"""End-to-end test of stage 2 on synthetic data and a mock API (no network).

1. synthetic SteemOps archives -> sample_authors.py; checks bots, eligibility, strata, buyers.
2. mock condenser_api.get_content server (HTTP errors 10%, JSON-RPC errors 5%, 2% deleted posts,
   rshares as strings and numbers) -> fetch_post_rewards.py in two runs with a cut line between
   them (interruption) -> parse_post_rewards.py; checks completeness, no duplicates, sums of
   rshares, bot and organic parts, payouts.

Usage: python validation/test_stage2.py   (from the repository root)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
OK = True


def check(cond, msg):
    global OK
    print(("  ok    " if cond else "  FAIL  ") + msg)
    OK &= bool(cond)


def run(*args, show=False):
    env = dict(os.environ, NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    if show:                                   # progress of the download is shown on screen
        r = subprocess.run([sys.executable, *args], env=env)
        if r.returncode:
            sys.exit("command failed: " + " ".join(args))
        return ""
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True, env=env)
    if r.returncode:
        print(r.stdout, r.stderr)
        sys.exit("command failed: " + " ".join(args))
    return r.stdout


def fake_post(author, permlink, bots):
    h = int(hashlib.md5(f"{author}/{permlink}".encode()).hexdigest(), 16)
    R = random.Random(h)
    if R.random() < 0.02:
        return None
    votes = []
    for i in range(R.randrange(0, 15)):
        voter = R.choice([author, R.choice(bots), f"x{R.randrange(50)}", f"x{R.randrange(50)}"])
        rs = R.choice([1, 1, 1, -1]) * R.randrange(10 ** 8, 10 ** 11)
        votes.append({"voter": voter, "rshares": rs if R.random() < 0.5 else str(rs), "percent": 10000,
                      "time": "2018-03-01T00:00:00", "weight": 1, "reputation": 0})
    pay = R.randrange(0, 50000) / 1000
    return {"author": author, "permlink": permlink, "category": "c", "created": "2018-03-01T00:00:00",
            "last_payout": "2018-03-08T00:00:00", "net_rshares": 0, "total_payout_value": f"{pay:.3f} SBD",
            "curator_payout_value": f"{pay / 3:.3f} SBD", "max_accepted_payout": "1000000.000 SBD",
            "percent_steem_dollars": 10000, "children": R.randrange(5), "beneficiaries": [],
            "body": "x" * 2000, "json_metadata": json.dumps({"tags": ["c", "d"]}), "active_votes": votes}


def serve(bots):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            R = random.random()
            if R < 0.10:
                self.send_response(503)
                self.end_headers()
                return
            if R < 0.15:
                out = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "busy"}}
            else:
                au, pl = req["params"]
                c = fake_post(au, pl, bots)
                if req["method"] == "condenser_api.get_active_votes":
                    out = {"jsonrpc": "2.0", "id": 1, "result": c["active_votes"] if c else []}
                else:
                    out = {"jsonrpc": "2.0", "id": 1, "result": c if c else
                           {"author": "", "permlink": "", "active_votes": []}}
            b = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    tmp = tempfile.mkdtemp(prefix="test4b_")
    print(f"=== 1. sample_authors.py on synthetic archives ({tmp})")
    run(os.path.join(HERE, "synth_steemops_4b.py"), "--out", tmp)
    out = run(os.path.join(SRC, "sample_authors.py"), "--data-root", tmp, "--n-authors", "900", "--n-buyers", "100",
              "--out-dir", tmp)
    print("  " + out.strip().splitlines()[-1])
    truth = json.load(open(os.path.join(tmp, "truth.json")))
    bots = {r["account"] for r in csv.DictReader(open(os.path.join(tmp, "bots.csv")))}
    check(bots == set(truth["bots"]), f"bots found {len(bots)} = true {len(truth['bots'])}, no human flagged")
    auth = list(csv.DictReader(open(os.path.join(tmp, "authors_sample.csv"))))
    hum = [r for r in auth if r["stratum"] != "buyer"]
    check(all(truth["posts"][r["account"]] >= 4 and truth["months"][r["account"]] >= 3 for r in hum),
          "human strata: every author has >= 4 posts in >= 3 months")
    check(all(int(r["posts"]) == truth["posts"][r["account"]] for r in auth), "post counts equal the truth (edits not double counted)")
    check(len({r["stratum"] for r in hum}) == 9 and len(hum) == 900, f"9 strata, {len(hum)} authors (900 asked)")
    check(all(r["account"] in truth["buyers"] for r in auth if r["buyer"] == "1")
          and sum(r["buyer"] == "1" for r in auth) == 100, "100 buyers, all true buyers")
    posts = list(csv.DictReader(open(os.path.join(tmp, "posts_sample.csv"))))
    exp = sum(truth["posts"][r["account"]] for r in auth)
    check(len(posts) == exp and len({(p["author"], p["permlink"]) for p in posts}) == exp,
          f"posts_sample: {len(posts)} posts = sum of sampled authors' posts, no duplicates")
    nb = sum(int(p["bought"]) > 0 for p in posts)
    nb_true = sum(1 for p in posts if f"{p['author']}/{p['permlink']}" in truth["bought"])
    check(nb == nb_true, f"bought posts flagged: {nb} = true {nb_true}")

    print("=== 2. fetch_post_rewards.py against a mock API with errors and an interruption")
    botlist = sorted(bots)
    srv = serve(botlist)
    node = f"http://127.0.0.1:{srv.server_address[1]}"
    import time
    import urllib.request
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    body = json.dumps({"jsonrpc": "2.0", "method": "condenser_api.get_content",
                       "params": [posts[0]["author"], posts[0]["permlink"]], "id": 1}).encode()
    for i in range(5):
        t = time.time()
        try:
            with op.open(urllib.request.Request(node, data=body, headers={"Content-Type": "application/json"}),
                         timeout=5) as r:
                print(f"  mock API {node}: answer {r.status} in {time.time() - t:.2f} s")
        except Exception as e:
            print(f"  mock API {node}: {type(e).__name__}: {e} after {time.time() - t:.2f} s")
    sub = os.path.join(tmp, "posts_sub.csv")
    with open(sub, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(posts[0]))
        w.writeheader()
        w.writerows(posts[:1500])
    jl = os.path.join(tmp, "post_rewards.jsonl")
    run(os.path.join(SRC, "fetch_post_rewards.py"), "--posts", sub, "--out", jl, "--node", node, "--rate", "500",
        "--limit", "600", "--timeout", "5", "--verbose", show=True)
    with open(jl, "a") as fh:
        fh.write('{"author": "h0001", "permlink": "cut-in-the-mid')     # interrupted write
    out = run(os.path.join(SRC, "fetch_post_rewards.py"), "--posts", sub, "--out", jl, "--node", node, "--rate", "500", "--timeout", "5", show=True)
    jl2 = os.path.join(tmp, "post_votes_light.jsonl")
    run(os.path.join(SRC, "fetch_post_rewards.py"), "--posts", sub, "--out", jl2, "--node", node, "--rate", "500",
        "--timeout", "5", "--light", "--limit", "400", show=True)
    srv.shutdown()
    csv2 = os.path.join(tmp, "post_votes_light.csv")
    run(os.path.join(SRC, "parse_post_rewards.py"), "--in", jl2, "--bots", os.path.join(tmp, "bots.csv"), "--out", csv2)
    bad2 = 0
    for r in csv.DictReader(open(csv2)):
        c = fake_post(r["author"], r["permlink"], botlist)
        pos = sum(int(v["rshares"]) for v in c["active_votes"] if int(v["rshares"]) > 0) if c else 0
        bad2 += int(r["rshares_pos"] or 0) != pos
    check(bad2 == 0, "light mode (get_active_votes): rshares equal the truth for 400 posts")
    lines = [json.loads(l) for l in open(jl) if l.strip().startswith("{") and l.strip().endswith("}")]
    keys = [(r["author"], r["permlink"]) for r in lines]
    check(len(keys) == len(set(keys)) == 1500, f"{len(set(keys))} posts fetched once each (1500 asked)")
    check(all("body" not in r for r in lines), "body dropped")
    csvp = os.path.join(tmp, "post_rewards.csv")
    run(os.path.join(SRC, "parse_post_rewards.py"), "--in", jl, "--bots", os.path.join(tmp, "bots.csv"), "--out", csvp)
    rows = {(r["author"], r["permlink"]): r for r in csv.DictReader(open(csvp))}
    bad = 0
    nmiss = 0
    for (au, pl), r in rows.items():
        c = fake_post(au, pl, botlist)
        if c is None:
            nmiss += 1
            bad += r["missing"] != "1"
            continue
        rs = [(v["voter"], int(v["rshares"])) for v in c["active_votes"]]
        pos = sum(x for _, x in rs if x > 0)
        botr = sum(x for v, x in rs if x > 0 and v in bots)
        selfr = sum(x for v, x in rs if x > 0 and v == au and v not in bots)
        org = pos - botr - selfr
        pay = float(c["total_payout_value"].split()[0])
        bad += not (int(r["rshares_pos"]) == pos and int(r["rshares_bot"]) == botr and int(r["rshares_organic"]) == org
                    and int(r["self_rshares"]) == selfr and abs(float(r["payout_author_sbd"]) - pay) < 1e-9)
    check(bad == 0, f"parsed rshares (total, bot, self, organic) and payouts equal the truth for all posts; "
                    f"{nmiss} deleted posts marked missing")
    print("ALL OK" if OK else "SOME CHECKS FAILED")
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
