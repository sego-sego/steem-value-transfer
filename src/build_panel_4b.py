"""Model C (channel I->M): post-level panel with audience and information shocks.

For the sampled authors, over the weeks of the window months (default 22-28):
  P_aw  root posts of author a in week w (all posts, SteemOps)
  F_aw  active followers at the start of week w: followers who voted or commented in the month
        of the week (follow graph rebuilt from all custom_json months up to the window)
  shock reblog of a PAST post of a by a big account: the post was created at least --past-days
        before the reblog (its payout window is closed, so the reblog cannot change its own reward
        and acts on the author only through attention); big = followers at the start of the
        window in the top --top share of the accounts that reblog in the window; bots and the
        author excluded.
and joins the rewards of the fetched posts (post_rewards.csv, posts_fetch.csv).

Outputs:
  panel_weeks.csv  author, week, month, P, F, shocks, shocks_any (past-post reblogs by anyone),
                   reach (sum of the followers of those rebloggers)
  reblogs_past.csv every reblog of a past post of the sample (the reblogged posts are dropped from
                   the outcome by the fits: they were chosen for their quality)
  panel_posts.csv  author, permlink, week, bought, weight, r_org, r_all, payout (fetched posts)
  shocks.csv       block, week, reblogger, author, permlink, reblogger_followers, post_age_days

Usage (from the data folder):
  python src/build_panel_4b.py --data-root . --hist-dir custom_json_hist
Standard library only. Memory: the whole follow graph (~20 M edges), a few GB.
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

BPD = 28800
WEEK = 7 * BPD
SHIFT = 21
RE_CJ = re.compile(r'^(\d+),(\[.*?\]),([^,]*),(.*)$')
RE_FOLLOWER = re.compile(r'"follower"\s*:\s*"([^"]*)"')
RE_FOLLOWING = re.compile(r'"following"\s*:\s*"([^"]*)"')
RE_WHAT = re.compile(r'"what"\s*:\s*\[([^\]]*)\]')
RE_ACCOUNT = re.compile(r'"account"\s*:\s*"([^"]*)"')
RE_AUTHOR = re.compile(r'"author"\s*:\s*"([^"]*)"')
RE_PERMLINK = re.compile(r'"permlink"\s*:\s*"([^"]*)"')


def zlines(path):
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            it = io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline="")
            next(it)
            for line in it:
                yield line.rstrip("\r\n")


def archive(dirs, op, m):
    for d in dirs:
        for name in (f"{op}-month-{m:02d}.zip", f"{op}-month-{m}.zip"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None


class Prog:
    def __init__(self, label, every=5_000_000):
        self.label, self.every, self.n, self.t0 = label, every, 0, time.time()

    def tick(self):
        self.n += 1
        if self.n % self.every == 0:
            print(f"  {self.label}: {self.n / 1e6:.0f} M lines, {(time.time() - self.t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--hist-dir", default="custom_json_hist")
    ap.add_argument("--months", type=int, nargs="+", default=list(range(22, 29)))
    ap.add_argument("--authors", default="authors_sample.csv")
    ap.add_argument("--posts", default="posts_fetch.csv")
    ap.add_argument("--rewards", default="post_rewards.csv")
    ap.add_argument("--bots", default="bots.csv")
    ap.add_argument("--top", type=float, default=0.05)
    ap.add_argument("--past-days", type=float, default=7.0)
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    t0 = time.time()
    mdir = lambda m: [os.path.join(a.data_root, f"data{m}"), a.data_root]
    with open(a.authors) as fh:
        authors = {r["account"] for r in csv.DictReader(fh)}
    bots = set()
    if os.path.exists(a.bots):
        with open(a.bots) as fh:
            bots = {r["account"] for r in csv.DictReader(fh)}
    months = [m for m in a.months if archive(mdir(m), "comment", m)]
    # ---- pass 1: posts of the sample, P_aw, activity per month, month starts
    created = {}                        # (author, permlink) -> block, sample authors only
    mstart, active_names = {}, {}
    for m in months:
        act = set()
        bmin = None
        p = Prog(f"comment {m}")
        for line in zlines(archive(mdir(m), "comment", m)):
            p.tick()
            f = line.split(",")
            if len(f) != 5:
                continue
            b = int(f[0])
            bmin = b if bmin is None else min(bmin, b)
            act.add(f[3])
            if f[1] == "" and f[3] in authors and (f[3], f[4]) not in created:
                created[(f[3], f[4])] = b
        mstart[m] = bmin
        vp = archive(mdir(m), "vote", m)
        if vp:
            p = Prog(f"vote {m}")
            for line in zlines(vp):
                p.tick()
                f = line.split(",", 2)
                if len(f) == 3:
                    act.add(f[1])
        else:
            print(f"WARNING: vote-month-{m} missing: activity of month {m} from comments only", flush=True)
        active_names[m] = act
        print(f"month {m}: {len(act)} active accounts; {len(created)} posts of the sample so far, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    b0 = mstart[months[0]]
    week = lambda b: (b - b0) // WEEK
    month_of = lambda b: max([m for m in months if mstart[m] <= b], default=months[0])
    P = collections.Counter((au, week(b)) for (au, pl), b in created.items())
    last_week = max(w for _, w in P)
    # ---- pass 2: follow graph through all custom_json months, weekly snapshots, reblogs
    ids = {}
    gid = lambda x: ids.setdefault(x, len(ids))
    edges, nfol = set(), collections.Counter()
    fol = {gid(x): set() for x in sorted(authors)}
    sample_ids = set(fol)
    active = {}
    F = {}
    nfol_b0 = None
    reblogs = []
    next_w = 0
    cj_months = [m for m in range(1, max(months) + 1)
                 if archive([a.hist_dir] + mdir(m), "custom_json", m)]
    missing = [m for m in range(1, max(months) + 1) if m not in cj_months]
    if missing:
        print(f"WARNING: custom_json missing for months {missing}", flush=True)
    last_cj = max(cj_months)

    def snapshot(w):
        nonlocal nfol_b0
        if w == 0:
            nfol_b0 = dict(nfol)
        m = month_of(b0 + w * WEEK)
        if m not in active:
            active[m] = {gid(x) for x in active_names[m]}   # ids for all active accounts, also future followers
        act = active[m]
        for aid, fs in fol.items():
            F[(aid, w)] = len(fs & act)

    for m in cj_months:
        p = Prog(f"custom_json {m}")
        for line in zlines(archive([a.hist_dir] + mdir(m), "custom_json", m)):
            p.tick()
            mm = RE_CJ.match(line)
            if not mm or mm.group(3) != "follow":
                continue
            b = int(mm.group(1))
            while next_w <= last_week and b >= b0 + next_w * WEEK:
                snapshot(next_w)
                next_w += 1
            js = mm.group(4)
            if '"reblog"' in js[:12]:
                if b >= b0:
                    acc, au, pl = RE_ACCOUNT.search(js), RE_AUTHOR.search(js), RE_PERMLINK.search(js)
                    if acc and au and pl and au.group(1) in authors:
                        reblogs.append((b, acc.group(1), au.group(1), pl.group(1)))
                continue
            fr, fg, wh = RE_FOLLOWER.search(js), RE_FOLLOWING.search(js), RE_WHAT.search(js)
            if not (fr and fg):
                continue
            x, t = gid(fr.group(1)), gid(fg.group(1))
            e = (x << SHIFT) | t
            if wh and '"blog"' in wh.group(1):
                if e not in edges:
                    edges.add(e)
                    nfol[t] += 1
                    if t in sample_ids:
                        fol[t].add(x)
            elif e in edges:
                edges.discard(e)
                nfol[t] -= 1
                if t in sample_ids:
                    fol[t].discard(x)
        print(f"custom_json {m}: {len(edges) / 1e6:.2f} M edges, {len(reblogs)} reblogs of sample posts, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    # weeks after the last custom_json month: graph unknown, F left empty
    while next_w <= last_week and month_of(b0 + next_w * WEEK) <= last_cj:
        snapshot(next_w)
        next_w += 1
    if next_w <= last_week:
        print(f"WARNING: no follow data after month {last_cj}: F missing from week {next_w} on", flush=True)
    if nfol_b0 is None:
        sys.exit("the custom_json archives do not reach the start of the window")
    # ---- shocks
    rebloggers = {r for _, r, _, _ in reblogs}
    vals = sorted(nfol_b0.get(ids.get(r, -1), 0) for r in rebloggers)
    k = max(1, int(round(a.top * len(vals))))          # number of big accounts: the top share
    thr = vals[len(vals) - k] if vals else 0
    shocks, shocks_any, reach = collections.Counter(), collections.Counter(), collections.Counter()
    srows, prows = [], []
    for b, r, au, pl in reblogs:
        if r == au or r in bots:
            continue
        c = created.get((au, pl))
        age = (b - c) / BPD if c is not None else float("inf")      # unknown: created before the window
        if age < a.past_days:
            continue
        w = week(b)
        shocks_any[(au, w)] += 1
        nf = nfol_b0.get(ids.get(r, -1), 0)
        reach[(au, w)] += nf                                         # followers of the rebloggers
        prows.append((b, w, r, au, pl, nf))
        if nf >= thr:
            shocks[(au, w)] += 1
            srows.append((b, w, r, au, pl, nf, f"{age:.1f}" if age != float("inf") else "before"))
    print(f"big rebloggers: followers >= {thr} (top {a.top:.0%} of {len(rebloggers)} rebloggers); "
          f"{len(srows)} shocks to {len({s[3] for s in srows})} authors; all past-post reblogs: {len(prows)} "
          f"to {len({x[3] for x in prows})} authors", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "shocks.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "week", "reblogger", "author", "permlink", "reblogger_followers", "post_age_days"])
        w.writerows(sorted(srows))
    with open(os.path.join(a.out_dir, "reblogs_past.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "week", "reblogger", "author", "permlink", "reblogger_followers"])
        w.writerows(sorted(prows))
    with open(os.path.join(a.out_dir, "panel_weeks.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["author", "week", "month", "P", "F", "shocks", "shocks_any", "reach"])
        for au in sorted(authors):
            aid = ids[au]
            for wk in range(last_week + 1):
                f_ = F.get((aid, wk))
                w.writerow([au, wk, month_of(b0 + wk * WEEK), P.get((au, wk), 0), "" if f_ is None else f_,
                            shocks.get((au, wk), 0), shocks_any.get((au, wk), 0), reach.get((au, wk), 0)])
    # ---- rewards of the fetched posts
    rew = {}
    if os.path.exists(a.rewards):
        with open(a.rewards) as fh:
            for r in csv.DictReader(fh):
                if r.get("missing") == "1":
                    continue
                rew[(r["author"], r["permlink"])] = r
    n = 0
    with open(a.posts) as fh, open(os.path.join(a.out_dir, "panel_posts.csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["author", "permlink", "week", "bought", "weight", "r_org", "r_all", "payout"])
        for r in csv.DictReader(fh):
            k = (r["author"], r["permlink"])
            q = rew.get(k)
            if q is None or k not in created:
                continue
            w.writerow([k[0], k[1], week(created[k]), r.get("bought", 0), r.get("weight", 1),
                        f"{int(q['rshares_organic']) / 1e9:.4f}", f"{int(q['rshares_pos']) / 1e9:.4f}",
                        q.get("payout_author_sbd", "")])
            n += 1
    print(f"saved panel_weeks.csv ({len(authors)} authors x {last_week + 1} weeks), panel_posts.csv ({n} posts "
          f"with rewards), shocks.csv; {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
