"""Reblog-pair sample for Model A (source side, information layer).

Changes with respect to version 1:
 1. Category placebo: a root post p'' by a different author in the same category (parent_permlink
    of a root post), created closest in time to p (within 12 h), not reblogged by v; engagements with
    p'' by v's followers (not following its author) in the same window. Unlike the same-author
    placebo, it cannot be reached through the author's blog after the reblog.
 2. Full source fan-out: k_items_day = reblogs + own root posts of v on that UTC day.
 3. Recipient load (capacity theta): for every active follower x, L_x = items per day produced by
    all accounts x follows (start-of-month graph). For each pair the followers exposed at the time
    of the reblog (following v, not following the author) and the credited engagers are counted in
    cells (load bin x activity bin): E[Y_cell] = N_cell * exp(alpha_p + ...) * h(k) * L^(-theta).
 4. Heavy rebloggers: v_maxday (largest number of reblogs by v on one day) to separate services.

Usage:
  python build_reblog_pairs.py --month-dir data25 --month 25 --hist-dir custom_json_hist --out pairs_month25.csv
Writes pairs_month25.csv and pairs_month25_cells.csv. Standard library only.
Memory: the follow graph is kept for all accounts that post or reblog in the month (more edges than
version 1). If the Mac runs out of memory, run with --no-load (skips item 3).
"""
from __future__ import annotations

import argparse
import array
import bisect
import collections
import csv
import heapq
import io
import os
import re
import sys
import time
import zipfile

BPD = 28800
WINDOW = 7 * BPD
CAT_WINDOW = BPD // 2
BOT_URL_MEMOS = 20
BOT_DAILY_VOTES = 400
MAX_K_CELLS = 200
SHIFT, MASK = 21, (1 << 21) - 1
LOAD_EDGES = [0, 10, 30, 100, 300, 1000, float("inf")]   # items per day in the follower's feed
ACT_EDGES = [0, 10, 30, 100, 300, float("inf")]          # votes + comments of the follower in the month

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


class Prog:
    def __init__(self, label, every=2_000_000):
        self.label, self.every, self.n, self.t0 = label, every, 0, time.time()

    def tick(self):
        self.n += 1
        if self.n % self.every == 0:
            print(f"  {self.label}: {self.n / 1e6:.0f} M lines, {(time.time() - self.t0) / 60:.1f} min", flush=True)


def parse_cj(line):
    m = RE_CJ.match(line)
    if not m or m.group(3) != "follow":
        return None
    js, b = m.group(4), int(m.group(1))
    if '"reblog"' in js[:12]:
        a, au, pl = RE_ACCOUNT.search(js), RE_AUTHOR.search(js), RE_PERMLINK.search(js)
        if a and au and pl:
            return ("reblog", a.group(1), (au.group(1), pl.group(1)), b)
        return None
    fr, fg, wh = RE_FOLLOWER.search(js), RE_FOLLOWING.search(js), RE_WHAT.search(js)
    if not (fr and fg):
        return None
    kind = "follow" if (wh and '"blog"' in wh.group(1)) else "unfollow"
    return (kind, fr.group(1), fg.group(1), b)


def binof(edges, x):
    return bisect.bisect_right(edges, x) - 1


# ---------------------------------------------------------------- pass A
def pass_a(mdir, M):
    zp = lambda op: os.path.join(mdir, f"{op}-month-{M}.zip")
    posts = {}                                  # (author, permlink) -> (block, category)
    posts_by_author = collections.defaultdict(list)
    posts_by_cat = collections.defaultdict(list)
    act = collections.Counter()                 # votes + comments per account
    own_posts_day = collections.Counter()
    bmin = bmax = None
    p = Prog("A comment")
    for line in zlines(zp("comment")):
        p.tick()
        f = line.split(",")
        if len(f) != 5:
            continue
        b = int(f[0])
        bmin = b if bmin is None else min(bmin, b)
        bmax = b if bmax is None else max(bmax, b)
        act[f[3]] += 1
        if f[1] == "":
            k = (f[3], f[4])
            if k not in posts:
                posts[k] = (b, f[2])
                posts_by_author[f[3]].append((b, f[4]))
                posts_by_cat[f[2]].append((b, f[3], f[4]))
    for k, (b, c) in posts.items():
        own_posts_day[(k[0], (b - bmin) // BPD)] += 1
    daily = collections.Counter()
    p = Prog("A vote")
    for line in zlines(zp("vote")):
        p.tick()
        f = line.split(",")
        if len(f) != 5:
            continue
        act[f[1]] += 1
        daily[(f[1], (int(f[0]) - bmin) // BPD)] += 1
    bots = {v for (v, d), c in daily.items() if c > BOT_DAILY_VOTES}
    del daily
    url_to = collections.Counter()
    p = Prog("A transfer")
    for line in zlines(zp("transfer")):
        p.tick()
        f = line.split(",", 4)
        if len(f) == 5 and "steemit.com/" in f[4] and "/@" in f[4]:
            url_to[f[2]] += 1
    bots |= {a for a, c in url_to.items() if c >= BOT_URL_MEMOS}
    reblogs = []
    p = Prog("A custom_json")
    for line in zlines(zp("custom_json")):
        p.tick()
        r = parse_cj(line)
        if r and r[0] == "reblog":
            reblogs.append((r[3], r[1], r[2]))
    print(f"A: {len(posts)} posts, {len(reblogs)} reblogs, {len(act)} active, {len(bots)} bots", flush=True)
    per_day = collections.Counter((v, (b - bmin) // BPD) for b, v, _ in reblogs)
    maxday = collections.defaultdict(int)
    for (v, d), c in per_day.items():
        maxday[v] = max(maxday[v], c)
    items_month = collections.Counter()
    for (v, d), c in per_day.items():
        items_month[v] += c
    for (au, d), c in own_posts_day.items():
        items_month[au] += c
    by_v = collections.defaultdict(list)
    for b, v, _ in reblogs:
        by_v[v].append(b)
    for v in by_v:
        by_v[v].sort()
    first = {}
    for b, v, post in sorted(reblogs):
        if post not in posts or v == post[0] or b < posts[post][0] or b + WINDOW > bmax:
            continue
        first.setdefault((post, v), b)
    rb_by_post = collections.defaultdict(list)
    for (post, v), b in first.items():
        rb_by_post[post].append((b, v))
    rebloggers_of = collections.defaultdict(set)
    for b, v, post in reblogs:
        rebloggers_of[post].add(v)
    targets = {post: sorted(l) for post, l in rb_by_post.items() if len(l) >= 2}
    for l in posts_by_author.values():
        l.sort()
    for l in posts_by_cat.values():
        l.sort()
    pairs = []
    for post, l in targets.items():
        a = post[0]
        created, cat = posts[post]
        cand_a = sorted(((abs(cb - created), pl) for cb, pl in posts_by_author[a] if pl != post[1]))
        cl = posts_by_cat[cat]
        lo = bisect.bisect_left(cl, (created - CAT_WINDOW,))
        hi = bisect.bisect_right(cl, (created + CAT_WINDOW, "￿"))
        cand_c = sorted(((abs(cb - created), au, pl) for cb, au, pl in cl[lo:hi]
                         if au != a and au not in bots))
        for b, v in l:
            plac_a = next((pl for _, pl in cand_a if v not in rebloggers_of.get((a, pl), ())), "")
            plac_c = next(((au, pl) for _, au, pl in cand_c
                           if au != v and v not in rebloggers_of.get((au, pl), ())), ("", ""))
            d = (b - bmin) // BPD
            lo_w, hi_w = bisect.bisect_left(by_v[v], b - 3 * BPD), bisect.bisect_right(by_v[v], b + 3 * BPD)
            pairs.append(dict(author=a, permlink=post[1], category=cat, v=v, reblog_block=b, post_block=created,
                              k_day=per_day[(v, d)], k_items_day=per_day[(v, d)] + own_posts_day[(v, d)],
                              k_week=hi_w - lo_w, v_maxday=maxday[v],
                              placebo_permlink=plac_a, cat_author=plac_c[0], cat_permlink=plac_c[1],
                              v_bot=int(v in bots), a_bot=int(a in bots)))
    print(f"A: {len(targets)} target posts, {len(pairs)} pairs; category placebo found for "
          f"{sum(1 for q in pairs if q['cat_permlink'])}", flush=True)
    items_day = collections.Counter(per_day)
    for k_, c in own_posts_day.items():
        items_day[k_] += c
    return dict(bmin=bmin, bmax=bmax, act=act, bots=bots, pairs=pairs, items_month=items_month,
                items_day=items_day)


# ---------------------------------------------------------------- graph
class Graph:
    def __init__(self, active, targets):
        self.fid = {a: i for i, a in enumerate(sorted(active))}
        self.fname = sorted(active)
        self.tid = {a: i for i, a in enumerate(sorted(targets))}
        self.edges = set()
        self.nfol = collections.Counter()

    def apply(self, kind, follower, following):
        f, t = self.fid.get(follower), self.tid.get(following)
        if f is None or t is None:
            return
        e = (f << SHIFT) | t
        if kind == "follow":
            if e not in self.edges:
                self.edges.add(e)
                self.nfol[t] += 1
        elif e in self.edges:
            self.edges.discard(e)
            self.nfol[t] -= 1

    def follows(self, follower, following):
        f, t = self.fid.get(follower), self.tid.get(following)
        return f is not None and t is not None and ((f << SHIFT) | t) in self.edges


def pass_b(hist_dir, M, g):
    for m in range(1, M):
        f = os.path.join(hist_dir, f"custom_json-month-{m:02d}.zip")
        if not os.path.exists(f):
            f = os.path.join(hist_dir, f"custom_json-month-{m}.zip")
        if not os.path.exists(f):
            sys.exit(f"missing {f}")
        p = Prog(f"B {os.path.basename(f)}", 5_000_000)
        for line in zlines(f):
            p.tick()
            if '"follower"' not in line:
                continue
            r = parse_cj(line)
            if r and r[0] != "reblog":
                g.apply(r[0], r[1], r[2])
        print(f"B: {os.path.basename(f)} done, {len(g.edges) / 1e6:.2f} M edges", flush=True)


# ---------------------------------------------------------------- pass C
def stream(mdir, M):
    zp = lambda op: os.path.join(mdir, f"{op}-month-{M}.zip")

    def cj():
        for line in zlines(zp("custom_json")):
            r = parse_cj(line)
            if r:
                yield (r[3], 0, r)

    def vote():
        for line in zlines(zp("vote")):
            f = line.split(",")
            if len(f) == 5:
                yield (int(f[0]), 1, ("vote", f[1], (f[2], f[3])))

    def comment():
        for line in zlines(zp("comment")):
            f = line.split(",")
            if len(f) == 5 and f[1]:
                yield (int(f[0]), 1, ("reply", f[3], (f[1], f[2])))

    return heapq.merge(cj(), vote(), comment(), key=lambda x: (x[0], x[1]))


def pass_c(mdir, M, A, g, do_load, n_controls=10, seed=20261029):
    pairs, bots = A["pairs"], A["bots"]
    tables = {"y": collections.defaultdict(list), "yp": collections.defaultdict(list),
              "yc": collections.defaultdict(list)}
    for idx, q in enumerate(pairs):
        tables["y"][(q["author"], q["permlink"])].append((q["reblog_block"], q["v"], idx, q["author"]))
        if q["placebo_permlink"]:
            tables["yp"][(q["author"], q["placebo_permlink"])].append((q["reblog_block"], q["v"], idx, q["author"]))
        if q["cat_permlink"]:
            tables["yc"][(q["cat_author"], q["cat_permlink"])].append((q["reblog_block"], q["v"], idx, q["cat_author"]))
        for pre in ("y", "yp", "yc"):
            q.update({f"{pre}_vote": 0, f"{pre}_reply": 0, f"{pre}_human": 0})
        q.update(n_v=None, y_multi=0, n_exposed=0)
    for t in tables.values():
        for l in t.values():
            l.sort()
    pending = collections.defaultdict(list)
    for idx, q in enumerate(pairs):
        pending[(q["reblog_block"], q["v"])].append(idx)
    # recipient load and start-of-month follower lists of rebloggers
    lbin = abin = fol_lists = None
    cells = collections.defaultdict(lambda: [0, 0, 0.0])
    lnload = None
    if do_load:
        items = A["items_month"]
        tname = sorted(g.tid, key=g.tid.get)
        titems = array.array("d", (items.get(t, 0) / 31.0 for t in tname))
        load = array.array("d", bytes(8 * len(g.fname)))
        reb_t = {g.tid[q["v"]] for q in pairs if q["v"] in g.tid}
        fol_lists = collections.defaultdict(lambda: array.array("i"))
        for e in g.edges:
            f, t = e >> SHIFT, e & MASK
            load[f] += titems[t]
            if t in reb_t:
                fol_lists[t].append(f)
        lbin = array.array("b", (binof(LOAD_EDGES, x) for x in load))
        import math
        lnload = array.array("d", (math.log(max(x, 0.5)) for x in load))
        abin = array.array("b", (binof(ACT_EDGES, A["act"].get(n, 0)) for n in g.fname))
        print(f"load: computed for {len(load)} followers; follower lists for {len(fol_lists)} rebloggers", flush=True)
        # case-control sample for the decomposition of recipient load (number of followees,
        # items per followee, day-to-day deviation of the feed volume)
        nfol_out = array.array("i", bytes(4 * len(g.fname)))
        for e in g.edges:
            nfol_out[e >> SHIFT] += 1
        ndays = int((A["bmax"] - A["bmin"]) // BPD) + 1
        tday = {}
        for (acct, d), c in A["items_day"].items():
            t = g.tid.get(acct)
            if t is not None and 0 <= d < ndays:
                if t not in tday:
                    tday[t] = array.array("f", bytes(4 * ndays))
                tday[t][d] += c
        out_adj = None                      # followee lists, built lazily for sampled followers
        need = set()
    cell_ok = [do_load and q["v_bot"] == 0 and q["a_bot"] == 0 and q["k_day"] <= MAX_K_CELLS for q in pairs]
    import random as _random
    rng = _random.Random(seed)
    cc_controls = {}                        # pair -> (n_exposed, sampled follower ids)
    fm_exp = collections.Counter()          # follower -> distinct target posts that reached her this month
    # first exposure only: a follower reached by several rebloggers of the same post is exposed once,
    # through the earliest reblog, matching the first-exposure rule used for crediting engagements
    post_id = {}
    remaining = collections.Counter()
    for idx, q in enumerate(pairs):
        if cell_ok[idx]:
            key = (q["author"], q["permlink"])
            post_id.setdefault(key, len(post_id))
            remaining[post_id[key]] += 1
    post_exposed = {}
    fm_eng = collections.Counter()          # follower -> credited (human) engagements
    cc_cases = collections.defaultdict(set)
    seen = {k: set() for k in tables}
    p = Prog("C merged", 5_000_000)
    for b, _, ev in stream(mdir, M):
        p.tick()
        kind = ev[0]
        if kind in ("follow", "unfollow"):
            g.apply(kind, ev[1], ev[2])
            continue
        if kind == "reblog":
            key = (b, ev[1])
            if key not in pending:
                continue
            tv = g.tid.get(ev[1])
            for idx in pending.pop(key):
                q = pairs[idx]
                q["n_v"] = g.nfol[tv] if tv is not None else 0
                if cell_ok[idx] and tv is not None:
                    ta = g.tid.get(q["author"])
                    fa = g.fid.get(q["author"])
                    ne = 0
                    res = []
                    pid = post_id[(q["author"], q["permlink"])]
                    pex = post_exposed.setdefault(pid, set())
                    for f in fol_lists.get(tv, ()):
                        if ((f << SHIFT) | tv) not in g.edges or f == fa:
                            continue
                        if ta is not None and ((f << SHIFT) | ta) in g.edges:
                            continue
                        if f in pex:
                            continue
                        pex.add(f)
                        cell = cells[(idx, lbin[f], abin[f])]
                        cell[0] += 1
                        cell[2] += lnload[f]
                        fm_exp[f] += 1
                        ne += 1
                        # reservoir sample of exposed followers (controls)
                        if ne <= n_controls:
                            res.append(f)
                        else:
                            j = rng.randrange(ne)
                            if j < n_controls:
                                res[j] = f
                    q["n_exposed"] = ne
                    cc_controls[idx] = (ne, list(res))
                    need.update(res)
                    remaining[pid] -= 1
                    if remaining[pid] <= 0:
                        post_exposed.pop(pid, None)
            continue
        x, post = ev[1], ev[2]
        for pre, table in tables.items():
            l = table.get(post)
            if not l:
                continue
            owner = post[0]
            if x == owner or g.follows(x, owner):
                continue
            exposed = [(rb, v, idx) for rb, v, idx, _ in l if rb < b <= rb + WINDOW and x != v and g.follows(x, v)]
            if not exposed:
                continue
            rb, v, idx = exposed[0]
            if (x, idx) in seen[pre]:
                continue
            seen[pre].add((x, idx))
            q = pairs[idx]
            q[f"{pre}_{kind}"] += 1
            if x not in bots:
                q[f"{pre}_human"] += 1
                if pre == "y" and cell_ok[idx]:
                    fx = g.fid.get(x)
                    if fx is not None:
                        cells[(idx, lbin[fx], abin[fx])][1] += 1
                        cc_cases[idx].add(fx)
                        need.add(fx)
                        fm_eng[fx] += 1
            if pre == "y" and len(exposed) > 1:
                q["y_multi"] += 1
    cc_rows = []
    if do_load:
        # daily feed volume of the followers in the case-control sample (start-of-month followees)
        out_adj = collections.defaultdict(lambda: array.array("i"))
        for e in g.edges:
            f = e >> SHIFT
            if f in need:
                out_adj[f].append(e & MASK)
        cache = {}

        def lday(f, d):
            key = (f, d)
            if key not in cache:
                cache[key] = sum(tday[t][d] for t in out_adj.get(f, ()) if t in tday)
            return cache[key]

        import math
        for idx, (ne, ctrl) in cc_controls.items():
            q = pairs[idx]
            d = (q["reblog_block"] - A["bmin"]) // BPD
            cases = cc_cases.get(idx, set())
            ctrl = [f for f in ctrl if f not in cases]
            w = ne / max(len(cc_controls[idx][1]), 1)
            for f, y, wt in [(f, 1, 1.0) for f in cases] + [(f, 0, w) for f in ctrl]:
                nf, lm = nfol_out[f], load[f]
                if nf <= 0 or lm <= 0:
                    continue
                ld = lday(f, d)
                cc_rows.append((idx, y, wt, math.log(nf), math.log(lm / nf), math.log(max(ld, 0.5) / lm),
                                math.log(max(ld, 0.5)), abin[f]))
    fm_rows = []
    if do_load:
        for f, ne in fm_exp.items():
            name = g.fname[f]
            if name in bots:
                continue
            fm_rows.append((name, ne, fm_eng.get(f, 0), nfol_out[f], load[f], A["act"].get(name, 0)))
    return pairs, cells, cc_rows, fm_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month-dir", required=True)
    ap.add_argument("--month", type=int, default=25)
    ap.add_argument("--hist-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-load", action="store_true", help="skip the recipient-load cells (less memory)")
    a = ap.parse_args()
    M = f"{a.month:02d}" if os.path.exists(os.path.join(a.month_dir, f"comment-month-{a.month:02d}.zip")) else str(a.month)
    t0 = time.time()
    A = pass_a(a.month_dir, M)
    if a.no_load:
        targets = {q["v"] for q in A["pairs"]} | {q["author"] for q in A["pairs"]} | \
                  {q["cat_author"] for q in A["pairs"] if q["cat_author"]}
    else:
        targets = set(A["items_month"]) | {q["author"] for q in A["pairs"]} | \
                  {q["cat_author"] for q in A["pairs"] if q["cat_author"]}
    print(f"graph: {len(A['act'])} active followers, {len(targets)} followed accounts", flush=True)
    g = Graph(set(A["act"]), targets)
    pass_b(a.hist_dir, a.month, g)
    pairs, cells, cc_rows, fm_rows = pass_c(a.month_dir, M, A, g, not a.no_load)
    out = a.out or os.path.join(a.month_dir, f"pairs_month{a.month}.csv")
    cols = ["author", "permlink", "category", "v", "post_block", "reblog_block", "k_day", "k_items_day", "k_week",
            "v_maxday", "n_v", "n_exposed", "y_vote", "y_reply", "y_human", "y_multi",
            "placebo_permlink", "yp_vote", "yp_reply", "yp_human",
            "cat_author", "cat_permlink", "yc_vote", "yc_reply", "yc_human", "v_bot", "a_bot"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair"] + cols, extrasaction="ignore")
        w.writeheader()
        for i, q in enumerate(pairs):
            w.writerow(dict(q, pair=i))
    if cells:
        cout = out.replace(".csv", "_cells.csv")
        with open(cout, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair", "load_bin", "act_bin", "N", "Y", "mean_lnL"])
            for (i, lb, ab), (n, y, sl) in sorted(cells.items()):
                w.writerow([i, lb, ab, n, y, f"{sl / n:.4f}" if n else ""])
        print(f"saved {cout}: {len(cells)} cells")
    if cc_rows:
        ccout = out.replace(".csv", "_cc.csv")
        with open(ccout, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair", "y", "w", "ln_nfol", "ln_items_per_followee", "ln_dev_day", "ln_load_day", "act_bin"])
            for row in cc_rows:
                w.writerow([row[0], row[1], f"{row[2]:.4f}"] + [f"{x:.4f}" for x in row[3:7]] + [row[7]])
        print(f"saved {ccout}: {len(cc_rows)} rows (cases + sampled controls)")
    if fm_rows:
        fmout = out.replace(".csv", "_followers.csv")
        with open(fmout, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["account", "exposures", "engagements", "n_followees", "load_per_day", "activity"])
            for row in fm_rows:
                w.writerow([row[0], row[1], row[2], row[3], f"{row[4]:.3f}", row[5]])
        print(f"saved {fmout}: {len(fm_rows)} followers (month summary for the panel)")
    print(f"saved {out}: {len(pairs)} pairs, {(time.time() - t0) / 60:.1f} min; edges {len(g.edges) / 1e6:.2f} M")


if __name__ == "__main__":
    main()
