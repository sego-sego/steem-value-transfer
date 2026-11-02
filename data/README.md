# Data

## Source archives (not redistributed here)

SteemOps (Li, Palanisamy, Xu, Xu and Wang, ACM CODASPY 2021) — operations of the Steem blockchain,
March 2016 to November 2019, one zip archive per operation type and month. Download link:
https://github.com/archerlclclc/SteemOps (section "SteemOps dataset download").

Month numbering: `month-01` is March 2016, so `month-22` = December 2017, `month-24` = February 2018,
`month-25` = March 2018 (blocks 20 279 029 – 21 170 375), `month-28` = June 2018. The whole window
December 2017 – June 2018 lies between hard forks 19 and 20, so the reward rules did not change in it
(linear rewards, 75% author / 25% curators, one payout 7 days after the post, a full vote spends 2% of
the voting power, which recovers by 20% a day).

Expected layout (archives are read without unpacking):

```
data22/ ... data28/   comment-month-NN.zip  vote-month-NN.zip  custom_json-month-NN.zip  transfer-month-NN.zip
custom_json_hist/     custom_json-month-01.zip ... custom_json-month-25.zip   (follow history)
```

Part A of `reproduce.sh` needs only `data24/`–`data26/` and `custom_json_hist/`.

Formats (first line of each CSV):

| Archive | Header |
|---|---|
| comment | `block_no,parent_author,parent_permlink,author,permlink` (root post: empty `parent_author`, `parent_permlink` = category) |
| vote | `block_no,voter,author,permlink,weight` (weight in 0.01%, negative = flag) |
| custom_json | `block_no,required_posting_auths,id,json` (follow, unfollow, reblog under `id=follow`) |
| transfer | `block_no,from,to,amount,memo` |

Rewards and vote rshares are not in SteemOps. They are retrieved through the public API
(`condenser_api.get_content`, `condenser_api.get_active_votes`) from `https://api.hive.blog`, which
serves the shared pre-2020 history, and `https://api.steemit.com`. After the payout the API reports
`net_rshares = 0`; the scripts therefore sum the rshares of the individual votes. No API key is needed.

## Derived files: information layer (written by `src/build_reblog_pairs.py`)

`pairs_monthNN.csv` — one row per (reblogged post p, reblogger v), posts created in the month with
at least two rebloggers and a full 7-day window:

| Column | Meaning |
|---|---|
| `author`, `permlink`, `category` | reblogged post |
| `v`, `reblog_block`, `post_block` | reblogger, block of the reblog and of the post |
| `k_day`, `k_items_day`, `k_week` | reblogs of v that day; reblogs + own posts that day; reblogs within ±3 days |
| `v_maxday` | largest number of reblogs by v on one day of the month (services) |
| `n_v` | active followers of v at the reblog (voted or commented in the month) |
| `n_exposed` | followers of v reached by p first through this reblog (not following the author) |
| `y_vote`, `y_reply`, `y_human`, `y_multi` | engagements credited to this reblog (votes, replies, excluding bots, reached through several rebloggers) |
| `placebo_permlink`, `yp_*` | same-author placebo post and its engagements |
| `cat_author`, `cat_permlink`, `yc_*` | category placebo (another author, same category, ±12 h) |
| `v_bot`, `a_bot` | reblogger / author flagged as bot |

`pairs_monthNN_cells.csv` — exposed readers (N) and engagements (Y) per pair × feed-load bin ×
activity bin, with the mean log load.

`pairs_monthNN_cc.csv` — case–control sample: all engagers (y = 1) and up to 10 sampled exposed
non-engagers per pair (y = 0, weight w), with log number of sources, log output per source, log
deviation of the day's load and activity bin.

`pairs_monthNN_followers.csv` — per reader and month: distinct reblogged posts that reached her,
engagements, number of followed active sources, feed load per day, activity.

Bots: recipients of ≥ 20 transfers with a post URL in the memo (bid bots) and voters with > 400
votes on a day.

## Derived files: reward data, channels and material layer

| File | Written by | Content |
|---|---|---|
| `bots.csv` | `sample_authors.py` | accounts flagged as bots in any month of the window |
| `authors_sample.csv` | `sample_authors.py` | 5000 human authors (≥ 4 posts in ≥ 3 months; 9 strata = terciles of posts × terciles of votes per post) and 3000 vote buyers |
| `posts_sample.csv`, `posts_fetch.csv` | `sample_authors.py`, `thin_posts.py` | their root posts; thinned list with weights |
| `rewards_*.jsonl` | `fetch_post_rewards.py` | one JSON line per post: reward fields and votes `[voter, rshares, percent, time]` |
| `post_rewards.csv` | `parse_post_rewards.py` | payouts; positive, bot, self and organic rshares; `declined`, `missing` |
| `panel_*/panel_weeks.csv` | `build_panel_4b.py` | author × week: active followers F, posts P, shocks, reach |
| `panel_*/panel_posts.csv`, `shocks.csv` | `build_panel_4b.py` | post-level rewards; shock events (reblogger, followers, block) |
| `mi_events.csv`, `mi_bins.csv`, `mi_recipients.csv` | `build_mi_events.py` | purchases, bot votes, refunds, services; organic votes and replies per 12-h bin and voter type; paid accounts |
| `material_panel.csv` | `build_material.py` | account × month: kind, outflow O, inflow I, row sum R, senders, stake |

Voter types in `mi_bins.csv`: new voters of the author who write posts themselves (`votes_new_poster`)
or not (`votes_new_other`), loyal voters (voted for the author before), very active voters (> 20 votes
per active day), vote trails and pools (≥ 10 votes on ≥ 3 authors, ≥ 50% of them within 10 min of a
bot vote or 2 h of a purchase; excluded). Account classes in `material_panel.csv`: exchange (≥ 1000
distinct senders and receivers, or a known exchange), bid bot, bot, service (≥ 200 senders), user.
