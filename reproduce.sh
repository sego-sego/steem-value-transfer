#!/usr/bin/env bash
# Full sequence of the paper. Run from the data folder (layout in data/README.md):
#   SRC=/path/to/steem-value-transfer/src bash /path/to/steem-value-transfer/reproduce.sh [A|B|all]
# Part A: information layer, February-April 2018 (transfer rules W ~ D^-theta A D^-gamma).
# Part B: inter-layer channels and material layer, December 2017 - June 2018.
# Steps that use the API (A4, B2) are resumable: rerun the same command after an interruption.
set -euo pipefail
SRC=${SRC:-src}
PART=${1:-all}

partA() {
# A0. inventory of March 2018
python $SRC/inventory_steemops.py data25 25 | tee results_inventory_mar.txt

# A1. reblog pairs, recipient-load cells, case-control sample, per-reader summaries (~6 min per month)
for m in 24 25 26; do
  python $SRC/build_reblog_pairs.py --month-dir data$m --month $m --hist-dir custom_json_hist --out data$m/pairs_month$m.csv
done

# A2. source side
python $SRC/fit_reblog.py data25/pairs_month25.csv                 | tee results_source_mar_all.txt
python $SRC/fit_reblog.py data25/pairs_month25.csv --max-maxday 50 | tee results_source_mar.txt
python $SRC/fit_reblog.py data26/pairs_month26.csv --max-maxday 50 | tee results_source_apr.txt

# A3. recipient side: case-control decomposition and within-reader panel
python $SRC/fit_recipient_cc.py data25/pairs_month25.csv --max-maxday 50 | tee results_cc_mar.txt
python $SRC/fit_recipient_cc.py data26/pairs_month26.csv --max-maxday 50 | tee results_cc_apr.txt
FOL="data24/pairs_month24_followers.csv data25/pairs_month25_followers.csv data26/pairs_month26_followers.csv"
python $SRC/fit_follower_panel.py $FOL | tee results_panel.txt

# A3b. asymmetry: follow decisions versus silence of sources
python $SRC/follow_decomposition.py --months 24 25 26 --month-dirs data24 data25 data26 --hist-dir custom_json_hist \
    --followers $FOL
python $SRC/fit_asymmetry.py --decomp follow_decomposition.csv --months 24 25 26 --followers $FOL \
    | tee results_asymmetry.txt

# A4. votes: protocol rule as positive control (API, ~1 h for 20 000 posts)
python $SRC/fetch_vote_rshares.py --month-dir data25 --month 25 --n-posts 20000 --out votes_rshares_month25.csv
python $SRC/fit_votes_model0.py --rshares votes_rshares_month25.csv --month-dir data25 --month 25 | tee results_votes.txt
}

partB() {
# B0. API check: reward fields, votes with rshares, net_rshares after payout
python $SRC/probe_api.py | tee results_probe_api.txt

# B1. sample of authors (5000 human authors in 9 strata + 3000 vote buyers), bots, their root posts;
#     thinning to <= 3 posts per author-week plus all bought posts (weights kept)
python $SRC/sample_authors.py --data-root . --months 22 23 24 25 26 27 28 | tee results_sample_authors.txt
python $SRC/thin_posts.py --posts posts_sample.csv --per-week 3 --out posts_fetch.csv | tee results_thin.txt

# B2. rewards and votes of the posts: two halves on two nodes in parallel (10-20 h; resumable).
#     Two threads per node: more threads make the nodes cut long answers. --light (get_active_votes)
#     gives the vote rshares without payouts and can finish a download that get_content cannot.
for pass in 1 2; do     # the second pass refetches only what failed in the first
  python $SRC/fetch_post_rewards.py --posts posts_fetch.csv --out rewards_hive.jsonl \
      --node https://api.hive.blog --shard 0/2 --timeout 30 --threads 2 >> fetch_hive.log 2>&1 &
  python $SRC/fetch_post_rewards.py --posts posts_fetch.csv --out rewards_steemit.jsonl \
      --node https://api.steemit.com --shard 1/2 --timeout 30 --threads 2 >> fetch_steemit.log 2>&1 &
  wait
done
python $SRC/check_rewards.py | tee results_check_rewards.txt
python $SRC/parse_post_rewards.py --in rewards_hive.jsonl rewards_steemit.jsonl --bots bots.csv --out post_rewards.csv

# B3. channel I->M: author-week panels (30-60 min each; follow graph of ~20 million edges)
#     main: reblogs of fresh posts by the top 5% of rebloggers by followers; check: reblogs of past posts, top 20%
python $SRC/build_panel_4b.py --data-root . --hist-dir custom_json_hist --top 0.05 --past-days 0 --out-dir panel_fresh05 \
    | tee results_panel_fresh05.txt
python $SRC/build_panel_4b.py --data-root . --hist-dir custom_json_hist --top 0.20 --out-dir panel_top20 \
    | tee results_panel_top20.txt

# B4. stacked event study (5000 clean controls per stack, 5 draws of the controls), distributed lags
{
for s in 1 2 3 4 5; do
  echo "=== stacked event study, fresh-post reblogs, top 5%, control draw $s"
  python $SRC/fit_model_c.py --panel-dir panel_fresh05 --controls 5000 --seed $s --no-F
done
echo "=== stacked event study, past-post reblogs, top 20%"
python $SRC/fit_model_c.py --panel-dir panel_top20 --controls 5000
echo "=== distributed lags, x = ln(1 + reach)"
python $SRC/fit_model_c_dl.py --panel-dir panel_fresh05
python $SRC/fit_model_c_dl.py --panel-dir panel_top20
} 2>&1 | tee results_model_c.txt

# B5. permutation tests (1-3 h)
{
python $SRC/perm_model_c.py --panel-dir panel_fresh05 --controls 5000 --control-seeds 5 --n-stacked 40 --n-dl 100
python $SRC/perm_model_c.py --panel-dir panel_top20 --n-stacked 30 --n-dl 0
} 2>&1 | grep -v "permutation [0-9]*/" | tee results_perm.txt

# B6. channel M->I: bought votes and the organic response by post age (~25 min)
python $SRC/build_mi_events.py --data-root . --posts posts_fetch.csv --bots bots.csv \
    --rewards rewards_hive.jsonl rewards_steemit.jsonl 2>&1 | tail -3 | tee results_mi_build.txt
{
for o in votes_new_poster votes_new_other votes_new votes_loyal votes_active replies; do
  echo "=== outcome $o"; python $SRC/fit_mi.py --dir . --outcome $o
done
for o in votes_new_poster votes_active; do
  echo "=== dose, $o"; python $SRC/fit_mi.py --dir . --outcome $o --dose
done
echo "=== strict placebo (refunds by bots that vote themselves on >= 90% of paid posts)"
python $SRC/fit_mi.py --dir . --outcome votes_new_poster --strict
} 2>&1 | tee results_mi.txt

# B7. material layer: transfer operator, stake, Result 1 of the PRL model (10-20 min)
python $SRC/build_material.py --data-root . --rewards rewards_hive.jsonl rewards_steemit.jsonl --bots bots.csv \
    | tail -2 | tee results_material_build.txt
{
python $SRC/fit_material.py --panel material_panel.csv --kind user
echo "=== services (>= 200 senders a month)"
python $SRC/fit_material.py --panel material_panel.csv --kind service
} 2>&1 | tee results_material.txt
}

case "$PART" in
  A) partA ;;
  B) partB ;;
  all) partA; partB ;;
  *) echo "usage: reproduce.sh [A|B|all]"; exit 1 ;;
esac

# Figures (to figures/ of the current folder). Figs. 1 and A1 are schemes; Figs. 2, 5-8 are drawn from
# the numbers in the results_*.txt files above, copied into the scripts; Figs. 3-4 need the pair files.
REPO=$(dirname "$SRC")
python $REPO/figures/make_fig1_pipeline.py
python $REPO/figures/make_figs_now.py
python $REPO/figures/make_figs_4b.py
[ -f data25/pairs_month25.csv ] && python $REPO/figures/make_figs_data.py --mar data25/pairs_month25.csv --apr data26/pairs_month26.csv
echo "DONE"
