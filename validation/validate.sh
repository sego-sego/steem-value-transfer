#!/usr/bin/env bash
# End-to-end validation of every estimator on synthetic data with known truth (Appendix A of the paper).
# Usage: bash validation/validate.sh   (from the repository root; ~15 min, no network, no real data)
set -euo pipefail
OUT=${OUT:-validation_out}
mkdir -p "$OUT"

echo "################ A. information layer: transfer rules"
for gt in "0.5 0.5" "0.0 0.0" "0.2 0.8"; do
  set -- $gt
  D="$OUT/reblogs_g$1_t$2"
  python validation/synth_reblogs.py --gamma "$1" --theta "$2" --out "$D"
  python src/build_reblog_pairs.py --month-dir "$D" --month 25 --hist-dir "$D" --out "$D/pairs_month25.csv" > "$D/build.log"
  echo "=== truth gamma=$1 theta=$2"
  python src/fit_reblog.py "$D/pairs_month25.csv" | grep -E "cited post \(k_items|category placebo|within pair|continuous"
  python src/fit_recipient_cc.py "$D/pairs_month25.csv" | grep -E "total|decomposition"
done
echo "=== within-reader panel: selection between readers, within-reader theta 0 and 1"
python src/fit_follower_panel.py --selftest
echo "=== asymmetry: follow decisions versus silence of sources"
python src/fit_asymmetry.py --selftest
echo "=== votes: protocol-like voting power with a break at 10 votes per day"
python validation/synth_votes_model0.py --out "$OUT/votes"
python src/fit_votes_model0.py --rshares "$OUT/votes/votes_rshares_month25.csv" --month-dir "$OUT/votes" --month 25 | grep -E "power|club"

echo "################ B. reward data: sample of authors, download through a mock API with errors, parsing"
python validation/test_stage2.py | tail -3

echo "################ C. channel I->M (model C)"
echo "=== stacked event study, 6 seeds, cases lambda = 0 and lambda > 0"
python src/fit_model_c.py --selftest | sed -n '/summary/,$p'
echo "=== distributed lags, 3 seeds"
python src/fit_model_c_dl.py --selftest | sed -n '/summary/,$p'

echo "################ D. channel M->I (bought votes)"
python src/fit_mi.py --selftest | grep "==="

echo "################ E. material layer (Result 1 of the PRL model)"
python src/fit_material.py --selftest | grep "==="
echo "VALIDATION DONE"
