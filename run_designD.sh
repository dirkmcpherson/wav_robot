#!/usr/bin/env bash
# Design D (board-mandated): BUDGET-LOCKED selection in the paper's regime.
#   - cumulative acquisition: 2 rounds x 60 = 120 unique episodes per strategy
#     (budget mode: prior selections excluded; random uses a fixed seed -> without
#     replacement; kills the coverage asymmetry that produced the earlier null)
#   - selection controls the whole non-expert batch half (MIX_RATIO=1.0)
#   - pool: covariate-shift eval (last 3 snapshots) + reserved MID snapshot eval
#     (paper-style reserved checkpoint, where hardness-seeking can pay)
#   - arms: random / idm (WAV-flavor) / oracle (true pred error, logged actions --
#     the paper's upper bound; if even oracle <= random, the regime is dead)
#   - reuses the existing full-scale DP rollouts + CLAM (no re-collection/retrain)
#
# Usage (cluster, conda env active):  ./run_designD.sh            # all 3 arms, seed 0
#        STRATS="oracle" SEED=1 ./run_designD.sh                  # subset/other seed
set -euo pipefail
cd "$(dirname "$0")"

export SCALE="${SCALE:-full}" TASK="${TASK:-can}" SEED="${SEED:-0}"
export POOL_TAG="${POOL_TAG:-_dd}"
export HOLDOUT_SNAPSHOTS="${HOLDOUT_SNAPSHOTS:-3}" RESERVE_MID=1
export BUDGET_MODE=True MIX_RATIO=1.0
# 2 selection rounds: itr 5000 and 30000; train to 55000 (25k itrs on the full budget).
export SAMPLE_START=5000 REFRESH_OVERRIDE=25000 SEL_OVERRIDE=60 ITRS_OVERRIDE=55000
# reuse the CLAM trained for the can_full pool (idm arm; oracle ignores it)
if [ -z "${CLAM_DIR:-}" ] && [ -f "scratch_dir/pools/${TASK}_${SCALE}/clam_dir.txt" ]; then
  export CLAM_DIR="$(cat "scratch_dir/pools/${TASK}_${SCALE}/clam_dir.txt")"
fi

STRATS="${STRATS:-random idm oracle}"
for STRAT in $STRATS; do
  echo "================ Design D: strategy=$STRAT seed=$SEED ================"
  STRATEGY=$STRAT ./run_faithful.sh 2>&1 | tee "designD_${STRAT}_seed${SEED}.log"
done

POOLS="scratch_dir/pools/${TASK}_${SCALE}${POOL_TAG}"
echo "================ Design D eval (open-loop image MSE, lower=better) ================"
for STRAT in $STRATS; do
  WM="scratch_dir/wm_final_${STRAT}_${SCALE}${POOL_TAG}_seed${SEED}.pt"
  [ -f "$WM" ] || { echo "$STRAT: missing $WM"; continue; }
  for EV in eval_pool eval_pool_mid; do
    for H in 8 31; do
      echo "--- $STRAT / $EV / horizon $H ---"
      python eval_wm_metrics.py "$WM" "$POOLS/${EV}.jsonl" $H | grep -a "openloop\|episodes="
    done
  done
done
echo "================ Design D DONE (seed $SEED) ================"
