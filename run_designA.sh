#!/usr/bin/env bash
# Option A — the FINAL robot experiment: Design D regime on a NOISE-FREE, paper-faithful pool.
#
# Design D (budget-locked, oracle arm) showed random < oracle < idm on the old pool: even the
# oracle loses -> hardness selection is counterproductive THERE. The one remaining fidelity gap
# to the paper is the pool itself: ours was collected with injected action noise (std 0.10);
# the paper's pool is checkpoint-diversity only (App E.3.1, no noise). This run re-collects
# with noise_std=0 (diversity from 30 linspace DP snapshots incl. step 0 + env resets + DP
# sampling stochasticity) and reruns the three budget-locked arms.
#   oracle > random on the clean pool  -> regime validates; noise was the poison; ladder reopens.
#   random still wins even vs oracle   -> strong negative: hardness selection loses at this
#                                         scale even with oracle knowledge on a faithful pool.
#
# SEED=1 gives the collection its own logdir (seed0's dp_rollouts hold the old noisy set).
# COMPROMISE (noted for the writeup): the idm arm reuses the CLAM trained on the OLD pool
# (retraining is hours; the idm scorer tracked the oracle within ~2% anyway). The ORACLE arm
# needs no CLAM and is the gate. If the gate flips, retrain CLAM on the clean pool before
# any topology work.
#
# Usage (cluster, conda env active):  ./run_designA.sh
set -euo pipefail
cd "$(dirname "$0")"

export SCALE="${SCALE:-full}" TASK="${TASK:-can}" SEED="${SEED:-1}"
export POOL_TAG="${POOL_TAG:-_clean}"
export NOISE_STD=0
export HOLDOUT_SNAPSHOTS="${HOLDOUT_SNAPSHOTS:-3}" RESERVE_MID=1
export BUDGET_MODE=True MIX_RATIO=1.0
export SAMPLE_START=5000 REFRESH_OVERRIDE=25000 SEL_OVERRIDE=60 ITRS_OVERRIDE=55000
# reuse the old-pool CLAM for the idm arm (see COMPROMISE above)
if [ -z "${CLAM_DIR:-}" ] && [ -f "scratch_dir/pools/${TASK}_full/clam_dir.txt" ]; then
  export CLAM_DIR="$(cat "scratch_dir/pools/${TASK}_full/clam_dir.txt")"
fi

STRATS="${STRATS:-random idm oracle}"
for STRAT in $STRATS; do
  echo "================ Design A (clean pool): strategy=$STRAT seed=$SEED ================"
  STRATEGY=$STRAT ./run_faithful.sh 2>&1 | tee "designA_${STRAT}_seed${SEED}.log"
done

POOLS="scratch_dir/pools/${TASK}_${SCALE}${POOL_TAG}"
echo "================ Design A eval (open-loop image MSE, lower=better) ================"
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
echo "================ Design A DONE (seed $SEED) ================"
