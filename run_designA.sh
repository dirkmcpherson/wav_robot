#!/usr/bin/env bash
# Option A — the FINAL robot experiment: Design D regime on a NOISE-FREE, paper-faithful pool.
#
# Design D (budget-locked, oracle arm) showed random < oracle < idm on the old pool: even the
# oracle loses -> hardness selection is counterproductive THERE. The one remaining fidelity gap
# to the paper is the pool: ours had injected action noise (std 0.10); the paper's pool is
# checkpoint-diversity only (App E.3.1). Gate question: oracle > random on the clean pool?
#
# SALVAGE NOTE (2026-07-12): the noise-free collection COMPLETED but landed in seed0's
# dp_rollouts, mixed into the same step_* dirs as the old noisy set (run_faithful.sh didn't
# pass --set seed; fixed since). The two sets are separated by FILENAME TIMESTAMP:
#   old/noisy = traj_2026070*.npz  (collected 2026-07-08/09)
#   new/clean = traj_2026071*.npz  (collected 2026-07-11, noise_std=0)
# CLEAN_GLOB below builds the pool from the clean files only. HAZARD: never rebuild a pool
# from seed0's dp_rollouts with an unfiltered glob — it would mix noisy+clean.
#
# Arms run at SEED=0 (where the rollouts live; also matches Design D's training seed).
# COMPROMISE (for the writeup): the idm arm reuses the CLAM trained on the OLD pool
# (the idm scorer tracked the oracle within ~2%; the ORACLE arm needs no CLAM and is the gate).
#
# Usage (cluster, conda env active):  ./run_designA.sh
set -euo pipefail
cd "$(dirname "$0")"

export SCALE="${SCALE:-full}" TASK="${TASK:-can}" SEED="${SEED:-0}"
export POOL_TAG="${POOL_TAG:-_clean}"
export NOISE_STD=0
export HOLDOUT_SNAPSHOTS="${HOLDOUT_SNAPSHOTS:-3}" RESERVE_MID=1
export BUDGET_MODE=True MIX_RATIO=1.0
export SAMPLE_START=5000 REFRESH_OVERRIDE=25000 SEL_OVERRIDE=60 ITRS_OVERRIDE=55000
# reuse the old-pool CLAM for the idm arm (see COMPROMISE above)
if [ -z "${CLAM_DIR:-}" ] && [ -f "scratch_dir/pools/${TASK}_full/clam_dir.txt" ]; then
  export CLAM_DIR="$(cat "scratch_dir/pools/${TASK}_full/clam_dir.txt")"
fi

POOLS="scratch_dir/pools/${TASK}_${SCALE}${POOL_TAG}"
CLEAN_GLOB="scratch_dir/logs/robomimic__${TASK}/None_demos50/seed0/dp_rollouts/**/traj_2026071*.npz"

# [pre] build the clean pool from the timestamp-filtered glob (clean files only)
if [ ! -f "$POOLS/sample_pool.jsonl" ]; then
  echo "===== [pre] build CLEAN pool from $CLEAN_GLOB ====="
  PYTHONPATH="$PWD" python datasets/build_pools.py \
    --rollouts_glob "$CLEAN_GLOB" --out_dir "$POOLS" \
    --holdout_by_snapshot --eval_snapshots ${HOLDOUT_SNAPSHOTS} --reserve_mid_snapshot
else
  echo "===== [pre] reuse existing clean pool at $POOLS ====="
fi

STRATS="${STRATS:-random idm oracle}"
for STRAT in $STRATS; do
  echo "================ Design A (clean pool): strategy=$STRAT seed=$SEED ================"
  STRATEGY=$STRAT ./run_faithful.sh 2>&1 | tee "designA_${STRAT}_seed${SEED}.log"
done

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
