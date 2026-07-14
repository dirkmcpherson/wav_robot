#!/usr/bin/env bash
# One-command faithful WAV pipeline (all steps validated at toy scale, 2026-06-27).
# Chains: DP collection -> build_pools -> CLAM IDM train -> WM-only training w/ selection.
# Scale via env (defaults = MEDIUM, a real-but-tractable run on the 12GB 3060).
#
# Usage:
#   STRATEGY=topology SCALE=medium ./run_faithful.sh
#   (STRATEGY in random|idm|topology ; idm/topology reuse the CLAM ckpt from step 3)
set -euo pipefail
cd "$(dirname "$0")"
# Local dev uses a venv; on the cluster the conda env is already active. Only source the
# venv if it exists — otherwise assume the caller (e.g. the sbatch) already activated the env.
if [ -f ../.venv_robot_faithful/bin/activate ]; then
  . ../.venv_robot_faithful/bin/activate
fi
# Respect an externally-set MUJOCO_GL; default egl (works on cluster GPU nodes). On boxes
# where egl's context teardown fails (e.g. some local GPUs) set MUJOCO_GL=osmesa before running.
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PP="$PWD:$PWD/IDM"

TASK="${TASK:-can}"; SEED="${SEED:-0}"; STRATEGY="${STRATEGY:-topology}"
NUM_EXP_TRAJS="${NUM_EXP_TRAJS:-50}"
DP_BATCH="${DP_BATCH:-32}"; TOPO_LAMBDA="${TOPO_LAMBDA:-0.1}"
# Stage-A positive-control knobs (defaults preserve the original behavior):
#   MIX_RATIO         selected-data dose in the non-expert batch half (was hard-coded 0.3)
#   HOLDOUT_SNAPSHOTS >0 -> eval = last K DP snapshots (covariate shift) vs the i.i.d. split
#   POOL_TAG          suffix for pool dir + WM stash so new pools/WMs don't clobber old ones
#   SAMPLE_START      override selection start iter ;  CLAM_DIR  reuse an existing CLAM dir
MIX_RATIO="${MIX_RATIO:-0.3}"; HOLDOUT_SNAPSHOTS="${HOLDOUT_SNAPSHOTS:-0}"; POOL_TAG="${POOL_TAG:-}"
# Image resolution: 64 (reduced scale) or 256 (faithful paper reproduction). Requires the
# matching image_<IMG_SIZE>_done0 expert hdf5 rendered on disk (see render_images.sh).
IMG_SIZE="${IMG_SIZE:-64}"; export WAV_IMG_SIZE="${IMG_SIZE}"
case "${SCALE:-medium}" in
  toy)    DP_STEPS=120;   SNAP=2;  SNAP_EVERY=60;   CLAM_UPD=20;    WM_ITRS=30;    SEL=4;   REFRESH=10;  START=5 ;;
  probe)  DP_STEPS=400;   SNAP=3;  SNAP_EVERY=133;  CLAM_UPD=300;   WM_ITRS=300;   SEL=8;   REFRESH=150; START=100 ;;
  medium) DP_STEPS=6000;  SNAP=10; SNAP_EVERY=600;  CLAM_UPD=10000; WM_ITRS=20000; SEL=64;  REFRESH=2000;START=2000 ;;
  full)   DP_STEPS=24000; SNAP=30; SNAP_EVERY=3000; CLAM_UPD=50000; WM_ITRS=200000;SEL=120; REFRESH=5000;START=5000 ;;
  paper)  DP_STEPS=24000; SNAP=30; SNAP_EVERY=3000; CLAM_UPD=500000;WM_ITRS=200000;SEL=120; REFRESH=5000;START=5000 ;;
  *) echo "SCALE must be toy|probe|medium|full|paper"; exit 1 ;;
esac
START="${SAMPLE_START:-$START}"   # allow overriding the selection start iter
# Design D overrides: budget-locked acquisition (see wm_only_sample_budget_mode) + custom cadence.
SEL="${SEL_OVERRIDE:-$SEL}"; REFRESH="${REFRESH_OVERRIDE:-$REFRESH}"; WM_ITRS="${ITRS_OVERRIDE:-$WM_ITRS}"
BUDGET_MODE="${BUDGET_MODE:-False}"; RESERVE_MID="${RESERVE_MID:-0}"
EXP="faithful_${TASK}_${SCALE}"
LOG="scratch_dir/logs/robomimic__${TASK}/${EXP}/seed${SEED}"
POOLS="$PWD/scratch_dir/pools/${TASK}_${SCALE}${POOL_TAG}"
ROLLOUTS_DIR="scratch_dir/logs/robomimic__${TASK}/None_demos${NUM_EXP_TRAJS}/seed${SEED}/dp_rollouts"
# NOTE: --set seed is REQUIRED for SEED to reach train_wm.py (its logdir seed comes from
# config, default 0). Without it, a SEED=1 run trains/collects into .../seed0/ (bug hit 2026-07-11).
COMMON="--configs cfg_dp_mppi robomimic --task robomimic__${TASK} --num_exp_trajs ${NUM_EXP_TRAJS} --use_wandb False --set done_mode 0 --set shape_rewards False --set seed ${SEED} --set image_size ${IMG_SIZE}"

# [1] DP collection — reuse existing rollouts if present (never re-collect for a new POOL_TAG),
# and skip entirely when the pool is already built (a pre-built pool needs no rollout dir).
if [ ! -f "$POOLS/sample_pool.jsonl" ] && [ -z "$(find "$ROLLOUTS_DIR" -name '*.npz' -print -quit 2>/dev/null)" ]; then
  echo "===== [1/4] DP collection (steps=${DP_STEPS}, snapshots=${SNAP}) ====="
  PYTHONPATH="$PP" python -u train_wm.py $COMMON \
    ${NUM_ENVS:+--set num_envs ${NUM_ENVS}} \
    --set train_dp_mppi False --set dp.batch_size ${DP_BATCH} \
    --set dp.train_steps ${DP_STEPS} --set dp.eval_freq $((DP_STEPS/2)) \
    --set dp.rollout_snapshot_count ${SNAP} --set dp.rollout_snapshot_steps ${SNAP_EVERY} \
    --set dp.rollout_snapshot_noise_std ${NOISE_STD:-0.10}
else
  echo "===== [1/4] reuse existing DP rollouts in $ROLLOUTS_DIR ====="
fi

# [2] build pools — skip if this pool dir already built (POOL_TAG isolates new splits).
if [ ! -f "$POOLS/sample_pool.jsonl" ]; then
  echo "===== [2/4] build pools (holdout_snapshots=${HOLDOUT_SNAPSHOTS}) -> $POOLS ====="
  HOLD_ARGS=""
  if [ "${HOLDOUT_SNAPSHOTS}" -gt 0 ]; then HOLD_ARGS="--holdout_by_snapshot --eval_snapshots ${HOLDOUT_SNAPSHOTS}"; fi
  if [ "${RESERVE_MID}" = "1" ]; then HOLD_ARGS="$HOLD_ARGS --reserve_mid_snapshot"; fi
  PYTHONPATH="$PWD" python datasets/build_pools.py \
    --rollouts_glob "$ROLLOUTS_DIR/**/*.npz" --out_dir "$POOLS" $HOLD_ARGS
else
  echo "===== [2/4] reuse existing pools at $POOLS ====="
fi
DPCKPT="$PWD/$(find scratch_dir -name DP_Pretrain_base_policy_latest.pt -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
echo "DP ckpt: $DPCKPT"

if [ ! -f "$POOLS/clam_dir.txt" ]; then
  if [ -n "${CLAM_DIR:-}" ]; then
    echo "===== [3/4] reuse CLAM from \$CLAM_DIR ($CLAM_DIR) ====="
    echo "$CLAM_DIR" > "$POOLS/clam_dir.txt"
  else
    echo "===== [3/4] CLAM IDM train (updates=${CLAM_UPD}) ====="
    PYTHONPATH="$PP" python IDM/scripts/train_idm_action_decoder.py \
      --config-name train_st_vivit_clam_stm env=robomimic_sailor env.env_id=${TASK} \
      data.source=sailor_pool data.sailor_pool_train_jsonl="$POOLS/train_pool.jsonl" data.sailor_pool_eval_jsonl="$POOLS/eval_pool.jsonl" \
      data.data_dir=./IDM/data data.data_type=n_step data.seq_len=8 data.batch_size="${CLAM_BATCH:-8}" \
      data.num_trajs=-1 env.image_obs=True data.use_images=True data.drop_images_after_obs=True \
      use_wandb=False num_updates=${CLAM_UPD} save_every=$((CLAM_UPD/2))
    ls -dt "$PWD"/results/*/st_vivit_clam_stm/*/ | head -1 > "$POOLS/clam_dir.txt"
  fi
else
  echo "===== [3/4] reuse existing CLAM ====="
fi
CLAMD="$(cat "$POOLS/clam_dir.txt")"
echo "CLAM dir: $CLAMD"

echo "===== [4/4] WM-only training (strategy=${STRATEGY}) ====="
KW=$(python -c "import json,sys;print(json.dumps({'suite':'robomimic','task':sys.argv[2],'device':'cuda:0','idm_ckpt_path':sys.argv[1]+'model_ckpts/latest.pkl','idm_config_path':sys.argv[1]+'config.yaml','idm_seq_len':8,'topology_weight':float(sys.argv[3]),'score_key':'idm_wm_latent_mismatch_mse'}))" "$CLAMD" "$TASK" "$TOPO_LAMBDA")
PYTHONPATH="$PP" python -u train_wm.py $COMMON --set wm_only_mode True \
  ${WM_BATCH:+--set batch_size ${WM_BATCH}} \
  --set wm_only_pool_jsonl "$POOLS/train_pool.jsonl" --set wm_only_sample_source_pool_jsonl "$POOLS/sample_pool.jsonl" \
  --set wm_only_eval_pool_jsonl "$POOLS/eval_pool.jsonl" \
  --set wm_only_sample_select_strategy ${STRATEGY} --set wm_only_sample_select_size ${SEL} \
  --set wm_only_sample_select_kwargs_json "$KW" \
  --set wm_only_sample_start_itr ${START} --set wm_only_sample_refresh_every ${REFRESH} --set wm_only_sample_mix_ratio ${MIX_RATIO} \
  --set wm_only_sample_budget_mode ${BUDGET_MODE} \
  --set wm_only_train_itrs ${WM_ITRS} --set wm_eval_every $((REFRESH)) \
  --set dp.rollout_snapshot_count 0 --set train_dp_mppi_params.use_discrim False --set dp.pretrained_ckpt "$DPCKPT"

# Stash the final WM per (strategy,scale,seed) BEFORE the next strategy overwrites
# latest_residual_checkpoint.pt. The seed in the name lets concurrent multi-seed runs
# (e.g. seed0 on one GPU, seed1 on another) coexist without clobbering each other's stash.
WM_SRC="scratch_dir/logs/robomimic__${TASK}/None_demos${NUM_EXP_TRAJS}/seed${SEED}/latest_residual_checkpoint.pt"
WM_DST="scratch_dir/wm_final_${STRATEGY}_${SCALE}${POOL_TAG}_seed${SEED}.pt"
if [ -f "$WM_SRC" ]; then
  cp "$WM_SRC" "$WM_DST"
  echo "stashed WM -> $WM_DST"
else
  echo "WARN: WM checkpoint not found at $WM_SRC (stash skipped)"
fi
echo "===== DONE (${STRATEGY}, ${SCALE}) ====="
