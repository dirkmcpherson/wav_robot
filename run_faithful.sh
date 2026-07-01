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
. ../.venv_robot_faithful/bin/activate
# Respect an externally-set MUJOCO_GL; default egl (works on cluster GPU nodes). On boxes
# where egl's context teardown fails (e.g. some local GPUs) set MUJOCO_GL=osmesa before running.
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PP="$PWD:$PWD/IDM"

TASK="${TASK:-can}"; SEED="${SEED:-0}"; STRATEGY="${STRATEGY:-topology}"
NUM_EXP_TRAJS="${NUM_EXP_TRAJS:-50}"
DP_BATCH="${DP_BATCH:-32}"; TOPO_LAMBDA="${TOPO_LAMBDA:-0.1}"
case "${SCALE:-medium}" in
  toy)    DP_STEPS=120;   SNAP=2;  SNAP_EVERY=60;   CLAM_UPD=20;    WM_ITRS=30;    SEL=4;   REFRESH=10;  START=5 ;;
  medium) DP_STEPS=6000;  SNAP=10; SNAP_EVERY=600;  CLAM_UPD=10000; WM_ITRS=20000; SEL=64;  REFRESH=2000;START=2000 ;;
  full)   DP_STEPS=24000; SNAP=30; SNAP_EVERY=3000; CLAM_UPD=50000; WM_ITRS=200000;SEL=120; REFRESH=5000;START=5000 ;;
  *) echo "SCALE must be toy|medium|full"; exit 1 ;;
esac
EXP="faithful_${TASK}_${SCALE}"
LOG="scratch_dir/logs/robomimic__${TASK}/${EXP}/seed${SEED}"
POOLS="$PWD/scratch_dir/pools/${TASK}_${SCALE}"
COMMON="--configs cfg_dp_mppi robomimic --task robomimic__${TASK} --num_exp_trajs ${NUM_EXP_TRAJS} --use_wandb False --set done_mode 0 --set shape_rewards False"

# Steps 1-3 produce SHARED artifacts (DP pool + CLAM); idempotent so random/idm/topology reuse them.
if [ ! -f "$POOLS/sample_pool.jsonl" ]; then
  echo "===== [1/4] DP collection (steps=${DP_STEPS}, snapshots=${SNAP}) ====="
  PYTHONPATH="$PP" python -u train_wm.py $COMMON \
    --set train_dp_mppi False --set dp.batch_size ${DP_BATCH} \
    --set dp.train_steps ${DP_STEPS} --set dp.eval_freq $((DP_STEPS/2)) \
    --set dp.rollout_snapshot_count ${SNAP} --set dp.rollout_snapshot_steps ${SNAP_EVERY} \
    --set dp.rollout_snapshot_noise_std 0.10
  echo "===== [2/4] build pools ====="
  PYTHONPATH="$PWD" python datasets/build_pools.py \
    --rollouts_glob "scratch_dir/logs/robomimic__${TASK}/None_demos${NUM_EXP_TRAJS}/seed${SEED}/dp_rollouts/**/*.npz" --out_dir "$POOLS"
else
  echo "===== [1-2/4] reuse existing pools at $POOLS ====="
fi
DPCKPT="$PWD/$(find scratch_dir -name DP_Pretrain_base_policy_latest.pt -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
echo "DP ckpt: $DPCKPT"

if [ ! -f "$POOLS/clam_dir.txt" ]; then
  echo "===== [3/4] CLAM IDM train (updates=${CLAM_UPD}) ====="
  PYTHONPATH="$PP" python IDM/scripts/train_idm_action_decoder.py \
    --config-name train_st_vivit_clam_stm env=robomimic_sailor env.env_id=${TASK} \
    data.source=sailor_pool data.sailor_pool_train_jsonl="$POOLS/train_pool.jsonl" data.sailor_pool_eval_jsonl="$POOLS/eval_pool.jsonl" \
    data.data_dir=./IDM/data data.data_type=n_step data.seq_len=8 data.batch_size="${CLAM_BATCH:-8}" \
    data.num_trajs=-1 env.image_obs=True data.use_images=True data.drop_images_after_obs=True \
    use_wandb=False num_updates=${CLAM_UPD} save_every=$((CLAM_UPD/2))
  ls -dt "$PWD"/results/*/st_vivit_clam_stm/*/ | head -1 > "$POOLS/clam_dir.txt"
else
  echo "===== [3/4] reuse existing CLAM ====="
fi
CLAMD="$(cat "$POOLS/clam_dir.txt")"
echo "CLAM dir: $CLAMD"

echo "===== [4/4] WM-only training (strategy=${STRATEGY}) ====="
KW=$(python -c "import json,sys;print(json.dumps({'suite':'robomimic','task':sys.argv[2],'device':'cuda:0','idm_ckpt_path':sys.argv[1]+'model_ckpts/latest.pkl','idm_config_path':sys.argv[1]+'config.yaml','idm_seq_len':8,'topology_weight':float(sys.argv[3]),'score_key':'idm_wm_latent_mismatch_mse'}))" "$CLAMD" "$TASK" "$TOPO_LAMBDA")
PYTHONPATH="$PP" python -u train_wm.py $COMMON --set wm_only_mode True \
  --set wm_only_pool_jsonl "$POOLS/train_pool.jsonl" --set wm_only_sample_source_pool_jsonl "$POOLS/sample_pool.jsonl" \
  --set wm_only_eval_pool_jsonl "$POOLS/eval_pool.jsonl" \
  --set wm_only_sample_select_strategy ${STRATEGY} --set wm_only_sample_select_size ${SEL} \
  --set wm_only_sample_select_kwargs_json "$KW" \
  --set wm_only_sample_start_itr ${START} --set wm_only_sample_refresh_every ${REFRESH} --set wm_only_sample_mix_ratio 0.3 \
  --set wm_only_train_itrs ${WM_ITRS} --set wm_eval_every $((REFRESH)) \
  --set dp.rollout_snapshot_count 0 --set train_dp_mppi_params.use_discrim False --set dp.pretrained_ckpt "$DPCKPT"

# Stash the final WM per (strategy,scale) BEFORE the next strategy overwrites latest_residual_checkpoint.pt.
WM_SRC="scratch_dir/logs/robomimic__${TASK}/None_demos${NUM_EXP_TRAJS}/seed${SEED}/latest_residual_checkpoint.pt"
if [ -f "$WM_SRC" ]; then
  cp "$WM_SRC" "scratch_dir/wm_final_${STRATEGY}_${SCALE}.pt"
  echo "stashed WM -> scratch_dir/wm_final_${STRATEGY}_${SCALE}.pt"
else
  echo "WARN: WM checkpoint not found at $WM_SRC (stash skipped)"
fi
echo "===== DONE (${STRATEGY}, ${SCALE}) ====="
