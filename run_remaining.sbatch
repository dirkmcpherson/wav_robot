#!/bin/bash
# Runs the two remaining faithful conditions (random, idm) sequentially, reusing the DP
# pool + CLAM already built by the topology run, stashes each WM, then evals all three.
# ==== EDIT THE 5 SBATCH LINES MARKED <EDIT> FOR YOUR CLUSTER, then: sbatch run_remaining.sbatch
#SBATCH --job-name=wav_can_full
#SBATCH --partition=gpu                 # <EDIT> your GPU partition (sinfo to list)
#SBATCH --gres=gpu:1                    # <EDIT> e.g. gpu:a100:1 ; WM-only fits <=16GB
#SBATCH --cpus-per-task=8               # <EDIT> data loading / sim
#SBATCH --mem=48G                       # <EDIT>
#SBATCH --time=24:00:00                 # <EDIT> 2 WM runs (~hrs each) + eval; lower if your partition caps
#SBATCH --output=slurm_wav_%j.out
#SBATCH --error=slurm_wav_%j.err

set -uo pipefail

# ---- paths (EDIT if different) ----
WAVDIR=/cluster/tufts/shortlab/jstale02/wav_robot
CONDA_ENV=/cluster/tufts/shortlab/jstale02/condaenv/wav_robot_faithful
SCALE=full
TASK=can
NUM_EXP_TRAJS=50
SEED=0

cd "$WAVDIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export MUJOCO_GL=egl
export PYTHONPATH="$PWD:$PWD/IDM"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="scratch_dir/logs/robomimic__${TASK}/None_demos${NUM_EXP_TRAJS}/seed${SEED}/latest_residual_checkpoint.pt"
POOL="scratch_dir/pools/${TASK}_${SCALE}/eval_pool.jsonl"

# Best-effort: preserve the already-completed topology WM before anything overwrites it.
if [ ! -f "scratch_dir/wm_final_topology_${SCALE}.pt" ] && [ -f "$CKPT" ]; then
  cp "$CKPT" "scratch_dir/wm_final_topology_${SCALE}.pt" && echo "stashed existing topology WM"
fi

# Run the two remaining strategies (idempotent driver reuses the shared DP pool + CLAM),
# stashing each WM before the next overwrites the shared checkpoint path.
set -e
for STRAT in random idm; do
  echo "================ running strategy=$STRAT (scale=$SCALE) ================"
  SCALE=$SCALE TASK=$TASK STRATEGY=$STRAT ./run_faithful.sh
  cp "$CKPT" "scratch_dir/wm_final_${STRAT}_${SCALE}.pt"
  echo "stashed -> scratch_dir/wm_final_${STRAT}_${SCALE}.pt"
done
set +e

echo "================ held-out WM eval (lower = better) ================"
for s in random idm topology; do
  echo -n "$s: "
  python eval_wm.py "scratch_dir/wm_final_${s}_${SCALE}.pt" "$POOL" 2>/dev/null | grep -a "held-out" || echo "(missing wm_final_${s}_${SCALE}.pt)"
done
echo "================ DONE ================"
