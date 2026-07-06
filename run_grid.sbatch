#!/bin/bash
# Faithful WAV grid as a SLURM job array: one task per SEED (= array index), each task runs
# its seed's 3 strategies sequentially (they share a per-seed logdir, so must be serial),
# different seeds run in parallel on separate GPUs. DP pool + CLAM are cached -> WM-only.
# Idempotent: skips any (strategy,seed) already stashed, so resubmits just fill gaps.
#
#   sbatch --array=0-1 run_grid.sbatch     # seeds 0 and 1  (n=2)
#   sbatch --array=1   run_grid.sbatch     # just seed 1    (if seed 0 is running manually)
#   sbatch --array=1-2 run_grid.sbatch     # seeds 1,2      (extend to n=3)
# ==== EDIT THE SBATCH LINES MARKED <EDIT> FOR YOUR CLUSTER ====
#SBATCH --job-name=wav_grid
#SBATCH --partition=gpu                 # <EDIT> your GPU partition (sinfo -s)
#SBATCH --gres=gpu:1                    # 1 H200 per seed/array-task
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00                 # per task = <=3 WM-only runs; DP/CLAM cached
#SBATCH --array=0-1                     # override on the command line if you like
#SBATCH --output=slurm_wav_seed%a_%A.out
#SBATCH --error=slurm_wav_seed%a_%A.err

set -uo pipefail

# ---- paths (EDIT if different) ----
WAVDIR=/cluster/tufts/shortlab/jstale02/wav_robot
CONDA_ENV=/cluster/tufts/shortlab/jstale02/condaenv/wav_robot_faithful
SCALE=full
TASK=can

cd "$WAVDIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export MUJOCO_GL=egl
export PYTHONPATH="$PWD:$PWD/IDM"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED="${SLURM_ARRAY_TASK_ID:-0}"
echo "================ grid task: seed=$SEED (scale=$SCALE, task=$TASK) ================"

for STRAT in topology random idm; do
  STASH="scratch_dir/wm_final_${STRAT}_${SCALE}_seed${SEED}.pt"
  if [ -f "$STASH" ]; then
    echo "----- skip ${STRAT} seed${SEED}: already stashed ($STASH) -----"
    continue
  fi
  echo "----- running ${STRAT} seed${SEED} -----"
  SCALE=$SCALE TASK=$TASK STRATEGY=$STRAT SEED=$SEED ./run_faithful.sh
  # run_faithful.sh (patched) auto-stashes to $STASH; sanity-check it landed.
  [ -f "$STASH" ] || echo "WARN: expected stash missing after run: $STASH"
done

echo "================ seed=$SEED done ================"
# Eval is cross-seed, so run it once after ALL array tasks finish (not here):
#   for SEED in 0 1; do for s in topology idm random; do
#     echo -n "$s seed$SEED: "
#     python eval_wm.py scratch_dir/wm_final_${s}_${SCALE}_seed${SEED}.pt scratch_dir/pools/${TASK}_${SCALE}/eval_pool.jsonl
#   done; done
