#!/bin/bash
#   sbatch probe256.sbatch     ;  results in probe256_<jobid>.out (see the PROBE REPORT block)
#SBATCH --job-name=wav_probe256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=12
#SBATCH --mem=256g
#SBATCH --time=02:00:00                 # render(12 demos) + tiny probe run; generous
#SBATCH --output=probe256_%j.out
#SBATCH --error=probe256_%j.out

set -uo pipefail
cd /cluster/tufts/shortlab/jstale02/wav_robot
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /cluster/tufts/shortlab/jstale02/condaenv/wav_robot_faithful

export MUJOCO_GL=egl
export PYTHONPATH="$PWD:$PWD/IDM"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# If the default WM batch OOMs at 256^2, resubmit with e.g.  --export=ALL,WM_BATCH=4
./run_probe256.sh
