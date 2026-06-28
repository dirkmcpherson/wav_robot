#!/usr/bin/env bash
# Phase 0: build the faithful robot env at the new location. Starting point — expect to
# iterate (sim installs are finicky). Run from the wav_robot/ directory.
set -euo pipefail
ENV=${ENV:-wav_robot_faithful}

echo "== creating conda env from robomimic_env.yml =="
conda env create -n "$ENV" -f env_ymls/robomimic_env.yml || \
  echo "  (env may exist; continuing)"

echo "== activate and add deps =="
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$ENV"

# Dreamer WM + CLAM IDM + topology infra deps (kept off the sim's CUDA: tensorflow-cpu)
pip install omegaconf einops termcolor imageio wandb \
            x-transformers vector-quantize-pytorch h5py \
            tensorflow-cpu rlds
# EXTRA deps discovered while driving train_wm.py (NOT in robomimic_env.yml; the diffusion
# policy + config loader need these): ruamel.yaml, timm, diffusers, gym 0.26.2.
# (robosuite/robomimic/mujoco/r3m/gym are in the yml; if NOT using conda, also pip-install
#  mujoco==2.3.5 robosuite==1.4.1 robomimic@<pin> r3m@<pin> gym==0.26.2 first.)
pip install "ruamel.yaml" timm diffusers "gym==0.26.2"
# CLAM IDM trainer (IDM/udrm) deps discovered by driving train_idm_action_decoder.py:
pip install accelerate hydra-core ml_collections moviepy safetensors dm-tree \
            opencv-python imageio-ffmpeg absl-py gdown torchsummary \
            jaxtyping typeguard jax jaxlib
# GOTCHA: IDM/udrm/utils/dataloader.py top-level `import rlds` pulls dm-reverb (heavy,
# TF-pinned) but is only needed for the TFDS path; the sailor_pool path guards on
# `rlds is None`. We wrapped that import in try/except -> rlds=None (see the file).
# NOTE: topology.py is now VENDORED into wav_robot/datasets/ — no wav_minigrid needed.

echo "== headless rendering =="
echo "  USE: export MUJOCO_GL=osmesa   (egl context-teardown errored on the 3060 box;"
echo "       on a cluster GPU egl is faster — try egl first, fall back to osmesa)"

# --- RUN GOTCHAS discovered locally (Phase 1/2) ---
# * train_wm.py drives the whole pipeline (train_sailor.py is missing but unneeded);
#   collection = DP training with rollout snapshots, via: --set train_dp_mppi False
#   --set dp.rollout_snapshot_count 30 --set dp.rollout_snapshot_steps 3000
#   --set dp.rollout_snapshot_noise_std 0.10   (saved under logdir/dp_rollouts/step_*/)
# * Expert dataset path: datasets/robomimic_datasets/<task>/ph/<name>_v141.hdf5 where
#   name = image_<image_size>[_shaped]_done<done_mode>. Config wants image_size=64,
#   shape_rewards -> "_shaped". We have image_64_done0 (no shaped); either render with
#   robomimic dataset_states_to_obs --shaped, OR --set shape_rewards False.
# * IMAGE data is NOT downloadable (404) -- render locally from low_dim states:
#   python -m robomimic.scripts.dataset_states_to_obs --dataset <low_dim_done0_v141.hdf5>
#     --output_name image_64_done0_v141.hdf5 --done_mode 0
#     --camera_names agentview robot0_eye_in_hand --camera_height 64 --camera_width 64
#   (~50 min for 200 demos on CPU/osmesa; much faster with GPU egl rendering on cluster.)
# * DP train batch_size=256 OOMs on a 12GB 3060 -> needs --set dp.batch_size 32 locally
#   (a faithfulness compromise). A >=24GB cluster GPU runs paper batch sizes -> USE CLUSTER
#   for the faithful run.

echo "== GATE checks =="
export MUJOCO_GL=${MUJOCO_GL:-egl}
python - <<'PY'
ok=True
for m in ["robosuite","robomimic","mujoco","torch","wav.dreamer.wm",
          "udrm.models.clam.space_time_clam","datasets.topology","datasets._scorers"]:
    try:
        __import__(m); print("OK  ", m)
    except Exception as e:
        ok=False; print("FAIL", m, "->", type(e).__name__, e)
print("PHASE-0 GATE", "PASSED" if ok else "FAILED (fix imports before Phase 1)")
PY
echo "NOTE: also verify a 1-step robosuite env renders an image (offscreen) before Phase 2."
echo "      PYTHONPATH must include wav_robot root + wav_robot/IDM for 'wav'/'udrm' imports."
