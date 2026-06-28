# Faithful WAV robot pipeline — file manifest & quickstart (cluster-ready)

`wav_robot/` is **self-contained**: copy this directory to the cluster and it runs (no
`wav_minigrid` needed — `topology.py` is vendored into `datasets/`).

## Files ADDED for the faithful pipeline (all reimplemented; not in upstream)
- `datasets/__init__.py`
- `datasets/build_pools.py`      — dp_rollouts → train/sample/eval `*.jsonl` pools
- `datasets/data_selection.py`   — `SelectionRequest` + `run_selection` (random + dispatch)
- `datasets/_scorers.py`         — WAV `idm` latent-mismatch scorer + `topology` supplement
- `datasets/topology.py`         — vendored relational topology metrics (RSA/CKA/dCor/kNN)
- `run_faithful.sh`              — one-command idempotent driver (DP→pools→CLAM→WM); SCALE=toy|medium|full, STRATEGY=random|idm|topology
- `run_lambda_sweep_robot.sh`    — topology λ sweep (reuses pool+CLAM)
- `eval_wm.py`                   — post-hoc held-out WM eval (fair cross-strategy metric)
- `setup_faithful_env.sh`        — env build (conda yml + discovered deps + fixes)
- `requirements_faithful.txt`    — EXACT pinned working set (148 pkgs) for `pip install -r`
- `fetch_data.sh`                — download robomimic demos
- `FAITHFUL_PLAN.md`, `CLUSTER_MIGRATION.md`, this file

## Files PATCHED in upstream (small, in-place — present if you copy the dir)
- `train_wm.py`                          — `MUJOCO_GL` overridable (was hard-coded egl)
- `IDM/udrm/utils/dataloader.py`         — `import rlds` → try/except (avoids dm-reverb)
- `wav/training/sample_selector_service.py` — inject WM snapshot for `topology` too

## Quickstart on the cluster
```bash
# 1. env  (prefer the exact pins; conda yml provides robosuite/mujoco/robomimic/r3m)
conda env create -n wav_robot_faithful -f env_ymls/robomimic_env.yml
conda activate wav_robot_faithful
pip install -r requirements_faithful.txt          # exact working set
export MUJOCO_GL=egl PYTHONPATH=$PWD:$PWD/IDM PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. data
bash fetch_data.sh
ln -sf $PWD/data/robomimic/can/low_dim_v141.hdf5 \
       datasets/robomimic_datasets/can/ph/low_dim_done0_v141.hdf5
python -m robomimic.scripts.dataset_states_to_obs \
  --dataset datasets/robomimic_datasets/can/ph/low_dim_done0_v141.hdf5 \
  --output_name image_64_done0_v141.hdf5 --done_mode 0 \
  --camera_names agentview robot0_eye_in_hand --camera_height 64 --camera_width 64

# 3. acceptance check (~10 min), then real runs
SCALE=toy STRATEGY=topology ./run_faithful.sh
for STRAT in random idm topology; do SCALE=full STRATEGY=$STRAT TASK=can SEED=0 ./run_faithful.sh; done
for s in random idm topology; do python eval_wm.py scratch_dir/wm_final_${s}_*.pt scratch_dir/pools/can_full/eval_pool.jsonl; done
```
For paper scale restore DP batch 256 / CLAM batch 32 (need ≥24 GB VRAM). GPU sizing + the
known-good acceptance checks are in `CLUSTER_MIGRATION.md`.
