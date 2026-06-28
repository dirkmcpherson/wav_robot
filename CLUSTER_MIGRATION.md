# Moving the faithful WAV robot pipeline to a cluster

The pipeline is **code-complete and validated end-to-end locally** (toy + medium scale on a
3060). Moving to a cluster is about (a) carrying the reconstructed code + fixes, (b) the env,
(c) using GPU `egl` rendering, and (d) enough GPU to run paper batch sizes × multiple seeds.

## A. What to copy
**Copy just `wav_robot/` — it is now self-contained** (topology vendored into `datasets/`;
no `wav_minigrid` dependency). It contains all reconstructed glue + patches + the reimplemented
`datasets/` package + `_scorers.py` + drivers + `requirements_faithful.txt`. See
`README_FAITHFUL.md` for the full file manifest.
- The rendered image dataset is large (~1.2 GB/task); cheaper to **re-render on the cluster
  with egl** (minutes) than to copy. See step C.

If instead you fresh-clone upstream, you MUST re-apply these (all small):
- `train_wm.py`: made `MUJOCO_GL` overridable (was hard-coded `egl`).
- `IDM/udrm/utils/dataloader.py`: wrapped `import rlds` in try/except → `rlds=None`
  (avoids the dm-reverb dependency; sailor_pool path already guards on `rlds is None`).
- `wav/training/sample_selector_service.py`: `elif strategy in ("idm","topology")` (inject WM snapshot for topology too).
- NEW files: `datasets/{__init__,build_pools,data_selection,_scorers}.py`,
  `run_faithful.sh`, `eval_wm.py`, `run_lambda_sweep_robot.sh`,
  `setup_faithful_env.sh`, `FAITHFUL_PLAN.md`.

## B. Environment
`conda env create -f env_ymls/robomimic_env.yml` then **`pip install -r requirements_faithful.txt`**
(the EXACT 148-package working set: torch 2.6.0+cu124, robosuite 1.4.1, mujoco 2.3.5,
robomimic/r3m @ pinned commits, tensorflow_cpu, jax, accelerate, hydra-core, timm, diffusers,
…). `setup_faithful_env.sh` does the same + the import gate. Key runtime env:
- **`export MUJOCO_GL=egl`** on the cluster (egl works on proper NVIDIA nodes and is MUCH
  faster than the osmesa CPU fallback we used locally; egl only failed on the local 3060 box).
- `export PYTHONPATH=$PWD:$PWD/IDM` (for `wav`, `udrm`, `datasets` imports).
- `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## C. Data
1. `bash fetch_data.sh` → robomimic `can`(/`square`) `low_dim_v141.hdf5`; symlink to the
   `_done0_` name the loader expects (see FAITHFUL_PLAN). 200 demos/task.
2. Render image obs from states (fast with egl):
   `python -m robomimic.scripts.dataset_states_to_obs --dataset <low_dim_done0_v141.hdf5>
    --output_name image_64_done0_v141.hdf5 --done_mode 0
    --camera_names agentview robot0_eye_in_hand --camera_height 64 --camera_width 64`

## D. Running (paper scale)
Use the idempotent driver (DP collect + build_pools + CLAM trained ONCE per task/seed, reused
across strategies). For each seed, set `wandb_exp_name`/run dir uniquely or use separate
`scratch_dir`s so the `None_demos<N>` paths don't collide across seeds.
```
# per (task, seed): generate shared artifacts + run each strategy
for STRAT in random progress curiosity idm topology; do
  SCALE=full STRATEGY=$STRAT TASK=can SEED=$s ./run_faithful.sh
done
# topology lambda sweep (reuses pool+CLAM): TOPO_LAMBDA in {0.05,0.1,0.2,0.3}
# compare held-out WM loss with eval_wm.py on the shared eval_pool.jsonl
```
`full` scale knobs (in run_faithful.sh): DP 24k steps / 30 snapshots, CLAM 50k, WM 200k,
select 120, refresh 5k. Restore paper batch sizes (DP 256, CLAM 32) — they OOM'd at 12 GB
but fit on >=24 GB. NOTE: also restore `shape_rewards`=paper value (we used False to reuse
the non-shaped dataset locally; on cluster render the `_shaped_` dataset or keep False).

## E. Known-good acceptance checks (run these first on the cluster)
1. `SCALE=toy STRATEGY=topology ./run_faithful.sh` completes "DONE" (≈10 min) — validates env+chain.
2. `idm` (WAV) **beats random** on held-out WM loss — the baseline ordering that must hold
   before topology numbers mean anything (locally WAV did NOT beat random at medium/under-trained scale).
3. Then the topology λ sweep — expect monotone improvement over `idm` (λ=0), as seen locally
   (4822→4779→4768) and decisively in MiniGrid.
