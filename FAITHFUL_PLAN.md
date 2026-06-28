# Faithful robot test of the topology supplement — plan

## Goal
Test whether the topology supplement that works in MiniGrid (real WAV + topology: +16% WM
loss, p<1e-4, n=20) helps in the **real** `wav_robot` setting — using the actual
components, not stand-ins.

## Why the current robot result is not yet a valid test
Faithfulness audit of what's been run so far:

| component   | MiniGrid (faithful) | Synthetic Dreamer | Robomimic (so far) | FAITHFUL needs |
|-------------|--------------------|-------------------|--------------------|----------------|
| FDM         | repo WorldModel ✅  | real Dreamer WM ✅ | real Dreamer WM ✅ | (done)         |
| IDM         | repo SparseIDM ✅   | toy CNN ⚠️         | toy MLP ⚠️         | real **CLAM** ST-ViViT+VQ |
| Base signal | real **WAV** cycle ✅| WM-surprise ⚠️     | WM-surprise ⚠️     | real **WAV** cycle-consistency |
| Data        | repo pool ✅        | synthetic ⚠️       | clean expert demos ⚠️ | diverse **DP-rollout** pools |
| Obs         | grid ✅             | image             | state-only ⚠️      | image |
| Result      | +16% p<1e-4         | helps             | null               | ? |

The robomimic "null" tests Dreamer↔(toy MLP) on clean demos with a WM-surprise base — several
substitutions from the real method. To actually answer the question we must make IDM, base
signal, and data all real. That requires the simulator (robosuite/MuJoCo) + the missing
`datasets/` glue + getting CLAM training working.

## Phases (each has a GATE that must pass before the next)

### Phase 0 — New-location environment
- Linux + NVIDIA GPU (>=16GB ideal, 12GB OK for `can`), recent driver; disk ~50GB for
  rollouts/checkpoints.
- `conda env create -f env_ymls/robomimic_env.yml` (python 3.10, robosuite 1.4.1, mujoco
  2.3.5, robomimic, r3m, torch 2.3.1+cu118). Then pip add: `omegaconf einops termcolor
  imageio wandb x-transformers vector-quantize-pytorch h5py` + `tensorflow-cpu rlds` (CLAM
  data pipeline) + editable `wav_minigrid` (for `topology.py`/`trajectory.py`).
- Headless render: `export MUJOCO_GL=egl` (or `osmesa`).
- GATE: `python -c "import robosuite, robomimic, mujoco"` works; a 1-step env produces a
  rendered image; `import wav.dreamer.wm` and `udrm.models.clam.space_time_clam` still import.

### Phase 1 — The missing `datasets/` package  ← BIGGEST BLOCKER
`wav/training/sample_selector_service.py:24` imports `from datasets.data_selection import
SelectionRequest, run_selection`, and `scripts/build_pools.sh` calls `datasets/build_pools.py`
— neither file is in this checkout (the READMEs say "cd release", implying a fuller tree).
- Option A (fast): obtain the original `release/datasets/` (build_pools.py + data_selection.py).
- Option B (fallback): reimplement. We know the interface (SelectionRequest{sample_pool_jsonl,
  output_jsonl, strategy, select_size, seed, strategy_kwargs}; run_selection→writes top-k
  refs; score_key e.g. `idm_wm_latent_mismatch_mse`). build_pools: dp_rollouts → train/sample/
  eval jsonl referencing `dp_rollout_npz`.
- GATE: build_pools produces the 3 jsonl pools; run_selection scores a pool.

### Phase 2 — Faithful data (diverse DP rollouts)
- Already downloaded: robomimic `can` PH low_dim (`data/robomimic/can/low_dim_v141.hdf5`,
  200 demos). For images, also get image hdf5 or render from states.
- `scripts/dp_collect_robomimic.sh` TASKS="can": trains a diffusion policy on demos, then
  collects diverse rollouts (NOISE_STD=0.10, 30 snapshots) in-sim → `dp_rollouts/*.npz`
  (with image obs). This is the heterogeneous, off-policy data the method actually uses
  (the thing clean expert demos lacked).
- `scripts/build_pools.sh` → pools.
- GATE: pools load via `sample_selector_service.load_pool_episodes_for_wm_local`.

### Phase 3 — Faithful IDM (CLAM)
- Train CLAM from pools: `scripts/train_idm_from_pool.sh` (avoids the TFDS/LIBERO path;
  trains the ST-ViViT+VQ CLAM directly from SAILOR pools). Fallback: TFDS path if pool path
  is incomplete.
- Resolve the Hydra config composition (cfg/train_st_vivit_clam.yaml defaults) and the
  tf/rlds dependency (tensorflow-cpu).
- GATE: CLAM checkpoint trains; can extract IDM latent (`la`/`encoder_out`) on a pool batch.

### Phase 4 — Real WAV base + topology supplement, in the selection service
- Implement the faithful **WAV cycle-consistency** score in the Dreamer setting: action-free
  prior rollout (`dynamics.img_step`) → CLAM-inferred latent action → action-conditioned
  rollout (`imagine_with_action`) → latent mismatch (the `idm_wm_latent_mismatch_mse` key).
- Add the **topology supplement**: combined = rank_norm(WAV) + λ·rank_norm(topology), with
  FDM latent = `get_feat`, IDM latent = CLAM `la`/`encoder_out`, reusing `wav_minigrid.topology`.
- Expose as a `data_selection` strategy so `sample_selector_service` drives it unchanged.
- GATE: selection produces per-episode scores; λ knob works.

### Phase 5 — Run the faithful experiment
- WM-only training loop (`round_loop` wm-only path) on real WM + CLAM + WAV base + diverse
  pools. Compare λ=0 (WAV only) vs λ>0 (WAV+topology), multi-seed (>=5).
- Eval: held-out WM loss; optionally online policy success via sim eval.
- This is the real answer to "does WAV+topology help on the robot benchmark."

### Phase 6 (optional) — full reproduction
`square`, ManiSkill, online eval, match paper numbers.

## What carries over from this location
- Code we wrote: `wav_robot/{bmech.py, mechanism.py, robomimic_mech.py}`, and the topology
  infra in `wav_minigrid/src/wav_minigrid/{topology.py, trajectory.py}` + the additive taps in
  `models/{wm.py,idm.py}` and the Hybrid/Topology strategies in `exps/wm_active_learning.py`.
- Data: `data/robomimic/can/low_dim_v141.hdf5` (re-fetchable; see `fetch_data.sh`).
- Findings: MiniGrid n=20 result + robot stand-in results (see project `CLAUDE.md` + memory).

## Assistance needed (see message)
1. Do you have the original `release/datasets/` (build_pools.py + data_selection.py)? Biggest time-saver.
2. New-location specs: GPU model/VRAM, OS, cluster module system, disk.
3. Scope: minimal-faithful (`can`, WM-only, λ sweep) vs full reproduction.
4. Wandb usage or disable (`USE_WANDB=False`).
5. OK to install the heavy sim env (robosuite/mujoco)?
