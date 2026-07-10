# WAV robot reproduction — neutral board brief (2026-07-09)

A prior session (with a 4-reviewer neutral panel) worked through the robot reproduction of
"World Action Verifier" (WAV). This brief is the **accurate current state** for a fresh board
to comment on. Treat prior conclusions as claims to verify. Numbers below supersede any in
`CLAUDE.md`, some of which were produced by a since-fixed eval bug (flagged inline).

## The question for the board
1. Is the robot result "WAV selection ≈ random (slightly worse) on held-out WM prediction" a
   **real limitation**, a **reproduction artifact**, or a **metric/regime mismatch**?
2. Given the signal is confirmed informative but unhelpful, which is worth doing:
   (A) test the collection-noise confound (re-collect a low/no-noise pool + retrain),
   (B) stand on the MiniGrid result and report the robot as a diagnosed null, or
   (C) build the paper's actual active-exploration WAV (subgoal generator + self-improving loop)?
3. Is the topology contribution (validated in MiniGrid) sound and worth foregrounding?

## What WAV is (paper: `wav_paper.pdf` in repo)
Three-component goal-oriented cycle: a **video-prior subgoal generator** `g_φ` (trained on
action-free video) proposes plausible subgoals; a **sparse inverse model** `h_ψ` infers actions
to reach them; the **forward world model** `f_θ` rolls forward; disagreement = `dist(subgoal,
forward_rollout)`; the **max-disagreement action is executed to collect new data** (self-improving
loop, 2 exploration rounds, 3 seeds). Headline metric: **downstream policy reward (+22%)** and
**32-frame open-loop WM prediction MSE**. Robot tasks include robomimic **Can**. Baselines: RND,
Uncertainty, Progress, Vanilla-IDM, WAV — **no uniform-random baseline** (on Can WM error:
WAV 18.2 < Vanilla-IDM 20.2 < Progress 22.0 < Uncertainty 22.9 < RND 26.5).

## The POSITIVE result — MiniGrid (holds, unaffected by anything below)
Full faithful 3-component WAV; eval on an **independent fixed test set under a hard labeling
budget** (a genuine active-learning regime); signal validated at **+0.82** corr with true WM
error. WAV beats random; the **topology supplement** (`rank_norm(idm) + λ·rank_norm(topo)`, λ≈0.05)
further improves WAV by ~16% (paired, p<1e-4, n=20). This is the sound contribution.

## The ROBOT reproduction — corrected state
**Provenance / faithfulness caveat.** `wav_robot` is the authors' code, but the data-selection
glue was missing and was **reimplemented** (`datasets/_scorers.py`, `data_selection.py`,
`build_pools.py`). The robot `idm` scorer is a **2-component** forward-inverse consistency
(`‖WM_onestep(IDM_action) − observed_next_latent‖²`) with **no subgoal generator** — it anchors on
the *real* next state, not an imagined subgoal. So it is ≈ the paper's **"Vanilla IDM"**, not full
WAV. (Dreamer WM + CLAM IDM are the authors' original code.)

**What was measured (all robomimic Can, full scale, seed 0):**

1. **Original `eval_wm.py`:** random 4312 < topology 4584 < idm 4868 (~13% spread).
   → **ARTIFACT. Discard.** Two bugs: `eval_wm.py` passed `wm.preprocess(b)` into
   `evaluate_batch_metrics`, which preprocesses again → images `/255` **twice** (≈255× too dark,
   OOD); and it reported the composite `model_loss` (recon NLL **+ KL**) mislabeled as "image loss"
   (there is no `image_loss` key). `eval_wm.py` is superseded by `eval_wm_metrics.py`.

2. **Corrected metric** (`eval_wm_metrics.py`: open-loop image MSE, single preprocess),
   **i.i.d.-split eval:** random ≈ idm ≈ topology, **all within noise** — openloop ~0.0092 for all
   three; `model_loss` ~118 (not 4800). The 13% ordering was purely the bug. **NULL.**

3. **Covariate-shift eval** (`build_pools --holdout_by_snapshot`: hold out the last 3 DP snapshots,
   paper-style) **+ higher dose** (`mix_ratio` 0.3→0.6), **retrained** WMs:
   openloop **random 0.00662 vs idm 0.00689** (idm +4.1%, ≈0.9 SE); idm uniformly a hair **worse**
   on every image metric (agentview +2.5%, eye-in-hand +4.9%, model_loss +5.2%). **NULL / slightly
   negative.** → eval-design and dose are **not** the blocker.

4. **Signal diagnostic** (`validate_idm_signal.py`, n=552 pool episodes): the `idm` score **is
   informative** — Pearson **+0.44** with per-episode WM open-loop error, **precision@10 = 0.70**
   (chance ≈0.018, ~39×), prec@25 0.48, prec@50 0.44 (~5×). Weaker than MiniGrid's +0.82 but real,
   and strongest at the top where selection acts. → **the CLAM/signal quality is NOT the blocker**
   (so retraining CLAM to paper's 500k, "Stage B", is not expected to help).

**Working diagnosis (for the board to scrutinize, not accept).** The `idm` signal correctly finds
high-WM-error episodes, yet training on them mildly *hurts* held-out prediction ⇒ those episodes
are **hard-but-not-helpful** (irreducible/aleatoric hardness). Mechanism: the pool is collected
with injected action noise (`dp.rollout_snapshot_noise_std=0.10`) across early→late policy
snapshots; the held-out eval is the **cleanest last snapshots**; so `idm` selects the noisy /
off-policy episodes, which don't transfer to the clean eval. WAV's premise (disagreement ⇒
*informative*) holds in clean, deterministic MiniGrid but is **confounded with collection noise**
in this **passive pool-filtering** robot regime. Fundamental gap vs the paper: paper WAV
**actively collects** fresh data by executing max-disagreement actions (+ subgoal generator); our
reimplementation passively **filters a fixed noisy pool**.

## Scale caveats (all applied equally across strategies, so not differential confounds, but they
cap faithfulness): CLAM 50k updates (paper 500k), images 64² (paper 256²), n=1 seed (paper 3),
DP batch 32 (paper 256), 50% of every training batch is fixed expert regardless of `mix_ratio`.

## Key files (all in `wav_robot/`)
- `eval_wm_metrics.py` — corrected eval (open-loop image MSE). **Use this, not `eval_wm.py`.**
- `validate_idm_signal.py` — signal-vs-WM-error correlation diagnostic.
- `datasets/_scorers.py` — the `idm` + `topology` scorers (`_episode_signals`).
- `datasets/build_pools.py` — pools; `--holdout_by_snapshot` for covariate-shift eval.
- `run_faithful.sh` — driver; knobs `MIX_RATIO / HOLDOUT_SNAPSHOTS / POOL_TAG / SAMPLE_START / CLAM_DIR`.
- `wav_paper.pdf` — the paper.
- Data (cluster, `jstale02@…tufts`): `scratch_dir/pools/can_full{,_hs}/`,
  `scratch_dir/wm_final_{random,idm,topology}_full{,_hs}_seed0.pt`.

## Prior 4-reviewer panel — convergent conclusions (context)
(1) The robot `idm` is not the paper's WAV (missing subgoal generator). (2) The original eval was
triple-broken (double-preprocess, mislabeled model_loss, and an i.i.d. split that mathematically
favors uniform random). (3) "topology > WAV" in the old numbers was a dilution-toward-random
artifact. (4) The paper's success axis is open-loop MSE / downstream reward, not teacher-forced
recon. The corrected experiments above are consistent with all four.

## Board verdict (2026-07-09, 4-member decision board)

**The working diagnosis above is OVERTURNED.** Code-verified corrections:
- The injected collection noise creates NO aleatoric hardness: the executed action (base+noise,
  clipped) is logged and conditions the WM (`rollout_utils.py:209-212`); robosuite is deterministic
  given (state, action). Noise is uniform across pool AND eval snapshots. Its only effect is
  off-manifold state visitation (covariate mismatch).
- Real explanation for the null (two structural artifacts): (1) **coverage asymmetry** — random
  re-permutes each refresh (`sample_selector_service.py:103`) and sweeps the whole ~552-episode pool
  over ~40 refreshes, while idm re-selects a near-static top-120 (Jaccard .86–.97) → the run compared
  full coverage vs a fixed hard subset under data abundance; (2) **non-binding budget** — 50% fixed
  expert half + mix-ratio coin flip → selection controls ~30% of batches (paper: binding 200-traj budget).
- **The paper's robot experiments are PASSIVE pool selection too** (App E.3.1: 9 imperfect DP
  checkpoints → pool, 10th reserved as validation, no injected noise). Fork C ("build active WAV")
  was premised on a misreading; rejected. The paper DOES beat uniform random (Vanilla-IDM, Fig 6,
  ~28% on Can) → our null is in real tension with the paper; pool/regime construction is the suspect.
- **STRUCK from the record:** the plan-B synthetic mechanism result (`bmech.py`/`mechanism.py`,
  "topology −22%, 5/5 seeds") — same double-preprocess bug class (trains+evals on ~black images,
  scored on fallback `model_loss`). Topology's ONLY clean support is MiniGrid.
- The "+0.44 informative signal" is partially circular (score and "true error" share the same WM);
  a recorded-action control is needed before citing it as validating inverse verification.

**Decision:** reject fork A as designed and fork C. Program:
1. FREE (cluster): coverage diagnostics on `sample_selection/selected_itr_*.jsonl` (unique-coverage
   counts, idm Jaccard, snapshot histograms) + recorded-action circularity control.
2. CRITICAL PATH (CPU, MiniGrid): **shuffled-topology placebo at matched λ, n=20** — the +16%
   headline is currently uncontrolled; this armors or kills the actual contribution.
3. GATED ROBOT (“Design D”, ≤2 H200-days): budget-locked selection on the EXISTING pool — 50-demo
   warm-up + cumulative binding budget of 120 episodes in 2 rounds; random drawn once without
   replacement; MIX_RATIO=1.0 with replay = acquired set; eval horizon-31 open-loop MSE on last-snapshot
   holdout + one reserved mid checkpoint; add an **Oracle arm** (true prediction error, logged actions).
   Gate: seed 0 (2 WM runs); stop on null (question closed), expand to 3 seeds only on ≥5% idm win.
