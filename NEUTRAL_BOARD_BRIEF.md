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

## Post-board execution results (2026-07-10)

**Track 2 (MiniGrid placebo) — KILL CRITERION FIRED; the topology claim is DEAD.** The earlier
statement above that the MiniGrid topology supplement "is the sound contribution" is RETRACTED:
- Shuffled-topology placebo (n=20, drift control bit-exact): WAV+λ·noise **beats** WAV+λ·topology
  in **20/20 seeds** (0.595 vs 0.800 at λ=0.05, t=−15.3; recovers 217% of the improvement).
- Sign-flip arm (−λ topology): 0.650 — better than +λ, still loses to noise (16/20).
- Noise λ sweep monotone: 0.05→0.595, 0.10→0.553, 0.20→0.550 = best strategy ever measured on the
  benchmark (WAV 0.96, Uncertainty 0.83, Random 1.34).
- Mechanism: corr(rank_wav, rank_topo)=+0.2 → topology reinforces WAV's redundant ranking
  (77/100 pick overlap vs WAV-only); noise diversifies (24–40/100), breaks trajectory clusters.
**Surviving results: (1) WAV > random in MiniGrid (faithful WAV); (2) NEW headline — "ε-diversified
WAV" (stochastic top-tier sampling) −43% MSE, beats everything; (3) the placebo methodology itself.**

Track 3 (robot Design D) implemented and pushed (budget-locked acquisition, oracle arm,
mid-snapshot eval; run_designD.sh) — not yet run.

## Final MiniGrid adjudication (2026-07-10, post-placebo follow-ups)

- Noise effect identified as prior art: Kirsch et al., "Stochastic Batch Acquisition"
  (arXiv:2106.12059, TMLR) — our rank-noise is their soft-rank variant. Domain replication, not novel.
- Residual test (topology ON TOP of the noise baseline, identical per-seed noise realization):
  +0.05·topo actively harmful (0.595 vs 0.553, p=0.0003). −0.05·topo (INVERTED: select FDM↔IDM
  topological AGREEMENT) trended exploratory (p=.079) and was **CONFIRMED pre-registered on fresh
  seeds 68-87: 0.537 vs 0.565, t=−3.52, one-sided p=0.0011 (15/20)**.
- Best known strategy: rank(wav) + 0.20·rank(noise) − 0.05·rank(topo_dis) = 0.537
  (WAV 0.96, Random 1.34). Interpretation: among hard transitions, FDM↔IDM representational
  AGREEMENT marks hard-but-learnable; disagreement marks hard-but-unhelpful — consistent with the
  robot diagnosis. Surviving novel contribution = this inverted residual + the control methodology.

## Robot line CLOSED — diagnosed negative (2026-07-13)

Design D (budget-locked selection + oracle arm, noisy pool) → random < oracle < idm (oracle +9.7%,
idm +12.4% vs random @ eval_pool h8). Design A (same regime, pool RE-COLLECTED noise-free —
noise_std=0, checkpoint-diversity only per paper App E.3.1, 840 clean episodes):

| eval / horizon | random | oracle | idm |
|---|---|---|---|
| eval_pool / 8  | 0.00671 | 0.00697 (+3.9%) | 0.00712 (+6.1%) |
| eval_pool / 31 | 0.00988 | 0.01046 | 0.01111 |
| eval_mid / 8   | 0.00773 | 0.00794 | 0.00818 |
| eval_mid / 31  | 0.01209 | 0.01244 | 0.01288 |

**Random best in all 8 cells.** Cleaning the collection noise ~HALVED the hard-selection penalty
(oracle 9.7%→3.9%) — confirming injected noise was a real contributor — but did NOT flip it. Even
the ORACLE (true prediction error, no CLAM/scorer) ties-to-slightly-loses to uniform random
(~0.5–1.4 SE, n=1). Every board-identified variable is now controlled (metric, eval distribution,
coverage asymmetry, collection noise, scorer quality via the oracle upper bound).

**Verdict:** at this scale, hardness-based selection provides no benefit over uniform random for
world-model learning on robomimic-can. Could not reproduce the paper's Oracle>Random ordering;
remaining untested differences are pure scale (64²→256², CLAM 50k→500k, remove 50%-expert anchor,
n=1→3) — the expensive full-scale campaign. The robot cannot host the inverted-topology residual
test. Final scope: MiniGrid carries the topology finding; the robot is a fully-diagnosed negative.

## Correction (2026-07-13) — the robot result is a reproduction FAILURE, not a settled negative

The "closed / diagnosed negative" framing above is walked back. A failure to reproduce is not a
finding: we have NO working robot positive control (nothing, including the oracle, ever beat
random), so the null is AMBIGUOUS between under-scaling, harness insensitivity, and true absence —
and our data cannot separate them. (Contrast MiniGrid, where WAV>random IS a working positive
control, which is what licenses trusting its negatives.)

Code-verified prime suspect for harness insensitivity: `mixed_sample` (wav/classes/rollout_utils.py:324)
HARDCODES 50% expert + 50% other in every batch; MIX_RATIO only picks whether the *other* half is
the selected pool vs replay. So selection never controlled >50% of the training diet, always fighting
a fixed half of 50 clean expert demos. Plausibly un-faithful (paper warm-starts then fine-tunes on
acquired data) and a plausible mask on any selection effect.

Also never tested: the paper's HEADLINE claim is downstream policy reward (+22%) / 2× sample
efficiency — we only ever measured WM prediction MSE (the axis that rewards typicality).

Load-bearing next experiment: BINDING-BUDGET positive control — shrink the expert half, rerun
Oracle vs Random on the clean pool. Oracle>Random → harness validated, WAV/topology get a fair
test. Still null at ~100% selection control → a stronger (still scale-caveated) negative.
