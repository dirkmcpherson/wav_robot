#!/usr/bin/env bash
# PHASE 0 — 256^2 feasibility + timing probe (run this FIRST, before any full-scale campaign).
# Answers the two make-or-break questions for the faithful reproduction on ONE H200:
#   (1) does 256^2 Dreamer WM + CLAM FIT in memory (and at what batch size)?
#   (2) how fast are WM iters / CLAM updates -> extrapolate full-run wall-clocks -> does a
#       single stage exceed the 48h session cap (=> we must add intra-run resume)?
#
# It renders a SMALL 256^2 dataset and runs the tiny 'probe' scale (few iters) at REAL batch
# sizes, polling GPU memory throughout. Nothing here is wasted: the render + probe pool live
# under _probe256 tags and don't touch the 64^2 data.
#
# Knobs: WM_BATCH (override WM batch if the default OOMs), CLAM_BATCH.
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYTHONPATH="$PWD:$PWD/IDM" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== [0/2] render 256^2 expert data (12 demos, quick) ====="
# NOTE: renders into image_256_done0_v141.hdf5 with only 12 demos — enough for the probe.
# Before the REAL campaign, delete it and re-render ALL demos: SIZE=256 ./render_images.sh
SIZE=256 TASK=can NUM=12 ./render_images.sh

echo "===== [1/2] start GPU memory poller ====="
MEMLOG=probe_mem.log; : > "$MEMLOG"
( while true; do nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits >> "$MEMLOG" 2>/dev/null; sleep 3; done ) &
POLL=$!
trap 'kill $POLL 2>/dev/null || true' EXIT

echo "===== [2/2] probe run (SCALE=probe IMG_SIZE=256, real batch sizes, few iters) ====="
T0=$SECONDS
SCALE=probe IMG_SIZE=256 STRATEGY=idm POOL_TAG=_probe256 SEED=2 NUM_EXP_TRAJS=10 \
  WM_BATCH="${WM_BATCH:-}" CLAM_BATCH="${CLAM_BATCH:-8}" \
  ./run_faithful.sh 2>&1 | tee probe256.log
RC=${PIPESTATUS[0]}
DT=$((SECONDS - T0))
kill $POLL 2>/dev/null || true

echo ""
echo "================ PROBE REPORT ================"
echo "exit code    : $RC   (nonzero => OOM or 256^2 breakage; see 'tail -40 probe256.log')"
echo "wall time    : ${DT}s for the whole probe chain (render excluded)"
PEAK=$(awk -F',' 'NR==1{max=$1} $1>max{max=$1} END{print max" / "$2}' "$MEMLOG" 2>/dev/null)
echo "peak GPU mem : ${PEAK:-?} MiB used/total"
echo "--- CLAM speed (updates) ---"; grep -aiE "update|it/s|loss" probe256.log | grep -aiE "clam|st_vivit|[0-9]+/300" | tail -6
echo "--- WM speed (iters) ---";     grep -aE "\[WM.*Itr:|Time taken" probe256.log | tail -6
cat <<'EOF'

READ IT AS:
  * peak mem << 80000 MiB  -> fits; note headroom for larger batch. peak ~OOM -> set WM_BATCH lower and rerun.
  * WM  hours(full) = 200000 / (WM it/s)  / 3600
  * CLAM hours(paper)= 500000 / (CLAM upd/s) / 3600
  * If BOTH stages < ~44h each -> inter-arm idempotency suffices (each of the 9 WM arms is a
    resumable chunk; restart skips finished stashes). No intra-run resume code needed.
  * If a single stage > 48h -> ping to add intra-run resume for THAT stage before committing.
EOF
echo "============================================="
