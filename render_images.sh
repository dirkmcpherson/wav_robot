#!/usr/bin/env bash
# Render robomimic expert demos to image observations at a given resolution, producing
# datasets/robomimic_datasets/<TASK>/ph/image_<SIZE>_done0_v141.hdf5 — the exact name the
# WM data loader derives from image_size. Idempotent (skips if the target exists).
#   SIZE=256 ./render_images.sh          # all demos @ 256^2 (the faithful-repro data)
#   SIZE=256 NUM=12 ./render_images.sh   # first 12 demos (quick)
set -euo pipefail
cd "$(dirname "$0")"
TASK="${TASK:-can}"; SIZE="${SIZE:-256}"; NUM="${NUM:-all}"
DIR="datasets/robomimic_datasets/${TASK}/ph"
SRC="$DIR/low_dim_done0_v141.hdf5"
OUT="image_${SIZE}_done0_v141.hdf5"
[ -f "$SRC" ] || { echo "ERROR: missing low_dim source $SRC (run fetch_data.sh first)"; exit 1; }
if [ -f "$DIR/$OUT" ]; then echo "exists: $DIR/$OUT (delete to re-render)"; exit 0; fi
NARG=""; [ "$NUM" != "all" ] && NARG="--n $NUM"
echo "rendering $NUM demos @ ${SIZE}x${SIZE} (agentview + robot0_eye_in_hand) -> $DIR/$OUT ..."
MUJOCO_GL="${MUJOCO_GL:-egl}" python -m robomimic.scripts.dataset_states_to_obs \
  --dataset "$SRC" --output_name "$OUT" --done_mode 0 \
  --camera_names agentview robot0_eye_in_hand --camera_height "$SIZE" --camera_width "$SIZE" $NARG
echo "done: $DIR/$OUT ($(du -h "$DIR/$OUT" | cut -f1))"
