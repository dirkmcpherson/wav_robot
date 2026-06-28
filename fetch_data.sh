#!/usr/bin/env bash
# Fetch robomimic demonstration data (the base demos; diverse DP rollouts are generated
# later in-sim via scripts/dp_collect_robomimic.sh). _v141 matches robosuite 1.4.1.
set -u
BASE="http://downloads.cs.stanford.edu/downloads/rt_benchmark"
mkdir -p data/robomimic/can data/robomimic/square

# can PH low_dim (already fetched here, ~47MB, 200 demos) + image (pre-rendered, larger)
curl -L -o data/robomimic/can/low_dim_v141.hdf5    "$BASE/can/ph/low_dim_v141.hdf5"
curl -L -o data/robomimic/can/image_v141.hdf5      "$BASE/can/ph/image_v141.hdf5"      # may 503; retry
# square PH (second task)
curl -L -o data/robomimic/square/low_dim_v141.hdf5 "$BASE/square/ph/low_dim_v141.hdf5"
curl -L -o data/robomimic/square/image_v141.hdf5   "$BASE/square/ph/image_v141.hdf5"   # may 503; retry

echo "done. (If image_v141 503s, retry, or render images from states in-sim during DP rollout collection.)"
