#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
: > lambda_sweep_results.txt
echo "lam=0.0(idm) mean=4821.9967" >> lambda_sweep_results.txt
echo "lam=0.1      mean=4767.5355" >> lambda_sweep_results.txt
for lam in 0.05 0.2 0.3; do
  SCALE=medium STRATEGY=topology TOPO_LAMBDA=$lam CLAM_BATCH=8 ./run_faithful.sh > run_medium_topo_lam${lam}.log 2>&1
  WMC=$(find scratch_dir/logs/robomimic__can/None_demos50 -name latest_residual_checkpoint.pt | head -1)
  cp "$WMC" scratch_dir/wm_final_topo_lam${lam}.pt
  res=$(MUJOCO_GL=osmesa PYTHONPATH="$PWD:$PWD/IDM" python eval_wm.py scratch_dir/wm_final_topo_lam${lam}.pt scratch_dir/pools/can_medium/eval_pool.jsonl 2>&1 | grep -aoE "mean=[0-9.]+ std=[0-9.]+")
  echo "lam=$lam      $res" >> lambda_sweep_results.txt
done
echo "LAMBDA_SWEEP_DONE" >> lambda_sweep_results.txt
