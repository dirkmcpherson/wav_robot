"""Corrected held-out WM eval — reports the paper-flavor metric (open-loop image MSE).

Fixes two bugs in eval_wm.py, which is why its numbers were ~4800 and mislabeled:
  (1) DOUBLE PREPROCESS: eval_wm.py called wm.evaluate_batch_metrics(wm.preprocess(b)),
      but evaluate_batch_metrics preprocesses internally (wm.py:513) and preprocess divides
      every *image* key by 255 unconditionally (wm.py:610-613) -> images normalized TWICE
      (~255x too dark, out-of-distribution). Here we pass the RAW batch (single preprocess).
  (2) MISLABELED METRIC: there is no "image_loss" key, so eval_wm.py fell back to model_loss
      (recon NLL + KL). Here we report:
        * openloop_img_pred_loss  -- action-conditioned N-step open-loop image MSE. This is
          the paper's WM-error flavor (multi-step open-loop prediction), the axis on which
          WAV is supposed to help. PRIMARY metric.
        * agentview_image_loss / robot0_eye_in_hand_image_loss -- per-camera TEACHER-FORCED
          recon NLL (now un-double-preprocessed). Secondary.
        * model_loss -- composite recon+KL, for continuity with the old eval_wm.py numbers.

We also mirror the in-training eval (wm_trainer.iter_eval_chunks): chunk each episode into
batch_length windows and force is_first[0]=True so the RSSM resets at each window start.

Usage: python eval_wm_metrics.py <wm_ckpt.pt> <eval_pool.jsonl> [openloop_horizon=8] [batch_length=32]
"""
import sys, json, collections
import numpy as np
import torch

from datasets._scorers import _load_wm, _wm_batch

IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
REPORT_KEYS = (
    "openloop_img_pred_loss",           # <- primary, paper-flavor (open-loop image MSE)
    "agentview_image_loss",
    "robot0_eye_in_hand_image_loss",
    "state_loss",
    "model_loss",                        # <- what old eval_wm.py reported (recon NLL + KL)
    "kl",
)


def _iter_chunks(ep, batch_length):
    """Slice an episode npz dict into [start:end] windows along time (mirrors iter_eval_chunks)."""
    T = int(np.asarray(ep["action"]).shape[0])
    for start in range(0, T - 1, batch_length):
        end = min(start + batch_length, T)
        if end - start < 2:
            continue
        sub = {}
        for k, v in ep.items():
            a = np.asarray(v)
            if a.ndim >= 1 and a.shape[0] == T:      # only time-indexed arrays
                sub[k] = a[start:end]
        yield sub


def main(wm_ckpt, eval_jsonl, horizon=8, batch_length=32,
         device="cuda:0" if torch.cuda.is_available() else "cpu"):
    horizon, batch_length = int(horizon), int(batch_length)
    wm = _load_wm(wm_ckpt, IMAGE_KEYS, state_dim=9, num_actions=7, device=device)
    # drive the open-loop rollout horizon (config default is 8; paper uses 32)
    wm._config.openloop_img_pred_horizon = horizon
    wm._config.log_openloop_img_pred = True

    refs = [json.loads(l) for l in open(eval_jsonl) if l.strip()]
    agg = collections.defaultdict(list)
    n_ep, n_chunk, n_fail = 0, 0, 0
    for ref in refs:
        try:
            with np.load(ref["file_path"], allow_pickle=False) as ep:
                ep = {k: ep[k] for k in ep.files}
        except Exception as e:
            n_fail += 1; continue
        n_ep += 1
        for sub in _iter_chunks(ep, batch_length):
            try:
                b = _wm_batch(sub, device)
                b["is_first"][:, 0] = 1.0            # reset RSSM at window start
                with torch.no_grad():
                    m = wm.evaluate_batch_metrics(b)  # RAW b -> single preprocess (bug fixed)
                for k in REPORT_KEYS:
                    if k in m:
                        agg[k].append(float(m[k]))
                n_chunk += 1
            except Exception as e:
                n_fail += 1

    print(f"# {wm_ckpt}")
    print(f"# episodes={n_ep} chunks={n_chunk} failures={n_fail} "
          f"openloop_horizon={horizon} batch_length={batch_length}")
    for k in REPORT_KEYS:
        if agg[k]:
            v = np.array(agg[k])
            print(f"{k:34s} mean={v.mean():.5f} std={v.std():.5f} n={len(v)}")
    # return the primary metric for scripting
    prim = np.array(agg["openloop_img_pred_loss"])
    return float(prim.mean()) if len(prim) else float("nan")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2],
         horizon=a[3] if len(a) > 3 else 8,
         batch_length=a[4] if len(a) > 4 else 32)
