"""Does the WAV `idm` selection score actually identify episodes the WM predicts poorly?

This is the cheap diagnostic that decides whether chasing a better CLAM is worth it. In
MiniGrid the WAV score correlated +0.82 with true WM error (prec@100 .46) -- that's WHY
selection helped there. On the robot it was never validated. Here, for each pool episode we
compute BOTH under the same WM:
  * idm_score  = _episode_signals(...) -> idm_wm_latent_mismatch_mse (the selection signal)
  * wm_err     = evaluate_batch_metrics(...) -> openloop_img_pred_loss (actual WM error)
and report Pearson/Spearman correlation + precision@k (do the top-k idm episodes overlap the
top-k highest-WM-error episodes?).

  corr >> 0  -> idm picks genuinely hard-for-the-WM episodes; signal is good, look elsewhere.
  corr ~ 0   -> idm is picking noise, not model-weak points -> the CLAM/scorer is the problem.
  corr < 0   -> idm is actively anti-correlated -> selection is worse than random by design.

Usage: python validate_idm_signal.py <wm_ckpt.pt> <clam_dir/> <pool.jsonl>
  (clam_dir is the trailing-slash dir from pools/<...>/clam_dir.txt)
"""
import sys, json
import numpy as np
import torch

from datasets._scorers import (_load_wm, _load_clam, _wm_batch, _episode_signals, IMAGE_KEYS)


def _pearson(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return _pearson(rx, ry)


def main(wm_ckpt, clam_dir, pool_jsonl,
         device="cuda:0" if torch.cuda.is_available() else "cpu"):
    wm = _load_wm(wm_ckpt, IMAGE_KEYS, state_dim=9, num_actions=7, device=device)
    wm._config.openloop_img_pred_horizon = 8
    wm._config.log_openloop_img_pred = True
    clam, adec, use_tr, distr = _load_clam(
        clam_dir + "model_ckpts/latest.pkl", clam_dir + "config.yaml", device)

    refs = [json.loads(l) for l in open(pool_jsonl) if l.strip()]
    idm_scores, wm_errs = [], []
    for ref in refs:
        try:
            with np.load(ref["file_path"], allow_pickle=False) as ep:
                ep = {k: ep[k] for k in ep.files}
            idm_mse, _ = _episode_signals(ep, wm, clam, adec, use_tr, distr, device)
            b = _wm_batch(ep, device)
            b["is_first"][:, 0] = 1.0
            with torch.no_grad():
                m = wm.evaluate_batch_metrics(b)
            err = m.get("openloop_img_pred_loss", m.get("model_loss"))
            if np.isfinite(idm_mse) and np.isfinite(err):
                idm_scores.append(float(idm_mse)); wm_errs.append(float(err))
        except Exception as e:
            print(f"  episode failed: {type(e).__name__}: {e}")

    idm_scores = np.array(idm_scores); wm_errs = np.array(wm_errs)
    n = len(idm_scores)
    print(f"# {wm_ckpt}")
    print(f"# n={n} episodes scored")
    if n < 3:
        print("too few episodes to correlate"); return
    print(f"Pearson (idm_score , wm_openloop_err)  = {_pearson(idm_scores, wm_errs):+.3f}")
    print(f"Spearman(idm_score , wm_openloop_err)  = {_spearman(idm_scores, wm_errs):+.3f}")
    for k in (10, 25, 50):
        if n >= 2 * k:
            top_idm = set(np.argsort(-idm_scores)[:k].tolist())
            top_err = set(np.argsort(-wm_errs)[:k].tolist())
            print(f"precision@{k} (top idm-score vs top WM-error) = {len(top_idm & top_err)/k:.2f}")
    # baseline: what does a random top-k get by chance?  ~ k/n
    print(f"(chance precision@k ~ k/n; n={n})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
