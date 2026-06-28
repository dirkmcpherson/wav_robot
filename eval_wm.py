"""Post-hoc held-out WM eval: load a final WM checkpoint + an eval pool, compute the
mean evaluate_batch_metrics over the eval episodes. Gives a controlled, identical metric
across selection strategies (idm/random/topology) for a fair comparison.

Usage: python eval_wm.py <wm_ckpt.pt> <eval_pool.jsonl>
"""
import sys, json, numpy as np, torch
from datasets._scorers import _load_wm, _wm_batch

def main(wm_ckpt, eval_jsonl, device="cuda:0" if torch.cuda.is_available() else "cpu"):
    wm = _load_wm(wm_ckpt, ("agentview_image", "robot0_eye_in_hand_image"),
                  state_dim=9, num_actions=7, device=device)
    refs = [json.loads(l) for l in open(eval_jsonl) if l.strip()]
    vals = []
    for ref in refs:
        with np.load(ref["file_path"], allow_pickle=False) as ep:
            ep = {k: ep[k] for k in ep.files}
        b = _wm_batch(ep, device)
        with torch.no_grad():
            m = wm.evaluate_batch_metrics(wm.preprocess(b))
        v = m.get("image_loss", m.get("model_loss"))
        vals.append(float(v.mean() if hasattr(v, "mean") else v))
    vals = np.array(vals)
    print(f"held-out WM eval over {len(vals)} episodes: mean={vals.mean():.4f} std={vals.std():.4f}")
    return vals.mean()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
