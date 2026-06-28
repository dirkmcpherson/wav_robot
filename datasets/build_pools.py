"""Reimplemented pool builder (original release/datasets/build_pools.py not shipped).

Turns collected DP-rollout npz episodes (and optionally expert hdf5 demos) into the
train/sample/eval `*.jsonl` pools the WM-only loop consumes. Each line is a ref dict
matching wav/runtime/dataset_loader.load_pool_episodes_for_wm:
  {"source_type": "dp_rollout_npz", "episode_id": <id>, "file_path": <abs npz>}
  {"source_type": "expert_hdf5",   "file_path": <abs hdf5>, "demo_key": "demo_N"}

Roles (faithful to the method):
  - sample_pool: the diverse DP rollouts -> candidates the WM actively selects from.
  - train_pool : initial WM training data (expert demos, or a slice of rollouts).
  - eval_pool  : held-out episodes for WM evaluation.
"""
import argparse, glob, json, os, random
from pathlib import Path


def _npz_refs(npz_paths, tag):
    refs = []
    for i, p in enumerate(npz_paths):
        refs.append({
            "source_type": "dp_rollout_npz",
            "episode_id": f"{tag}_{i}_{Path(p).stem}",
            "file_path": str(Path(p).resolve()),
        })
    return refs


def _expert_refs(hdf5_path, demo_keys):
    return [{"source_type": "expert_hdf5", "file_path": str(Path(hdf5_path).resolve()),
             "demo_key": dk} for dk in demo_keys]


def _write(path, refs):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in refs:
            f.write(json.dumps(r) + "\n")
    return len(refs)


def build_pools(rollouts_glob, out_dir, seed=0,
                sample_frac=0.6, eval_frac=0.2,
                expert_hdf5=None, expert_train_demos=0, expert_eval_demos=0):
    """sample_pool <- rollouts (candidates); train/eval <- remaining rollouts and/or expert."""
    npz = sorted(glob.glob(rollouts_glob, recursive=True))
    if not npz:
        raise FileNotFoundError(f"No rollout npz matched: {rollouts_glob}")
    random.Random(seed).shuffle(npz)
    n = len(npz)
    n_sample = max(1, int(n * sample_frac))
    n_eval = max(1, int(n * eval_frac))
    sample_npz = npz[:n_sample]
    eval_npz = npz[n_sample:n_sample + n_eval]
    train_npz = npz[n_sample + n_eval:]

    out = Path(out_dir)
    counts = {}
    train_refs = _npz_refs(train_npz, "train")
    eval_refs = _npz_refs(eval_npz, "eval")
    # Optionally fold expert demos into train/eval (more faithful: WM seeds on expert data).
    if expert_hdf5 and (expert_train_demos or expert_eval_demos):
        import h5py
        with h5py.File(expert_hdf5, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        train_refs += _expert_refs(expert_hdf5, demos[:expert_train_demos])
        eval_refs += _expert_refs(expert_hdf5, demos[expert_train_demos:expert_train_demos + expert_eval_demos])

    counts["sample"] = _write(out / "sample_pool.jsonl", _npz_refs(sample_npz, "sample"))
    counts["train"] = _write(out / "train_pool.jsonl", train_refs)
    counts["eval"] = _write(out / "eval_pool.jsonl", eval_refs)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # explicit mode
    ap.add_argument("--rollouts_glob", default=None, help="glob for dp_rollout npz")
    ap.add_argument("--out_dir", default=None)
    # env/script mode (SUITE/TASK/EXP_NAME -> infer rollouts dir under scratch_dir/logs)
    ap.add_argument("--suite", default=os.environ.get("SUITE", "robomimic"))
    ap.add_argument("--task", default=os.environ.get("TASK", "can"))
    ap.add_argument("--exp_name", default=os.environ.get("EXP_NAME", ""))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    ap.add_argument("--sample_frac", type=float, default=0.6)
    ap.add_argument("--eval_frac", type=float, default=0.2)
    ap.add_argument("--expert_hdf5", default=None)
    ap.add_argument("--expert_train_demos", type=int, default=0)
    ap.add_argument("--expert_eval_demos", type=int, default=0)
    a = ap.parse_args()

    rollouts_glob = a.rollouts_glob
    out_dir = a.out_dir
    if rollouts_glob is None:
        base = f"scratch_dir/logs/{a.suite}__{a.task}/{a.exp_name or '*'}/seed{a.seed}/dp_rollouts"
        rollouts_glob = f"{base}/**/*.npz"
        out_dir = out_dir or f"scratch_dir/pools/{a.suite}__{a.task}/seed{a.seed}"
    counts = build_pools(rollouts_glob, out_dir, seed=a.seed,
                         sample_frac=a.sample_frac, eval_frac=a.eval_frac,
                         expert_hdf5=a.expert_hdf5,
                         expert_train_demos=a.expert_train_demos,
                         expert_eval_demos=a.expert_eval_demos)
    print(f"built pools in {out_dir}: {counts}")
