"""Reimplemented sample-selection scorer (original release/datasets/data_selection.py
not shipped). Matches the interface used by wav/training/sample_selector_service.py:

    from datasets.data_selection import SelectionRequest, run_selection
    result = run_selection(request=SelectionRequest(...), context=None)
    result.metadata["num_selected"], result.metadata["num_scored"]

Strategies (per scripts/run_wm_only.sh score_key conventions):
  - random      : uniform top-k                                  (no model)        [DONE]
  - progress    : learning-progress  old_loss - new_loss          (2 WM ckpts)     [DONE]
  - curiosity   : ensemble latent-prior feature variance          (>=2 WM ckpts)   [DONE]
  - idm         : WAV  idm_wm_latent_mismatch_mse                  (WM + CLAM IDM)  [Phase 4a]
  - topology    : idm + lambda * FDM<->IDM topology disagreement   (supplement)     [Phase 4b]

Selection writes the chosen refs (highest score) to output_jsonl, top `select_size`.
Episode scoring loads each candidate npz via the same fields the WM uses.
"""
from dataclasses import dataclass, field
import json
import pathlib
import numpy as np


@dataclass
class SelectionRequest:
    sample_pool_jsonl: pathlib.Path
    output_jsonl: pathlib.Path
    strategy: str
    select_size: int
    seed: int = 0
    strategy_kwargs: dict = field(default_factory=dict)


@dataclass
class SelectionResult:
    metadata: dict


def _read_refs(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _write_refs(p, refs):
    p = pathlib.Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in refs:
            f.write(json.dumps(r) + "\n")


def _load_episode(ref):
    with np.load(ref["file_path"], allow_pickle=False) as ep:
        return {k: ep[k] for k in ep.files}


def _topk_by_score(refs, scores, k):
    order = np.argsort(-np.asarray(scores, dtype=np.float64))  # high score first
    return [refs[i] for i in order[:k]]


def run_selection(request: SelectionRequest, context=None) -> SelectionResult:
    refs = _read_refs(request.sample_pool_jsonl)
    n = len(refs)
    kw = dict(request.strategy_kwargs or {})
    strat = request.strategy

    if strat == "random":
        rng = np.random.RandomState(request.seed)
        chosen = [refs[i] for i in rng.permutation(n)[: request.select_size]]
        num_scored = n
    elif strat in ("idm", "topology", "oracle", "progress", "curiosity", "uncertainty"):
        from datasets._scorers import score_episodes  # heavy imports kept lazy
        scores = score_episodes(strat, refs, kw, seed=request.seed)
        chosen = _topk_by_score(refs, scores, request.select_size)
        num_scored = int(np.isfinite(scores).sum())
    else:
        raise ValueError(f"unknown strategy {strat!r}")

    _write_refs(request.output_jsonl, chosen)
    return SelectionResult(metadata={
        "num_selected": len(chosen),
        "num_scored": num_scored,
        "strategy": strat,
        "score_key": kw.get("score_key", ""),
    })
