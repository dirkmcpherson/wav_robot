"""Per-episode scorers for model-based selection strategies.

`random` is handled directly in data_selection.run_selection. This module implements the
WAV signal `idm` (score_key=idm_wm_latent_mismatch_mse): how much the world model (FDM)
and the CLAM inverse model (IDM) disagree about an episode's dynamics, in WM latent space.

Algorithm (port of the MiniGrid WAV forward->inverse->forward cycle to Dreamer+CLAM):
  - WM encodes the episode -> posterior latents (get_feat(post)); this is "what happened".
  - CLAM infers the action for each transition from the (image) observations -> action_pred.
  - WM rolls one step forward under the IDM-inferred action (img_step) -> predicted latent.
  - mismatch_t = ||predicted_latent_t - posterior_latent_{t+1}||^2 ; episode score = mean_t.
High mismatch = the IDM's action doesn't reproduce the observed dynamics under the WM.

topology (Phase 4b) = idm + lambda * FDM<->IDM topology disagreement (added later).
progress/curiosity still pending (need WM snapshot ensembles).
"""
import numpy as np
import torch
from omegaconf import OmegaConf

IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
CLAM_CAMERA = "agentview_image"


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def _build_wm(image_keys, state_dim, num_actions, device):
    cfg = OmegaConf.load("wav/configs.yaml").defaults
    OmegaConf.set_struct(cfg, False)
    cfg.device = device; cfg.precision = 32; cfg.num_actions = num_actions
    cfg.action_dim = num_actions; cfg.state_only = False; cfg.compile = False
    cfg.use_wandb = False; cfg.image_size = 64
    import wav.dreamer.wm as wmmod
    class Sp:
        def __init__(s, sh): s.shape = tuple(sh)
    class OS:
        def __init__(s, d): s.spaces = d
    spaces = {k: Sp((64, 64, 3)) for k in image_keys}; spaces["state"] = Sp((state_dim,))
    return wmmod.WorldModel(obs_space=OS(spaces), step=0, config=cfg).to(device)


def _load_wm(wm_ckpt_path, image_keys, state_dim, num_actions, device):
    wm = _build_wm(image_keys, state_dim, num_actions, device)
    sd = torch.load(wm_ckpt_path, map_location=device, weights_only=False)
    wm_sd = {}
    for k, v in sd.items():
        if k.startswith("_wm."):
            nk = k[len("_wm."):]
            if nk.startswith("_orig_mod."):
                nk = nk[len("_orig_mod."):]
            wm_sd[nk] = v
    wm.load_state_dict(wm_sd, strict=False)
    wm.eval()
    return wm


def _load_clam(idm_ckpt_path, idm_config_path, device):
    import udrm.resolvers  # noqa: F401  (register OmegaConf resolvers)
    from udrm.models.utils.clam_utils import get_clam_cls, get_la_dim
    from udrm.models.mlp_policy import MLPPolicy
    cfg = OmegaConf.load(idm_config_path); OmegaConf.resolve(cfg); OmegaConf.set_struct(cfg, False)
    cfg.use_wandb = False
    clam = get_clam_cls(cfg.name)(cfg.model, input_dim=(3, 64, 64), la_dim=get_la_dim(cfg)).to(device)
    adec = MLPPolicy(cfg=cfg.model.action_decoder, input_dim=get_la_dim(cfg),
                     output_dim=cfg.env.action_dim).to(device)
    ck = torch.load(idm_ckpt_path, map_location=device, weights_only=False)
    clam.load_state_dict(ck["model"])
    if "action_decoder" in ck:
        adec.load_state_dict(ck["action_decoder"])
    clam.eval(); adec.eval()
    use_transformer = "transformer" in cfg.model.idm.net.name
    distributional = bool(getattr(cfg.model, "distributional_la", False))
    return clam, adec, use_transformer, distributional


# --------------------------------------------------------------------------- #
# episode -> model inputs
# --------------------------------------------------------------------------- #
def _destack(a):
    """Drop the obs-horizon stacking axis (last dim, size 2) -> take the latest frame."""
    a = np.asarray(a)
    return a[..., -1] if a.ndim and a.shape[-1] == 2 else a


def _wm_batch(ep, device):
    """Episode npz -> WM batch dict [1, T, ...] (preprocess de-stacks internally)."""
    b = {}
    for k in IMAGE_KEYS:
        if k in ep:
            b[k] = torch.tensor(np.asarray(ep[k])[None]).float()     # [1,T,H,W,3,2]
    b["state"] = torch.tensor(np.asarray(ep["state"])[None]).float()  # [1,T,9,2]
    act = np.asarray(ep["action"])                                    # [T,7,8] chunk
    b["action"] = torch.tensor(act[None]).float()
    T = b["state"].shape[1]
    isf = np.asarray(ep["is_first"]).astype(np.float32) if "is_first" in ep else np.zeros(T, np.float32)
    ist = np.asarray(ep["is_terminal"]).astype(np.float32) if "is_terminal" in ep else np.zeros(T, np.float32)
    b["is_first"] = torch.tensor(isf[None]); b["is_terminal"] = torch.tensor(ist[None])
    b["reward"] = torch.tensor(np.asarray(ep.get("reward", np.zeros(T)))[None]).float()
    return {k: v.to(device) for k, v in b.items()}


def _clam_obs(ep, device):
    """Episode -> CLAM observations [1, T, 3, 64, 64] (agentview, de-stacked, CHW, /255) + states/timesteps."""
    img = _destack(ep[CLAM_CAMERA]).astype(np.float32) / 255.0    # [T,64,64,3]
    img = np.transpose(img, (0, 3, 1, 2))                          # [T,3,64,64]
    st = _destack(ep["state"]).astype(np.float32)                 # [T,9]
    T = img.shape[0]
    obs = torch.tensor(img[None], device=device)                  # [1,T,3,64,64]
    states = torch.tensor(st[None], device=device)                # [1,T,9]
    timesteps = torch.arange(T, device=device)[None]              # [1,T]
    return obs, states, timesteps


# --------------------------------------------------------------------------- #
# the idm / WAV score
# --------------------------------------------------------------------------- #
def _rank_norm(x):
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    r = np.full(len(x), 0.5, dtype=np.float64)
    if finite.sum() > 1:
        order = np.argsort(np.argsort(x[finite]))
        r[finite] = order / (finite.sum() - 1)
    return r


@torch.no_grad()
def _episode_signals(ep, wm, clam, adec, use_transformer, distributional, device):
    """Return (idm_wm_latent_mismatch_mse, topology_disagreement) for one episode."""
    # WM posterior latents over the episode (the FDM latent sequence)
    b = _wm_batch(ep, device)
    data = wm.preprocess(b)
    embed = wm.encoder(data)
    post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
    feat = wm.dynamics.get_feat(post)                              # [1,T,F]

    # CLAM latent action `la` (the IDM latent) + decoded env action
    obs, states, timesteps = _clam_obs(ep, device)
    if use_transformer:
        out = clam(obs, timesteps=timesteps, states=states); la = out.la[:, 1:]
    else:
        out = clam(obs); la = out.la
    if distributional:
        la = clam.reparameterize(la)
    idm_action = adec(la)                                          # [1, T-1, 7]

    # (1) WAV idm score: WM one-step rollout under the IDM action vs the observed posterior
    T = feat.shape[1]
    n = min(idm_action.shape[1], T - 1)
    errs = []
    state_t = {k: v[:, 0] for k, v in post.items()}
    for t in range(n):
        prior = wm.dynamics.img_step(state_t, idm_action[:, t])
        pred = wm.dynamics.get_feat(prior)
        errs.append(float(((pred - feat[:, t + 1]) ** 2).mean().item()))
        state_t = {k: v[:, t + 1] for k, v in post.items()}
    idm_mse = float(np.mean(errs)) if errs else float("nan")

    # (2) topology disagreement: relational (RSA) gap between FDM latent curve and IDM latent curve
    from datasets import topology as topo  # vendored (was wav_minigrid.topology)
    f = feat[0].detach().cpu().numpy()                            # [T,F]
    g = la[0].detach().cpu().numpy()                              # [n_la, la_dim]
    m = min(len(f), len(g))
    topo_dis = float("nan")
    if m >= 3:
        agr = topo.snippet_rsa_g(f[:m], g[:m], "euclidean", "pearson")
        if agr is not None and not np.isnan(agr):
            topo_dis = 1.0 - float(agr)
    return idm_mse, topo_dis


def score_episodes(strategy, refs, kwargs, seed=0):
    if strategy in ("progress", "curiosity", "uncertainty"):
        raise NotImplementedError(
            f"strategy '{strategy}' scoring pending (needs WM snapshot ckpt[s]). kwargs={list(kwargs)}")
    if strategy not in ("idm", "topology"):
        raise ValueError(f"no scorer for strategy {strategy!r}")

    device = kwargs.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    wm = _load_wm(kwargs["wm_ckpt_path"], IMAGE_KEYS, state_dim=9, num_actions=7, device=device)
    clam, adec, use_tr, distr = _load_clam(kwargs["idm_ckpt_path"], kwargs["idm_config_path"], device)

    idm = np.full(len(refs), np.nan); topo = np.full(len(refs), np.nan)
    for i, ref in enumerate(refs):
        try:
            with np.load(ref["file_path"], allow_pickle=False) as ep:
                ep = {k: ep[k] for k in ep.files}
            idm[i], topo[i] = _episode_signals(ep, wm, clam, adec, use_tr, distr, device)
        except Exception as e:
            print(f"[{strategy} scorer] episode {i} failed: {type(e).__name__}: {e}")

    def fill(a):
        return np.where(np.isnan(a), np.nanmedian(a) if np.isfinite(a).any() else 0.0, a)
    idm = fill(idm)
    if strategy == "idm":
        return idm
    # topology supplement: rank_norm(WAV idm) + lambda * rank_norm(topology disagreement)
    lam = float(kwargs.get("topology_weight", kwargs.get("lambda", 0.1)))
    topo = fill(topo)
    return _rank_norm(idm) + lam * _rank_norm(topo)
