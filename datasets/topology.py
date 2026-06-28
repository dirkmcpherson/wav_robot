"""
Per-snippet topology measures comparing the FDM (world model) and IDM latent spaces.

A trajectory snippet gives two *paired* latent sequences -- one from each model,
indexed by the same timesteps. The FDM latent (e.g. WorldModel `h_t_map` pooled, or
the action latent `z`) and the IDM latent (SparseIDM `_last_embedding`) live in
DIFFERENT spaces with different dimensionality and semantics. So the only valid
comparison is *relational* -- who-is-near-whom -- which is invariant to dimensionality,
rotation and scaling. Never compare raw coordinates across the two spaces.

Two relational measures are provided:

  A. `snippet_rsa`         -- Representational Similarity Analysis on the snippet's
                              own points: correlate the two LxL pairwise-distance
                              matrices (RDMs). One scalar per snippet, no reference
                              set. Simple but noisy for small L.

  B. `knn_agreement`       -- Local neighbourhood agreement against a shared reference
                              cloud: for each snippet point, compare its k-NN set in
                              FDM space vs IDM space (Jaccard). Stable per-point signal;
                              average over the snippet.

Both return *agreement in [0, 1]* (higher = the two models organise the snippet the
same way). Use `1 - agreement` as a disagreement / acquisition score.
"""

import numpy as np

try:  # optional, only for a slightly faster/cleaner Spearman
    from scipy.stats import spearmanr as _scipy_spearmanr
except Exception:  # pragma: no cover
    _scipy_spearmanr = None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pdist(X):
    """Euclidean pairwise-distance matrix for X: [L, d] -> [L, L]."""
    X = np.asarray(X, dtype=np.float64)
    sq = np.sum(X * X, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def _upper(M):
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


def _spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    if _scipy_spearmanr is not None:
        r = _scipy_spearmanr(a, b).correlation
        return float(r)
    # numpy fallback: Pearson on ranks
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else np.nan


def _l2norm(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


# --------------------------------------------------------------------------- #
# A. snippet-level RSA
# --------------------------------------------------------------------------- #
def snippet_rsa(fdm_latents, idm_latents):
    """RDM-correlation between the two latent organisations of one snippet.

    fdm_latents, idm_latents: [L, d_fdm], [L, d_idm] -- MUST be aligned (same L,
    paired by timestep/transition). Returns Spearman correlation of the two RDMs'
    off-diagonal entries, mapped to agreement in [0, 1] via (r + 1) / 2.
    Returns NaN if the snippet is degenerate (constant distances).
    """
    fdm_latents = np.asarray(fdm_latents, dtype=np.float64)
    idm_latents = np.asarray(idm_latents, dtype=np.float64)
    L = min(len(fdm_latents), len(idm_latents))
    if L < 3:
        return np.nan
    rd_f = _upper(_pdist(fdm_latents[:L]))
    rd_i = _upper(_pdist(idm_latents[:L]))
    r = _spearman(rd_f, rd_i)
    if np.isnan(r):
        return np.nan
    return 0.5 * (r + 1.0)  # [-1,1] -> [0,1]


# --------------------------------------------------------------------------- #
# B. local k-NN agreement against a shared reference cloud
# --------------------------------------------------------------------------- #
def _knn_indices(queries, reference, k, cosine=True):
    """Indices of the k nearest reference points for each query. [Nq, k]."""
    if cosine:
        Q = _l2norm(queries)
        R = _l2norm(reference)
        sim = Q @ R.T            # higher = nearer
        # exclude self if query is literally in reference is handled by caller
        return np.argsort(-sim, axis=1)[:, :k]
    else:
        Q = np.asarray(queries, dtype=np.float64)
        R = np.asarray(reference, dtype=np.float64)
        d = (np.sum(Q * Q, 1)[:, None] + np.sum(R * R, 1)[None, :] - 2 * Q @ R.T)
        return np.argsort(d, axis=1)[:, :k]


def knn_agreement(q_fdm, q_idm, ref_fdm, ref_idm, k=10, cosine=True):
    """Per-query Jaccard overlap of k-NN sets across the two spaces.

    q_fdm/q_idm:   [Nq, d*]  query points (e.g. a snippet's timesteps) in each space.
    ref_fdm/ref_idm: [Nr, d*] the SAME reference rows encoded by each model.
    Returns `agreement[Nq]` in [0, 1]; average over a snippet's points for a
    per-snippet score. The two reference clouds must be row-aligned (same underlying
    transitions), so neighbour *indices* are comparable across spaces.
    """
    nn_f = _knn_indices(q_fdm, ref_fdm, k, cosine=cosine)
    nn_i = _knn_indices(q_idm, ref_idm, k, cosine=cosine)
    out = np.empty(len(nn_f), dtype=np.float64)
    for j in range(len(nn_f)):
        a = set(nn_f[j].tolist())
        b = set(nn_i[j].tolist())
        inter = len(a & b)
        union = len(a | b)
        out[j] = inter / union if union else np.nan
    return out


def snippet_knn_agreement(win_idx, ref_fdm, ref_idm, k=10, cosine=True):
    """Per-snippet score using the reference cloud itself as the query source.

    `win_idx` indexes rows of the reference clouds that belong to one snippet.
    Returns the mean k-NN Jaccard agreement over the snippet's points (self excluded).
    """
    win_idx = np.asarray(win_idx)
    q_f = ref_fdm[win_idx]
    q_i = ref_idm[win_idx]
    # k+1 then drop self-match (each query is a reference row).
    nn_f = _knn_indices(q_f, ref_fdm, k + 1, cosine=cosine)
    nn_i = _knn_indices(q_i, ref_idm, k + 1, cosine=cosine)
    vals = []
    for row, qi in enumerate(win_idx):
        a = [x for x in nn_f[row].tolist() if x != qi][:k]
        b = [x for x in nn_i[row].tolist() if x != qi][:k]
        a, b = set(a), set(b)
        u = len(a | b)
        vals.append(len(a & b) / u if u else np.nan)
    return float(np.nanmean(vals)) if vals else np.nan


def disagreement(agreement):
    """Convert an agreement score in [0,1] to a disagreement / acquisition score."""
    return 1.0 - np.asarray(agreement, dtype=np.float64)


# --------------------------------------------------------------------------- #
# C. additional snippet-level relational measures (all cross-space valid)
# --------------------------------------------------------------------------- #
def _rdm(X, metric="euclidean"):
    X = np.asarray(X, dtype=np.float64)
    if metric == "cosine":
        Xn = _l2norm(X)
        return 1.0 - (Xn @ Xn.T)
    return _pdist(X)


def snippet_rsa_g(fdm, idm, rdm_metric="euclidean", corr="spearman"):
    """Generalised RSA: choose the RDM distance and the RDM-correlation."""
    fdm = np.asarray(fdm, dtype=np.float64); idm = np.asarray(idm, dtype=np.float64)
    L = min(len(fdm), len(idm))
    if L < 3:
        return np.nan
    a = _upper(_rdm(fdm[:L], rdm_metric)); b = _upper(_rdm(idm[:L], rdm_metric))
    if corr == "pearson":
        if np.allclose(a, a[0]) or np.allclose(b, b[0]):
            return np.nan
        am, bm = a - a.mean(), b - b.mean()
        den = np.sqrt((am * am).sum() * (bm * bm).sum())
        r = float((am * bm).sum() / den) if den > 0 else np.nan
    else:
        r = _spearman(a, b)
    return np.nan if np.isnan(r) else 0.5 * (r + 1.0)


def snippet_cka(fdm, idm):
    """Linear CKA between the two latent sets over a snippet. Already in [0,1]."""
    X = np.asarray(fdm, dtype=np.float64); Y = np.asarray(idm, dtype=np.float64)
    L = min(len(X), len(Y))
    if L < 3:
        return np.nan
    X = X[:L] - X[:L].mean(0); Y = Y[:L] - Y[:L].mean(0)
    num = np.linalg.norm(Y.T @ X, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / den) if den > 0 else np.nan


def snippet_dcor(fdm, idm):
    """Distance correlation between the two latent sets over a snippet. In [0,1].

    Captures nonlinear dependence and is dimension-agnostic -- a natural fit for
    comparing two differently-shaped latent spaces.
    """
    X = np.asarray(fdm, dtype=np.float64); Y = np.asarray(idm, dtype=np.float64)
    L = min(len(X), len(Y))
    if L < 3:
        return np.nan
    A = _pdist(X[:L]); B = _pdist(Y[:L])
    A = A - A.mean(0)[None, :] - A.mean(1)[:, None] + A.mean()
    B = B - B.mean(0)[None, :] - B.mean(1)[:, None] + B.mean()
    dcov2 = (A * B).mean()
    dvarx = (A * A).mean(); dvary = (B * B).mean()
    den = np.sqrt(dvarx * dvary)
    if den <= 0:
        return np.nan
    return float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(den))


# Registry of snippet-level agreement methods (return agreement in [0,1]).
SNIPPET_METHODS = {
    "rsa_eucl_spear": lambda f, i: snippet_rsa_g(f, i, "euclidean", "spearman"),
    "rsa_cos_spear":  lambda f, i: snippet_rsa_g(f, i, "cosine", "spearman"),
    "rsa_eucl_pear":  lambda f, i: snippet_rsa_g(f, i, "euclidean", "pearson"),
    "cka":            snippet_cka,
    "dcor":           snippet_dcor,
}
