"""
psck_mmd.correlator — Van den Nest Fourier Monte Carlo engine for IQP circuits.

Exact identity (Van den Nest 2010, arXiv:0911.1624): for an IQP circuit
  U(θ) = prod_j exp(i θ_j P_j)   acting on |+⟩^⊗n,
with P_j a Z-Pauli on subset gate_j ⊆ [n] (single or pair), the marginal
Pauli-Z expectation value

  ⟨Z_S⟩(θ) = (1/2^n) · sum_{x∈{0,1}^n} cos(2 · sum_{j: |gate_j ∩ S| odd} θ_j·(-1)^{x·gate_j})

The inner sum over x is estimated by a Monte-Carlo average with M Bernoulli-1/2
samples of x. The index set {j : |gate_j ∩ S| odd} is the "active set" of S:
it depends only on the graph, so it is precomputed once.

Backward-mode gradient (analytic, unbiased for the same M samples):
  ∂⟨Z_S⟩/∂θ_j = -2 · E_x[(-1)^{x·gate_j} · sin(arg_S(x))]     if j ∈ active(S)
              = 0                                              otherwise

Everything is strictly polynomial in n, M, and the number of gates. No
statevector, no 2^n storage, no 2^n time.
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Latent sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_latents(n: int, M: int, rng: np.random.Generator,
                   antithetic: bool = True) -> np.ndarray:
    """Return (M, n) int8 array in {0,1}. Antithetic pairing halves variance
    for single-qubit observables; pair observables are weakly affected."""
    if antithetic:
        half = M // 2
        x = (rng.random((half, n)) < 0.5).astype(np.int8)
        return np.concatenate([x, 1 - x], axis=0)
    return (rng.random((M, n)) < 0.5).astype(np.int8)


# ─────────────────────────────────────────────────────────────────────────────
# ParityCache: gate_sign[j, m] = (-1)^{latent[m]·gate_j}
# ─────────────────────────────────────────────────────────────────────────────

class ParityCache:
    """Precomputed per-gate sign signals over the latent batch.

    Attributes:
        latent : (M, n) int8 in {0,1}
        signs  : (n_gates, M) float32 in {-1, +1}
        gates  : list of tuples
    """

    __slots__ = ("latent", "signs", "gates")

    def __init__(self, latent: np.ndarray, gates: list[tuple[int, ...]]):
        self.latent = latent.astype(np.int8, copy=False)
        self.gates = gates
        M = latent.shape[0]
        ng = len(gates)
        # singles[q, m] = (-1)^{latent[m, q]} in {-1, +1}
        singles = (1 - 2 * latent.astype(np.int16, copy=False).T).astype(np.float32)
        signs = np.empty((ng, M), dtype=np.float32)
        for j, g in enumerate(gates):
            if len(g) == 1:
                signs[j] = singles[g[0]]
            else:
                s = singles[g[0]].copy()
                for q in g[1:]:
                    s = s * singles[q]
                signs[j] = s
        self.signs = signs

    def get_signs(self, idx: np.ndarray) -> np.ndarray:
        """Return signs[idx, :] as a contiguous 2-D view."""
        return self.signs[idx]

    def get_signs_single(self, j: int) -> np.ndarray:
        return self.signs[j]


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass
# ─────────────────────────────────────────────────────────────────────────────

def forward_batch(params: np.ndarray, cache: ParityCache,
                   active_lists: list[list[int]]):
    """Compute Van den Nest MC estimates of ⟨Z_{S_k}⟩(params) for k = 1..K.

    Returns:
        z    : (K,) float64 MC estimates
        args : list of length K, each (M,) float32 array arg_k = 2 Σ θ_j·sign_j
               (needed for backward_batch to avoid recomputing)
    """
    K = len(active_lists)
    M = cache.signs.shape[1]
    z = np.empty(K)
    args: list[np.ndarray] = [None] * K  # type: ignore
    for k, active in enumerate(active_lists):
        if not active:
            z[k] = 1.0
            args[k] = np.zeros(M, dtype=np.float32)
            continue
        a = np.asarray(active, dtype=np.int64)
        arg = 2.0 * (params[a][:, None] * cache.signs[a]).sum(axis=0)
        z[k] = float(np.mean(np.cos(arg)))
        args[k] = arg
    return z, args


# ─────────────────────────────────────────────────────────────────────────────
# Backward pass (chain rule for an upstream dL/dz)
# ─────────────────────────────────────────────────────────────────────────────

def backward_batch(params: np.ndarray, cache: ParityCache,
                    active_lists: list[list[int]], args: list[np.ndarray],
                    dL_dz: np.ndarray) -> np.ndarray:
    """Accumulate dL/dθ = Σ_k (dL/dz_k) · (-2) · E_x[sign_j·sin(arg_k)].

    Parameters
    ----------
    params       : (n_gates,) current angles (only used for shape)
    cache        : ParityCache returned by ParityCache(latent, gates)
    active_lists : list of active-gate-indices per observable
    args         : list of (M,) arrays arg_k from forward_batch
    dL_dz        : (K,) upstream gradient wrt z

    Returns
    -------
    grad : (n_gates,) dL/dθ accumulated over all observables
    """
    ng = params.shape[0]
    grad = np.zeros(ng, dtype=np.float64)
    M = cache.signs.shape[1]
    inv_M = 1.0 / M
    for k, active in enumerate(active_lists):
        d = float(dL_dz[k])
        if d == 0.0 or not active:
            continue
        a = np.asarray(active, dtype=np.int64)
        sin_arg = np.sin(args[k])
        e_signs_sin = (cache.signs[a] * sin_arg[None, :]).sum(axis=1) * inv_M
        grad[a] -= 2.0 * d * e_signs_sin
    return grad


# ─────────────────────────────────────────────────────────────────────────────
# Exact data Z-correlators (no MC)
# ─────────────────────────────────────────────────────────────────────────────

def z_data_batch(binary: np.ndarray, observables: list[tuple[int, ...]]) -> np.ndarray:
    """⟨Z_S⟩_data = E[prod_{i∈S} (1 - 2 b_i)] computed directly from the
    binary-encoded data matrix.
    """
    signs = 1.0 - 2.0 * binary.astype(np.float64)
    K = len(observables)
    z = np.empty(K)
    for k, S in enumerate(observables):
        if not S:
            z[k] = 1.0
            continue
        p = np.ones(binary.shape[0])
        for q in S:
            p = p * signs[:, q]
        z[k] = float(p.mean())
    return z
