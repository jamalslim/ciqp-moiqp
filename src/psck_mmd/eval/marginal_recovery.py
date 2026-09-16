"""
psck_mmd.eval.marginal_recovery — IQP-native exact per-feature marginals.

Mathematical identity
---------------------
Let f index a feature, with B qubits {q_{f,0}, …, q_{f,B-1}}. The reduced
density matrix on these B qubits is diagonal in the computational basis
(IQP states are diagonal up to the input H^⊗n), so the per-feature marginal
distribution is a probability vector p_f ∈ ℝ^{2^B} on the B-bit patterns.

The Walsh expansion gives, for any β ⊆ {0, …, B-1} and any x ∈ {0,1}^B,
    χ_β(x) = ∏_{i ∈ β} (1 − 2 x_i) = (-1)^{β·x}.
The orthonormal basis {χ_β} on {0,1}^B inverts the moment representation:
    p_f(x) = (1/2^B) Σ_β ⟨Z_β⟩_f · (-1)^{β·x}
where ⟨Z_β⟩_f = Σ_x p_f(x) (-1)^{β·x} is the IQP-native marginal expectation
on the qubits of β (a subset of feature-f qubits).

This is a 2^B × 2^B Walsh-Hadamard matrix-vector product per feature. At
B = 8: 256 × 256 ≈ 65,500 flops per feature × D features = trivial.

The cost lives in obtaining the 2^B − 1 nontrivial expectations per feature:
each requires one Van den Nest MC forward pass. We provide
`enumerate_intra_feature_observables(D, B)` for that.

The recovered p_f on integer levels v ∈ {0, …, 2^B − 1} can be read against
the data's empirical level histogram directly. From p_f we can compute
exact W₁ and KS distances against any reference distribution, no sampling.

This module does NOT touch a statevector. It only requires the IQP oracle's
ability to return ⟨Z_β⟩ for low-order Z observables — exactly what Van den
Nest MC gives in polynomial time.

Public API
----------
    enumerate_intra_feature_observables(D, B)
        → (obs_list, slot_index)  full set of (D × 2^B) tuples + lookup map

    pattern_to_level(pattern_bits, encoding, bits)
        → integer level ∈ {0, …, 2^B−1}  (inverse of the encoding table row map)

    recover_feature_distributions(z_full, feature_obs_index, D, B, encoding)
        → (D, 2^B) ndarray of per-feature probability vectors over LEVELS

    moment_mmd_from_correlators(z_model, z_data, observables,
                                 kernel_coeffs)
        → scalar moment-domain MMD² between model and held-out target

Notes
-----
Encoding-aware level mapping: the Hadamard inversion gives p_f(x) on the
2^B BIT patterns x. To compare against the data's per-feature INTEGER
LEVEL distribution we must map each bit pattern back to the level it
encodes. This is encoding-specific and uses the inverse of the encoding
table from psck_mmd.data.
"""

from __future__ import annotations
import numpy as np
from itertools import combinations


# ─────────────────────────────────────────────────────────────────────────────
# Observable enumeration (intra-feature, all subsets up to weight B)
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_intra_feature_observables(D: int, B: int):
    """All Z observables that live entirely within a single feature, for
    every feature, including the empty set (which we drop and treat as
    ⟨I⟩ = 1 implicitly during inversion).

    Returns
    -------
    obs_list : list of tuples of qubit indices, length D * (2^B − 1).
               Each tuple is sorted ascending. Excludes the empty set.
    slot_index : (D, 2^B) int array. slot_index[f, mask] gives the index in
                 obs_list of the observable on feature f selected by the
                 B-bit `mask` (where bit i of mask says whether qubit
                 q_{f,i} is in β). slot_index[f, 0] = -1 (empty set marker).
    """
    obs_list: list[tuple[int, ...]] = []
    slot_index = -np.ones((D, 1 << B), dtype=np.int64)
    for f in range(D):
        base = f * B
        qubits = [base + i for i in range(B)]
        for mask in range(1, 1 << B):
            tup = tuple(qubits[i] for i in range(B) if (mask >> i) & 1)
            slot_index[f, mask] = len(obs_list)
            obs_list.append(tup)
    return obs_list, slot_index


# ─────────────────────────────────────────────────────────────────────────────
# Encoding-table inversion: bit pattern → integer level
# ─────────────────────────────────────────────────────────────────────────────

def build_pattern_to_level_map(bits: int, encoding: str) -> np.ndarray:
    """Return an int array `lookup` of shape (2^B,) such that for the
    natural ordering of bit patterns (LSB at qubit index 0), `lookup[mask]`
    gives the integer LEVEL that the encoding assigns to mask.

    For our convention (psck_mmd.data quantile encoding):
        binary[:, f*B + k]  is bit k of feature f, with k=0 the MSB.
    So a bit pattern over the B qubits of feature f, written as
    (b_0, b_1, …, b_{B-1}) in qubit-index order, corresponds to encoding
    table row v iff table[v] == (b_0, b_1, …, b_{B-1}).

    We assemble `mask` here in the LSB convention used by `slot_index` from
    enumerate_intra_feature_observables: bit i of mask corresponds to qubit
    q_{f, i}, i.e., position i in the encoding table row.
    """
    from psck_mmd.data import ENCODING_TABLES

    if encoding not in ENCODING_TABLES:
        raise ValueError(f"unknown encoding {encoding!r}")
    nb = 1 << bits
    table = ENCODING_TABLES[encoding](nb, bits)   # (nb, bits) in {0,1}

    # lookup[mask] = level v such that table[v] (bits in qubit-position order)
    # equals the bit pattern (b_0, b_1, ..., b_{B-1}) where b_i = (mask>>i)&1.
    lookup = -np.ones(nb, dtype=np.int64)
    for v in range(nb):
        mask = 0
        for k in range(bits):
            if int(table[v, k]):
                mask |= (1 << k)
        if lookup[mask] != -1:
            raise RuntimeError(
                f"Encoding {encoding!r} at B={bits}: pattern collision at "
                f"mask={mask} (levels {lookup[mask]} and {v})."
            )
        lookup[mask] = v
    if (lookup < 0).any():
        raise RuntimeError(
            f"Encoding {encoding!r} at B={bits} does not span all 2^B bit "
            f"patterns; recovery would be incomplete."
        )
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Core: Walsh-Hadamard inversion per feature
# ─────────────────────────────────────────────────────────────────────────────

def _walsh_hadamard_basis(B: int) -> np.ndarray:
    """Return W of shape (2^B, 2^B) where W[mask, x] = (-1)^{mask·x}.

    This is the un-normalized Hadamard / character matrix. Both row and
    column index a B-bit string; the entry is the value of character χ_mask
    at point x.
    """
    nb = 1 << B
    masks = np.arange(nb, dtype=np.int64)[:, None]
    xs    = np.arange(nb, dtype=np.int64)[None, :]
    # popcount of (mask & x) modulo 2 → sign
    bits = np.bitwise_and(masks, xs)
    parity = np.zeros_like(bits)
    for k in range(B):
        parity ^= (bits >> k) & 1
    return (1 - 2 * parity).astype(np.float64)


def recover_feature_distributions(
    z_full: np.ndarray,
    slot_index: np.ndarray,
    D: int,
    B: int,
    encoding: str,
    clip: bool = True,
) -> np.ndarray:
    """Reconstruct exact per-feature distributions over integer levels.

    Parameters
    ----------
    z_full     : 1-D array of length len(obs_list) from
                 enumerate_intra_feature_observables(D, B). Entry j must be
                 ⟨Z_{obs_list[j]}⟩ as estimated from the trained IQP model.
    slot_index : (D, 2^B) int array from the same enumerator.
    D, B       : architecture sizes.
    encoding   : one of {"binary","gray","hadamard"} — used to map bit
                 patterns back to integer levels.
    clip       : if True, clip per-pattern probabilities to [0,1] and
                 re-normalize, to absorb negative tails caused by MC noise
                 in z_full. Default True. Set False to inspect the raw
                 (signed) reconstruction for diagnostics.

    Returns
    -------
    p : (D, 2^B) ndarray. p[f, v] = probability that feature f takes level v
        under the trained model, on a fixed lattice of 2^B integer levels.
    """
    nb = 1 << B
    W = _walsh_hadamard_basis(B)              # (2^B, 2^B), W[mask, x]
    lookup = build_pattern_to_level_map(B, encoding)   # mask → level

    # Per feature, build the moment vector indexed by mask ∈ {0, ..., 2^B-1}
    # with mom[0] = 1 (trivial observable) and mom[mask] = z_full[slot_index[f,mask]] for mask>0.
    p = np.zeros((D, nb), dtype=np.float64)
    for f in range(D):
        mom = np.empty(nb, dtype=np.float64)
        mom[0] = 1.0
        for mask in range(1, nb):
            mom[mask] = z_full[slot_index[f, mask]]
        # Walsh inversion: p_pattern(x) = (1/2^B) Σ_mask mom[mask] * (-1)^{mask·x}
        p_patterns = (W.T @ mom) / nb         # (2^B,), indexed by bit pattern x
        # Re-index by integer level v
        p_levels = np.zeros(nb, dtype=np.float64)
        for mask in range(nb):
            v = int(lookup[mask])
            p_levels[v] = p_patterns[mask]
        if clip:
            p_levels = np.clip(p_levels, 0.0, None)
            s = p_levels.sum()
            if s > 0:
                p_levels /= s
        p[f] = p_levels
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Empirical per-feature distributions (data side) for direct comparison
# ─────────────────────────────────────────────────────────────────────────────

def empirical_feature_distributions(
    binary: np.ndarray,
    D: int,
    B: int,
    encoding: str,
) -> np.ndarray:
    """Compute the empirical per-feature integer-level distribution from a
    bit-encoded sample matrix `binary` of shape (N, D*B).

    Returns p_emp of shape (D, 2^B) where p_emp[f, v] is the empirical
    frequency of level v on feature f.
    """
    nb = 1 << B
    lookup = build_pattern_to_level_map(B, encoding)
    N = binary.shape[0]
    p = np.zeros((D, nb), dtype=np.float64)
    for f in range(D):
        bits_f = binary[:, f * B:(f + 1) * B].astype(np.int64)
        # Compose mask from bit slice using qubit-position-to-LSB convention
        masks = np.zeros(N, dtype=np.int64)
        for k in range(B):
            masks |= (bits_f[:, k] << k)
        levels = lookup[masks]
        counts = np.bincount(levels, minlength=nb).astype(np.float64)
        p[f] = counts / counts.sum()
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Joint moment-MMD: closed-form distance on low-order Z-correlator vectors
# ─────────────────────────────────────────────────────────────────────────────

def moment_mmd_from_correlators(
    z_model: np.ndarray,
    z_data:  np.ndarray,
    kernel_coeffs: np.ndarray,
) -> float:
    """Closed-form MMD² between model and data restricted to a finite
    Z-correlator basis with non-negative Fourier coefficients.

        MMD² = Σ_β k̂_β · (⟨Z_β⟩_θ − ⟨Z_β⟩_data)²

    Parameters
    ----------
    z_model        : (K,) ndarray of model-side ⟨Z_β⟩ for β ∈ basis
    z_data         : (K,) ndarray of held-out data-side ⟨Z_β⟩
    kernel_coeffs  : (K,) ndarray of non-negative Fourier coefficients k̂_β
                     (e.g. from psck_mmd.psck_kernel.heat_kernel_coeffs).

    Returns
    -------
    mmd2 : float ≥ 0.

    Notes
    -----
    This is the natural training-objective-aligned distance: it evaluates
    the very kernel the model was (or could have been) trained against, but
    on independent ⟨Z_β⟩_data computed from a HELD-OUT sample. It is
    unaffected by sampling because it does not require sampling — it is
    computed entirely from low-order Z-correlator estimates that the IQP
    oracle returns natively.
    """
    if not (z_model.shape == z_data.shape == kernel_coeffs.shape):
        raise ValueError(
            f"shape mismatch: z_model={z_model.shape}, "
            f"z_data={z_data.shape}, k̂={kernel_coeffs.shape}"
        )
    if (kernel_coeffs < 0).any():
        raise ValueError("kernel_coeffs must be non-negative for valid MMD.")
    delta = z_model - z_data
    return float((kernel_coeffs * delta * delta).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Audit
# ─────────────────────────────────────────────────────────────────────────────

def audit_recovery(D: int = 4, B: int = 3, encoding: str = "binary",
                    seed: int = 7) -> bool:
    """Self-test: build a random discrete distribution per feature, compute
    its full Z-correlator vector exactly, then invert and verify recovery
    matches to machine precision."""
    rng = np.random.default_rng(seed)
    nb = 1 << B
    obs_list, slot_index = enumerate_intra_feature_observables(D, B)
    p_target = rng.dirichlet(np.ones(nb), size=D)        # (D, 2^B), simplex per row
    # Compute z = ⟨Z_β⟩ = Σ_x p(x) (-1)^{β·x}, indexed in obs_list order.
    W = _walsh_hadamard_basis(B)
    lookup = build_pattern_to_level_map(B, encoding)
    z = np.empty(len(obs_list), dtype=np.float64)
    for f in range(D):
        # First express p_target on bit patterns (inverse of lookup).
        p_pat = np.empty(nb)
        for mask in range(nb):
            v = int(lookup[mask])
            p_pat[mask] = p_target[f, v]
        moments_per_mask = W @ p_pat   # mom[mask] = Σ_x (-1)^{mask·x} p(x)
        for mask in range(1, nb):
            z[slot_index[f, mask]] = moments_per_mask[mask]
    p_recov = recover_feature_distributions(z, slot_index, D, B, encoding,
                                              clip=False)
    err = float(np.max(np.abs(p_recov - p_target)))
    ok = err < 1e-12
    print(f"  marginal_recovery audit (D={D}, B={B}, enc={encoding}): "
          f"max abs err = {err:.2e}  {'✓' if ok else '✗'}")
    return ok


if __name__ == "__main__":
    audit_recovery(D=4, B=3, encoding="binary")
    audit_recovery(D=4, B=3, encoding="gray")
    audit_recovery(D=4, B=4, encoding="binary")
    # NB: hadamard encoding has duplicate rows at B=2 (provably non-invertible
    # because Sylvester WH matrix has a constant first column). Marginal
    # recovery is therefore meaningful for hadamard only at B ≥ 3. We do not
    # include hadamard at B=2 here.
    audit_recovery(D=8, B=8, encoding="binary")
