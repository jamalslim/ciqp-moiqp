"""
psck_mmd.feature_obs — Enumeration of Z-observables for feature-structured data.

For feature-vector data with D features × B bits each (so n = D·B qubits
laid out as q = f·B + k for feature f ∈ [D] and bit k ∈ [B]), we enumerate
all Z-observables up to a chosen max_weight that are relevant to
reconstructing feature-level moments.

Weight-1:  ⟨Z_{f,k}⟩                 → E[b_{f,k}]
Weight-2 intra-feature (f,k)-(f,l):    → E[b_{f,k} b_{f,l}]  → Var(S_f)
Weight-2 cross-feature (f,k)-(g,l):    → E[b_{f,k} b_{g,l}]  → Cov(S_f, S_g)
Weight-3 (optional):                   → triple correlators, needed for a
                                         rank-augmented Pearson Jacobian.
"""

from __future__ import annotations
import numpy as np


def enumerate_feature_observables(D: int, B: int, max_weight: int = 2):
    """Enumerate all feature-relevant Z observables up to max_weight.

    Returns:
        obs_list : sorted list of tuples of qubit indices
        weights  : ndarray of |S| for each observable in obs_list
    """
    assert max_weight >= 1
    obs: set[tuple[int, ...]] = set()
    n = D * B
    for q in range(n):
        obs.add((q,))
    if max_weight >= 2:
        for f in range(D):
            for k in range(B):
                for l in range(k + 1, B):
                    obs.add((f * B + k, f * B + l))
        for f in range(D):
            for g in range(f + 1, D):
                for k in range(B):
                    for l in range(B):
                        i, j = f * B + k, g * B + l
                        obs.add((min(i, j), max(i, j)))
    if max_weight >= 3:
        # v4.14 (review item B5): INTRA-feature triples, D*C(B,3) observables.
        # These fix the third cumulants of the single-feature marginals, which is
        # what the weight-3 physics argument (heavy-tailed inner-shower cells)
        # requires; the previous cross-feature enumeration did not implement it.
        for f in range(D):
            for k in range(B):
                for l in range(k + 1, B):
                    for m in range(l + 1, B):
                        obs.add((f * B + k, f * B + l, f * B + m))
    obs_list = sorted(obs, key=lambda t: (len(t), t))
    weights = np.array([len(t) for t in obs_list], dtype=np.int32)
    return obs_list, weights
