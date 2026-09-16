"""
corr_jacobian.py — Analytical Jacobian of the Pearson correlation matrix
with respect to Z-moments.

Setup:
  Feature f ∈ {0,...,D-1} encoded as B bits b_{f,0}, ..., b_{f,B-1} ∈ {0,1}.
  Feature value   S_f = sum_{k=0..B-1} w_k b_{f,k},  w_k = 2^{B-1-k}  (binary).
  z1_{f,k}        = <Z_{f·B+k}> = 1 - 2 E[b_{f,k}]
  z2_{fg,kl}      = <Z_{f·B+k} Z_{g·B+l}> for (f,k) != (g,l).

From these:
  E[b_{f,k}]                 = (1 - z1_{f,k})/2
  E[b_{f,k} b_{g,l}]         = (1 - z1_{f,k} - z1_{g,l} + z2_{fg,kl})/4  ((f,k)!=(g,l))
  Cov(b_{f,k}, b_{g,l})      = E[..] - E[b_{f,k}] E[b_{g,l}]

Feature covariance:
  Cov(S_f, S_g) = sum_{k,l} w_k w_l Cov(b_{f,k}, b_{g,l})

Feature variance (diagonal, needs k==l term):
  Var(S_f) = sum_{k,l} w_k w_l Cov(b_{f,k}, b_{f,l})
           = sum_k w_k^2 Var(b_{f,k}) + sum_{k!=l} w_k w_l Cov(b_{f,k}, b_{f,l})
  Var(b_{f,k}) = E[b]*(1-E[b]) = (1-z1^2)/4.

Pearson correlation:
  rho_fg = Cov(S_f, S_g) / sqrt( Var(S_f) Var(S_g) )

We need d rho_fg / d z_beta for every beta (Z-observable index). Since Cov and
Var are POLYNOMIAL (in fact bilinear/quadratic) in the z1, z2 entries, and the
square-roots in the denominator are the only non-trivial nonlinearity, the
Jacobian is available analytically. We compute it by symbolic backprop.

What this module provides:
    pearson_value(z, observables, D, B)  -> rho (D, D)
    pearson_jacobian(z, observables, D, B)
        -> (P, K)  where P = D*(D-1)/2 and K = len(observables)

The Jacobian is the row-major flattening of d rho_{fg} / d z_k for f < g.
"""

from __future__ import annotations
import numpy as np


def _build_index_maps(observables, D, B):
    """Build integer maps from (f,k) -> observable index for z1, and
    from ((f,k),(g,l)) -> observable index for z2. These let us look up
    the right entry of the big z-vector when building rho / Jacobian.

    Returns:
        z1_idx   (D, B) int   — index into z of the single-qubit Z on feature f bit k
        z2_idx   dict  keyed by frozenset({f*B+k, g*B+l}) → index into z
    """
    obs_to_idx = {tuple(sorted(o)): i for i, o in enumerate(observables)}
    n = D * B
    z1_idx = np.full((D, B), -1, dtype=np.int64)
    for f in range(D):
        for k in range(B):
            q = f * B + k
            t = (q,)
            if t in obs_to_idx:
                z1_idx[f, k] = obs_to_idx[t]
    z2_idx: dict[tuple[int, int], int] = {}
    for f in range(D):
        for k in range(B):
            for g in range(D):
                for l in range(B):
                    qi, qj = f * B + k, g * B + l
                    if qi >= qj:
                        continue
                    t = (qi, qj)
                    if t in obs_to_idx:
                        z2_idx[t] = obs_to_idx[t]
    return z1_idx, z2_idx


def _expand_bit_weights(B: int) -> np.ndarray:
    """w_k = 2^{B-1-k} for binary encoding. For Gray encoding, the
    appropriate linear-reconstruction weights differ but are not supported
    here — use binary.
    """
    return np.array([2 ** (B - 1 - k) for k in range(B)], dtype=np.float64)


def pearson_value_and_jacobian(z: np.ndarray,
                                observables,
                                D: int, B: int,
                                eps_std: float = 1e-10):
    """Compute rho (D, D) AND its Jacobian (P, K) where P = D*(D-1)/2.

    The Jacobian is expressed in the z ordering given by `observables`, so
    column j of the Jacobian matches entry z[j] of the MC forward pass.

    This Jacobian is the analytical backprop of the composition
        z  ->  cov(D,D), var(D)  ->  rho(D,D)
    using chain rule. Each step is a polynomial (or sqrt) so exact.
    """
    z = np.asarray(z, dtype=np.float64)
    K = z.shape[0]
    n = D * B
    w = _expand_bit_weights(B)
    z1_idx, z2_idx = _build_index_maps(observables, D, B)

    # -- Step 1: extract z1, z2 in structured arrays
    z1 = np.where(z1_idx >= 0, z[z1_idx.clip(0)], 0.0)  # (D, B)

    # Build z2 as sparse pair lookup
    pair_keys = list(z2_idx.keys())             # list of (qi, qj)
    pair_obs_idx = np.array([z2_idx[k] for k in pair_keys], dtype=np.int64)  # indices into z
    pair_fk = np.array([(qi // B, qi % B) for (qi, _) in pair_keys], dtype=np.int32)
    pair_gl = np.array([(qj // B, qj % B) for (_, qj) in pair_keys], dtype=np.int32)
    z2_vals = z[pair_obs_idx] if len(pair_obs_idx) else np.zeros(0)

    # -- Step 2: expectations
    #   Eb[f,k]  = (1 - z1[f,k]) / 2
    Eb = 0.5 * (1.0 - z1)                           # (D, B)

    # -- Step 3: Cov(b_fk, b_gl) for k==l same feature: Var
    Varb = Eb * (1.0 - Eb)                          # (D, B)

    # -- Step 4: Per-pair Cov contribution
    #   Cov(b_fk, b_gl) = E[b_fk b_gl] - Eb[f,k] Eb[g,l]
    # for f==g, k!=l this is intra; for f!=g this contributes to cross-cov.
    # E[b_fk b_gl] = (1 - z1[f,k] - z1[g,l] + z2)/4
    if len(pair_obs_idx):
        f_pair = pair_fk[:, 0]
        k_pair = pair_fk[:, 1]
        g_pair = pair_gl[:, 0]
        l_pair = pair_gl[:, 1]
        E_ij = (1.0 - z1[f_pair, k_pair] - z1[g_pair, l_pair] + z2_vals) / 4.0
        cov_ij = E_ij - Eb[f_pair, k_pair] * Eb[g_pair, l_pair]
    else:
        f_pair = k_pair = g_pair = l_pair = np.zeros(0, dtype=np.int32)
        cov_ij = np.zeros(0)

    # -- Step 5: Assemble Var(S_f) and Cov(S_f, S_g)
    VarS = np.zeros(D)
    # diagonal k==k term:
    for f in range(D):
        VarS[f] += float((w * w * Varb[f]).sum())
    # intra-feature k != l off-diagonals (f==g, k!=l) — included in cov_ij when f==g
    intra_mask = (f_pair == g_pair)
    if intra_mask.any():
        wwi = w[k_pair[intra_mask]] * w[l_pair[intra_mask]]
        for idx, f in enumerate(f_pair[intra_mask]):
            pass  # we use scatter below
        # scatter-add into VarS
        for cnt_idx in np.where(intra_mask)[0]:
            f = f_pair[cnt_idx]
            VarS[f] += 2.0 * w[k_pair[cnt_idx]] * w[l_pair[cnt_idx]] * cov_ij[cnt_idx]

    CovS = np.zeros((D, D))
    cross_mask = (f_pair != g_pair)
    wwc = w[k_pair[cross_mask]] * w[l_pair[cross_mask]]
    cc = cov_ij[cross_mask]
    for cnt_idx, is_cross in enumerate(cross_mask):
        if not is_cross:
            continue
        f, g = int(f_pair[cnt_idx]), int(g_pair[cnt_idx])
        val = w[k_pair[cnt_idx]] * w[l_pair[cnt_idx]] * cov_ij[cnt_idx]
        CovS[f, g] += val
        CovS[g, f] += val

    # -- Step 6: rho
    sigma = np.sqrt(np.maximum(VarS, eps_std))     # (D,)
    rho = np.eye(D)
    for f in range(D):
        for g in range(f + 1, D):
            denom = sigma[f] * sigma[g] + eps_std
            rho[f, g] = CovS[f, g] / denom
            rho[g, f] = rho[f, g]

    # ─────────────────────────────────────────────────────────────────────
    # Jacobian: reverse mode
    # ─────────────────────────────────────────────────────────────────────
    # We accumulate d rho_{fg} / d z[*] for each (f,g) with f<g.
    # We do this by: for each (f,g), the chain rule contributions are
    #   d rho / d Cov_{fg}    = 1 / (sigma_f sigma_g)
    #   d rho / d Var_f        = -rho/(2 Var_f)
    #   d rho / d Var_g        = -rho/(2 Var_g)
    # and Var, Cov depend linearly (bilinearly in Eb) on z1, z2.
    #
    # We return a flat (P, K) matrix where P = D*(D-1)/2, K = |observables|,
    # ordered by (f<g) lex.
    P = D * (D - 1) // 2
    J = np.zeros((P, K), dtype=np.float64)

    # Precompute d Eb[f,k] / d z1_{f,k} = -1/2
    d_Eb_d_z1 = -0.5                                # scalar, same everywhere
    # Precompute d Varb[f,k] / d Eb[f,k] = 1 - 2 Eb[f,k]
    d_Varb_d_Eb = 1.0 - 2.0 * Eb                    # (D, B)

    # For each pair (f,g):
    pair_idx_rho = 0
    for f in range(D):
        for g in range(f + 1, D):
            # Components of the gradient of rho_{fg}:
            #   rho_{fg} = Cov_{fg} / (sqrt(Var_f) sqrt(Var_g))
            sf, sg = sigma[f], sigma[g]
            inv_denom = 1.0 / (sf * sg + eps_std)
            # Partials:
            partial_Cov_fg = inv_denom                           # dRho/dCov_fg
            partial_Var_f  = -0.5 * rho[f, g] / max(VarS[f], eps_std)
            partial_Var_g  = -0.5 * rho[f, g] / max(VarS[g], eps_std)

            # (a) Contribution of Cov_{fg} to each z:
            # Cov_{fg} = sum_{kl} w_k w_l * cov_{fg, kl}
            # where cov = E[b_{f,k} b_{g,l}] - Eb[f,k] Eb[g,l]
            # d Cov_{fg} / d z2_{fg,kl} = w_k w_l / 4    (via dE[..]/dz2 = 1/4)
            # d Cov_{fg} / d z1_{f,k}   = w_k w_l * (-1/4) - w_k w_l * (-1/2) * Eb[g,l]
            #                           = w_k w_l * ( -1/4 + Eb[g,l]/2 )   (summed over l)
            # d Cov_{fg} / d z1_{g,l}   = w_k w_l * ( -1/4 + Eb[f,k]/2 )   (summed over k)
            # where summation reflects that a single z1_{f,k} appears across all l.

            # Precompute per-bit sums
            # T_g_k = sum_l w_l * Eb[g,l]    (weight-1 in l)
            # T_g   = sum_l w_l              -- constant
            sum_wl = w.sum()
            T_gk = (w * Eb[g]).sum()      # ≡ sum_l w_l Eb[g,l]
            T_fk = (w * Eb[f]).sum()      # ≡ sum_k w_k Eb[f,k]
            # For fixed k, d Cov_{fg}/d z1_{f,k} = w_k * ( -sum_l w_l /4 + T_gk / 2 )
            # Wait — let me redo carefully:
            # Cov_fg = sum_{k,l} w_k w_l [ E[b_fk b_gl] - Eb[f,k] Eb[g,l] ]
            # ∂/∂z1_{f,k*}:
            #   ∂E[b_fk b_gl]/∂z1_{f,k} = (-1/4) [k=k*]
            #   ∂Eb[f,k]/∂z1_{f,k} = -1/2 [k=k*]
            #   So ∂Cov_fg/∂z1_{f,k*} = sum_l w_{k*} w_l * (-1/4) - sum_l w_{k*} w_l * (-1/2) Eb[g,l]
            #                         = w_{k*} * ( -sum_l w_l /4 + (1/2) sum_l w_l Eb[g,l] )
            #                         = w_{k*} * ( -sum_wl/4 + T_gk/2 )
            # and similarly for g.
            dCov_dz1_f = w * (-sum_wl / 4.0 + T_gk / 2.0)   # (B,), indexed by k
            dCov_dz1_g = w * (-sum_wl / 4.0 + T_fk / 2.0)   # (B,), indexed by l

            # (b) Contribution of Var_f to each z: analogous for pair (f, f)
            # Var_f = sum_{k,l} w_k w_l Cov(b_fk, b_fl)     with k==l -> Var(b_fk)
            # Var_f = sum_k w_k^2 Varb[f,k] + 2 sum_{k<l} w_k w_l cov_{ff,kl}
            # Let's derive ∂Var_f/∂z1_{f,k*}:
            #   ∂Varb[f,k*]/∂z1_{f,k*} = (1 - 2 Eb[f,k*]) * (-1/2) = -(1 - 2 Eb[f,k*])/2
            #   ∂(cov_{ff,k*,l})/∂z1_{f,k*} for k*!=l:
            #       = -1/4 - (-1/2) Eb[f,l] = -1/4 + Eb[f,l]/2
            #   ∂(cov_{ff,k,k*})/∂z1_{f,k*} for k!=k*:
            #       same by symmetry
            # So ∂Var_f/∂z1_{f,k*} = w_{k*}^2 * (-(1-2Eb[f,k*])/2)
            #                     + 2 sum_{l != k*} w_{k*} w_l * (-1/4 + Eb[f,l]/2)
            # and ∂Var_f/∂z2_{f,k,l} = 2 w_k w_l * (1/4)  for k<l, else 0 via symmetry.
            T_f = (w * Eb[f]).sum()
            sum_wl_not = sum_wl - w  # sum_{l != k*} w_l as array over k*
            T_f_minus = np.array([T_f - w[k] * Eb[f, k] for k in range(B)])  # T_f without the k* term
            # ∂Var_f/∂z1_{f,k*}:
            dVarf_dz1_f = (
                w * w * (-(1.0 - 2.0 * Eb[f]) / 2.0)
                + 2.0 * w * (-(sum_wl_not) / 4.0 + T_f_minus / 2.0)
            )  # (B,)

            T_g = (w * Eb[g]).sum()
            T_g_minus = np.array([T_g - w[l] * Eb[g, l] for l in range(B)])
            dVarg_dz1_g = (
                w * w * (-(1.0 - 2.0 * Eb[g]) / 2.0)
                + 2.0 * w * (-(sum_wl_not) / 4.0 + T_g_minus / 2.0)
            )

            # ── Now fill in J ─────────────────────────────────────────
            # z1 contributions
            for k in range(B):
                jidx = z1_idx[f, k]
                if jidx >= 0:
                    J[pair_idx_rho, jidx] += partial_Cov_fg * dCov_dz1_f[k]
                    J[pair_idx_rho, jidx] += partial_Var_f * dVarf_dz1_f[k]
                jidx = z1_idx[g, k]
                if jidx >= 0:
                    J[pair_idx_rho, jidx] += partial_Cov_fg * dCov_dz1_g[k]
                    J[pair_idx_rho, jidx] += partial_Var_g * dVarg_dz1_g[k]

            # z2 contributions — cross (f,g) pairs and intra (f,f), (g,g)
            # For each z2 pair recorded:
            for cnt_idx in range(len(pair_obs_idx)):
                jidx = int(pair_obs_idx[cnt_idx])
                pf, pk = int(f_pair[cnt_idx]), int(k_pair[cnt_idx])
                pg, pl = int(g_pair[cnt_idx]), int(l_pair[cnt_idx])
                # Cov_fg term (only cross):
                if (pf, pg) == (f, g) or (pf, pg) == (g, f):
                    # Cov_fg contains this pair with coefficient w_{pk} w_{pl}
                    # where orientation is (pf -> f) consistent with bit index
                    w_coef = w[pk] * w[pl]
                    # The pair was stored as (min, max). If pf,pg == f,g (in that order)
                    # then k=pk, l=pl; otherwise swap.
                    J[pair_idx_rho, jidx] += partial_Cov_fg * (w_coef / 4.0)
                # Var_f term: pair must be both bits of feature f
                if (pf, pg) == (f, f):
                    # factor of 2 from the (k,l) and (l,k) pair symmetry already
                    # incorporated when we enumerated only k<l.
                    w_coef = w[pk] * w[pl]
                    J[pair_idx_rho, jidx] += partial_Var_f * (2.0 * w_coef / 4.0)
                if (pf, pg) == (g, g):
                    w_coef = w[pk] * w[pl]
                    J[pair_idx_rho, jidx] += partial_Var_g * (2.0 * w_coef / 4.0)

            pair_idx_rho += 1

    return rho, J
