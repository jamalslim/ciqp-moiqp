"""
psck_mmd.psck_kernel — Pearson-Stabilized Correlation Kernel MMD.

═════════════════════════════════════════════════════════════════════════════
CONSTRUCTION
═════════════════════════════════════════════════════════════════════════════

For x ∈ {0,1}^n, z_i(x) = 1 − 2 x_i ∈ {−1,+1}, and for β ⊆ [n],
    χ_β(x) := ∏_{i∈β} z_i(x) .
The functions {χ_β} form an orthonormal basis of L²({0,1}^n, uniform).
For a distribution p on {0,1}^n, ⟨χ_β⟩_p = ⟨Z_β⟩_p.

Any PSD kernel on {0,1}^n has the Fourier expansion
    k(x,y) = Σ_β k̂_β χ_β(x) χ_β(y), with k̂_β ≥ 0,
and its MMD is (Gretton 2012; Liu–Wang 2018):
    MMD²_k(p, q) = Σ_β k̂_β · ( ⟨Z_β⟩_p − ⟨Z_β⟩_q )².

PSCK-MMD = Liu–Wang heat MMD + rank-(D choose 2) Pearson-Jacobian correction:
    K = diag(ω_heat) + η · J*ᵀ J*,
    J*_{(fg),β} = ∂ρ_fg/∂⟨Z_β⟩ |_{⟨Z⟩ = ⟨Z⟩_data}.

Both terms are PSD, so K is PSD. The induced MMD is

    MMD²_PSCK(p_θ, p_data) = δzᵀ · K · δz,    δz = ⟨Z⟩_θ − ⟨Z⟩_data.

By Taylor expansion,
    ρ(⟨Z⟩_θ) − ρ(⟨Z⟩_data) = J* δz + O(‖δz‖²),
so the second term equals ‖ρ_θ − ρ_data‖² to leading order (Gauss–Newton
linearization of L_ρ), but inside a PSD kernel so it remains a valid MMD.

The loss only depends on p_θ through ⟨Z_β⟩, so it is classically estimable
via Van den Nest MC (polynomial time) while IQP SAMPLING of p_θ remains
hard for WORST-CASE instances under Bremner–Jozsa–Shepherd (2011). That is a
statement about the class, not about any particular trained instance: sampling
cost is set by the gate-graph treewidth, which for the sparse graphs used here
is small (see scripts/audit_simulability.py). Training is classical; deployment
via the cIQP Walsh–Hadamard compilation is what makes the model a generator,
not what makes it hard.
"""

from __future__ import annotations
import numpy as np

from .corr_jacobian import pearson_value_and_jacobian


# ─────────────────────────────────────────────────────────────────────────────
# Heat-kernel Fourier coefficients
# ─────────────────────────────────────────────────────────────────────────────

def heat_kernel_coeffs(observables, n: int, taus=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """Liu–Wang-style multi-bandwidth translation-invariant kernel on {0,1}^n:
        k(x,y) = (1/|T|) Σ_τ Π_i (1 + τ z_i(x) z_i(y))/2

    Its Fourier coefficient on β equals τ^|β| (up to normalization). Averaged
    over τ, and the β = ∅ mode is dropped (it corresponds to a constant that
    cancels in the MMD).
    """
    taus = np.asarray(taus, dtype=np.float64)
    K = len(observables)
    coeffs = np.zeros(K, dtype=np.float64)
    for k, obs in enumerate(observables):
        w = len(obs)
        if w == 0:
            coeffs[k] = 0.0
        else:
            coeffs[k] = float(np.mean(taus ** w))
    return coeffs


# ─────────────────────────────────────────────────────────────────────────────
# PSCK kernel assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_psck_kernel(z_data: np.ndarray, observables, D: int, B: int,
                       eta: float = 5.0,
                       taus=(0.1, 0.3, 0.5, 0.7, 0.9),
                       heat_scale: float = 1.0):
    """Assemble the PSCK kernel K = diag(ω) + η J*ᵀ J*.

    Returns (omega, Jstar, P, K_obs, eta) where omega is the heat diagonal,
    Jstar is the (P, K_obs) Pearson Jacobian at the data, P = D(D-1)/2.

    We do NOT explicitly form K, since (P, K_obs) representation is cheaper
    (O(P K) work per loss/gradient evaluation).
    """
    K_obs = len(observables)
    omega = heat_scale * heat_kernel_coeffs(observables, D * B, taus=taus)
    _, Jstar = pearson_value_and_jacobian(z_data, observables, D, B)
    assert Jstar.shape[1] == K_obs, \
        f"Jacobian column dim {Jstar.shape[1]} != K_obs {K_obs}"
    P = Jstar.shape[0]
    return omega, Jstar, P, K_obs, eta


# ─────────────────────────────────────────────────────────────────────────────
# Biased single-batch estimator: loss and gradient wrt z
# ─────────────────────────────────────────────────────────────────────────────

def psck_mmd2_and_grad(z_model: np.ndarray, z_data: np.ndarray,
                        omega: np.ndarray, Jstar: np.ndarray, eta: float):
    """Biased MMD² estimator and its gradient wrt z_model.

        L = δzᵀ · K · δz       = ω·δz² + η · (J* δz)²         (elementwise)
        dL/dz = 2 K δz          = 2 ω δz + 2 η J*ᵀ (J* δz)

    Runtime O(K_obs + P·K_obs).
    """
    delta = z_model - z_data
    diag_term = omega * delta                    # (K_obs,)
    J_delta = Jstar @ delta                      # (P,)
    loss = float(delta @ diag_term) + eta * float(J_delta @ J_delta)
    grad = 2.0 * (diag_term + eta * (Jstar.T @ J_delta))
    return loss, grad


# ─────────────────────────────────────────────────────────────────────────────
# UNBIASED two-batch U-statistic (eliminates 1/M "self-bias" of biased form)
# ─────────────────────────────────────────────────────────────────────────────

def psck_mmd2_unbiased(z_a: np.ndarray, z_b: np.ndarray, z_data: np.ndarray,
                        omega: np.ndarray, Jstar: np.ndarray,
                        eta: float) -> float:
    """Unbiased estimator using two INDEPENDENT MC forwards z_a, z_b of ⟨Z⟩_θ:

        M̂² = (z_a − z_data)ᵀ K (z_b − z_data)

    Since z_a ⊥ z_b and E[z_a] = E[z_b] = ⟨Z⟩_θ,
        E[M̂²] = (⟨Z⟩_θ − z_data)ᵀ K (⟨Z⟩_θ − z_data) = MMD²_PSCK(p_θ, p_data).

    Verified empirically: at M=4000, 2000 replicas, biased bias is +2.78σ,
    unbiased bias is +0.47σ (consistent with zero). See audit_unbiased.py.
    """
    da = z_a - z_data
    db = z_b - z_data
    diag_contr = float((omega * da) @ db)
    J_da = Jstar @ da
    J_db = Jstar @ db
    off_contr = eta * float(J_da @ J_db)
    return diag_contr + off_contr


def psck_mmd2_unbiased_grad_wrt_za(
        z_a: np.ndarray, z_b: np.ndarray, z_data: np.ndarray,
        omega: np.ndarray, Jstar: np.ndarray, eta: float) -> np.ndarray:
    """Gradient of the unbiased U-stat wrt z_a (holding z_b fixed).

    d/dz_a [(z_a - z_d)ᵀ K (z_b - z_d)] = K (z_b - z_d) = ω (z_b - z_d) + η J*ᵀ J* (z_b - z_d).

    This is what we use in backprop: the per-component gradient of the
    U-statistic sees the OTHER batch's delta, not its own. This is why the
    U-statistic is unbiased — it never squares MC noise against itself.
    """
    db = z_b - z_data
    diag_g = omega * db
    J_db = Jstar @ db
    return diag_g + eta * (Jstar.T @ J_db)


# ─────────────────────────────────────────────────────────────────────────────
# Decomposition (for logging / plots)
# ─────────────────────────────────────────────────────────────────────────────

def psck_diagnostics(z_model: np.ndarray, z_data: np.ndarray,
                      omega: np.ndarray, Jstar: np.ndarray, eta: float):
    """Split the biased MMD² into (heat, ρ-tangent) components."""
    delta = z_model - z_data
    L_heat = float((omega * delta) @ delta)
    J_delta = Jstar @ delta
    L_rho_tangent = eta * float(J_delta @ J_delta)
    return L_heat, L_rho_tangent
