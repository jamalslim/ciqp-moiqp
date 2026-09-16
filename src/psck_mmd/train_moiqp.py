"""
psck_mmd.train_moiqp — MoIQP (mixture-of-IQP) Born machine trainer with
three interchangeable losses:

  "corrmse" : baseline    L = L_z + λ · L_ρ        (MSE of Z moments + Pearson MSE)
  "lw_mmd"  : Liu–Wang    L = δzᵀ diag(ω_heat) δz   (heat-kernel MMD²)
  "psck"    : this work   L = δzᵀ (diag(ω_heat) + η J*ᵀ J*) δz   (PSCK-MMD²)

The mixture of L IQP circuits shares a base graph; each component has its
own angle vector θ_ℓ. Since Z-observables are LINEAR functionals of the
distribution, the mixture ⟨Z_S⟩ equals the mean over components:
    ⟨Z_S⟩_mix = (1/L) Σ_ℓ ⟨Z_S⟩_{θ_ℓ}.
Chain rule gives dL/dθ_ℓ = (1/L) · dL/dz_mix · dz_ℓ/dθ_ℓ.

At deployment, the cIQP / Walsh–Hadamard deferred-measurement construction
compiles the MoIQP into a single IQP on n + ⌈log₂ L⌉ qubits whose marginals
on the n feature qubits reproduce (1/L) Σ_ℓ p_{θ_ℓ} exactly.
"""

from __future__ import annotations
import time
from typing import Literal
import numpy as np

from .correlator import (
    ParityCache, sample_latents, forward_batch, backward_batch, z_data_batch,
)
from .graph import precompute_active_lists
from .corr_jacobian import pearson_value_and_jacobian
from .psck_kernel import (
    build_psck_kernel, psck_mmd2_and_grad, psck_mmd2_unbiased,
    psck_mmd2_unbiased_grad_wrt_za, heat_kernel_coeffs, psck_diagnostics,
)


LossKind = Literal["corrmse", "lw_mmd", "psck"]


# ─────────────────────────────────────────────────────────────────────────────
# Loss interface (returns scalar loss, dL/dz_mix, per-loss diagnostics)
# ─────────────────────────────────────────────────────────────────────────────

def _loss_and_dLdz(kind: LossKind,
                    z: np.ndarray,
                    z_b: np.ndarray | None,
                    z_data: np.ndarray,
                    context: dict):
    """Returns (L_val, dL/dz, diag_dict).

    For "psck" with unbiased U-statistic, z_b is a SECOND independent MC
    forward and the gradient is computed via the U-stat formula
        dL/dz = K (z_b − z_data),
    which is what keeps the estimator unbiased (MC noise of z is never
    squared against itself).
    """
    if kind == "corrmse":
        D = context["D"]; B = context["B"]
        lam_rho = context["lambda_rho"]
        observables = context["observables"]
        iu = context["iu"]
        rho_target = context["rho_target"]
        K = len(z)

        delta = z - z_data
        L_z = float(np.mean(delta ** 2))
        dL_dz_z = (2.0 / K) * delta

        rho, J = pearson_value_and_jacobian(z, observables, D, B)
        rho_flat = rho[iu[0], iu[1]]
        drho = rho_flat - rho_target
        L_rho = float(np.mean(drho ** 2))
        P = J.shape[0]
        dL_dz_rho = (2.0 / P) * (J.T @ drho)

        L_total = L_z + lam_rho * L_rho
        dL_dz = dL_dz_z + lam_rho * dL_dz_rho
        return L_total, dL_dz, {"L_z": L_z, "L_rho": L_rho}

    elif kind == "lw_mmd":
        omega = context["omega"]
        delta = z - z_data
        L = float((omega * delta) @ delta)
        dL_dz = 2.0 * omega * delta
        return L, dL_dz, {"L_heat": L}

    elif kind == "psck":
        omega = context["omega"]
        Jstar = context["Jstar"]
        eta = context["eta"]
        if z_b is None:
            # biased single-batch
            L, dL_dz = psck_mmd2_and_grad(z, z_data, omega, Jstar, eta)
        else:
            # unbiased U-stat
            L = psck_mmd2_unbiased(z, z_b, z_data, omega, Jstar, eta)
            dL_dz = psck_mmd2_unbiased_grad_wrt_za(
                z, z_b, z_data, omega, Jstar, eta
            )
        L_heat, L_rho_t = psck_diagnostics(z, z_data, omega, Jstar, eta)
        return L, dL_dz, {"L_heat": L_heat, "L_rho_tangent": L_rho_t}
    else:
        raise ValueError(f"unknown loss kind: {kind}")


# ─────────────────────────────────────────────────────────────────────────────
# MoIQP trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_moiqp(
    binary: np.ndarray, raw: np.ndarray,
    gates: list[tuple[int, ...]], q2g: dict[int, list[int]],
    D: int, B: int, L: int,
    observables,
    loss_kind: LossKind,
    n_epochs: int = 400,
    mc_batch: int = 4096,
    lr: float = 0.015,
    lambda_rho: float = 40.0,
    eta_psck: float = 5.0,
    psck_taus=(0.1, 0.3, 0.5, 0.7, 0.9),
    heat_scale: float = 1.0,
    seed: int = 42,
    grad_clip: float = 10.0,
    T_restart: int | None = None,
    unbiased: bool = True,
    verbose: bool = True,
):
    """Train an L-component MoIQP Born machine. Adam + cosine restarts.

    Returns:
        best_params : list of L per-component angle vectors
        hist        : dict of training history arrays
    """
    n = D * B
    assert binary.shape[1] == n

    z_data = z_data_batch(binary, observables)
    K = len(observables)
    iu = np.triu_indices(D, k=1)
    corr_raw = np.corrcoef(raw.T)
    rho_target = corr_raw[iu]
    active = precompute_active_lists(observables, gates, q2g)

    # Loss-specific context
    context: dict = {"D": D, "B": B, "lambda_rho": lambda_rho,
                     "observables": observables, "iu": iu,
                     "rho_target": rho_target}
    if loss_kind == "lw_mmd":
        context["omega"] = heat_scale * heat_kernel_coeffs(observables, n, taus=psck_taus)
    elif loss_kind == "psck":
        omega, Jstar, _, _, _ = build_psck_kernel(
            z_data, observables, D, B,
            eta=eta_psck, taus=psck_taus, heat_scale=heat_scale,
        )
        context.update(omega=omega, Jstar=Jstar, eta=eta_psck)

    ng = len(gates)
    rngs = [np.random.default_rng(seed + 1000 * ell) for ell in range(L)]
    params = [rngs[ell].normal(0, 0.1, ng).astype(np.float64) for ell in range(L)]
    adam_m = [np.zeros(ng) for _ in range(L)]
    adam_v = [np.zeros(ng) for _ in range(L)]

    rng_mc = np.random.default_rng(seed + 99991)
    if T_restart is None:
        T_restart = max(50, n_epochs // 5)

    best_loss = np.inf
    best_params = [p.copy() for p in params]
    hist = {"loss": [], "z_err": [], "rho_err": [], "diag": [], "dt": []}
    t_adam = 0
    t_start = time.time()

    for ep in range(n_epochs):
        t0 = time.time(); t_adam += 1
        cyc = ep % T_restart
        lr_t = lr * 0.5 * (1.0 + np.cos(np.pi * cyc / T_restart))
        if ep > 0 and cyc == 0:
            for ell in range(L):
                adam_m[ell][:] = 0.0
                adam_v[ell][:] = 0.0
            t_adam = 1

        # FORWARD batch A (always)
        caches_A = []; args_A = []; z_A = []
        for ell in range(L):
            lat = sample_latents(n, mc_batch, rng_mc, antithetic=True)
            c = ParityCache(lat, gates)
            z_l, a_l = forward_batch(params[ell], c, active)
            z_A.append(z_l); args_A.append(a_l); caches_A.append(c)
        z_mix_A = np.mean(z_A, axis=0)

        # FORWARD batch B (only for unbiased PSCK)
        z_mix_B = None
        if loss_kind == "psck" and unbiased:
            z_B = []
            for ell in range(L):
                lat = sample_latents(n, mc_batch, rng_mc, antithetic=True)
                z_l, _ = forward_batch(
                    params[ell], ParityCache(lat, gates), active,
                )
                z_B.append(z_l)
            z_mix_B = np.mean(z_B, axis=0)

        L_val, dL_dzmix, diag = _loss_and_dLdz(
            loss_kind, z_mix_A, z_mix_B, z_data, context,
        )

        # Backprop into each component: dL/dθ_ℓ = (1/L) · dL/dz_mix · dz_ℓ/dθ_ℓ
        dL_dz_l = dL_dzmix / L
        grads = []
        for ell in range(L):
            g = backward_batch(
                params[ell], caches_A[ell], active, args_A[ell], dL_dz_l,
            )
            gn = float(np.linalg.norm(g))
            if gn > grad_clip:
                g *= (grad_clip / gn)
            grads.append(g)

        # Adam per component
        for ell in range(L):
            adam_m[ell] = 0.9 * adam_m[ell] + 0.1 * grads[ell]
            adam_v[ell] = 0.999 * adam_v[ell] + 0.001 * grads[ell] ** 2
            mh = adam_m[ell] / (1 - 0.9 ** t_adam)
            vh = adam_v[ell] / (1 - 0.999 ** t_adam)
            params[ell] -= lr_t * mh / (np.sqrt(vh) + 1e-8)

        # Logging
        z_err = float(np.mean(np.abs(z_mix_A - z_data)))
        rho_A, _ = pearson_value_and_jacobian(z_mix_A, observables, D, B)
        rho_flat_A = rho_A[iu[0], iu[1]]
        rho_err = float(np.mean(np.abs(rho_flat_A - rho_target)))
        dt = time.time() - t0
        hist["loss"].append(float(L_val))
        hist["z_err"].append(z_err)
        hist["rho_err"].append(rho_err)
        hist["diag"].append(diag)
        hist["dt"].append(dt)
        if L_val < best_loss:
            best_loss = float(L_val)
            best_params = [p.copy() for p in params]

        if verbose and (ep % 25 == 0 or ep == n_epochs - 1):
            elapsed = time.time() - t_start
            eta_s = elapsed / (ep + 1) * (n_epochs - ep - 1)
            diag_str = " ".join(f"{k}={v:.4g}" for k, v in diag.items())
            print(
                f"  ep {ep:4d}/{n_epochs}  L={L_val:.5e}  |Δz|={z_err:.4f}  "
                f"|Δρ|={rho_err:.4f}  {diag_str}  "
                f"[{elapsed:.0f}s<{eta_s:.0f}s]"
            )

    return best_params, hist


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_mixture(
    params_list: list[np.ndarray],
    gates, q2g, D: int, B: int,
    observables, binary, raw,
    M_eval: int = 50_000, seed: int = 77777,
):
    """MC-evaluate a trained MoIQP at a given budget M_eval."""
    n = D * B
    active = precompute_active_lists(observables, gates, q2g)
    z_data = z_data_batch(binary, observables)
    iu = np.triu_indices(D, k=1)
    corr_raw = np.corrcoef(raw.T)
    rho_target = corr_raw[iu]

    rng = np.random.default_rng(seed)
    L = len(params_list)
    z_mix = np.zeros(len(observables))
    for ell in range(L):
        lat = sample_latents(n, M_eval, rng, antithetic=True)
        c = ParityCache(lat, gates)
        z_l, _ = forward_batch(params_list[ell], c, active)
        z_mix += z_l / L

    rho_model, _ = pearson_value_and_jacobian(z_mix, observables, D, B)
    rho_flat = rho_model[iu[0], iu[1]]
    rho_mae = float(np.mean(np.abs(rho_flat - rho_target)))
    rho_r = float(np.corrcoef(rho_flat, rho_target)[0, 1])
    z_mae = float(np.mean(np.abs(z_mix - z_data)))

    # Encoding-fidelity floor: rho recomputed from exact data z directly
    rho_data, _ = pearson_value_and_jacobian(z_data, observables, D, B)
    enc_fid = float(np.mean(np.abs(rho_data[iu[0], iu[1]] - rho_target)))

    return {
        "z_mae": z_mae, "rho_mae": rho_mae, "rho_r": rho_r,
        "enc_fid": enc_fid, "rho_model": rho_model,
        "rho_data_from_z": rho_data, "z_mix": z_mix,
    }
