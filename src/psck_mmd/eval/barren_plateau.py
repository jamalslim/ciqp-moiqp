"""
psck_mmd.eval.barren_plateau — Gradient-variance scaling diagnostic.

Protocol (McClean et al. 2018, Nat. Commun. 9:4812; Cerezo et al. 2021):
At each qubit count n, draw K_init random parameter vectors θ ~ N(0, σ²I)
from the same distribution used at training initialisation, evaluate the
training-loss gradient ∂L/∂θ at each, and report the variance of a
representative gradient component (or the mean ‖∇L‖²) across the K_init
draws. Plotting Var[∂L/∂θ_j] vs n on a log scale tells you whether the
optimization landscape becomes exponentially flat as n grows (barren
plateau) or stays trainable.

For our setting:
  - n = D · B   ranges across the scaling sweep {16, 24, 32, 48, 64}
  - L = 1 (single component) for the diagnostic — no need for the mixture,
    we are characterizing the loss-landscape geometry, not the model
  - Gradient is via the existing Van den Nest backward pass
  - Both PSCK and LW-MMD losses are exercised, on the SAME random graphs
    and SAME random θ draws (paired comparison)

Outputs are intentionally machine-readable for a band/scaling plot.
This module does not perform any TRAINING — it only measures gradients
at random initialisations.
"""

from __future__ import annotations
import time
from typing import Literal
import numpy as np

# All imports use the package's existing modules. This file lives at
# src/psck_mmd/eval/barren_plateau.py once dropped into the repo.
from psck_mmd.correlator import (
    ParityCache, sample_latents, forward_batch, backward_batch, z_data_batch,
)
from psck_mmd.graph import build_er_graph, precompute_active_lists
from psck_mmd.feature_obs import enumerate_feature_observables
from psck_mmd.psck_kernel import (
    build_psck_kernel, psck_mmd2_and_grad, heat_kernel_coeffs,
)
from psck_mmd.data_split import load_calorimeter_split


LossKind = Literal["psck", "lw_mmd"]


# ─────────────────────────────────────────────────────────────────────────────
# Single-init gradient evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _loss_grad_at_init(
    params: np.ndarray,
    binary: np.ndarray,
    gates, q2g,
    observables, active_lists,
    D: int, B: int,
    loss_kind: LossKind,
    z_data: np.ndarray,
    omega: np.ndarray,
    Jstar: np.ndarray | None,
    eta: float,
    M: int,
    rng_mc: np.random.Generator,
):
    """Evaluate (loss value, dL/dθ) at a single param init using one MC batch.

    Reused machinery: forward_batch + backward_batch + the chosen loss's
    closed-form (loss, dL/dz). Returns (L, grad) where grad has shape (n_gates,).
    """
    n = D * B
    lat = sample_latents(n, M, rng_mc, antithetic=True)
    cache = ParityCache(lat, gates)
    z, args = forward_batch(params, cache, active_lists)

    delta = z - z_data
    if loss_kind == "lw_mmd":
        L_val = float((omega * delta) @ delta)
        dL_dz = 2.0 * omega * delta
    elif loss_kind == "psck":
        L_val, dL_dz = psck_mmd2_and_grad(z, z_data, omega, Jstar, eta)
    else:
        raise ValueError(f"unknown loss_kind {loss_kind!r}")
    grad = backward_batch(params, cache, active_lists, args, dL_dz)
    return float(L_val), grad


# ─────────────────────────────────────────────────────────────────────────────
# Per-n diagnostic: many random inits, paired across losses
# ─────────────────────────────────────────────────────────────────────────────

def bp_diagnostic_at_n(
    bits: int,
    K_init: int = 200,
    M_grad: int = 2048,
    sigma_init: float = 0.1,
    avg_deg: float = 6.0,
    encoding: str = "binary",
    eta_psck: float = 5.0,
    seed_graph: int = 3,
    seed_inits: int = 12345,
    losses: tuple[LossKind, ...] = ("psck", "lw_mmd"),
    verbose: bool = True,
) -> dict:
    """Sample K_init random θ ~ N(0, σ²I), evaluate ‖∇L‖² and per-component
    gradient variance for each loss in `losses`. Same graph and same θ
    samples are used across all losses for paired comparison.

    Returns a dict with per-loss arrays of shape (K_init,) for ‖∇L‖² and
    per-component variances Var_θ[∂L/∂θ_j], plus aggregate scalars.
    """
    D = 8
    n_feat = D * bits
    binary, raw, _, _ = load_calorimeter_split(
        data_path="data/cal_shower_img_8q.npy",
        bits=bits, encoding=encoding,
    )
    obs, _ = enumerate_feature_observables(D, bits, max_weight=2)
    gates, q2g = build_er_graph(n_feat, avg_deg=avg_deg, seed=seed_graph)
    active = precompute_active_lists(obs, gates, q2g)
    n_gates = len(gates)
    K_obs = len(obs)

    z_data = z_data_batch(binary, obs)
    omega = heat_kernel_coeffs(obs, n_feat)
    omega_psck, Jstar, _, _, eta_used = build_psck_kernel(
        z_data, obs, D, bits, eta=eta_psck, heat_scale=1.0,
    )
    # Use the same omega across both losses for fair comparison.
    # For PSCK we also pass Jstar; for lw_mmd we ignore Jstar.

    rng_init = np.random.default_rng(seed_inits)
    rng_mc   = np.random.default_rng(seed_inits + 99991)

    # (K_init, n_gates) parameter draws
    Theta = rng_init.normal(0.0, sigma_init, size=(K_init, n_gates))

    out: dict = {
        "n_feat":      n_feat,
        "bits":        bits,
        "K_init":      K_init,
        "M_grad":      M_grad,
        "sigma_init":  sigma_init,
        "n_gates":     n_gates,
        "K_obs":       K_obs,
        "avg_deg":     avg_deg,
        "encoding":    encoding,
        "seed_graph":  seed_graph,
        "seed_inits":  seed_inits,
        "losses":      list(losses),
    }

    for loss_kind in losses:
        if verbose:
            print(f"  [{loss_kind}]  n={n_feat}, K_init={K_init}, M_grad={M_grad}")
        t0 = time.time()
        grad_norm2 = np.empty(K_init, dtype=np.float64)
        # We collect per-gate gradients for variance-per-component.
        # (K_init, n_gates) is fine to hold for K_init=200, n_gates ≲ 250 → 50K floats.
        all_grads = np.empty((K_init, n_gates), dtype=np.float64)
        loss_vals = np.empty(K_init, dtype=np.float64)
        for k in range(K_init):
            L_val, g = _loss_grad_at_init(
                Theta[k], binary, gates, q2g, obs, active, D, bits,
                loss_kind, z_data,
                omega_psck if loss_kind == "psck" else omega,
                Jstar if loss_kind == "psck" else None,
                eta_used,
                M=M_grad, rng_mc=rng_mc,
            )
            loss_vals[k] = L_val
            all_grads[k] = g
            grad_norm2[k] = float((g * g).sum())
        per_gate_var = all_grads.var(axis=0, ddof=1)   # (n_gates,)
        gn2_mean = float(grad_norm2.mean())
        gn2_var  = float(grad_norm2.var(ddof=1))
        per_var_mean = float(per_gate_var.mean())
        per_var_med  = float(np.median(per_gate_var))
        if verbose:
            print(f"            ⟨‖∇L‖²⟩      = {gn2_mean:.4e}")
            print(f"            mean Var/gate = {per_var_mean:.4e}")
            print(f"            median Var/gate = {per_var_med:.4e}")
            print(f"            elapsed: {time.time()-t0:.1f}s")
        out[loss_kind] = {
            "loss_values":      loss_vals.tolist(),
            "grad_norm2":       grad_norm2.tolist(),
            "grad_norm2_mean":  gn2_mean,
            "grad_norm2_var":   gn2_var,
            "per_gate_var":     per_gate_var.tolist(),
            "per_gate_var_mean":   per_var_mean,
            "per_gate_var_median": per_var_med,
            "elapsed_s":        float(time.time() - t0),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Scaling sweep across qubit counts
# ─────────────────────────────────────────────────────────────────────────────

def bp_scaling_sweep(
    bits_list: list[int] = (2, 3, 4, 6, 8),
    K_init: int = 200,
    M_grad: int = 2048,
    sigma_init: float = 0.1,
    losses: tuple[LossKind, ...] = ("psck", "lw_mmd"),
    encoding: str = "binary",
    seed_graph: int = 3,
    seed_inits: int = 12345,
    verbose: bool = True,
) -> dict:
    """Run bp_diagnostic_at_n for each B ∈ bits_list and aggregate."""
    results = []
    for B in bits_list:
        if verbose:
            print(f"\n{'='*72}")
            print(f"  BP diagnostic at B={B}, n={8*B}q")
            print(f"{'='*72}")
        r = bp_diagnostic_at_n(
            bits=B, K_init=K_init, M_grad=M_grad, sigma_init=sigma_init,
            losses=losses, encoding=encoding,
            seed_graph=seed_graph, seed_inits=seed_inits, verbose=verbose,
        )
        results.append(r)
    return {
        "bits_list":   list(bits_list),
        "qubits_list": [8 * B for B in bits_list],
        "K_init":      K_init,
        "M_grad":      M_grad,
        "sigma_init":  sigma_init,
        "losses":      list(losses),
        "by_n":        results,
    }
