"""psck_mmd test suite.

Run from the package root:
    PYTHONPATH=src python tests/test_core.py

Six audits:
    1. pearson_value_and_jacobian vs FD (absolute + relative)
    2. K-matrix PSD-ness
    3. MC forward self-consistency
    4. MC backward vs FD (fixed latents)
    5. End-to-end PSCK gradient vs FD
    6. Unbiased U-statistic vs biased estimator
    7. cIQP deferred-measurement marginal exactness
"""

from __future__ import annotations
import sys
import os
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "src"),
)

from psck_mmd import (
    build_complete_graph, build_er_graph, precompute_active_lists,
    sample_latents, ParityCache, forward_batch, backward_batch,
    z_data_batch, enumerate_feature_observables, load_calorimeter,
    pearson_value_and_jacobian, build_psck_kernel,
    psck_mmd2_and_grad, psck_mmd2_unbiased, heat_kernel_coeffs,
    build_ciqp_circuit,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pearson Jacobian
# ─────────────────────────────────────────────────────────────────────────────

def test_jacobian_fd():
    D, B = 8, 2
    binary, _ = load_calorimeter("data/cal_shower_img_8q.npy",
                                    bits=B, encoding="binary")
    obs, _ = enumerate_feature_observables(D, B, max_weight=2)
    z_data = z_data_batch(binary, obs)
    rho, J = pearson_value_and_jacobian(z_data, obs, D, B)

    K = len(obs); P = D*(D-1)//2
    iu0, iu1 = np.triu_indices(D, k=1)
    eps = 1e-5
    max_abs = 0.0
    max_rel = 0.0
    for jcol in range(K):
        zp = z_data.copy(); zp[jcol] += eps
        zm = z_data.copy(); zm[jcol] -= eps
        rp, _ = pearson_value_and_jacobian(zp, obs, D, B)
        rm, _ = pearson_value_and_jacobian(zm, obs, D, B)
        fd = (rp[iu0, iu1] - rm[iu0, iu1]) / (2*eps)
        an = J[:, jcol]
        err = np.abs(fd - an)
        max_abs = max(max_abs, float(err.max()))
        mag = np.maximum(np.abs(fd), np.abs(an))
        big = mag > 1e-8
        if big.any():
            max_rel = max(max_rel, float((err[big]/mag[big]).max()))
    ok = (max_abs < 1e-7) and (max_rel < 1e-4)
    print(f"  [1] Jacobian FD:   abs={max_abs:.3e}  rel={max_rel:.3e}  "
          f"{'✓' if ok else '✗'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 2. K-matrix PSD
# ─────────────────────────────────────────────────────────────────────────────

def test_kernel_psd():
    D, B = 8, 2
    binary, _ = load_calorimeter("data/cal_shower_img_8q.npy",
                                    bits=B, encoding="binary")
    obs, _ = enumerate_feature_observables(D, B, max_weight=2)
    z_data = z_data_batch(binary, obs)
    omega, Jstar, _, _, _ = build_psck_kernel(z_data, obs, D, B,
                                                 eta=5.0, heat_scale=1.0)
    Kmat = np.diag(omega) + 5.0 * (Jstar.T @ Jstar)
    eig = np.linalg.eigvalsh(Kmat)
    ok = eig.min() > -1e-10
    cond = eig.max() / max(eig.min(), 1e-20)
    print(f"  [2] K PSD:         eig_min={eig.min():.3e}  cond={cond:.2f}  "
          f"{'✓' if ok else '✗'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 3. MC forward self-consistency
# ─────────────────────────────────────────────────────────────────────────────

def test_mc_forward_consistency():
    n = 24
    gates, q2g = build_er_graph(n, avg_deg=5.0, seed=1)
    rng_p = np.random.default_rng(7)
    params = rng_p.normal(0, 0.3, len(gates))
    obs, _ = enumerate_feature_observables(D=4, B=6, max_weight=2)
    active = precompute_active_lists(obs, gates, q2g)
    M = 40_000
    lat_a = sample_latents(n, M, np.random.default_rng(111), antithetic=True)
    lat_b = sample_latents(n, M, np.random.default_rng(222), antithetic=True)
    z_a, _ = forward_batch(params, ParityCache(lat_a, gates), active)
    z_b, _ = forward_batch(params, ParityCache(lat_b, gates), active)
    gap = np.abs(z_a - z_b)
    expected = 2.0 / np.sqrt(M)
    frac = float(np.mean(gap < 5 * expected))
    ok = frac > 0.95
    print(f"  [3] MC forward:    mean|Δ|={gap.mean():.3e} vs 1/√M={1/np.sqrt(M):.3e}"
          f"   {frac*100:.0f}% within 5σ  {'✓' if ok else '✗'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 4. MC backward vs FD
# ─────────────────────────────────────────────────────────────────────────────

def test_mc_backward_fd():
    n = 16
    gates, q2g = build_er_graph(n, avg_deg=6.0, seed=13)
    ng = len(gates)
    rng_p = np.random.default_rng(7)
    params = rng_p.normal(0, 0.3, ng)
    obs, _ = enumerate_feature_observables(D=8, B=2, max_weight=2)
    active = precompute_active_lists(obs, gates, q2g)

    lat = sample_latents(n, 8000, np.random.default_rng(31), antithetic=True)
    K = len(obs)
    dL_dz = np.random.default_rng(41).normal(0, 1.0, K)

    cache = ParityCache(lat, gates)
    z0, args0 = forward_batch(params, cache, active)
    grad_an = backward_batch(params, cache, active, args0, dL_dz)

    idxs = np.random.default_rng(9).choice(ng, size=30, replace=False)
    eps = 1e-4
    max_rel = 0.0
    for j in idxs:
        pp = params.copy(); pp[j] += eps
        pm = params.copy(); pm[j] -= eps
        zp, _ = forward_batch(pp, ParityCache(lat, gates), active)
        zm, _ = forward_batch(pm, ParityCache(lat, gates), active)
        fd = float((dL_dz @ (zp - zm)) / (2*eps))
        an = float(grad_an[j])
        rel = abs(fd - an) / max(abs(fd), abs(an), 1e-10)
        max_rel = max(max_rel, rel)
    ok = max_rel < 1e-2
    print(f"  [4] MC backward:   max rel FD err={max_rel:.3e}   {'✓' if ok else '✗'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 5. End-to-end PSCK gradient vs FD
# ─────────────────────────────────────────────────────────────────────────────

def test_psck_gradient_fd():
    D, B = 8, 2
    n = D * B
    binary, _ = load_calorimeter("data/cal_shower_img_8q.npy",
                                    bits=B, encoding="binary")
    obs, _ = enumerate_feature_observables(D, B, max_weight=2)
    K = len(obs)
    gates, q2g = build_er_graph(n, avg_deg=6.0, seed=3)
    active = precompute_active_lists(obs, gates, q2g)
    ng = len(gates)

    z_data = z_data_batch(binary, obs)
    omega, Jstar, _, _, eta = build_psck_kernel(z_data, obs, D, B,
                                                   eta=1.0, heat_scale=1.0)

    rng_p = np.random.default_rng(5)
    params = rng_p.normal(0, 0.25, ng)
    lat = sample_latents(n, 8000, np.random.default_rng(101), antithetic=True)

    def loss_of(p):
        c = ParityCache(lat, gates)
        z, args = forward_batch(p, c, active)
        L, dL_dz = psck_mmd2_and_grad(z, z_data, omega, Jstar, eta)
        return L, c, args, dL_dz

    L0, c0, args0, dL_dz = loss_of(params)
    grad_an = backward_batch(params, c0, active, args0, dL_dz)

    idxs = np.random.default_rng(9).choice(ng, size=25, replace=False)
    eps = 1e-4
    max_rel = 0.0
    for j in idxs:
        pp = params.copy(); pp[j] += eps
        pm = params.copy(); pm[j] -= eps
        Lp, _, _, _ = loss_of(pp)
        Lm, _, _, _ = loss_of(pm)
        fd = (Lp - Lm) / (2*eps)
        an = grad_an[j]
        max_rel = max(max_rel,
                       abs(fd - an) / max(abs(fd), abs(an), 1e-10))
    ok = max_rel < 2e-2
    print(f"  [5] PSCK grad FD:  max rel err={max_rel:.3e}   {'✓' if ok else '✗'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 6. Unbiased U-statistic
# ─────────────────────────────────────────────────────────────────────────────

def test_unbiased_u_stat():
    D, B = 8, 2
    n = D * B
    binary, _ = load_calorimeter("data/cal_shower_img_8q.npy",
                                    bits=B, encoding="binary")
    obs, _ = enumerate_feature_observables(D, B, max_weight=2)
    gates, q2g = build_er_graph(n, avg_deg=6.0, seed=3)
    active = precompute_active_lists(obs, gates, q2g)

    z_data = z_data_batch(binary, obs)
    omega, Jstar, _, _, eta = build_psck_kernel(z_data, obs, D, B,
                                                   eta=1.0, heat_scale=1.0)
    rng_p = np.random.default_rng(5)
    params = rng_p.normal(0, 0.25, len(gates))

    # High-M reference
    lat0 = sample_latents(n, 1_500_000, np.random.default_rng(999),
                            antithetic=True)
    z_inf, _ = forward_batch(params, ParityCache(lat0, gates), active)
    L_true, _ = psck_mmd2_and_grad(z_inf, z_data, omega, Jstar, eta)

    M = 4000
    n_rep = 500
    biased = np.empty(n_rep)
    unbiased = np.empty(n_rep)
    rng = np.random.default_rng(7)
    for r in range(n_rep):
        la = sample_latents(n, M, rng, antithetic=True)
        lb = sample_latents(n, M, rng, antithetic=True)
        z_a, _ = forward_batch(params, ParityCache(la, gates), active)
        z_b, _ = forward_batch(params, ParityCache(lb, gates), active)
        Lb, _ = psck_mmd2_and_grad(z_a, z_data, omega, Jstar, eta)
        Lu = psck_mmd2_unbiased(z_a, z_b, z_data, omega, Jstar, eta)
        biased[r] = Lb
        unbiased[r] = Lu
    bias_b = biased.mean() - L_true
    bias_u = unbiased.mean() - L_true
    se_u = unbiased.std(ddof=1) / np.sqrt(n_rep)
    # unbiased should be within 3σ of zero, biased should be positive
    ok = (abs(bias_u) < 3 * se_u) and (bias_b > 0)
    print(f"  [6] Unbiased:      bias_u={bias_u:+.3e}(±{se_u:.3e})  "
          f"bias_b={bias_b:+.3e}   {'✓' if ok else '✗'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 7. cIQP deployment
# ─────────────────────────────────────────────────────────────────────────────

def test_ciqp_deployment():
    D, B, L = 8, 2, 4
    n = D * B
    gates, q2g = build_er_graph(n, avg_deg=6.0, seed=3)
    ng = len(gates)
    rng_p = np.random.default_rng(42)
    params_list = [rng_p.normal(0, 0.3, ng) for _ in range(L)]

    obs, _ = enumerate_feature_observables(D, B, max_weight=2)
    active_feat = precompute_active_lists(obs, gates, q2g)
    M = 100_000
    rng = np.random.default_rng(98765)
    z_mo = np.zeros(len(obs))
    for ell in range(L):
        lat = sample_latents(n, M, rng, antithetic=True)
        z_l, _ = forward_batch(params_list[ell],
                                ParityCache(lat, gates), active_feat)
        z_mo += z_l / L

    deploy_gates, deploy_params, info = build_ciqp_circuit(params_list, gates, n)
    n_total = info["n_total"]
    dep_q2g: dict[int, list[int]] = {q: [] for q in range(n_total)}
    for j, g in enumerate(deploy_gates):
        for q in g:
            dep_q2g[q].append(j)
    active_dep = precompute_active_lists(obs, deploy_gates, dep_q2g)
    lat_d = sample_latents(n_total, M, np.random.default_rng(98765),
                             antithetic=True)
    z_dep, _ = forward_batch(deploy_params, ParityCache(lat_d, deploy_gates),
                              active_dep)
    diff = z_dep - z_mo
    mae = float(np.mean(np.abs(diff)))
    mc = 1.0 / np.sqrt(M)
    ok = (mae / mc) < 5.0
    print(f"  [7] cIQP vs MoIQP: MAE={mae:.4e}  MAE/MC={mae/mc:.2f}×   "
          f"{'✓' if ok else '✗'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tests() -> bool:
    print("\n  psck_mmd — Test suite\n")
    results = [
        test_jacobian_fd(),
        test_kernel_psd(),
        test_mc_forward_consistency(),
        test_mc_backward_fd(),
        test_psck_gradient_fd(),
        test_unbiased_u_stat(),
        test_ciqp_deployment(),
    ]
    n_pass = sum(results)
    n_tot = len(results)
    status = "✓" if n_pass == n_tot else "✗"
    print(f"\n  {status}  {n_pass}/{n_tot} audits passed.\n")
    return n_pass == n_tot


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
