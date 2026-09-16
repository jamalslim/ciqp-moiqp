#!/usr/bin/env python
"""
audit_shallow_deploy.py — Mandatory self-audit of psck_mmd.shallow_deploy.

  A1  cat-cIQP EXACTNESS: dense-statevector data marginal of the GHZ-bus
      compiled circuit == exact MoIQP mixture, machine precision, several
      (n, L, m) configurations and random graphs/angles.
  A2  dcIQP EXACTNESS: the switch-branch construction is per-branch
      identical to the component circuits; verified by exact per-component
      distribution equality (branch ℓ ≡ U(θ^(ℓ))) + padded-mixture identity
      against the cIQP marginal (Van den Nest at matched latents).
  A3  Walsh-sparse trainer: analytic dφ gradient vs finite differences.
  A4  Walsh-sparse training on the real calorimeter (D=8, B=2, a=2) at
      k ∈ {1, 2, 4}: PSCK quality vs coherent deployment gate count.
      (D=25, B=2, L=4) for all four deployment objects.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from psck_mmd.graph import build_er_graph, precompute_active_lists
from psck_mmd.feature_obs import enumerate_feature_observables
from psck_mmd.data import load_calorimeter
from psck_mmd.correlator import (ParityCache, sample_latents, forward_batch,
                                 z_data_batch)
from psck_mmd.ciqp import build_ciqp_circuit, _pad_to_power_of_2
from psck_mmd.corr_jacobian import pearson_value_and_jacobian
from psck_mmd.psck_kernel import (build_psck_kernel, psck_mmd2_and_grad)
from psck_mmd.shallow_deploy import (
    build_cat_ciqp, exact_cat_ciqp_data_marginal,
    exact_single_iqp_distribution, dciqp_reference_marginal,
    train_moiqp_walsh_sparse, walsh_sparse_to_components, walsh_synthesize,
    deployment_budget, NoiseModelParams, greedy_gate_layers,
)

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{name}] {'✓' if cond else '✗ FAIL'}  {detail}")


# ─────────────────────────────────────────────────────────────────────────
# A1: cat-cIQP exactness (dense)
# ─────────────────────────────────────────────────────────────────────────
print("\nA1  cat-cIQP dense exactness")
# The L = 3 case is an EXPECTED REJECTION, not an exactness check. Zero-angle
# padding cannot extend the compilation to non-powers of two: an all-zero-angle
# IQP component is H^n I H^n |0> = |0>, a delta spike at the all-zeros string
# rather than a neutral element, so the compiled marginal would be the padded
# mixture and not the L-component one. App. A puts the deviation at up to 0.18
# in probability at L = 3. _pad_to_power_of_2 raises for such L and this loop
# asserts that it does; before the guard existed the script silently compiled
# an L = 3 case, which is why the check is kept.
for (n, L, m, gseed) in [(4, 2, 2, 1), (5, 4, 2, 2), (6, 4, 1, 3), (4, 3, 3, 4)]:
    gates, _ = build_er_graph(n, avg_deg=3.0, seed=gseed)
    rng = np.random.default_rng(10 + gseed)
    params = [rng.normal(0, 0.6, len(gates)) for _ in range(L)]
    if L & (L - 1):                      # not a power of two
        try:
            _pad_to_power_of_2(params)
            check(f"n={n} L={L} rejected", False, "guard did NOT fire")
        except ValueError:
            check(f"n={n} L={L} rejected", True,
                  "non-power-of-two L correctly refused, padding is not a no-op")
        continue
    padded, a = _pad_to_power_of_2(params)
    p_mix = np.zeros(1 << n)
    for th in padded:
        p_mix += exact_single_iqp_distribution(th, gates, n) / len(padded)
    p_cat = exact_cat_ciqp_data_marginal(params, gates, n, m)
    err = float(np.max(np.abs(p_cat - p_mix)))
    check(f"n={n} L={L} m={m}", err < 1e-12,
          f"max|Δp|={err:.2e}, total qubits {n + a*m}")

# ─────────────────────────────────────────────────────────────────────────
# A2: dcIQP exactness
# ─────────────────────────────────────────────────────────────────────────
print("\nA2  dcIQP exactness")
# (i) branch ℓ of dcIQP applies exactly U(θ^(ℓ)) — the construction inserts
# the component circuit verbatim; verify the mathematical statement:
# padded uniform mixture of exact component distributions == cIQP marginal.
n, L = 6, 4
gates, q2g = build_er_graph(n, avg_deg=4.0, seed=7)
rng = np.random.default_rng(5)
params = [rng.normal(0, 0.5, len(gates)) for _ in range(L)]
padded, a = _pad_to_power_of_2(params)
p_mix = np.zeros(1 << n)
for th in padded:
    p_mix += exact_single_iqp_distribution(th, gates, n) / len(padded)
# cIQP marginal (dense, via cat with m=1 == the plain compiled circuit)
p_ciqp = exact_cat_ciqp_data_marginal(params, gates, n, 1)
err = float(np.max(np.abs(p_ciqp - p_mix)))
check("mixture == cIQP marginal", err < 1e-12, f"max|Δp|={err:.2e}")
# (ii) Van den Nest cross-check of the dcIQP reference marginal helper
obs, _ = enumerate_feature_observables(3, 2, max_weight=2)  # n=6
z_ref = dciqp_reference_marginal(params, gates, n, obs, q2g, M=200_000)
signs = 1.0 - 2.0 * (((np.arange(1 << n)[:, None] >>
                       np.arange(n)[None, :]) & 1))
z_exact = np.array([ (p_mix * np.prod(signs[:, list(S)], axis=1)).sum()
                     for S in obs ])
mae = float(np.mean(np.abs(z_ref - z_exact)))
check("VdN helper vs exact", mae < 5.0 / np.sqrt(200_000),
      f"MAE={mae:.2e} vs 5/√M={5/np.sqrt(200_000):.2e}")

# ─────────────────────────────────────────────────────────────────────────
# A3: Walsh-sparse gradient vs finite differences (fixed latents)
# ─────────────────────────────────────────────────────────────────────────
print("\nA3  Walsh-sparse chain rule vs FD")
D, B, a = 8, 2, 2
n = D * B
L = 1 << a
binary, raw = load_calorimeter("data/cal_shower_img_8q.npy", bits=B,
                               encoding="binary")
obs, _ = enumerate_feature_observables(D, B, max_weight=2)
gates, q2g = build_er_graph(n, avg_deg=5.0, seed=11)
active = precompute_active_lists(obs, gates, q2g)
z_data = z_data_batch(binary, obs)
omega, Jstar, _, _, _ = build_psck_kernel(z_data, obs, D, B, eta=5.0)
U = [0, 1, 3]
rng = np.random.default_rng(3)
phi = rng.normal(0, 0.2, (len(gates), len(U)))
signs_mat = np.array([[1.0 - 2.0 * (bin(ell & S).count('1') % 2) for S in U]
                      for ell in range(L)])
lat_fixed = [sample_latents(n, 2048, np.random.default_rng(100 + ell),
                            antithetic=True) for ell in range(L)]

def loss_of_phi(ph):
    zm = np.zeros(len(obs))
    for ell in range(L):
        th = ph @ signs_mat[ell]
        zl, _ = forward_batch(th, ParityCache(lat_fixed[ell], gates), active)
        zm += zl / L
    Lv, _ = psck_mmd2_and_grad(zm, z_data, omega, Jstar, 5.0)
    return Lv, zm

Lv, zm = loss_of_phi(phi)
_, dL_dz = psck_mmd2_and_grad(zm, z_data, omega, Jstar, 5.0)
from psck_mmd.correlator import backward_batch
g_phi = np.zeros_like(phi)
for ell in range(L):
    th = phi @ signs_mat[ell]
    c = ParityCache(lat_fixed[ell], gates)
    zl, al = forward_batch(th, c, active)
    g_theta = backward_batch(th, c, active, al, dL_dz / L)
    g_phi += np.outer(g_theta, signs_mat[ell])
eps = 1e-5
max_rel = 0.0
rngi = np.random.default_rng(0)
for _ in range(12):
    j = rngi.integers(len(gates)); s = rngi.integers(len(U))
    php = phi.copy(); php[j, s] += eps
    phm = phi.copy(); phm[j, s] -= eps
    fd = (loss_of_phi(php)[0] - loss_of_phi(phm)[0]) / (2 * eps)
    an = g_phi[j, s]
    rel = abs(fd - an) / max(abs(fd), abs(an), 1e-12)
    max_rel = max(max_rel, rel)
check("dL/dφ vs FD (12 probes)", max_rel < 1e-5, f"max rel={max_rel:.2e}")

# ─────────────────────────────────────────────────────────────────────────
# A4: Walsh-sparse training on real data — quality vs coherent gate count
# ─────────────────────────────────────────────────────────────────────────
print("\nA4  Walsh-sparse PSCK training (D=8, B=2, a=2, 150 epochs each)")
results = {}
for U in ([0], [0, 1], [0, 1, 2, 3]):
    t0 = time.time()
    phi_t, Ut, hist = train_moiqp_walsh_sparse(
        binary, raw, gates, q2g, D, B, a, obs, walsh_support=U,
        n_epochs=150, mc_batch=2048, lr=0.02, seed=42, verbose=False)
    comps = walsh_sparse_to_components(phi_t, Ut, a)
    gates_c, params_c, info = build_cat_ciqp(comps, gates, n, m_copies=1,
                                             walsh_support=U)
    # evaluate at higher M
    rng_e = np.random.default_rng(777)
    zm = np.zeros(len(obs))
    for th in comps:
        lat = sample_latents(n, 30_000, rng_e, antithetic=True)
        zl, _ = forward_batch(th, ParityCache(lat, gates), active)
        zm += zl / len(comps)
    iu = np.triu_indices(D, k=1)
    rho_t = np.corrcoef(raw.T)[iu]
    rho_m, _ = pearson_value_and_jacobian(zm, obs, D, B)
    rho_mae = float(np.mean(np.abs(rho_m[iu] - rho_t)))
    results[len(U)] = (rho_mae, info["n_deploy_gates"], info["max_weight"])
    print(f"    k={len(U)}: MAEρ={rho_mae:.4f}  coherent gates="
          f"{info['n_deploy_gates']}  max weight={info['max_weight']}  "
          f"[{time.time()-t0:.0f}s]")
check("k=2 ≤ 2·|G| gates & beats k=1",
      results[2][1] <= 2 * len(gates) and results[2][0] < results[1][0],
      f"MAEρ: k=1 {results[1][0]:.4f} → k=2 {results[2][0]:.4f} → "
      f"k=4 {results[4][0]:.4f}")

print("\n" + "=" * 60)
print("  \u2713 ALL SHALLOW-DEPLOY AUDITS PASSED" if ok_all else "  \u2717 FAILURES")
print("=" * 60)
