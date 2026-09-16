#!/usr/bin/env python
"""
audit_iqp_membership.py — Is the MoIQP mixture an n-qubit IQP distribution?

The user's claim: "the mixture of the IQP is not IQP, only cIQP is the IQP."
This audit makes the claim PRECISE and tests it at three resource levels,
with positive controls throughout, then certifies the one object that IS a
bona fide IQP circuit at NISQ cost.

  N1  DIMENSION: rank of the distribution-space Jacobian. Single n-qubit
      IQP with ALL weight-≤2 gates (complete graph) spans a manifold of
      dimension ≤ |G| = n + C(n,2). The MoIQP family Jacobian rank at a
      generic point exceeds that bound → the mixture family leaves EVERY
      weight-≤2 single-IQP manifold on n qubits. (Necessary-condition test:
      generic non-membership by parameter counting, verified numerically.)
  N2  FULL-DIAGONAL CLASS (exponential resources): mixture ∈ IQP_n(full
      diagonal) ⟺ ∃ phases θ(x) with |IWHT[√p·e^{iθ}]| constant — a
      constant-modulus phase-retrieval feasibility problem. Probed by
      Gerchberg–Saxton alternating projections with multi-restart, WITH a
      known-feasible positive control. GS failure is evidence, not proof;
      GS success is a certificate of membership. Reported as found.
  N3  BEST SINGLE-IQP FIT: directly minimize ‖p_θ − p_mix‖² over the
      complete-graph single IQP at n=8; positive control = fitting a target
      that IS single-IQP. Residual TV gap quantifies HOW far the mixture
      sits from the weight-≤2 IQP class.
  N4  THE CONSTRUCTIVE CERTIFICATE: the degree-3 Walsh-sparse cIQP joint on
      n+a qubits IS a genuine IQP distribution — verified by recomputing the
      dense joint directly from the compiled circuit's IQP normal form
      (H^{⊗(n+a)} D H^{⊗(n+a)}) and checking degree ≤ 3. "IQP-ness" is
      retained at (a+1)|G| gates.

Honest scope notes are printed inline; every test carries a control.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from psck_mmd.graph import build_er_graph, build_complete_graph
from psck_mmd.ciqp import _pad_to_power_of_2
from psck_mmd.shallow_deploy import (
    exact_single_iqp_distribution, walsh_synthesize, build_cat_ciqp,
)

ok_all = True
def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{name}] {'✓' if cond else '✗ FAIL'}  {detail}")


def mixture_dist(comps, gates, n):
    p = np.zeros(1 << n)
    for th in comps:
        p += exact_single_iqp_distribution(th, gates, n) / len(comps)
    return p


# ─────────────────────────────────────────────────────────────────────────
# N1: Jacobian rank — mixture family vs the weight-≤2 single-IQP bound
# ─────────────────────────────────────────────────────────────────────────
print("\nN1  distribution-manifold dimensions (n=8, complete weight-≤2 graph)")
n = 8
gates, _ = build_complete_graph(n)           # |G| = 8 + 28 = 36
G = len(gates)
L = 4
rng = np.random.default_rng(11)
comps = [rng.uniform(0, 2 * np.pi, G) for _ in range(L)]
eps = 1e-6

def jac_single(theta):
    p0 = exact_single_iqp_distribution(theta, gates, n)
    J = np.zeros((1 << n, G))
    for j in range(G):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        J[:, j] = (exact_single_iqp_distribution(tp, gates, n)
                   - exact_single_iqp_distribution(tm, gates, n)) / (2 * eps)
    return J

J_single = jac_single(comps[0])
J_mix = np.hstack([jac_single(c) / L for c in comps])
def numrank(J):
    s = np.linalg.svd(J, compute_uv=False)
    return int(np.sum(s > 1e-7 * s[0])), s

r_s, _ = numrank(J_single)
r_m, _ = numrank(J_mix)
print(f"    single-IQP manifold rank = {r_s}  (bound |G| = {G})")
print(f"    MoIQP(L={L}) family rank = {r_m}  (params = {L*G})")
check("mixture family dim > ANY weight-≤2 single-IQP dim", r_m > G,
      f"{r_m} > {G} → generic mixture is NOT an n-qubit weight-≤2 IQP")

# ─────────────────────────────────────────────────────────────────────────
# N2: full-diagonal class — constant-modulus phase retrieval (GS probe)
# ─────────────────────────────────────────────────────────────────────────
print("\nN2  full-diagonal IQP membership (n=4): Gerchberg–Saxton probe")
n2 = 4
dim = 1 << n2
def wht_mat(N):
    H = np.array([[1.0]])
    while H.shape[0] < N:
        H = np.block([[H, H], [H, -H]])
    return H
Hm = wht_mat(dim) / np.sqrt(dim)             # unitary WHT

def gs_probe(p_target, n_restart=300, n_iter=3000, rng=None):
    """Feasibility of |H u|² = 2^n p with |u| ≡ 1. Returns min residual."""
    rng = rng or np.random.default_rng(0)
    tgt = np.sqrt(dim * p_target)
    best = np.inf
    for _ in range(n_restart):
        u = np.exp(1j * rng.uniform(0, 2 * np.pi, dim))
        for _ in range(n_iter):
            v = Hm @ u
            ph = np.where(np.abs(v) > 1e-14, v / np.abs(v),
                          np.exp(1j * rng.uniform(0, 2 * np.pi, dim)))
            v = tgt * ph
            u = Hm.conj().T @ v
            u = np.where(np.abs(u) > 1e-14, u / np.abs(u), 1.0)
        res = float(np.linalg.norm(np.abs(Hm @ u) - tgt))
        best = min(best, res)
    return best

gates2, _ = build_er_graph(n2, avg_deg=3.0, seed=2)
rng2 = np.random.default_rng(7)
# positive control: a target that IS full-diagonal IQP (random phases)
phi_ctrl = rng2.uniform(0, 2 * np.pi, dim)
p_ctrl = np.abs(Hm @ np.exp(1j * phi_ctrl)) ** 2 / dim
res_ctrl = gs_probe(p_ctrl, rng=np.random.default_rng(1))
check("GS positive control (feasible target) → residual ≈ 0",
      res_ctrl < 1e-6, f"res={res_ctrl:.2e}")
# the mixture target
comps2 = [rng2.normal(0, 0.6, len(gates2)) for _ in range(2)]
p_mix2 = mixture_dist(comps2, gates2, n2)
res_mix = gs_probe(p_mix2, rng=np.random.default_rng(2))
print(f"    mixture target: min GS residual over 300 restarts = {res_mix:.3e}")
if res_mix < 1e-6:
    print("    → this mixture IS representable with an EXPONENTIAL-size "
          "diagonal: 'IQP-ness' is resource-relative. Irrelevant to poly-size"
          " circuits and to hardness, but honesty demands reporting it.")
else:
    print("    → no unimodular preimage found: evidence (not proof) that even"
          " exponential diagonals cannot represent this mixture on n qubits.")

# ─────────────────────────────────────────────────────────────────────────
# N3: best single-IQP fit to the mixture — the TV gap
# ─────────────────────────────────────────────────────────────────────────
print("\nN3  best weight-≤2 single-IQP fit (n=8, analytic-gradient Adam)")
Wm = wht_mat(1 << n)                          # unscaled WHT (entries ±1)
bits8 = ((np.arange(1 << n, dtype=np.uint64)[:, None] >>
          np.arange(n, dtype=np.uint64)[None, :]) & 1).astype(np.int8)
Sg = np.empty((1 << n, G))
for j, g in enumerate(gates):
    Sg[:, j] = 1.0 - 2.0 * (bits8[:, list(g)].sum(axis=1) % 2)

def p_and_grad(theta, p_target):
    E = np.exp(1j * (Sg @ theta))             # e^{iφ(z)}
    F = (Wm @ E) / (1 << n)                   # amplitudes f(x)
    p = np.abs(F) ** 2
    dF = 1j * (Wm @ (Sg * E[:, None])) / (1 << n)     # (dim, G)
    grad_p = 2.0 * np.real(np.conj(F)[:, None] * dF)  # ∂p/∂θ
    r = p - p_target
    return float(r @ r), 2.0 * (r @ grad_p), p

def fit_single(p_target, inits, steps=4000, lr=0.05):
    best_tv = np.inf
    for th0 in inits:
        th = th0.copy(); m = np.zeros(G); v = np.zeros(G)
        for t in range(1, steps + 1):
            lr_t = lr * 0.5 * (1 + np.cos(np.pi * (t % 800) / 800))
            _, gr, _ = p_and_grad(th, p_target)
            m = 0.9 * m + 0.1 * gr; v = 0.999 * v + 0.001 * gr * gr
            th -= lr_t * (m / (1 - 0.9 ** t)) / (
                np.sqrt(v / (1 - 0.999 ** t)) + 1e-9)
        _, _, pf = p_and_grad(th, p_target)
        best_tv = min(best_tv, 0.5 * float(np.abs(pf - p_target).sum()))
    return best_tv

rngN = np.random.default_rng(3)
inits = [rngN.normal(0, s0, G) for s0 in (0.1, 0.2, 0.3, 0.4)
         for _ in range(3)]
th_true = rngN.normal(0, 0.4, G)
p_ctrl8 = exact_single_iqp_distribution(th_true, gates, n)
tv_ctrl = fit_single(p_ctrl8, inits)
check("fit positive control (target IS single-IQP) -> TV ~ 0",
      tv_ctrl < 2e-2, f"TV={tv_ctrl:.3e}")
p_mix8 = mixture_dist(comps, gates, n)
# give the optimizer its best shot at the mixture: also seed at each component
inits_mix = inits + [c.copy() for c in comps]
tv_mix = fit_single(p_mix8, inits_mix)
check("mixture sits FAR from the single-IQP class", tv_mix > 10 * max(tv_ctrl, 1e-3),
      f"best-fit TV={tv_mix:.3f} vs control {tv_ctrl:.3e}")

# ─────────────────────────────────────────────────────────────────────────
# N4: the constructive certificate — degree-3 Walsh-sparse cIQP IS IQP
# ─────────────────────────────────────────────────────────────────────────
print("\nN4  degree-3 Walsh-sparse cIQP: bona fide IQP normal form on n+a")
n4, a4 = 5, 2
g4, _ = build_er_graph(n4, avg_deg=3.0, seed=5)
U = [0] + [1 << k for k in range(a4)]
phi = np.random.default_rng(6).normal(0, 0.4, (len(g4), len(U)))
comps4 = walsh_synthesize(phi, U, a4)
gates_c, params_c, info = build_cat_ciqp(comps4, g4, n4, m_copies=1)
# The compiled object is by construction H^{⊗(n+a)} D H^{⊗(n+a)} with D a
# product of Z-string rotations of weight ≤ 3 → its FULL joint is an IQP
# distribution computable by the single-IQP formula on n+a qubits:
p_joint = exact_single_iqp_distribution(params_c, gates_c, n4 + a4)
p_mix4 = mixture_dist(comps4, g4, n4)
p_marg = p_joint.reshape((1 << n4, 1 << a4), order="F").sum(axis=1)
check("joint = IQP normal form, degree ≤ 3",
      info["max_weight"] <= 3 and abs(p_joint.sum() - 1) < 1e-12,
      f"wmax={info['max_weight']}, gates={info['n_deploy_gates']}"
      f"=(a+1)|G|={(a4+1)*len(g4)}")
check("its data-marginal = the (non-IQP) mixture",
      float(np.max(np.abs(p_marg - p_mix4))) < 1e-12)

print(f"\n{'='*64}\n  {'✓ ALL MEMBERSHIP AUDITS PASSED' if ok_all else '✗ FAILURES'}"
      f"\n{'='*64}")
sys.exit(0 if ok_all else 1)
