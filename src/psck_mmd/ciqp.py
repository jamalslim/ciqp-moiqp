"""
psck_mmd.ciqp — Compile a L-component MoIQP into a single IQP circuit on
n + a qubits where a = ⌈log₂ L⌉, via the Walsh–Hadamard Transform / deferred-
measurement construction.

─────────────────────────────────────────────────────────────────────────────
CONSTRUCTION (Bremner-Jozsa-Shepherd compatible)
─────────────────────────────────────────────────────────────────────────────

Let the base IQP graph have G gates {P_j} (each P_j is a Z-Pauli on one or two
feature qubits). A MoIQP is the ensemble
    { |psi_l> = prod_j exp(i theta_{l,j} P_j) |+>^n, l = 0..L-1 }
with uniform weight 1/L. The generator (mixture) is
    rho_mix = (1/L) sum_l |psi_l><psi_l|.

We want a PURE state on n + a qubits (a = ceil(log2 L)) whose PARTIAL TRACE
over the a "control" qubits equals rho_mix. Starting from |+>^{n+a}, we will
apply an IQP unitary of the form
    U = prod_{j in base} prod_{b in {0,1}^a} exp(i phi_{j,b} P_j otimes Pi_b)
where Pi_b = (1/2^a) prod_{k} (I + z_k^{b_k} Z_{n+k}) is the projector on
control state |b>. Since each Z-Pauli commutes with every other Z-Pauli,
the product unfolds to:
    U = prod_{j in base} exp( i P_j otimes T_j(theta) )
where T_j(theta) = sum_b phi_{j,b} Pi_b is a DIAGONAL operator on control
qubits. Expanding Pi_b in Pauli Zs on the a controls gives a sum over
subsets S of {n, n+1, ..., n+a-1} with coefficients
    phi_{j,b} <-> tilde_phi_{j,S} := WHT[phi_{j,*}](S) / 2^a
i.e. the WHT of the per-component angles, spread over all 2^a control-subset
combinations. In terms of IQP gates on the enlarged graph:
    U = prod_{j in base} prod_{S subseteq controls} exp( i tilde_phi_{j,S} P_j Z_S )
The CRITICAL IDENTITY is: when the controls are initialised as |+>^a and we
measure them in the computational basis at the end (uniformly selecting some
index l in {0, ..., L-1}), the resulting post-measurement state on the n data
qubits is exactly |psi_l>. Equivalently, tracing out the controls gives
rho_mix. This is the deferred-measurement trick (aka cIQP).

Why this keeps the circuit inside the IQP class: the enlarged U is a Z-polynomial-degree unitary
conjugated by H^{(n+a)}, which IS a valid IQP unitary. Sampling its output
(including the control register) is conjectured classically hard for WORST-CASE
instances under BJS 2011, because
any classical simulator for the enlarged circuit would also simulate each
|psi_l> via post-selection.

─────────────────────────────────────────────────────────────────────────────
WHAT'S COMPUTED HERE
─────────────────────────────────────────────────────────────────────────────

L MUST be a power of 2 (review item A3): a zero-angle 'identity' component
is H*I*H|0..0> = |0..0>, a delta spike at the all-zeros bitstring, so padding
is NOT distributionally neutral (dense check: up to 0.18 deviation at L=3).
A Hadamard layer cannot prepare a uniform superposition over a non-power-of-two
number of ancilla states; the IQP form itself imposes the constraint. Historic text:
L_eff = 2^a. The Walsh-Hadamard matrix H_a has entries H_a[b, S] = (-1)^{b·S}
and H_a/2^a applied to the padded theta vector gives tilde_phi.

Output:
  deploy_gates : list of tuples (q0,) or (q0, q1, ...) of arbitrary weight
  deploy_params: (n_gates_deploy,) float array of deployed angles
  info         : diagnostic dict

Note: the deployed gates may have weight up to 1 + a (for controls only) or
up to 2 + a (for base pair-gate on data x subset-of-controls). This stays
polynomial in n, not exponential. For the run we do here (a = 2), weights
stay <= 4.

The marginals <Z_S>_deploy on the FEATURE qubits alone equal the MoIQP
marginals (1/L) sum_l <Z_S>_{theta_l}, EXACTLY. This is the verification we
check numerically.
"""

from __future__ import annotations
import numpy as np


def _pad_to_power_of_2(params_list: list[np.ndarray]) -> tuple[list[np.ndarray], int]:

    L = len(params_list)
    if L & (L - 1) != 0:
        raise ValueError(
            f"MoIQP->cIQP compilation requires L to be a power of 2 (got L={L}). "
            "Zero-angle padding is invalid: an all-zero-angle IQP component produces "
            "a delta distribution at the all-zeros bitstring, not a neutral element, "
            "so the compiled marginal would equal the PADDED mixture, not the L-component "
            "mixture (deviation up to 0.18 in probability at L=3). Retrain at L in {1,2,4,8,...}."
        )
    a = int(np.ceil(np.log2(max(L, 1))))
    L_pad = 1 << a
    ng = len(params_list[0])
    if L_pad > L:
        zero = np.zeros(ng, dtype=params_list[0].dtype)
        padded = list(params_list) + [zero] * (L_pad - L)
    else:
        padded = list(params_list)
    return padded, a


def _walsh_hadamard(v: np.ndarray) -> np.ndarray:
    """Unscaled WHT: out[S] = sum_b (-1)^{b·S} v[b], for v of length 2^a."""
    v = v.astype(np.float64).copy()
    L = len(v)
    h = 1
    while h < L:
        for i in range(0, L, 2 * h):
            for j in range(i, i + h):
                x, y = v[j], v[j + h]
                v[j] = x + y
                v[j + h] = x - y
        h *= 2
    return v


def build_ciqp_circuit(params_list: list[np.ndarray],
                       base_gates: list[tuple[int, ...]],
                       n_feat: int):
    """Build the cIQP deployment circuit.

    Inputs:
        params_list : L per-component angle vectors, each length = |base_gates|
        base_gates  : gates of the SHARED base IQP graph
        n_feat      : number of feature qubits

    Outputs:
        deploy_gates  : list of tuples of qubit indices (data and controls)
        deploy_params : ndarray of deployed angles, same length as deploy_gates
        info          : dict with diagnostics
    """
    padded, a = _pad_to_power_of_2(params_list)
    L_pad = 1 << a
    G = len(base_gates)

    # WHT per gate-index over components: (G, L_pad) -> tilde_phi[j, S]
    per_gate = np.stack(padded, axis=1)          # (G, L_pad) where col = component
    tilde = np.zeros_like(per_gate)
    for j in range(G):
        tilde[j] = _walsh_hadamard(per_gate[j]) / L_pad    # NORMALIZED WHT

    control_qubits = list(range(n_feat, n_feat + a))

    deploy_gates: list[tuple[int, ...]] = []
    deploy_params_list: list[float] = []
    for j, base in enumerate(base_gates):
        for S_int in range(L_pad):
            # interpret S_int as a subset S ⊆ {0, ..., a-1}
            S_qubits = tuple(control_qubits[k] for k in range(a) if (S_int >> k) & 1)
            angle = float(tilde[j, S_int])
            if abs(angle) < 1e-15:
                continue
            new_gate = tuple(sorted(list(base) + list(S_qubits)))
            deploy_gates.append(new_gate)
            deploy_params_list.append(angle)

    deploy_params = np.array(deploy_params_list, dtype=np.float64)
    n_total = n_feat + a

    # Deduplicate identical qubit-set gates by summing their angles (shouldn't
    # happen if base_gates are unique, but defensive)
    dedup: dict[tuple[int, ...], float] = {}
    for g, p in zip(deploy_gates, deploy_params):
        dedup[g] = dedup.get(g, 0.0) + float(p)
    deploy_gates_out = list(dedup.keys())
    deploy_params_out = np.array([dedup[g] for g in deploy_gates_out], dtype=np.float64)

    # Weight histogram
    weights = [len(g) for g in deploy_gates_out]
    wh = {w: weights.count(w) for w in sorted(set(weights))}

    info = {
        "n_feat": n_feat, "a": a, "n_total": n_total,
        "L": len(params_list), "L_padded": L_pad,
        "n_base_gates": G,
        "n_deploy_gates": len(deploy_gates_out),
        "max_weight": int(max(weights)) if weights else 0,
        "weight_histogram": wh,
    }
    return deploy_gates_out, deploy_params_out, info


def print_ciqp_info(info: dict):
    print(f"  cIQP: {info['n_feat']} feat + {info['a']} ctrl "
          f"= {info['n_total']} total qubits  (L={info['L']} → L_pad={info['L_padded']})")
    print(f"  Base gates: {info['n_base_gates']}  →  Deployed gates: {info['n_deploy_gates']}")
    print(f"  Max gate weight: {info['max_weight']}")
    print(f"  Weight histogram: {info['weight_histogram']}")
