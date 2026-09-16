"""
psck_mmd.graph — IQP graph constructors and active-gate precomputation.

An IQP graph is an (unordered) collection of gates P_j, each of which is a
Z-Pauli on a subset gate_j ⊆ [n]. For IQP Born machines we restrict to
weight-1 (single qubit Z) and weight-2 (pair ZZ) gates, which is standard.

The two constructors match the usages in your original iqp_storm codebase:
  - build_complete_graph:    all singles + all pairs   (for small n ≲ 32)
  - build_er_graph:          singles + Erdős–Rényi pairs with avg degree d
                              — the BJS-style sparse IQP graph that keeps
                              classical simulation costly in the worst case while keeping the
                              circuit depth polynomial.
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constructors
# ─────────────────────────────────────────────────────────────────────────────

def build_complete_graph(n: int):
    """All n singles + C(n,2) pairs. Returns (gates, q2g)."""
    gates = [(q,) for q in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            gates.append((i, j))
    q2g: dict[int, list[int]] = {q: [] for q in range(n)}
    for idx, g in enumerate(gates):
        for q in g:
            q2g[q].append(idx)
    return gates, q2g


def build_er_graph(n: int, avg_deg: float = 6.0, seed: int = 42):
    """Singles + Erdős–Rényi pair-gate layer with target average degree.

    Edge probability p = min(1, avg_deg/(n-1)). Returns (gates, q2g).
    """
    rng = np.random.default_rng(seed)
    p = min(1.0, avg_deg / max(1, n - 1))
    gates = [(q,) for q in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                gates.append((i, j))
    q2g: dict[int, list[int]] = {q: [] for q in range(n)}
    for idx, g in enumerate(gates):
        for q in g:
            q2g[q].append(idx)
    return gates, q2g


def build_graph(n: int, graph_type: str = "erdos_renyi",
                avg_deg: float = 6.0, seed: int = 42):
    """Uniform entry-point. graph_type ∈ {"complete", "erdos_renyi"}."""
    if graph_type == "complete":
        return build_complete_graph(n)
    elif graph_type in ("erdos_renyi", "er"):
        return build_er_graph(n, avg_deg=avg_deg, seed=seed)
    else:
        raise ValueError(f"unknown graph_type: {graph_type}")


def graph_stats(gates, n: int):
    """Diagnostic summary used by run scripts."""
    singles = [g for g in gates if len(g) == 1]
    pairs = [g for g in gates if len(g) == 2]
    deg = np.zeros(n, dtype=int)
    for i, j in pairs:
        deg[i] += 1; deg[j] += 1
    return {
        "n_gates": len(gates),
        "n_singles": len(singles),
        "n_pairs": len(pairs),
        "avg_degree": float(deg.mean()),
        "max_degree": int(deg.max()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Active-set precomputation (needed by the forward pass)
# ─────────────────────────────────────────────────────────────────────────────

def active_gates_for_observable(obs, gates, q2g) -> list[int]:
    """Sorted list of gate indices j with |gate_j ∩ obs| odd (the only gates
    that contribute to the argument of cos for observable obs)."""
    obs_set = set(obs)
    touched: set[int] = set()
    for q in obs:
        touched.update(q2g[q])
    out: list[int] = []
    for j in touched:
        overlap = sum(1 for q in gates[j] if q in obs_set)
        if overlap & 1:
            out.append(j)
    out.sort()
    return out


def precompute_active_lists(observables, gates, q2g) -> list[list[int]]:
    """Vectorized version over a list of observables."""
    return [active_gates_for_observable(o, gates, q2g) for o in observables]


# ─────────────────────────────────────────────────────────────────────────────
# Hardware-native graph: induced subgraph of a device coupling map
#
# Motivation: a random ER pair layer has long-range edges that the transpiler
# must implement with SWAP chains on a planar (heavy-hex) device, inflating the
# 2-qubit-gate count ~3x and crushing the global fidelity F. If instead the IQP
# pair gates are exactly the device's NATIVE edges, routing is unnecessary and F
# rises sharply. This MUST be done at TRAINING time, because the trained angles
# are attached to specific edges; you cannot relabel the graph after the fact.
#
# Trade-off (be honest): a heavy-hex native subgraph has avg degree ~2-3 vs the
# ER default of 6, i.e. FEWER ZZ terms and thus lower expressivity. Whether net
# generation quality improves is empirical — compare the recovered rho
# against the data rho for both graphs. Native graph helps the shallow `mixture`
# object directly; the cIQP's control-feature couplings are non-native anyway,
# so native+mixture is the intended combination.
# ─────────────────────────────────────────────────────────────────────────────

from collections import defaultdict as _defaultdict
import json as _json


def _adjacency(coupling_map):
    adj = _defaultdict(set)
    for e in coupling_map:
        a, b = int(e[0]), int(e[1])
        if a != b:
            adj[a].add(b); adj[b].add(a)
    return adj


def linear_coupling_map(n):
    """Nearest-neighbor chain (trivially correct; for tests/sanity)."""
    return [[i, i + 1] for i in range(n - 1)]


def grid_coupling_map(rows, cols):
    """2-D grid coupling map (planar, like a square-lattice device)."""
    edges = []
    def idx(r, c):
        return r * cols + c
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append([idx(r, c), idx(r, c + 1)])
            if r + 1 < rows:
                edges.append([idx(r, c), idx(r + 1, c)])
    return edges


def save_coupling_map(coupling_map, path):
    _json.dump([[int(a), int(b)] for a, b in coupling_map], open(path, "w"))


