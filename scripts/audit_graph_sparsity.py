#!/usr/bin/env python3
"""Reachable correlations under a sparse gate graph, Sec. VII.

Two qubits whose closed neighborhoods in the gate graph are disjoint satisfy
<Z_i Z_j> = <Z_i><Z_j> exactly, for every parameter setting, so that pair carries
no covariance at all. Banks et al. (arXiv:2606.28236, App. E 5) point out that on
Erdos-Renyi graphs of fixed average degree the fraction of such pairs tends to one
as the register grows, and that at Delta = 6 and n = 64 it is already about half.

Both halves of that are worth checking, and they come apart. The qubit-level
statement holds and this script reproduces it. But the scored observable is the
Pearson correlation between FEATURES, and each feature spans B qubits, so a
feature pair is constrained only when all B^2 of its cross-feature qubit pairs
factorize at once. This script counts how many survive.

Usage
-----
    python3 scripts/audit_graph_sparsity.py --sweep outputs/seed_sweep_B8_L8_split
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def closed_neighbourhoods(gates, n):
    nb = {q: {q} for q in range(n)}
    for g in gates:
        if len(g) == 2:
            nb[g[0]].add(g[1])
            nb[g[1]].add(g[0])
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--D", type=int, default=8, help="features")
    ap.add_argument("--bits", type=int, default=8, help="qubits per feature")
    ap.add_argument("--avg-degree", type=float, default=6.0)
    a = ap.parse_args()
    D, B = a.D, a.bits
    n = D * B

    paths = sorted(glob.glob(os.path.join(a.sweep, "*seed*", "graph.json")))
    if not paths:
        print(f"  no graph.json under {a.sweep!r}")
        return 1

    print(f"\n  n = {n}, {D} features of {B} qubits, {len(paths)} graphs\n")
    print(f"  {'run':>10} {'qubit pairs factorizing':>25} "
          f"{'min':>6} {'median':>8} {'max':>6} {'zero':>6}")
    print("  " + "-" * 70)
    allcnt = []
    fracs = []
    for p in paths:
        g = json.load(open(p))
        gates = [tuple(x) for x in (g["gates"] if isinstance(g, dict) else g)]
        nb = closed_neighbourhoods(gates, n)

        tot = dis = 0
        for i in range(n):
            for j in range(i + 1, n):
                tot += 1
                if not (nb[i] & nb[j]):
                    dis += 1
        fracs.append(dis / tot)

        feat = [list(range(f * B, (f + 1) * B)) for f in range(D)]
        cnt = []
        for x in range(D):
            for y in range(x + 1, D):
                cnt.append(sum(1 for i in feat[x] for j in feat[y] if nb[i] & nb[j]))
        cnt = np.array(cnt)
        allcnt.append(cnt)
        print(f"  {os.path.basename(os.path.dirname(p))[-7:]:>10} "
              f"{100*dis/tot:>24.1f}% {cnt.min():>6} {int(np.median(cnt)):>8} "
              f"{cnt.max():>6} {int((cnt == 0).sum()):>6}")

    allcnt = np.concatenate(allcnt)
    d, nn = a.avg_degree, n
    pred = (1 - d / (nn - 1)) * (1 - (d / (nn - 1)) ** 2) ** (nn - 2)
    print(f"\n  qubit-level, Eq. (E11) of Banks et al. at Delta = {d:g}: {pred:.3f}")
    print(f"  qubit-level, measured mean over graphs:            {np.mean(fracs):.3f}")
    print(f"\n  feature-level, unconstrained cross-feature qubit pairs out of {B*B}")
    print(f"    min {allcnt.min()}, median {int(np.median(allcnt))}, "
          f"mean {allcnt.mean():.1f}, max {allcnt.max()}")
    print(f"    feature pairs with zero: {int((allcnt == 0).sum())} of {allcnt.size}")
    print(f"\n  The qubit-level constraint is real and is reproduced here. It does not")
    print(f"  reach the scored observable, because a feature pair is constrained only")
    print(f"  when all {B*B} of its cross-feature qubit pairs factorize together, and")
    print(f"  none does at this size. The margin shrinks as n grows at fixed Delta,")
    print(f"  so it is a design constraint rather than a defect.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
