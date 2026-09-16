#!/usr/bin/env python
"""Per-gate gradient variance against qubit count, App. D.

Produces bp_results.json, which Tables IV and V quote directly and Fig. 7 plots.
The scan is at a single initialisation amplitude and at initialisation only, so
it bounds the landscape at that point and not along a training trajectory. The
appendix says so; this note is here because the file name suggests more.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from psck_mmd.eval.barren_plateau import bp_scaling_sweep
from psck_mmd.plotting import set_root_style, ROOT_COLORS, _add_minor_ticks

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_both(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"),
                     dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_dir}/{stem}.{{png,pdf}}")


def plot_grad_norm2(results: dict, out_dir: str):
    qs = np.array(results["qubits_list"])
    losses = results["losses"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for loss in losses:
        means = np.array([r[loss]["grad_norm2_mean"] for r in results["by_n"]])
        # Per-replica standard deviation, scaled to SEM via sqrt(K)
        K_init = results["K_init"]
        # Use SD of grad_norm2 across replicas as an envelope
        sds = np.sqrt(np.array([r[loss]["grad_norm2_var"] for r in results["by_n"]]))
        sems = sds / np.sqrt(K_init)
        c = ROOT_COLORS.get(loss, "#7f8c8d")
        ax.errorbar(qs, means, yerr=sems, fmt="o-", color=c, linewidth=1.7,
                     capsize=3, label=loss)
    ax.set_xlabel("number of qubits  $n$")
    ax.set_ylabel(r"$\langle\|\nabla L\|^2\rangle_\theta$")
    ax.set_yscale("log")

    _add_minor_ticks(ax, logy=True)
    ax.legend(fontsize=11)
    _save_both(fig, out_dir, "bp_grad_norm2_vs_n")


def plot_per_gate_var(results: dict, out_dir: str):
    qs = np.array(results["qubits_list"])
    losses = results["losses"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for loss in losses:
        means = np.array([r[loss]["per_gate_var_mean"] for r in results["by_n"]])
        meds  = np.array([r[loss]["per_gate_var_median"] for r in results["by_n"]])
        c = ROOT_COLORS.get(loss, "#7f8c8d")
        ax.plot(qs, means, "o-", color=c, linewidth=1.7,
                  label=f"{loss} (mean)")
        ax.plot(qs, meds, "s--", color=c, linewidth=1.3, alpha=0.7,
                  label=f"{loss} (median)")
    ax.set_xlabel("number of qubits  $n$")
    ax.set_ylabel(r"$\mathrm{Var}_\theta[\partial L/\partial\theta_j]$  (per-gate)")
    ax.set_yscale("log")

    _add_minor_ticks(ax, logy=True)
    ax.legend(fontsize=11)
    _save_both(fig, out_dir, "bp_per_gate_var_vs_n")


def plot_loss_at_init(results: dict, out_dir: str):
    qs = np.array(results["qubits_list"])
    losses = results["losses"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for loss in losses:
        means = np.array([float(np.mean(r[loss]["loss_values"]))
                            for r in results["by_n"]])
        sds   = np.array([float(np.std(r[loss]["loss_values"], ddof=1))
                            for r in results["by_n"]])
        c = ROOT_COLORS.get(loss, "#7f8c8d")
        ax.errorbar(qs, means, yerr=sds, fmt="o-", color=c, linewidth=1.7,
                     capsize=3, label=loss)
    ax.set_xlabel("number of qubits  $n$")
    ax.set_ylabel(r"$L(\theta_0)$  at random init")
    ax.set_yscale("log")

    _add_minor_ticks(ax, logy=True)
    ax.legend(fontsize=11)
    _save_both(fig, out_dir, "bp_loss_value_vs_n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits-list", type=int, nargs="+", default=[2, 3, 4, 6, 8],
                     help="list of B values (qubits = 8*B); default 2 3 4 6 8")
    ap.add_argument("--K-init", type=int, default=200,
                     help="number of random parameter inits per n")
    ap.add_argument("--M-grad", type=int, default=2048,
                     help="MC batch size per gradient evaluation")
    ap.add_argument("--sigma-init", type=float, default=0.1,
                     help="std of N(0, σ²I) parameter init")
    ap.add_argument("--losses", type=str, nargs="+", default=["psck", "lw_mmd"],
                     choices=["psck", "lw_mmd"])
    ap.add_argument("--encoding", type=str, default="binary",
                     choices=["binary", "gray", "hadamard"])
    ap.add_argument("--seed-graph", type=int, default=3)
    ap.add_argument("--seed-inits", type=int, default=12345)
    ap.add_argument("--out-dir", type=str, required=True,
                     help="where to write bp_results.json and plots")
    ap.add_argument("--style", type=str, default="ROOT",
                     choices=["ROOT", "CMS", "ATLAS", "LHCb", "ALICE"])
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_root_style(args.style)

    print(f"\n{'='*72}")
    print(f"  Barren-plateau scaling sweep")
    print(f"  bits_list = {args.bits_list}  (qubits = {[8*B for B in args.bits_list]})")
    print(f"  losses    = {args.losses}")
    print(f"  K_init    = {args.K_init}, M_grad = {args.M_grad}, σ_init = {args.sigma_init}")
    print(f"  encoding  = {args.encoding}")
    print(f"  out_dir   = {args.out_dir}")
    print(f"{'='*72}")

    t0 = time.time()
    results = bp_scaling_sweep(
        bits_list=args.bits_list, K_init=args.K_init, M_grad=args.M_grad,
        sigma_init=args.sigma_init, losses=tuple(args.losses),
        encoding=args.encoding, seed_graph=args.seed_graph,
        seed_inits=args.seed_inits, verbose=True,
    )
    total = time.time() - t0
    print(f"\n  Total elapsed: {total/60:.1f} min")

    json.dump(results, open(os.path.join(args.out_dir, "bp_results.json"), "w"),
              indent=2)
    print(f"  bp_results.json → {args.out_dir}")

    if not args.skip_plots:
        plot_grad_norm2(results, args.out_dir)
        plot_per_gate_var(results, args.out_dir)
        plot_loss_at_init(results, args.out_dir)
    print()


if __name__ == "__main__":
    main()
