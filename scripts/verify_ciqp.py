#!/usr/bin/env python
"""Verify that the cIQP deployment of a trained MoIQP reproduces its marginals.

Reads a previously-trained run directory (produced by one of scripts/run_*.py),
rebuilds the base graph deterministically from the config, compiles the MoIQP
to a single IQP circuit on n + ⌈log₂ L⌉ qubits via the Walsh–Hadamard
deferred-measurement trick, runs Van den Nest MC on both, and compares
marginal ⟨Z_S⟩ agreement to 1/√M.

Writes `verification.json` and cIQP gate/param files into the same run
directory. Does NOT re-train.

Usage:
    python scripts/verify_ciqp.py outputs/psck_B2_binary_L4_ep500_seed42/
    python scripts/verify_ciqp.py <run_dir> --M 500000
"""
import os, sys, argparse, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from psck_mmd.data import load_calorimeter
from psck_mmd.graph import build_er_graph, build_complete_graph, precompute_active_lists
from psck_mmd.feature_obs import enumerate_feature_observables
from psck_mmd.correlator import ParityCache, sample_latents, forward_batch
from psck_mmd.ciqp import build_ciqp_circuit, print_ciqp_info
from psck_mmd.corr_jacobian import pearson_value_and_jacobian


def rebuild_graph(cfg):
    n = cfg["D"] * cfg["B"]
    if cfg.get("graph", "erdos_renyi") == "erdos_renyi":
        return build_er_graph(n, avg_deg=cfg.get("avg_deg", 6.0),
                                seed=cfg["seed"] + 1)
    return build_complete_graph(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str, help="path to a run folder")
    ap.add_argument("--M", type=int, default=200_000,
                     help="MC sample count for marginal verification")
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "metrics.json")) as f:
        cfg = json.load(f)
    D, B, L = cfg["D"], cfg["B"], cfg["L"]
    n_feat = D * B
    a = int(np.ceil(np.log2(max(L, 1))))
    n_total = n_feat + a

    print(f"\n{'='*72}")
    print(f"  cIQP verification: {args.run_dir}")
    print(f"  D={D} B={B} L={L}  →  n+a = {n_feat}+{a} = {n_total}q")
    print(f"{'='*72}\n")

    binary, raw = load_calorimeter(
        "data/cal_shower_img_8q.npy", bits=B,
        encoding=cfg.get("encoding", "binary"),
    )
    obs, _ = enumerate_feature_observables(D, B, max_weight=2)
    gates, q2g = rebuild_graph(cfg)

    params_list = [np.load(os.path.join(args.run_dir, f"params_{ell}.npy"))
                    for ell in range(L)]

    # MoIQP reference
    print(f"[1/3] MoIQP reference marginals (M={args.M})...")
    active_feat = precompute_active_lists(obs, gates, q2g)
    rng_v = np.random.default_rng(424242)
    z_mo = np.zeros(len(obs))
    for ell in range(L):
        lat = sample_latents(n_feat, args.M, rng_v, antithetic=True)
        z_l, _ = forward_batch(params_list[ell],
                                 ParityCache(lat, gates), active_feat)
        z_mo += z_l / L

    # cIQP deployment
    print(f"[2/3] Building cIQP circuit (Walsh–Hadamard)...")
    deploy_gates, deploy_params, info = build_ciqp_circuit(
        params_list, gates, n_feat,
    )
    print_ciqp_info(info)

    print(f"[3/3] Simulating deployed cIQP on {n_total} qubits...")
    dep_q2g = {q: [] for q in range(n_total)}
    for j, g in enumerate(deploy_gates):
        for q in g:
            dep_q2g[q].append(j)
    active_dep = precompute_active_lists(obs, deploy_gates, dep_q2g)
    lat_d = sample_latents(n_total, args.M, np.random.default_rng(424242),
                             antithetic=True)
    z_dep, _ = forward_batch(
        deploy_params, ParityCache(lat_d, deploy_gates), active_dep,
    )

    diff = z_dep - z_mo
    mae = float(np.mean(np.abs(diff)))
    mc_noise = 1.0 / np.sqrt(args.M)
    iu = np.triu_indices(D, k=1)
    rho_mo, _ = pearson_value_and_jacobian(z_mo, obs, D, B)
    rho_dep, _ = pearson_value_and_jacobian(z_dep, obs, D, B)
    rho_mae = float(np.mean(np.abs(rho_mo[iu] - rho_dep[iu])))

    print(f"\n  ⟨Z_S⟩ MAE (cIQP vs MoIQP) = {mae:.6f}")
    print(f"  MC noise 1/√M             = {mc_noise:.6f}")
    print(f"  ratio                     = {mae / mc_noise:.2f}× MC noise")
    print(f"  ρ MAE (cIQP vs MoIQP)     = {rho_mae:.6f}")
    passed = bool(mae < 5 * mc_noise)
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}: cIQP deployment "
          f"{'is exact' if passed else 'deviates from MoIQP'} within MC noise.")

    np.save(os.path.join(args.run_dir, "ciqp_deploy_gates.npy"),
            np.array(deploy_gates, dtype=object), allow_pickle=True)
    np.save(os.path.join(args.run_dir, "ciqp_deploy_params.npy"), deploy_params)
    json.dump({
        "n_feat": n_feat, "n_control": a, "n_total": n_total,
        "L": L, "M_verify": args.M,
        "mae_ciqp_vs_moiqp": mae,
        "mc_noise_scale": mc_noise,
        "ratio": mae / mc_noise,
        "rho_mae_ciqp_vs_moiqp": rho_mae,
        "n_deploy_gates": info["n_deploy_gates"],
        "weight_histogram": info["weight_histogram"],
        "max_weight": info["max_weight"],
        "passed": passed,
    }, open(os.path.join(args.run_dir, "verification.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
