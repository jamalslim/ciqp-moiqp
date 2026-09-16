#!/usr/bin/env python
"""Evaluate a trained run on the held-out split.

Everything in Tables I and X other than the training-split values comes from
here. The quantile edges are read from the committed split rather than refitted,
which is what makes the test numbers honest.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from psck_mmd.data_split import (
    load_calorimeter_split, ensure_split_exists,
    compute_encoding_fidelity_with_encoding,
)
from psck_mmd.graph import build_er_graph, build_complete_graph, precompute_active_lists
from psck_mmd.feature_obs import enumerate_feature_observables
from psck_mmd.correlator import (
    ParityCache, sample_latents, forward_batch, z_data_batch,
)
from psck_mmd.corr_jacobian import pearson_value_and_jacobian
from psck_mmd.psck_kernel import heat_kernel_coeffs

from psck_mmd.eval.marginal_recovery import (
    enumerate_intra_feature_observables,
    recover_feature_distributions,
    empirical_feature_distributions,
)
from psck_mmd.eval.distributional_metrics import (
    summarize_distribution_metrics, moment_mmd,
)
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


def rebuild_graph(cfg):
    n = cfg["D"] * cfg["B"]
    if cfg.get("graph", "erdos_renyi") == "erdos_renyi":
        return build_er_graph(n, avg_deg=cfg.get("avg_deg", 6.0),
                                seed=cfg["seed"] + 1)
    return build_complete_graph(n)


def evaluate_z_marginals(params_list, gates, q2g, observables, n_feat, M, seed):
    """Run forward pass on observables for each component, return mixture z."""
    L = len(params_list)
    active = precompute_active_lists(observables, gates, q2g)
    rng = np.random.default_rng(seed)
    z_mix = np.zeros(len(observables))
    for ell in range(L):
        lat = sample_latents(n_feat, M, rng, antithetic=True)
        z_l, _ = forward_batch(params_list[ell],
                                 ParityCache(lat, gates), active)
        z_mix += z_l / L
    return z_mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--eval-M", type=int, default=100_000,
                     help="MC sample count for low-order correlator estimation")
    ap.add_argument("--marg-M", type=int, default=50_000,
                     help="MC sample count for intra-feature high-weight correlators")
    ap.add_argument("--skip-marginal-recovery", action="store_true",
                     help="skip the (D × (2^B-1))-correlator marginal recovery step")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--style", type=str, default="ROOT",
                     choices=["ROOT", "CMS", "ATLAS", "LHCb", "ALICE"])
    ap.add_argument("--data", type=str, default="data/cal_shower_img_8q.npy")
    ap.add_argument("--split", type=str, default="data/split_indices.npz")
    ap.add_argument("--out-subdir", type=str, default="eval")
    args = ap.parse_args()

    out_dir = os.path.join(args.run_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    log_lines: list[str] = []

    def log(s: str = ""):
        print(s); log_lines.append(s)

    set_root_style(args.style)

    # ── Load trained model config & params ──
    with open(os.path.join(args.run_dir, "metrics.json")) as f:
        cfg = json.load(f)
    D, B, L = cfg["D"], cfg["B"], cfg["L"]
    n_feat = D * B
    encoding = cfg.get("encoding", "binary")
    log(f"\n{'='*72}")
    log(f"  Evaluation: {args.run_dir}")
    log(f"  D={D}, B={B}, n={n_feat}q, L={L}, enc={encoding}, loss={cfg['loss']}")
    log(f"{'='*72}\n")

    params_list = [np.load(os.path.join(args.run_dir, f"params_{ell}.npy"))
                    for ell in range(L)]
    gates, q2g = rebuild_graph(cfg)

    # ── Held-out split ──
    ensure_split_exists(data_path=args.data, split_path=args.split)
    tr_bin, tr_raw, te_bin, te_raw = load_calorimeter_split(
        data_path=args.data, bits=B, encoding=encoding, split_path=args.split,
    )
    log(f"  train: {tr_bin.shape}   test: {te_bin.shape}")

    enc_tr = compute_encoding_fidelity_with_encoding(tr_bin, tr_raw, B, encoding)
    enc_te = compute_encoding_fidelity_with_encoding(te_bin, te_raw, B, encoding)
    log(f"  encoding floor train: ρ MAE = {enc_tr['rho_mae_floor']:.6f}")
    log(f"  encoding floor test : ρ MAE = {enc_te['rho_mae_floor']:.6f}")

    metrics: dict = {
        "run_dir":  args.run_dir,
        "config":   {"D": D, "B": B, "L": L, "encoding": encoding,
                       "loss": cfg["loss"], "epochs": cfg["epochs"],
                       "seed": cfg["seed"]},
        "split":    {"n_train": int(tr_bin.shape[0]),
                       "n_test":  int(te_bin.shape[0]),
                       "encoding_floor_train": enc_tr["rho_mae_floor"],
                       "encoding_floor_test":  enc_te["rho_mae_floor"]},
    }

    # ── (A) Low-order ⟨Z_β⟩ on train and test ──
    log(f"\n[A] Low-order Z-correlator evaluation (M={args.eval_M}) ...")
    obs, _ = enumerate_feature_observables(D, B, max_weight=2)
    z_data_train = z_data_batch(tr_bin, obs)
    z_data_test  = z_data_batch(te_bin, obs)

    t0 = time.time()
    z_model = evaluate_z_marginals(
        params_list, gates, q2g, obs, n_feat, args.eval_M,
        seed=cfg["seed"] + 77777,
    )
    log(f"    model ⟨Z_β⟩ MC done in {time.time()-t0:.1f}s "
        f"(K_obs = {len(obs)})")

    z_mae_train = float(np.mean(np.abs(z_model - z_data_train)))
    z_mae_test  = float(np.mean(np.abs(z_model - z_data_test)))
    log(f"    z MAE  train: {z_mae_train:.6f}")
    log(f"    z MAE  test : {z_mae_test:.6f}")

    iu = np.triu_indices(D, k=1)
    rho_model, _ = pearson_value_and_jacobian(z_model, obs, D, B)
    rho_train_raw = np.corrcoef(tr_raw.T)
    rho_test_raw  = np.corrcoef(te_raw.T)
    rho_mae_train = float(np.mean(np.abs(rho_model[iu] - rho_train_raw[iu])))
    rho_mae_test  = float(np.mean(np.abs(rho_model[iu] - rho_test_raw[iu])))
    rho_r_train = float(np.corrcoef(rho_model[iu], rho_train_raw[iu])[0, 1])
    rho_r_test  = float(np.corrcoef(rho_model[iu], rho_test_raw[iu])[0, 1])
    log(f"    ρ MAE  train: {rho_mae_train:.6f}, r = {rho_r_train:.4f}")
    log(f"    ρ MAE  test : {rho_mae_test:.6f},  r = {rho_r_test:.4f}")

    metrics["lowZ"] = {
        "K_obs": len(obs), "M_eval": args.eval_M,
        "z_mae_train": z_mae_train, "z_mae_test":  z_mae_test,
        "rho_mae_train": rho_mae_train, "rho_mae_test": rho_mae_test,
        "rho_r_train": rho_r_train, "rho_r_test": rho_r_test,
    }

    # ── (B) Heat-kernel moment MMD on train & test ──
    log(f"\n[B] Heat-kernel moment MMD ...")
    omega_heat = heat_kernel_coeffs(obs, n_feat)
    mmd_heat_train = moment_mmd(z_model, z_data_train, omega_heat)
    mmd_heat_test  = moment_mmd(z_model, z_data_test,  omega_heat)
    # Independent-uniform-kernel MMD (k̂_β = 1 for all β) — characteristic on
    # the basis up to weight-2.
    omega_uniform = np.ones_like(omega_heat)
    mmd_unif_train = moment_mmd(z_model, z_data_train, omega_uniform)
    mmd_unif_test  = moment_mmd(z_model, z_data_test,  omega_uniform)
    log(f"    heat MMD²    train = {mmd_heat_train:.4e}, test = {mmd_heat_test:.4e}")
    log(f"    uniform MMD² train = {mmd_unif_train:.4e}, test = {mmd_unif_test:.4e}")

    metrics["mmd"] = {
        "heat_train": mmd_heat_train, "heat_test": mmd_heat_test,
        "uniform_train": mmd_unif_train, "uniform_test": mmd_unif_test,
    }

    # ── (C) Per-feature exact marginal recovery + W₁/KS/TV ──
    if not args.skip_marginal_recovery:
        log(f"\n[C] Per-feature exact-marginal recovery ...")
        intra_obs, slot_index = enumerate_intra_feature_observables(D, B)
        n_intra = len(intra_obs)
        log(f"    intra-feature observables to estimate: {n_intra}")
        log(f"    MC budget per observable: {args.marg_M}")
        log(f"    note: weight-1 and weight-2 intra-feature observables "
            f"are already in obs (low-order). We compute the higher-weight "
            f"complement with a separate MC pass.")

        # Compute z on the intra-feature observable set with a fresh active list.
        intra_active = precompute_active_lists(intra_obs, gates, q2g)
        t0 = time.time()
        z_intra = evaluate_z_marginals(
            params_list, gates, q2g, intra_obs, n_feat, args.marg_M,
            seed=cfg["seed"] + 88888,
        )
        log(f"    intra-feature ⟨Z_β⟩ MC done in {time.time()-t0:.1f}s")

        p_model = recover_feature_distributions(
            z_intra, slot_index, D, B, encoding, clip=True,
        )
        p_test = empirical_feature_distributions(te_bin, D, B, encoding)
        p_train = empirical_feature_distributions(tr_bin, D, B, encoding)

        np.savez(os.path.join(out_dir, "feature_marginals.npz"),
                 p_model=p_model, p_test=p_test, p_train=p_train)

        summary_train = summarize_distribution_metrics(p_model, p_train)
        summary_test  = summarize_distribution_metrics(p_model, p_test)
        log(f"    per-feature W₁ (levels) test: "
            f"mean = {summary_test['wasserstein1_mean_levels']:.4f}, "
            f"max = {summary_test['wasserstein1_max_levels']:.4f}")
        log(f"    per-feature KS         test: "
            f"mean = {summary_test['ks_mean']:.4f}, "
            f"max = {summary_test['ks_max']:.4f}")
        log(f"    per-feature TV         test: "
            f"mean = {summary_test['tv_mean']:.4f}, "
            f"max = {summary_test['tv_max']:.4f}")

        metrics["per_feature_train"] = summary_train
        metrics["per_feature_test"]  = summary_test

        # ── Plots ──
        if not args.skip_plots:
            color = ROOT_COLORS.get(cfg["loss"], "#c0392b")
            # W1 per feature
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(D)
            ax.bar(x - 0.18, summary_train["wasserstein1_per_feature_levels"],
                    0.34, color=color, alpha=0.6, label="vs train")
            ax.bar(x + 0.18, summary_test["wasserstein1_per_feature_levels"],
                    0.34, color=color, edgecolor="black", linewidth=1.0,
                    label="vs test")
            ax.set_xticks(x); ax.set_xticklabels([f"f{i}" for i in range(D)])
            ax.set_ylabel(r"$W_1$ (level units)")
            _add_minor_ticks(ax)
            ax.legend(fontsize=10)
            _save_both(fig, out_dir, "per_feature_w1")

            # KS per feature
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x - 0.18, summary_train["ks_per_feature"], 0.34,
                    color=color, alpha=0.6, label="vs train")
            ax.bar(x + 0.18, summary_test["ks_per_feature"], 0.34,
                    color=color, edgecolor="black", linewidth=1.0,
                    label="vs test")
            ax.set_xticks(x); ax.set_xticklabels([f"f{i}" for i in range(D)])
            ax.set_ylabel("KS statistic")

            _add_minor_ticks(ax)
            ax.legend(fontsize=10)
            _save_both(fig, out_dir, "per_feature_ks")

            # Per-feature histograms (one panel per feature is OK here —
            # this IS the per-feature-distribution plot)
            n_levels = 1 << B
            fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
            for f, ax in enumerate(axes.flat):
                lvls = np.arange(n_levels)
                ax.bar(lvls - 0.20, p_test[f], 0.38, color="#2c3e50",
                        alpha=0.85, label="test")
                ax.bar(lvls + 0.20, p_model[f], 0.38, color=color,
                        alpha=0.85, label="model")

                ax.set_xlabel("level")
                if f == 0:
                    ax.set_ylabel("probability")
                if f == 0:
                    ax.legend(fontsize=9)
                ax.grid(False)

            fig.tight_layout()
            _save_both(fig, out_dir, "feature_histograms")

    # ── Save everything ──
    json.dump(metrics, open(os.path.join(out_dir, "eval_metrics.json"), "w"),
              indent=2)
    with open(os.path.join(out_dir, "eval_log.txt"), "w") as f:
        f.write("\n".join(log_lines))
    log(f"\n  Eval artifacts under: {out_dir}\n")


if __name__ == "__main__":
    main()