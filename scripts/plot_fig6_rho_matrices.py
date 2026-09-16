#!/usr/bin/env python3
"""plot_fig6_rho_matrices.py — reproduce manuscript Fig. 6: pairwise Pearson
correlation matrices and residual structure at the 64-qubit headline
(D=8 features, B=8 bits), from the raw data and trained run directories.

Panels (files named exactly as included by the manuscript):
  rho_matrix_data.pdf          train-split raw-data Pearson  (top left)
  rho_matrix_model_psck.pdf    PSCK-MoIQP model Pearson      (top middle)
  rho_matrix_model_lwmmd.pdf   LW-MMD baseline Pearson       (top right)
  rho_diff_psck.pdf            PSCK - data                   (bottom left)
  rho_diff_lwmmd.pdf           LW - data                     (bottom middle)
  rho_diff_advantage.pdf       |Delta_LW| - |Delta_PSCK|     (bottom right)
  fig6_combined.pdf            all six, for quick inspection

Model Pearson matrices are computed EXACTLY as in the paper: Van den Nest
Monte Carlo estimates of the weight-<=2 Z-correlators of the trained model,
pushed through the closed-form level-Pearson map (bit expectations
E[b] = (1-<Z>)/2, pair expectations E[bb'] = (1-<Z_a>-<Z_b>+<Z_aZ_b>)/4,
S_f = sum_k 2^{B-1-k} b_{f,k}). The data matrix is the raw Pearson of the
TRAINING split (split-correct; edges never touch the model side here).

Accepted model-artifact formats (auto-detected in the run dir):
  (a) MoIQP components: graph.json + params_0.npy ... params_{L-1}.npy
      (per-component angle vectors over the shared gate list; vectors longer
      than the gate list are treated as [trainable | frozen] and used whole)
  (b) compiled degree-3 object: hcgt_deg3_gates.npy + hcgt_deg3_params.npy
      (single VdN on the n+a compiled register; marginal equals the mixture
      by the equivalence theorem)

Usage:
  PYTHONPATH=src python scripts/plot_fig6_rho_matrices.py \
      --psck-run <run_dir> [--lw-run <run_dir>] \
      [--data data/cal_shower_img_8q.npy] [--split data/split_indices.npz] \
      [--bits 8] [-M 200000] [-o figures_fig6]

Without --lw-run the LW column is skipped (four panels). Monte Carlo default
M=2e5 matches the paper's evaluation protocol (correlator noise ~2e-3).
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import mplhep as hep
    hep.style.use("ROOT")
except Exception:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 14, "axes.labelsize": 16,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "axes.linewidth": 1.2,
        "figure.facecolor": "white", "savefig.bbox": "tight",
        "axes.grid": False})


# ── model correlators ────────────────────────────────────────────────────────
def load_model(run_dir):
    """Return (gate_lists, param_vectors, n_register) in a unified form:
    a list of (gates, params) 'branches' whose correlators are averaged."""
    g3, p3 = (os.path.join(run_dir, "hcgt_deg3_gates.npy"),
              os.path.join(run_dir, "hcgt_deg3_params.npy"))
    if os.path.exists(g3) and os.path.exists(p3):
        gates = [list(g) for g in np.load(g3, allow_pickle=True)]
        params = np.load(p3)
        n_reg = max(max(g) for g in gates) + 1
        return [(gates, params)], n_reg, f"compiled deg-3 ({len(gates)} gates)"
    gj = os.path.join(run_dir, "graph.json")
    pf = sorted(glob.glob(os.path.join(run_dir, "params_*.npy")))
    if os.path.exists(gj) and pf:
        gates = [list(g) for g in json.load(open(gj))]
        comps = [np.load(f) for f in pf]
        n_reg = max(max(g) for g in gates) + 1
        assert all(len(c) >= len(gates) for c in comps), \
            "component vector shorter than gate list"
        for c in comps:
            tail = c[len(gates):]
            assert tail.size == 0 or np.max(np.abs(tail)) < 1e-12, (
                "component vectors carry non-zero frozen-gate angles beyond "
                "graph.json; use the compiled deg-3 artifact "
                "(hcgt_deg3_gates/params) instead - slicing would drop them")
        comps = [c[:len(gates)] for c in comps]
        return [(gates, c) for c in comps], n_reg, \
            f"MoIQP L={len(comps)} ({len(gates)} gates)"
    sys.exit(f"no recognized model artifact in {run_dir}")


def vdn_correlators(branches, n_reg, betas, M, seed=0):
    """Mixture-averaged <Z_beta> for every beta, by Van den Nest MC."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=(M, n_reg), dtype=np.int8)
    out = np.zeros(len(betas))
    for gates, params in branches:
        xi = np.empty((len(gates), M))
        for i, G in enumerate(gates):
            xi[i] = 1.0 - 2.0 * (y[:, G].sum(axis=1) & 1)
        gsets = [frozenset(G) for G in gates]
        for j, beta in enumerate(betas):
            b = set(beta)
            act = [i for i, G in enumerate(gsets) if len(G & b) & 1]
            out[j] += 1.0 if not act else float(
                np.cos(2.0 * (params[act] @ xi[act])).mean())
    return out / len(branches)


def model_level_pearson(run_dir, D, B, M, seed=0):
    branches, n_reg, desc = load_model(run_dir)
    print(f"  [{os.path.basename(run_dir)}] {desc}, register {n_reg}, M={M}")
    n = D * B
    betas = [(q,) for q in range(n)]
    pair_idx = {}
    for f in range(D):
        for g in range(f, D):
            for k in range(B):
                for m in range(B):
                    qa, qb = f * B + k, g * B + m
                    if qa < qb and (qa, qb) not in pair_idx:
                        pair_idx[(qa, qb)] = len(betas)
                        betas.append((qa, qb))
    z = vdn_correlators(branches, n_reg, betas, M, seed)
    z1 = z[:n]
    Eb = (1.0 - z1) / 2.0
    w = (1 << np.arange(B - 1, -1, -1)).astype(float)

    def Ebb(qa, qb):
        if qa == qb:
            return Eb[qa]
        a_, b_ = min(qa, qb), max(qa, qb)
        return (1 - z1[a_] - z1[b_] + z[pair_idx[(a_, b_)]]) / 4.0

    mu = np.array([w @ Eb[f * B:(f + 1) * B] for f in range(D)])
    ES = np.zeros((D, D))
    for f in range(D):
        for g in range(f, D):
            s = sum(w[k] * w[m] * Ebb(f * B + k, g * B + m)
                    for k in range(B) for m in range(B))
            ES[f, g] = ES[g, f] = s
    var = np.diag(ES) - mu ** 2
    rho = (ES - np.outer(mu, mu)) / np.sqrt(np.outer(var, var))
    np.fill_diagonal(rho, 1.0)
    return rho


# ── panels ───────────────────────────────────────────────────────────────────
def panel(M, D, out, name, vlim, cmap="RdBu_r", cbar_label=r"$\rho_{fg}$"):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(M, vmin=-vlim, vmax=vlim, cmap=cmap, origin="lower")
    ax.set_xlabel("calorimeter cell")
    ax.set_ylabel("calorimeter cell")
    ax.set_xticks(range(D)); ax.set_yticks(range(D))
    ax.minorticks_off()
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(cbar_label)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=220)
    plt.close(fig)
    print(f"  [fig] {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psck-run", required=True)
    ap.add_argument("--lw-run", default=None)
    ap.add_argument("--data", default="data/cal_shower_img_8q.npy")
    ap.add_argument("--split", default="data/split_indices.npz")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("-M", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", default="figures_fig6")
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    raw = np.load(args.data)
    D, B = raw.shape[1], args.bits
    tr = np.load(args.split)["train_idx"]
    rho_dat = np.corrcoef(raw[tr].T)
    iu = np.triu_indices(D, 1)

    rho_psck = model_level_pearson(args.psck_run, D, B, args.M, args.seed)
    print(f"  MAErho(PSCK, train raw) = "
          f"{np.mean(np.abs(rho_psck[iu] - rho_dat[iu])):.4f}")
    rho_lw = None
    if args.lw_run:
        rho_lw = model_level_pearson(args.lw_run, D, B, args.M, args.seed)
        print(f"  MAErho(LW,   train raw) = "
              f"{np.mean(np.abs(rho_lw[iu] - rho_dat[iu])):.4f}")

    panel(rho_dat, D, args.output, "rho_matrix_data", 1.0)
    panel(rho_psck, D, args.output, "rho_matrix_model_psck", 1.0)
    panel(rho_psck - rho_dat, D, args.output, "rho_diff_psck", 0.3,
          cbar_label=r"$\Delta\rho_{fg}$")
    if rho_lw is not None:
        panel(rho_lw, D, args.output, "rho_matrix_model_lwmmd", 1.0)
        panel(rho_lw - rho_dat, D, args.output, "rho_diff_lwmmd", 0.3,
              cbar_label=r"$\Delta\rho_{fg}$")
        adv = np.abs(rho_lw - rho_dat) - np.abs(rho_psck - rho_dat)
        np.fill_diagonal(adv, 0.0)
        panel(adv, D, args.output, "rho_diff_advantage", 0.3,
              cbar_label=r"$|\Delta_{\mathrm{LW}}| - |\Delta_{\mathrm{PSCK}}|$")

    # combined quick-look
    mats = [(rho_dat, 1.0, "data"), (rho_psck, 1.0, "PSCK"),
            (rho_psck - rho_dat, 0.3, "PSCK $-$ data")]
    if rho_lw is not None:
        mats = [(rho_dat, 1.0, "data"), (rho_psck, 1.0, "PSCK"),
                (rho_lw, 1.0, "LW"),
                (rho_psck - rho_dat, 0.3, "PSCK $-$ data"),
                (rho_lw - rho_dat, 0.3, "LW $-$ data"),
                (np.abs(rho_lw - rho_dat) - np.abs(rho_psck - rho_dat), 0.3,
                 "advantage")]
    ncol = 3
    nrow = int(np.ceil(len(mats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.5 * nrow),
                             constrained_layout=True)
    for ax, (Mx, vl, ttl) in zip(np.atleast_1d(axes).flat, mats):
        im = ax.imshow(Mx, vmin=-vl, vmax=vl, cmap="RdBu_r", origin="lower")
        ax.set_title(ttl, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in np.atleast_1d(axes).flat[len(mats):]:
        ax.axis("off")
    fig.savefig(os.path.join(args.output, "fig6_combined.pdf"))
    plt.close(fig)
    print(f"  [fig] fig6_combined\n[done] {args.output}/")


if __name__ == "__main__":
    main()
