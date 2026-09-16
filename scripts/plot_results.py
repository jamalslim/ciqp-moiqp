#!/usr/bin/env python
"""Plot results from a trained run directory.

Produces INDIVIDUAL plots (one file per quantity), not a large multi-panel
figure. Each plot is saved as both .png and .pdf in a `plots/` subfolder of
the run directory (or a custom output path).

Plots produced:
    loss.{png,pdf}          — total-loss curve vs epoch, log scale
    z_err.{png,pdf}          — |Δz| vs epoch
    rho_err.{png,pdf}        — |Δρ| vs epoch, with encoding-floor line
    rho_matrix_data.{png,pdf}  — data (ground truth) ρ matrix
    rho_matrix_model.{png,pdf} — model ρ matrix
    rho_scatter.{png,pdf}    — pairwise ρ_model vs ρ_data
    ciqp_weights.{png,pdf}   — cIQP gate-weight histogram (if verification.json exists)

mplhep is used if available (ROOT/CMS/ATLAS style). If not, a hand-tuned
matplotlib rcParams fallback reproduces the same look.

Usage:
    python scripts/plot_results.py outputs/psck_B2_binary_L4_ep500_seed42/
    python scripts/plot_results.py <run_dir> --style CMS
    python scripts/plot_results.py <run_dir> --out-dir my_plots/

To overlay multiple runs on a single curve (comparison mode), use
scripts/plot_compare.py instead.
"""

import os, sys, argparse, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from psck_mmd.data import load_calorimeter
from psck_mmd.graph import build_er_graph, build_complete_graph, precompute_active_lists
from psck_mmd.feature_obs import enumerate_feature_observables
from psck_mmd.correlator import ParityCache, sample_latents, forward_batch
from psck_mmd.corr_jacobian import pearson_value_and_jacobian
from psck_mmd.plotting import set_root_style, ROOT_COLORS, _add_minor_ticks

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSS_DISPLAY_SHORT = {
    "corrmse": "CorrMSE",
    "lw_mmd":  "LW-MMD",
    "psck":    "PSCK-MMD",
}


def _save_both(fig, out_dir: str, stem: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_dir}/{stem}.{{png,pdf}}")


# ─────────────────────────────────────────────────────────────────────────────
# Individual plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curve(history, metrics, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    color = ROOT_COLORS.get(metrics["loss"], ROOT_COLORS["neutral"])
    ax.plot(history["loss"], color=color, linewidth=1.8,
             label=LOSS_DISPLAY_SHORT.get(metrics["loss"], metrics["loss"]))
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")

    _add_minor_ticks(ax, logy=True)
    ax.legend(loc="upper right", fontsize=12)
    _save_both(fig, out_dir, "loss")


def plot_z_err(history, metrics, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    color = ROOT_COLORS.get(metrics["loss"], ROOT_COLORS["neutral"])
    ax.plot(history["z_err"], color=color, linewidth=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\langle|\Delta\langle Z_S\rangle|\rangle$")

    _add_minor_ticks(ax)
    _save_both(fig, out_dir, "z_err")


def plot_rho_err(history, metrics, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    color = ROOT_COLORS.get(metrics["loss"], ROOT_COLORS["neutral"])
    ax.plot(history["rho_err"], color=color, linewidth=1.8,
             label=r"$\langle|\Delta\rho|\rangle$")
    ax.axhline(metrics["enc_fid"], color="black", linestyle="--", linewidth=1.3,
                label=f"encoding floor = {metrics['enc_fid']:.3f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\langle|\Delta\rho|\rangle$")

    ax.legend(loc="upper right", fontsize=12)
    _add_minor_ticks(ax)
    _save_both(fig, out_dir, "rho_err")


def plot_rho_matrix(mat: np.ndarray, title: str, out_dir: str, stem: str,
                     vmax=None):
    D = mat.shape[0]
    if vmax is None:
        iu = np.triu_indices(D, k=1)
        vmax = max(abs(mat[iu]).max(), 0.3)
        vmax = 1

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    interpolation="nearest")
    ax.set_xticks(range(D)); ax.set_yticks(range(D))


    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(direction="in", length=3)
    _save_both(fig, out_dir, stem)


def plot_rho_scatter(rho_data, rho_model, metrics, out_dir):
    D = rho_data.shape[0]
    iu = np.triu_indices(D, k=1)
    color = ROOT_COLORS.get(metrics["loss"], ROOT_COLORS["neutral"])
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(rho_data[iu], rho_model[iu], s=60, color=color,
                alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3,
                label=f"{LOSS_DISPLAY_SHORT.get(metrics['loss'], metrics['loss'])} "
                        f"(r = {metrics['rho_r']:.4f})")
    lim = max(abs(rho_data[iu]).max(), 0.3) + 0.15
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1.3, alpha=0.6, zorder=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\rho_{ij}$ (data)")
    ax.set_ylabel(r"$\rho_{ij}$ (model)")

    ax.legend(loc="upper left", fontsize=11)
    _add_minor_ticks(ax)
    _save_both(fig, out_dir, "rho_scatter")


def plot_ciqp_weights(verification, out_dir):
    wh = verification["weight_histogram"]
    wh_int = {int(k): v for k, v in wh.items()}
    weights = sorted(wh_int.keys())
    counts = [wh_int[w] for w in weights]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(weights, counts, 0.62, color="#8e44ad",
                   edgecolor="black", linewidth=1.3)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, c * 1.04, str(c),
                 ha="center", va="bottom", fontsize=11)
    ax.set_xlabel(r"gate weight $|g|$")
    ax.set_ylabel("count")
    ax.set_xticks(weights)

    _add_minor_ticks(ax)
    _save_both(fig, out_dir, "ciqp_weights")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def rebuild_graph(cfg):
    n = cfg["D"] * cfg["B"]
    if cfg.get("graph", "erdos_renyi") == "erdos_renyi":
        return build_er_graph(n, avg_deg=cfg.get("avg_deg", 6.0),
                                seed=cfg["seed"] + 1)
    return build_complete_graph(n)


def recover_rho_model(run_dir, cfg, M_eval=50_000):
    D, B, L = cfg["D"], cfg["B"], cfg["L"]
    n = D * B
    binary, raw = load_calorimeter(
        "data/cal_shower_img_8q.npy", bits=B, encoding=cfg.get("encoding", "binary"),
    )
    obs, _ = enumerate_feature_observables(D, B, max_weight=2)
    gates, q2g = rebuild_graph(cfg)
    active = precompute_active_lists(obs, gates, q2g)

    params_list = [np.load(os.path.join(run_dir, f"params_{ell}.npy"))
                    for ell in range(L)]
    rng = np.random.default_rng(cfg["seed"] + 77777)
    z_mix = np.zeros(len(obs))
    for ell in range(L):
        lat = sample_latents(n, M_eval, rng, antithetic=True)
        z_l, _ = forward_batch(params_list[ell],
                                 ParityCache(lat, gates), active)
        z_mix += z_l / L
    rho, _ = pearson_value_and_jacobian(z_mix, obs, D, B)
    corr_raw = np.corrcoef(raw.T)
    return rho, corr_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str, help="path to a run folder")
    ap.add_argument("--out-dir", type=str, default=None,
                     help="where to save plots (default: <run_dir>/plots/)")
    ap.add_argument("--style", type=str, default="ROOT",
                     choices=["ROOT", "CMS", "ATLAS", "LHCb", "ALICE"])
    ap.add_argument("--eval-M", type=int, default=50_000)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.run_dir, "plots")

    set_root_style(args.style)

    with open(os.path.join(args.run_dir, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(args.run_dir, "history.json")) as f:
        history = json.load(f)

    print(f"\n  plotting {args.run_dir}  →  {out_dir}  (style = {args.style})\n")
    plot_loss_curve(history, metrics, out_dir)
    plot_z_err(history, metrics, out_dir)
    plot_rho_err(history, metrics, out_dir)

    # Recover ρ model and data from saved params + raw data
    rho_model, corr_raw = recover_rho_model(args.run_dir, metrics,
                                                M_eval=args.eval_M)
    plot_rho_matrix(corr_raw, "ρ data (ground truth)",
                     out_dir, "rho_matrix_data")
    plot_rho_matrix(rho_model,
                     f"ρ model — {metrics.get('loss_display', metrics['loss'])}",
                     out_dir, "rho_matrix_model")
    plot_rho_scatter(corr_raw, rho_model, metrics, out_dir)

    # cIQP weights, if available
    ver_path = os.path.join(args.run_dir, "verification.json")
    if os.path.exists(ver_path):
        with open(ver_path) as f:
            ver = json.load(f)
        plot_ciqp_weights(ver, out_dir)
    else:
        print(f"  (no verification.json — run scripts/verify_ciqp.py first "
              f"for cIQP weights plot)")

    print()


if __name__ == "__main__":
    main()
