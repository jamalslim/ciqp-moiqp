#!/usr/bin/env python
"""Overlay training curves and final metrics from multiple run directories.

Each plot contains ONE quantity with one curve or bar per run, color-coded.
No multi-panel figures — individual files saved to --out-dir.

Plots produced:
    compare_loss.{png,pdf}         — loss curves overlaid (log y)
    compare_z_err.{png,pdf}        — |Δz| vs epoch
    compare_rho_err.{png,pdf}      — |Δρ| vs epoch, with first run's enc floor
    compare_rho_mae_bar.{png,pdf}  — final ρ MAE bar chart
    compare_z_mae_bar.{png,pdf}    — final z MAE bar chart
    compare_rho_r_bar.{png,pdf}    — Pearson r bar chart
    compare_rho_scatter.{png,pdf}  — ρ_model vs ρ_data, all runs overlaid

Usage:
    python scripts/plot_compare.py \\
        outputs/corrmse_B2_binary_L4_ep500_seed42 \\
        outputs/lw_mmd_B2_binary_L4_ep500_seed42 \\
        outputs/psck_B2_binary_L4_ep500_seed42 \\
        --out-dir plots_B2_compare --style ROOT

Runs are labeled by their loss_display metric; use --labels to override.
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
from matplotlib.ticker import LogLocator

LOSS_DISPLAY_SHORT = {
    "corrmse": "CorrMSE", "lw_mmd": "LW-MMD", "psck": "PSCK-MMD",
}

FALLBACK_COLORS = ["#1f3a93", "#e67e22", "#c0392b", "#16a085",
                   "#8e44ad", "#2c3e50", "#d35400", "#27ae60"]


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
    return rho, np.corrcoef(raw.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+",
                     help="one or more trained run folders")
    ap.add_argument("--out-dir", type=str, required=True,
                     help="folder where comparison plots are written")
    ap.add_argument("--labels", type=str, default=None,
                     help="comma-separated labels (one per run). "
                            "Default: metrics.loss_display.")
    ap.add_argument("--style", type=str, default="ROOT",
                     choices=["ROOT", "CMS", "ATLAS", "LHCb", "ALICE"])
    ap.add_argument("--eval-M", type=int, default=50_000)
    args = ap.parse_args()

    set_root_style(args.style)

    runs = []
    for rd in args.run_dirs:
        with open(os.path.join(rd, "metrics.json")) as f:
            m = json.load(f)
        with open(os.path.join(rd, "history.json")) as f:
            h = json.load(f)
        runs.append({"dir": rd, "metrics": m, "history": h})

    # Labels
    if args.labels:
        labels = [x.strip() for x in args.labels.split(",")]
        if len(labels) != len(runs):
            raise SystemExit(
                f"--labels has {len(labels)} entries but there are {len(runs)} runs"
            )
    else:
        labels = [r["metrics"].get("loss_display", r["metrics"]["loss"])
                   for r in runs]

    # Colors: prefer the per-loss palette, fall back to a sequence
    colors = []
    used_loss = set()
    for i, r in enumerate(runs):
        lk = r["metrics"]["loss"]
        if lk in ROOT_COLORS and lk not in used_loss:
            colors.append(ROOT_COLORS[lk]); used_loss.add(lk)
        else:
            colors.append(FALLBACK_COLORS[i % len(FALLBACK_COLORS)])

    print(f"\n  overlaying {len(runs)} runs → {args.out_dir}  (style={args.style})\n")

    # ── Loss curves ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for r, lab, c in zip(runs, labels, colors):
        ax.plot(r["history"]["loss"], color=c, linewidth=1.8,
                 alpha=0.9, label=lab)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_yscale("log")
    _add_minor_ticks(ax, logy=True)
    ax.legend(fontsize=11, loc="upper right")
    _save_both(fig, args.out_dir, "compare_loss")

    # ── z MAE curves ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for r, lab, c in zip(runs, labels, colors):
        ax.plot(r["history"]["z_err"], color=c, linewidth=1.8,
                 alpha=0.9, label=lab)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\langle|\Delta\langle Z_S\rangle|\rangle$")

    _add_minor_ticks(ax)
    ax.legend(fontsize=11, loc="upper right")
    _save_both(fig, args.out_dir, "compare_z_err")

    # ── rho MAE curves ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for r, lab, c in zip(runs, labels, colors):
        ax.plot(r["history"]["rho_err"], color=c, linewidth=1.8,
                 alpha=0.9, label=lab)
    ax.axhline(runs[0]["metrics"]["enc_fid"], color="black",
                linestyle="--", linewidth=1.3,
                label=f"enc. floor = {runs[0]['metrics']['enc_fid']:.3f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\langle|\Delta\rho|\rangle$")

    _add_minor_ticks(ax)
    ax.legend(fontsize=11, loc="upper right")
    _save_both(fig, args.out_dir, "compare_rho_err")

    # ── Bar: final rho MAE ──
    x = np.arange(len(runs))
    rho_maes = [r["metrics"]["rho_mae"] for r in runs]
    enc_fid = runs[0]["metrics"]["enc_fid"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, rho_maes, 0.62, color=colors,
                    edgecolor="black", linewidth=1.3)
    for b, v in zip(bars, rho_maes):
        ax.text(b.get_x() + b.get_width()/2, v * 1.07,
                 f"{v:.4f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax.axhline(enc_fid, color="black", linestyle="--", linewidth=1.2,
                label=f"enc. floor = {enc_fid:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11, rotation=20)
    ax.set_ylabel(r"$\langle|\Delta\rho|\rangle$")
    ax.set_yscale("log")
    ymin = max(min(rho_maes) * 0.4, 1e-4)
    ax.set_ylim(ymin, max(max(rho_maes), enc_fid) * 3.0)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=10)
    )

    ax.legend(fontsize=10, loc="upper right")
    _save_both(fig, args.out_dir, "compare_rho_mae_bar")

    # ── Bar: final z MAE ──
    z_maes = [r["metrics"]["z_mae"] for r in runs]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, z_maes, 0.62, color=colors, edgecolor="black", linewidth=1.3)
    for b, v in zip(bars, z_maes):
        ax.text(b.get_x() + b.get_width()/2, v * 1.05,
                 f"{v:.4f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11, rotation=20)
    ax.set_ylabel(r"$\langle|\Delta\langle Z_S\rangle|\rangle$")
    _add_minor_ticks(ax)

    _save_both(fig, args.out_dir, "compare_z_mae_bar")

    # ── Bar: rho_r ──
    rs = [r["metrics"]["rho_r"] for r in runs]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, rs, 0.62, color=colors, edgecolor="black", linewidth=1.3)
    for b, v in zip(bars, rs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.015,
                 f"{v:.4f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11, rotation=20)
    ax.set_ylabel(r"Pearson $r$ (pairwise $\rho$)")
    ax.set_ylim(0, 1.05)
    _add_minor_ticks(ax)

    _save_both(fig, args.out_dir, "compare_rho_r_bar")

    # ── ρ scatter overlay ──
    # Use first run to recover ground-truth ρ
    rho_data = None
    rho_models = []
    for r in runs:
        rho_m, rho_d = recover_rho_model(r["dir"], r["metrics"],
                                            M_eval=args.eval_M)
        rho_models.append(rho_m)
        if rho_data is None:
            rho_data = rho_d
    D = rho_data.shape[0]
    iu = np.triu_indices(D, k=1)
    fig, ax = plt.subplots(figsize=(7, 7))
    markers = ["o", "s", "^", "v", "D", "P", "X"]
    for r, lab, c, mk, rho_m in zip(runs, labels, colors, markers, rho_models):
        ax.scatter(rho_data[iu], rho_m[iu], s=55, color=c, alpha=0.80,
                     edgecolor="white", linewidth=0.5, marker=mk, zorder=3,
                     label=f"{lab} (r={r['metrics']['rho_r']:.3f})")
    lim = max(abs(rho_data[iu]).max(), 0.3) + 0.15
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1.3, alpha=0.6, zorder=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel(r"$\rho_{ij}$ (data)"); ax.set_ylabel(r"$\rho_{ij}$ (model)")
    _add_minor_ticks(ax)

    ax.legend(loc="upper left", fontsize=10)
    _save_both(fig, args.out_dir, "compare_rho_scatter")

    print()


if __name__ == "__main__":
    main()
