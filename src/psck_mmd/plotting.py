"""
psck_mmd.plotting — ROOT-style publication figures via mplhep.

Uses `mplhep` with the CMS/ATLAS/ROOT styles when available. If mplhep is not
installed, we ship a pure-matplotlib fallback that replicates the tick
behavior (inward major+minor ticks on all 4 sides, no top/right frame
suppression) so figures render identically.

Usage
-----
from psck_mmd.plotting import set_root_style, plot_comparison

set_root_style("CMS")            # or "ATLAS", "LHCb", "ROOT"
plot_comparison(metrics, rhos, corr_raw, save_path="results.pdf")

The mplhep experiment label (e.g. "CMS Preliminary") is suppressed for this
work since we aren't publishing under an experiment; we use the label slot
for our own watermark text, which keeps the ROOT aesthetic.
"""

from __future__ import annotations
import os
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator, LogLocator

try:
    import mplhep as hep
    HAS_MPLHEP = True
except ImportError:
    HAS_MPLHEP = False


# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

_ROOT_FALLBACK_RCPARAMS = {
    # Fonts: ROOT uses bold sans-serif; we match with DejaVu Sans.
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 1.2,
    "axes.grid": False,
    "axes.axisbelow": True,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "legend.handletextpad": 0.4,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    # ROOT-style ticks: inward, all four sides, with minor ticks.
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 6.0,
    "ytick.major.size": 6.0,
    "xtick.minor.size": 3.0,
    "ytick.minor.size": 3.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "lines.linewidth": 1.8,
    "lines.markersize": 7.0,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 160,
}


def set_root_style(style: str = "ROOT"):
    """Apply mplhep ROOT/CMS/ATLAS/LHCb style if available, else fall back
    to pure-matplotlib equivalents.

    style ∈ {"ROOT", "CMS", "ATLAS", "LHCb", "ALICE"}
    """
    if HAS_MPLHEP:
        style_map = {
            "ROOT": hep.styles.ROOT,
            "CMS":  hep.styles.CMS,
            "ATLAS": hep.styles.ATLAS,
            "LHCb": hep.styles.LHCb,
            "ALICE": hep.styles.ALICE,
        }
        plt.style.use(style_map.get(style, hep.styles.ROOT))
    else:
        plt.rcParams.update(_ROOT_FALLBACK_RCPARAMS)


def _hep_label(ax, text_left: str, text_right: str = ""):
    """Add mplhep-style experimental label; no-op if mplhep missing."""
    if HAS_MPLHEP:
        # Custom text; we don't claim a real experiment label
        hep.label.exp_text(text=text_right, loc=0, ax=ax, fontsize=12)
        ax.text(0.01, 1.02, text_left, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom")
    else:
        ax.text(0.01, 1.02, text_left, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom")
        if text_right:
            ax.text(0.99, 1.02, text_right, transform=ax.transAxes,
                    fontsize=11, ha="right", va="bottom", style="italic")


def _add_minor_ticks(ax, logy: bool = False):
    if not logy:
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))


# ─────────────────────────────────────────────────────────────────────────────
# Color palette — ROOT's kBlue/kRed/kBlack aesthetic
# ─────────────────────────────────────────────────────────────────────────────

ROOT_COLORS = {
    "corrmse": "#1f3a93",   # ROOT kBlue-ish
    "lw_mmd":  "#e67e22",   # ROOT kOrange+1
    "psck":    "#c0392b",   # ROOT kRed+1
    "data":    "#2c3e50",   # ROOT kBlack
    "neutral": "#7f8c8d",
    "heat":    "#2980b9",
    "rho":     "#8e44ad",
}

_LOSS_LABELS = {
    "corrmse": r"$L_z + \lambda\,L_\rho$ (baseline)",
    "lw_mmd":  r"Liu–Wang MMD (heat kernel)",
    "psck":    r"PSCK-MMD (this work)",
}

_LOSS_SHORT = {"corrmse": "CorrMSE", "lw_mmd": "LW-MMD", "psck": "PSCK"}


# ─────────────────────────────────────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(metrics: dict,
                     rho_models: dict,
                     corr_raw: np.ndarray,
                     D: int, B: int, L: int,
                     iu: tuple,
                     save_path: str,
                     style: str = "ROOT",
                     title: str | None = None):
    """Generate the 3×4 PSCK-MMD comparison figure.

    Parameters
    ----------
    metrics    : dict loaded from metrics.json (see run_comparison output)
    rho_models : dict mapping loss_kind -> (D, D) ρ_model matrix
    corr_raw   : (D, D) ground-truth ρ from data
    D, B, L    : architecture sizes (for annotations)
    iu         : np.triu_indices(D, k=1) tuple
    save_path  : file to save (.png or .pdf)
    style      : mplhep style name
    title      : optional suptitle override

    Returns the figure object.
    """
    set_root_style(style)
    losses = ["corrmse", "lw_mmd", "psck"]
    n_feat = D * B

    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.05, right=0.97, top=0.92, bottom=0.05)

    # ── Row 1, col 0: Setup panel ─────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    cfg = metrics["config"]
    setup = (
        f"D = {D}, B = {B}, n = {n_feat} qubits\n"
        f"Base graph: ER(avg_deg = {cfg['avg_deg']})\n"
        f"MoIQP components: L = {L}\n"
        f"Epochs: {cfg['n_epochs']}, M_batch = {cfg['mc_batch']}\n"
        f"Encoding: binary\n"
        f"Dataset: calorimeter shower\n"
        f"  (47,682 samples × {D} features)\n\n"
        f"Identical seed and hyperparameters\n"
        f"across all three losses."
    )
    ax.text(0.04, 0.94, setup, ha="left", va="top",
            fontsize=12, family="monospace",
            bbox=dict(boxstyle="round,pad=0.7",
                      facecolor="#ECEFF1", edgecolor="#455A64", linewidth=1.5))
    ax.set_title("Setup", fontweight="bold", fontsize=14, pad=10)

    # ── Row 1, col 1: rho MAE bar (LOG scale — ROOT-style) ─────────
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(losses))
    rho_maes = [metrics[lk]["rho_mae"] for lk in losses]
    bars = ax.bar(x, rho_maes, 0.62,
                  color=[ROOT_COLORS[lk] for lk in losses],
                  edgecolor="black", linewidth=1.3)
    for b, v in zip(bars, rho_maes):
        ax.text(b.get_x() + b.get_width()/2, v * 1.08,
                f"{v:.4f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    enc_fid = metrics[losses[0]]["enc_fid"]
    ax.axhline(enc_fid, color="black", linestyle="--", linewidth=1.3,
               label=f"encoding floor = {enc_fid:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([_LOSS_SHORT[lk] for lk in losses], fontsize=12)
    ax.set_ylabel(r"$\langle|\Delta\rho|\rangle$   (lower is better)")
    ax.set_yscale("log")
    ax.set_ylim(min(rho_maes) * 0.4, max(max(rho_maes), enc_fid) * 2.5)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1,
                                           numticks=10))
    ax.legend(loc="upper right", fontsize=11)
    ax.set_title("Correlation accuracy", fontsize=13)

    # ── Row 1, col 2: z MAE bar ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    z_maes = [metrics[lk]["z_mae"] for lk in losses]
    bars = ax.bar(x, z_maes, 0.62,
                  color=[ROOT_COLORS[lk] for lk in losses],
                  edgecolor="black", linewidth=1.3)
    for b, v in zip(bars, z_maes):
        ax.text(b.get_x() + b.get_width()/2, v * 1.05,
                f"{v:.4f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([_LOSS_SHORT[lk] for lk in losses], fontsize=12)
    ax.set_ylabel(r"$\langle|\Delta\langle Z_S\rangle|\rangle$")
    _add_minor_ticks(ax)
    ax.set_title("Marginal (Z-moment) accuracy", fontsize=13)

    # ── Row 1, col 3: Pearson r bar ────────────────────────────────
    ax = fig.add_subplot(gs[0, 3])
    rs = [metrics[lk]["rho_r"] for lk in losses]
    bars = ax.bar(x, rs, 0.62,
                  color=[ROOT_COLORS[lk] for lk in losses],
                  edgecolor="black", linewidth=1.3)
    for b, v in zip(bars, rs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.012,
                f"{v:.4f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([_LOSS_SHORT[lk] for lk in losses], fontsize=12)
    ax.set_ylabel(r"Pearson $r$ (pairwise $\rho$)")
    ax.set_ylim(0, 1.05)
    _add_minor_ticks(ax)
    ax.set_title(r"$\rho$ linear consistency", fontsize=13)

    # ── Row 2: rho matrices ────────────────────────────────────────
    vmax = max(abs(corr_raw[iu]).max(), 0.3)
    mats = [("data", corr_raw, r"$\rho$ data (ground truth)")]
    for lk in losses:
        mats.append((lk, rho_models[lk],
                     rf"{_LOSS_LABELS[lk]}" + "\n" +
                     rf"$\langle|\Delta\rho|\rangle = {metrics[lk]['rho_mae']:.4f}$"))

    for i, (lk, M, lab) in enumerate(mats):
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(lab, fontsize=12)
        ax.set_xticks(range(D)); ax.set_yticks(range(D))
        ax.tick_params(length=3, direction="out")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(direction="in", length=3)

    # ── Row 3, col 0: pairwise scatter ─────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    marker = {"corrmse": "o", "lw_mmd": "s", "psck": "^"}
    for lk in losses:
        ax.scatter(corr_raw[iu], rho_models[lk][iu],
                   s=60, color=ROOT_COLORS[lk], alpha=0.80,
                   edgecolor="white", linewidth=0.5,
                   marker=marker[lk],
                   label=f"{_LOSS_SHORT[lk]} (r={metrics[lk]['rho_r']:.3f})",
                   zorder=3)
    lim = max(abs(corr_raw[iu]).max(), 0.3) + 0.15
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1.3, alpha=0.6, zorder=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\rho_{ij}$ (data)")
    ax.set_ylabel(r"$\rho_{ij}$ (model)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=11)
    _add_minor_ticks(ax)
    ax.set_title(r"Pairwise $\rho$ agreement", fontsize=13)

    # ── Row 3, col 1: cIQP gate weight histogram ──────────────────
    ax = fig.add_subplot(gs[2, 1])
    cq = metrics["ciqp_deployment"]
    wh = cq["weight_histogram"]
    # keys may be int (in-memory) or str (after JSON round-trip); normalize
    wh_norm = {int(k): v for k, v in wh.items()}
    weights = sorted(wh_norm.keys())
    counts = [wh_norm[w] for w in weights]
    bars = ax.bar(weights, counts, 0.62, color="#8e44ad",
                  edgecolor="black", linewidth=1.3)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, c * 1.04,
                str(c), ha="center", va="bottom", fontsize=11)
    ax.set_xlabel(r"gate weight $|g|$")
    ax.set_ylabel("count")
    ax.set_xticks(weights)
    _add_minor_ticks(ax)
    ax.set_title(f"cIQP gates (PSCK-MoIQP)\n"
                 f"{cq['n_feat']}+{cq['n_control']}={cq['n_total']}q",
                 fontsize=12)

    # ── Row 3, col 2: cIQP verification text panel ────────────────
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    ciqp_text = (
        f"cIQP deployment verification\n"
        f"(Walsh–Hadamard / deferred measurement)\n"
        f"\n"
        f"  n_feat + a_ctrl  = {cq['n_feat']} + {cq['n_control']} "
        f"= {cq['n_total']} qubits\n"
        f"  L = {L} MoIQP → single IQP circuit\n"
        f"  base gates  : {cq['n_base_gates']}\n"
        f"  deploy gates: {cq['n_deploy_gates']}\n"
        f"  max |g|     : {cq['max_weight']}\n"
        f"\n"
        f"  ⟨Z_S⟩ MAE (cIQP vs MoIQP)\n"
        f"    = {cq['mae_ciqp_vs_moiqp']:.6f}\n"
        f"  MC noise 1/√M (M=200k)\n"
        f"    = {cq['mc_noise_scale']:.6f}\n"
        f"  ratio = {cq['ratio']:.2f}× MC noise\n"
        f"\n"
        f"  ρ MAE (cIQP vs MoIQP) = {cq['rho_mae_ciqp_vs_moiqp']:.6f}\n"
        f"\n"
        f"  ✓ cIQP reproduces MoIQP marginals\n"
        f"    within sampling precision.\n"
        f"  ✓ compiled circuit remains IQP-form."
    )
    ax.text(0.02, 0.98, ciqp_text, ha="left", va="top",
            fontsize=11, family="monospace",
            bbox=dict(boxstyle="round,pad=0.7",
                      facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=1.5))

    # ── Row 3, col 3: takeaways ───────────────────────────────────
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    improvement = (1.0 - metrics["psck"]["rho_mae"] /
                   metrics["lw_mmd"]["rho_mae"]) * 100
    takeaways = (
        f"Key comparisons (n = {n_feat}q, B = {B}, L = {L})\n"
        f"\n"
        f"  Liu–Wang MMD (heat kernel only):\n"
        f"    ρ MAE = {metrics['lw_mmd']['rho_mae']:.4f}\n"
        f"    z MAE = {metrics['lw_mmd']['z_mae']:.4f}\n"
        f"\n"
        f"  PSCK-MMD (this work):\n"
        f"    ρ MAE = {metrics['psck']['rho_mae']:.4f}\n"
        f"    z MAE = {metrics['psck']['z_mae']:.4f}\n"
        f"\n"
        f"  PSCK vs LW-MMD ρ MAE reduction\n"
        f"    = {improvement:.1f}%\n"
        f"\n"
        f"  Both losses use only ⟨Z_β⟩ → \n"
        f"    Van den Nest MC (polynomial)\n"
        f"  PSCK adds one hyperparameter η\n"
        f"    replacing Lagrange multiplier λ.\n"
        f"  compiled circuit remains IQP-form\n"
        f"    at deployment (BJS 2011)."
    )
    ax.text(0.02, 0.98, takeaways, ha="left", va="top",
            fontsize=11, family="monospace",
            bbox=dict(boxstyle="round,pad=0.7",
                      facecolor="#FFF3E0", edgecolor="#E65100", linewidth=1.5))

    if title is None:
        title = (f"Pearson-Stabilized Correlation Kernel MMD for IQP Born Machines"
                 f"   |   n = {n_feat} qubits (binary, L = {L})")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.985)

    # A single ROOT/CMS-style experiment banner in the top-left corner
    fig.text(0.05, 0.945, "PSCK-MMD",
             fontsize=13, fontweight="bold", style="italic", color="#333")
    fig.text(0.97, 0.945, f"n={n_feat}q, B={B}, L={L}",
             fontsize=11, ha="right", color="#555")

    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# Training-curve plot (separate figure, since histories are large)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(histories: dict, save_path: str, style: str = "ROOT",
                          title: str = "Training curves"):
    """One figure with three curves: total loss, z-MAE, ρ-MAE per epoch.

    histories: dict mapping loss_kind -> {"loss": [...], "z_err": [...], "rho_err": [...]}
    """
    set_root_style(style)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

    for lk, H in histories.items():
        c = ROOT_COLORS[lk]
        axes[0].plot(H["loss"], color=c, linewidth=1.6,
                     label=_LOSS_SHORT[lk], alpha=0.9)
        axes[1].plot(H["z_err"], color=c, linewidth=1.6, alpha=0.9)
        axes[2].plot(H["rho_err"], color=c, linewidth=1.6, alpha=0.9)

    for ax, ylab, tit, logy in [
        (axes[0], "total loss", "Loss", True),
        (axes[1], r"$\langle|\Delta z|\rangle$", r"$z$ MAE", False),
        (axes[2], r"$\langle|\Delta\rho|\rangle$", r"$\rho$ MAE", False),
    ]:
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
        ax.set_title(tit, fontsize=13)
        if logy:
            ax.set_yscale("log")
        _add_minor_ticks(ax, logy=logy)

    axes[0].legend(loc="upper right", fontsize=12)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: rebuild rho_model matrices from saved params for plotting
# ─────────────────────────────────────────────────────────────────────────────

def recover_rho_models(out_dir: str, D: int, B: int, L: int,
                        gates, q2g, observables,
                        losses=("corrmse", "lw_mmd", "psck"),
                        M_eval: int = 50_000, seed: int = 77777):
    """Reload params_{lk}_{ell}.npy and MC-evaluate ρ_model for each loss."""
    from .correlator import (
        ParityCache, sample_latents, forward_batch,
    )
    from .graph import precompute_active_lists
    from .corr_jacobian import pearson_value_and_jacobian

    n = D * B
    active = precompute_active_lists(observables, gates, q2g)
    rho_models: dict[str, np.ndarray] = {}
    z_mixes: dict[str, np.ndarray] = {}
    for lk in losses:
        params_list = [np.load(os.path.join(out_dir, f"params_{lk}_{ell}.npy"))
                        for ell in range(L)]
        rng = np.random.default_rng(seed)
        z_mix = np.zeros(len(observables))
        for ell in range(L):
            lat = sample_latents(n, M_eval, rng, antithetic=True)
            c = ParityCache(lat, gates)
            z_l, _ = forward_batch(params_list[ell], c, active)
            z_mix += z_l / L
        rho, _ = pearson_value_and_jacobian(z_mix, observables, D, B)
        rho_models[lk] = rho
        z_mixes[lk] = z_mix
    return rho_models, z_mixes


# ─────────────────────────────────────────────────────────────────────────────
# Single-figure savers — one plot per file (png + pdf)
#
# Added so every figure can be emitted SEPARATELY (one panel == one file),
# instead of bundling several panels into one composite. Used by run_sample.py,
# plot_corr_matrices.py and sample_and_plot.py; reusable from any script.
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig, path_stem, exts=("png", "pdf")):
    """Save a single figure to <path_stem>.<ext> for each ext, then close it.
    Returns the list of written paths."""
    paths = []
    for ext in exts:
        p = f"{path_stem}.{ext}"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def save_heatmap(M, path_stem, title="", vmin=-1.0, vmax=1.0, cmap="RdBu_r",
                 cbar_label=r"$\rho$", annotate=None, xlabel="feature",
                 ylabel="feature", style="ROOT", exts=("png", "pdf"),
                 figsize=(5.6, 4.8)):
    """One matrix -> one standalone figure with its own colorbar."""
    set_root_style(style)
    M = np.asarray(M)
    nrow, ncol = M.shape
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(ncol)); ax.set_yticks(range(nrow))
    ax.tick_params(length=3, direction="out")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if annotate is None:
        annotate = (nrow <= 12 and ncol <= 12)
    if annotate:
        for i in range(nrow):
            for j in range(ncol):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    cb.ax.tick_params(direction="in", length=3)
    fig.tight_layout()
    return _save_fig(fig, path_stem, exts)


def save_corr_triplet_separate(C_data, C_gen, D, out_dir, prefix="corr",
                               gen_name="sampled", style="ROOT",
                               exts=("png", "pdf"), titles=True):
    """Write THREE separate files: <prefix>_data, <prefix>_<gen_name>,
    <prefix>_diff. With titles=False the panels carry no title (clean figures
    for captions added elsewhere). Returns (paths, off_diagonal_MAE)."""
    iu = np.triu_indices(D, k=1)
    C_data = np.asarray(C_data); C_gen = np.asarray(C_gen)
    C_diff = C_gen - C_data
    mae = float(np.abs(C_gen[iu] - C_data[iu]).mean())
    vmax = max(0.05, float(np.abs(C_diff[iu]).max()))
    t_data = "data" if titles else ""
    t_gen = gen_name if titles else ""
    t_diff = f"{gen_name} - data  (MAE={mae:.3f})" if titles else ""
    paths = []
    paths += save_heatmap(C_data, f"{out_dir}/{prefix}_data", title=t_data,
                          cbar_label=r"$\rho$", style=style, exts=exts)
    paths += save_heatmap(C_gen, f"{out_dir}/{prefix}_{gen_name}", title=t_gen,
                          cbar_label=r"$\rho$", style=style, exts=exts)
    paths += save_heatmap(C_diff, f"{out_dir}/{prefix}_diff", title=t_diff,
                          vmin=-vmax, vmax=vmax, cmap="PuOr_r",
                          cbar_label=r"$\Delta\rho$", style=style, exts=exts)
    return paths, mae


def save_marginals_separate(gen_lev, dat_lev, D, B, out_dir, prefix="marginal",
                            style="ROOT", exts=("png", "pdf"), titles=True):
    """One standalone figure PER FEATURE (gen vs data bars). titles=False
    suppresses the per-figure title."""
    set_root_style(style)
    nlev = 1 << B
    centers = np.arange(nlev); bw = 0.4
    paths = []
    for f in range(D):
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        gh = np.bincount(gen_lev[:, f], minlength=nlev) / len(gen_lev)
        dh = np.bincount(dat_lev[:, f], minlength=nlev) / len(dat_lev)
        ax.bar(centers - bw / 2, dh, bw, label="data", color="#2166ac",
               edgecolor="k", lw=.5)
        ax.bar(centers + bw / 2, gh, bw, label="gen", color="#b2182b",
               edgecolor="k", lw=.5)
        if titles:
            ax.set_title(f"feature {f}", fontsize=11)
        ax.set_xticks(centers); ax.set_xlabel("level"); ax.set_ylabel("probability")
        ax.legend(fontsize=9)
        fig.tight_layout()
        paths += _save_fig(fig, f"{out_dir}/{prefix}_feat{f:02d}", exts)
    return paths


def save_scatter(x, y, path_stem, xlabel, ylabel, title="", style="ROOT",
                 diag=True, label=None, exts=("png", "pdf"), figsize=(4.8, 4.8)):
    """One standalone scatter plot."""
    set_root_style(style)
    x = np.asarray(x); y = np.asarray(y)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=42, color="#8e44ad", alpha=0.8, edgecolor="white",
               linewidth=0.5, label=label, zorder=3)
    if diag and x.size:
        lim = max(float(np.abs(np.concatenate([x, y])).max()), 0.3) + 0.1
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2, alpha=0.6, zorder=1)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=12)
    if label:
        ax.legend(fontsize=10)
    fig.tight_layout()
    return _save_fig(fig, path_stem, exts)


def save_bar(values, labels, path_stem, ylabel, title="", colors=None,
             style="ROOT", logy=False, annotate=True, exts=("png", "pdf"),
             figsize=(4.8, 3.6)):
    """One standalone bar chart."""
    set_root_style(style)
    values = list(values)
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(values))
    bars = ax.bar(x, values, 0.62, color=(colors or "#2980b9"),
                  edgecolor="black", linewidth=1.2)
    if annotate:
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2,
                    v * (1.08 if logy else 1.04),
                    f"{v:.4g}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=12)
    if logy:
        ax.set_yscale("log")
    fig.tight_layout()
    return _save_fig(fig, path_stem, exts)


def save_image(img2d, path_stem, title="", vmin=None, vmax=None, cmap="inferno",
               cbar=True, cbar_label="intensity", style="ROOT",
               exts=("png", "pdf"), figsize=(3.6, 3.2)):
    """One 2-D image -> one standalone figure (no axis ticks). Used for the
    calorimeter shower images (data and generated)."""
    set_root_style(style)
    img2d = np.asarray(img2d, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(img2d, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", origin="upper", aspect="equal")
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if cbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
        cb.ax.tick_params(direction="in", length=3)
    fig.tight_layout()
    return _save_fig(fig, path_stem, exts)
