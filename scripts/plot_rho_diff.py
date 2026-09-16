#!/usr/bin/env python3
"""plot_rho_diff.py
===================

Residual-correlation-matrix diagnostics for the PSCK-MoIQP paper.

Produces five figures that complement the absolute ρ-matrices written by
`make_paper_figures.py`. They show where each model disagrees with the
data, quantitatively and graphically, and where PSCK beats the Liu-Wang
baseline (and vice versa).

Figures written to --out-dir
----------------------------
    rho_diff_psck.{pdf,png}
        Δ = ρ_PSCK − ρ_data, diverging colormap centered at 0.

    rho_diff_lwmmd.{pdf,png}
        Δ = ρ_LW − ρ_data, same color scale as the PSCK panel for
        apples-to-apples comparison.

    rho_diff_advantage.{pdf,png}
        |Δ_LW| − |Δ_PSCK|.  Positive (red) = PSCK beats LW at that
        (f,g) pair;  negative (blue) = LW beats PSCK there.

    rho_diff_triptych.{pdf,png}
        3-panel composite: PSCK residual, LW residual, PSCK-advantage,
        all on a shared symmetric color scale with a single colorbar.
        This is the figure most suitable for a Section-5 insert.

    rho_diff_histogram.{pdf,png}
        Distribution of off-diagonal residuals for PSCK (red) and LW
        (orange), with mean, median, and RMS annotated.  Shows the
        LW amplitude-compression signature as a bias in the LW
        histogram mean.

Also prints to stdout a numerical summary for each model:
    mean |Δ|, max |Δ|, Frobenius norm of Δ, number of (f,g) pairs where
    |Δ_LW| > |Δ_PSCK|, and the mean-bias of each residual (non-zero
    mean-bias is the systematic compression/dilation signature).

Usage
-----
    python plot_rho_diff.py \\
        --archive-root /path/to/iqp_strom_psck_mmd \\
        --out-dir      /path/to/figures \\
        --best-psck-seed 46 \\
        --lw-seed 42 \\
        --m-eval 50000

All defaults match `make_paper_figures.py` so the outputs from both
scripts are directly stackable in a single LaTeX \\begin{figure*}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ---------------------------------------------------------------------------
# Style — matches make_paper_figures.py verbatim
# ---------------------------------------------------------------------------

PRX_STYLE: dict = {
    "font.family":       "serif",
    "font.serif":        ["Times", "STIXGeneral"],
    "mathtext.fontset":  "stix",
    "font.size":         10.0,
    "axes.labelsize":    10.0,
    "axes.titlesize":    10.0,
    "legend.fontsize":    8.5,
    "xtick.labelsize":    9.0,
    "ytick.labelsize":    9.0,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "axes.linewidth":    0.7,
    "lines.linewidth":   1.0,
    "legend.frameon":    False,
    "figure.dpi":        160,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _seed_dir(archive_root: str, which: str, seed: int) -> str:
    if which == "L8":
        return os.path.join(
            archive_root, "outputs", "seed_sweep_B8_PSCK",
            f"psck_B8_binary_L8_ep1500_seed{seed}",
        )
    if which == "LW_L8":
        return os.path.join(
            archive_root, "outputs",
            f"lw_mmd_B8_binary_L8_ep1500_seed{seed}",
        )
    raise ValueError(f"unknown run class {which!r}")


def _save(fig: plt.Figure, out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  wrote  {out_dir}/{stem}.{{pdf,png}}")


def _rebuild_rho_model(run_dir: str,
                       archive_root: str,
                       m_eval: int,
                       pkg) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild ρ_data (train split, from continuous raw features) and
    ρ_model (from trained params, via Van den Nest MC).

    Returns (rho_model (D,D), rho_data (D,D)).
    """
    cfg = _read_json(os.path.join(run_dir, "config.json"))
    D = cfg["D"]
    B = cfg["bits"]
    L = int(cfg.get("L", 1))
    seed_user = cfg["seed"]
    n = D * B
    if (D, B) != (8, 8):
        raise ValueError(f"expected D=8, B=8; got D={D}, B={B}")

    data_path = os.path.join(archive_root, "data", "cal_shower_img_8q.npy")
    split_path = os.path.join(archive_root, "data", "split_indices.npz")
    _tr_bin, tr_raw, _te_bin, _te_raw = pkg.data_split.load_calorimeter_split(
        data_path=data_path, bits=B, encoding=cfg.get("encoding", "binary"),
        split_path=split_path, auto_create_split=False,
    )
    rho_data = np.corrcoef(tr_raw.T)

    gates, q2g = pkg.graph.build_er_graph(
        n, avg_deg=cfg.get("avg_deg", 6.0), seed=seed_user + 1,
    )
    observables, _ = pkg.feature_obs.enumerate_feature_observables(
        D, B, max_weight=2,
    )
    active = pkg.graph.precompute_active_lists(observables, gates, q2g)

    # Load all components, mix uniformly.
    params_list = [np.load(os.path.join(run_dir, f"params_{ell}.npy"))
                   for ell in range(L)]
    rng = np.random.default_rng(seed_user + 0xCAFE)
    z_mix = np.zeros(len(observables))
    for params in params_list:
        lat = pkg.correlator.sample_latents(n, m_eval, rng, antithetic=True)
        cache = pkg.correlator.ParityCache(lat, gates)
        signs = cache.signs
        z_l = np.empty(len(observables))
        for k, a in enumerate(active):
            if not a:
                z_l[k] = 1.0
                continue
            idx = np.asarray(a, dtype=np.int64)
            arg = 2.0 * (params[idx][:, None] * signs[idx]).sum(axis=0)
            z_l[k] = float(np.mean(np.cos(arg)))
        z_mix += z_l / L
        del lat, cache, signs, z_l

    rho_model, _ = pkg.corr_jacobian.pearson_value_and_jacobian(
        z_mix, observables, D, B,
    )
    return rho_model, rho_data


# ---------------------------------------------------------------------------
# Diagnostic summary
# ---------------------------------------------------------------------------

def _residual_stats(diff: np.ndarray, label: str) -> dict:
    """Off-diagonal residual statistics."""
    D = diff.shape[0]
    iu = np.triu_indices(D, k=1)
    vals = diff[iu]
    stats = {
        "mean_abs":       float(np.mean(np.abs(vals))),
        "max_abs":        float(np.max(np.abs(vals))),
        "frobenius":      float(np.sqrt(np.sum(vals ** 2))),
        "mean_bias":      float(np.mean(vals)),
        "median_bias":    float(np.median(vals)),
        "rms":            float(np.sqrt(np.mean(vals ** 2))),
        "n_pairs":        int(vals.size),
    }
    print(f"  {label}:")
    print(f"    mean |Δρ|   : {stats['mean_abs']:.4f}")
    print(f"    max  |Δρ|   : {stats['max_abs']:.4f}")
    print(f"    rms   Δρ    : {stats['rms']:.4f}")
    print(f"    Frob ||Δρ|| : {stats['frobenius']:.4f}  (on {stats['n_pairs']} off-diag pairs)")
    print(f"    mean  Δρ    : {stats['mean_bias']:+.4f}   "
          f"(nonzero = systematic over/under-estimation)")
    return stats


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _draw_residual_matrix(ax: plt.Axes, diff: np.ndarray,
                          vmax: float, cbar: bool = True):
    """Draw a residual matrix on a given axis, return the image handle."""
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=+vmax)
    im = ax.imshow(diff, cmap="RdBu_r", norm=norm, aspect="equal",
                   origin="upper")
    D = diff.shape[0]
    ax.set_xticks(range(D))
    ax.set_yticks(range(D))
    ax.set_xticklabels([f"$f_{i}$" for i in range(D)])
    ax.set_yticklabels([f"$f_{i}$" for i in range(D)])

    # Annotate each cell with its value (small font)
    for i in range(D):
        for j in range(D):
            v = diff[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=6.0,
                    color="white" if abs(v) > 0.6 * vmax else "black")
    if cbar:
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    return im


def plot_residual_single(diff: np.ndarray, vmax: float,
                         stem: str, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    _draw_residual_matrix(ax, diff, vmax=vmax, cbar=True)
    _save(fig, out_dir, stem)


def plot_residual_triptych(diff_psck: np.ndarray,
                           diff_lw: np.ndarray,
                           advantage: np.ndarray,
                           vmax: float,
                           out_dir: str,
                           best_psck_seed: int,
                           lw_seed: int) -> None:
    """Three-panel comparison with a single shared colorbar."""
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.8),
                             constrained_layout=True)
    
    mats = [diff_psck, diff_lw, advantage]
    for ax, M, t in zip(axes, mats):
        _draw_residual_matrix(ax, M, vmax=vmax, cbar=False)

    # Shared colorbar on the right
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=+vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.02,
                        label=r"$\Delta \rho$ (panels a, b)  /  "
                              r"$|\Delta_{\mathrm{LW}}| - |\Delta_{\mathrm{PSCK}}|$ (c)")

    _save(fig, out_dir, "rho_diff_triptych")


def plot_residual_histogram(diff_psck: np.ndarray,
                            diff_lw: np.ndarray,
                            stats_psck: dict,
                            stats_lw: dict,
                            out_dir: str) -> None:
    """Histogram of off-diagonal residuals for PSCK vs LW."""
    D = diff_psck.shape[0]
    iu = np.triu_indices(D, k=1)
    res_psck = diff_psck[iu]
    res_lw = diff_lw[iu]

    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    bins = np.linspace(
        min(res_psck.min(), res_lw.min()) - 0.005,
        max(res_psck.max(), res_lw.max()) + 0.005,
        25,
    )
    ax.hist(res_psck, bins=bins, color="#C8102E", alpha=0.55,
            label=f"PSCK, bias = {stats_psck['mean_bias']:+.3f}, "
                  f"RMS = {stats_psck['rms']:.3f}",
            edgecolor="#7A0A1C", linewidth=0.5)
    ax.hist(res_lw, bins=bins, color="#E48400", alpha=0.55,
            label=f"LW-MMD, bias = {stats_lw['mean_bias']:+.3f}, "
                  f"RMS = {stats_lw['rms']:.3f}",
            edgecolor="#A05800", linewidth=0.5)
    ax.axvline(0, color="k", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.axvline(stats_psck["mean_bias"], color="#C8102E",
               linestyle="--", linewidth=0.8)
    ax.axvline(stats_lw["mean_bias"], color="#E48400",
               linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"residual $\rho^{\mathrm{model}}_{fg}"
                  r" - \rho^{\mathrm{data}}_{fg}$")
    ax.set_ylabel("off-diagonal pairs")
    ax.legend(loc="upper left")
    _save(fig, out_dir, "rho_diff_histogram")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot ρ_model − ρ_data residual matrices for PSCK and LW-MMD, "
            "a PSCK-advantage map, a 3-panel triptych, and a residual "
            "histogram.  All plots use the same VdN machinery and style "
            "as make_paper_figures.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--archive-root", required=True,
                        help="extracted iqp_strom_psck_mmd directory.")
    parser.add_argument("--src-root", default=None,
                        help="overrides <archive-root>/src for psck_mmd import.")
    parser.add_argument("--out-dir", required=True,
                        help="where to write the residual figures.")
    parser.add_argument("--best-psck-seed", type=int, default=46,
                        help="PSCK L=8/1500 seed.")
    parser.add_argument("--lw-seed", type=int, default=42,
                        help="LW-MMD L=8/1500 seed.")
    parser.add_argument("--m-eval", type=int, default=50_000,
                        help="Van den Nest Monte Carlo latents for ρ-reconstruction.")
    parser.add_argument("--vmax", type=float, default=None,
                        help="symmetric color-scale limit (auto if not given).")
    args = parser.parse_args(argv)

    src_root = args.src_root or os.path.join(args.archive_root, "src")
    sys.path.insert(0, src_root)
    try:
        import psck_mmd  # noqa: F401
        from psck_mmd import (
            data_split, graph, feature_obs, correlator, corr_jacobian,
        )

        class _PkgHandle:
            pass
        pkg = _PkgHandle()
        pkg.data_split = data_split
        pkg.graph = graph
        pkg.feature_obs = feature_obs
        pkg.correlator = correlator
        pkg.corr_jacobian = corr_jacobian
    except ImportError as err:
        print(f"ERROR: could not import psck_mmd from {src_root!r}: {err}",
              file=sys.stderr)
        return 2

    plt.rcParams.update(PRX_STYLE)
    t0 = time.time()

    # ---- Rebuild the two model ρ matrices ------------------------------
    print(f"[1/2] PSCK L=8 seed {args.best_psck_seed} "
          f"(Van den Nest, M = {args.m_eval})")
    psck_dir = _seed_dir(args.archive_root, "L8", args.best_psck_seed)
    t1 = time.time()
    rho_psck, rho_data = _rebuild_rho_model(
        psck_dir, args.archive_root, args.m_eval, pkg,
    )
    print(f"        done in {time.time()-t1:.1f} s")

    print(f"[2/2] LW-MMD L=8 seed {args.lw_seed} "
          f"(Van den Nest, M = {args.m_eval})")
    lw_dir = _seed_dir(args.archive_root, "LW_L8", args.lw_seed)
    t1 = time.time()
    rho_lw, _ = _rebuild_rho_model(
        lw_dir, args.archive_root, args.m_eval, pkg,
    )
    print(f"        done in {time.time()-t1:.1f} s")

    # ---- Compute residuals --------------------------------------------
    diff_psck = rho_psck - rho_data
    diff_lw   = rho_lw   - rho_data
    advantage = np.abs(diff_lw) - np.abs(diff_psck)  # +ve = PSCK wins

    # Zero out the diagonal of advantage (always 0 by definition, but be safe)
    np.fill_diagonal(diff_psck, 0.0)
    np.fill_diagonal(diff_lw,   0.0)
    np.fill_diagonal(advantage, 0.0)

    print("\nResidual statistics (off-diagonal, upper-triangle only):")
    stats_psck = _residual_stats(diff_psck, "PSCK-MoIQP")
    stats_lw   = _residual_stats(diff_lw,   "LW-MMD")

    # Pair-level PSCK advantage count
    D = diff_psck.shape[0]
    iu = np.triu_indices(D, k=1)
    psck_wins = int(np.sum(np.abs(diff_lw[iu]) > np.abs(diff_psck[iu])))
    lw_wins   = int(np.sum(np.abs(diff_lw[iu]) < np.abs(diff_psck[iu])))
    print(f"\nPair-level comparison on {iu[0].size} off-diagonal pairs:")
    print(f"  PSCK strictly better : {psck_wins} / {iu[0].size}  "
          f"({100.0 * psck_wins / iu[0].size:.1f} %)")
    print(f"  LW strictly better   : {lw_wins} / {iu[0].size}")
    print(f"  ties                 : {iu[0].size - psck_wins - lw_wins}")

    # ---- Plot ----------------------------------------------------------
    if args.vmax is None:
        vmax = max(np.max(np.abs(diff_psck)), np.max(np.abs(diff_lw)),
                   np.max(np.abs(advantage)))
        # round up to a nice 0.05 step
        vmax = np.ceil(vmax / 0.05) * 0.05
        vmax = max(vmax, 0.05)
    else:
        vmax = args.vmax
    print(f"\nsymmetric color limits: ± {vmax:.3f}\n")

    print("[plot 1/5] rho_diff_psck")
    plot_residual_single(
        diff_psck, vmax=vmax,
        stem="rho_diff_psck", out_dir=args.out_dir,
    )

    print("[plot 2/5] rho_diff_lwmmd")
    plot_residual_single(
        diff_lw, vmax=vmax,


        stem="rho_diff_lwmmd", out_dir=args.out_dir,
    )

    print("[plot 3/5] rho_diff_advantage")
    plot_residual_single(
        advantage, vmax=vmax,

        stem="rho_diff_advantage", out_dir=args.out_dir,
    )




    # Save numerical summary for reproducibility
    summary = {
        "m_eval": args.m_eval,
        "best_psck_seed": args.best_psck_seed,
        "lw_seed": args.lw_seed,
        "vmax_plot": float(vmax),
        "psck_stats": stats_psck,
        "lw_stats":   stats_lw,
        "pair_comparison": {
            "n_pairs": int(iu[0].size),
            "psck_strictly_better": psck_wins,
            "lw_strictly_better":   lw_wins,
            "ties":                 int(iu[0].size - psck_wins - lw_wins),
        },
    }
    summary_path = os.path.join(args.out_dir, "rho_diff_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary JSON written to {summary_path}")
    print(f"total wall time: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
