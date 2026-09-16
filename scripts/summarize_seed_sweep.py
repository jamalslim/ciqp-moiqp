#!/usr/bin/env python
"""Aggregate a multi-seed sweep and produce error-bar plots + summary table.

Reads every run directory under a sweep root (matching the harness naming
convention <loss>_B<..>_<enc>_L<..>_ep<..>_seed<s>), loads each run's
metrics.json and history.json, and produces:

    <sweep_root>/summary.json          # structured per-seed + aggregate stats
    <sweep_root>/summary_table.txt     # plain-text table (easy to paste)
    <sweep_root>/summary_table.tex     # LaTeX booktabs table
    <sweep_root>/plots_sweep/
        rho_mae_err.{png,pdf}          # per-seed final ρ MAE with mean ± std
        z_mae_err.{png,pdf}            # per-seed final z MAE with mean ± std
        rho_r_err.{png,pdf}            # per-seed Pearson r with mean ± std
        rho_curves_band.{png,pdf}      # mean rho_err trajectory with ±1σ band
        z_curves_band.{png,pdf}        # mean z_err trajectory with ±1σ band
        loss_curves_band.{png,pdf}     # mean loss trajectory with ±1σ band
        ciqp_mae_err.{png,pdf}         # cIQP deployment MAE per seed

USAGE
-----
    python scripts/summarize_seed_sweep.py <sweep_root>
    python scripts/summarize_seed_sweep.py <sweep_root> --style CMS
    python scripts/summarize_seed_sweep.py <sweep_root> --baseline <ref_run_dir>

The --baseline flag lets you overlay a single reference run (e.g. LW-MMD
at the same scale) as a dashed line on the band plots, for direct
visual comparison without running a second sweep.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

# Load the package's plotting style helpers from src/psck_mmd/plotting.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from psck_mmd.plotting import set_root_style, ROOT_COLORS, _add_minor_ticks

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSS_DISPLAY_SHORT = {"corrmse": "CorrMSE", "lw_mmd": "LW-MMD", "psck": "PSCK-MMD"}


def _save_both(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"),
                     dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_dir}/{stem}.{{png,pdf}}")


# ─────────────────────────────────────────────────────────────────────────────
# Discovery and loading
# ─────────────────────────────────────────────────────────────────────────────

RUN_DIR_RE = re.compile(
    r"^(?P<loss>corrmse|lw_mmd|psck)_B(?P<B>\d+)_(?P<enc>[a-z]+)"
    r"_L(?P<L>\d+)_ep(?P<ep>\d+)_seed(?P<seed>\d+)(_.*)?$"
)


def discover_runs(sweep_root):
    """Find every subdirectory of sweep_root that matches the harness naming
    convention AND has a metrics.json inside."""
    runs = []
    for name in sorted(os.listdir(sweep_root)):
        path = os.path.join(sweep_root, name)
        if not os.path.isdir(path):
            continue
        m = RUN_DIR_RE.match(name)
        if not m:
            continue
        mpath = os.path.join(path, "metrics.json")
        hpath = os.path.join(path, "history.json")
        if not (os.path.exists(mpath) and os.path.exists(hpath)):
            continue
        with open(mpath) as f:
            metrics = json.load(f)
        with open(hpath) as f:
            history = json.load(f)
        verification = None
        vpath = os.path.join(path, "verification.json")
        if os.path.exists(vpath):
            with open(vpath) as f:
                verification = json.load(f)
        runs.append({
            "dir": path,
            "name": name,
            "seed": int(m.group("seed")),
            "loss": m.group("loss"),
            "B": int(m.group("B")),
            "enc": m.group("enc"),
            "L": int(m.group("L")),
            "epochs": int(m.group("ep")),
            "metrics": metrics,
            "history": history,
            "verification": verification,
        })
    return runs


def aggregate(runs):
    """Compute per-loss mean / std across seeds for final metrics."""
    by_loss = {}
    for r in runs:
        by_loss.setdefault(r["loss"], []).append(r)

    agg = {}
    for loss, rs in by_loss.items():
        rho_mae = np.array([r["metrics"]["rho_mae"] for r in rs])
        z_mae   = np.array([r["metrics"]["z_mae"]   for r in rs])
        rho_r   = np.array([r["metrics"]["rho_r"]   for r in rs])
        enc_fid = np.array([r["metrics"]["enc_fid"] for r in rs])
        ttime   = np.array([r["metrics"]["train_time"] for r in rs])
        ciqp_mae = np.array([
            r["verification"]["mae_ciqp_vs_moiqp"]
            if r["verification"] is not None else np.nan
            for r in rs
        ])
        ciqp_ratio = np.array([
            r["verification"]["ratio"]
            if r["verification"] is not None else np.nan
            for r in rs
        ])
        agg[loss] = {
            "n_seeds": len(rs),
            "seeds": [r["seed"] for r in rs],
            "rho_mae": {"mean": float(rho_mae.mean()), "std": float(rho_mae.std(ddof=1)) if len(rho_mae)>1 else 0.0, "values": rho_mae.tolist()},
            "z_mae":   {"mean": float(z_mae.mean()),   "std": float(z_mae.std(ddof=1))   if len(z_mae)>1   else 0.0, "values": z_mae.tolist()},
            "rho_r":   {"mean": float(rho_r.mean()),   "std": float(rho_r.std(ddof=1))   if len(rho_r)>1   else 0.0, "values": rho_r.tolist()},
            "enc_fid": {"mean": float(enc_fid.mean()), "std": float(enc_fid.std(ddof=1)) if len(enc_fid)>1 else 0.0, "values": enc_fid.tolist()},
            "train_time": {"mean": float(ttime.mean()), "std": float(ttime.std(ddof=1)) if len(ttime)>1 else 0.0, "values": ttime.tolist()},
            "ciqp_mae":   {"mean": float(np.nanmean(ciqp_mae)),   "std": float(np.nanstd(ciqp_mae, ddof=1))   if np.sum(~np.isnan(ciqp_mae))>1 else 0.0, "values": ciqp_mae.tolist()},
            "ciqp_ratio": {"mean": float(np.nanmean(ciqp_ratio)), "std": float(np.nanstd(ciqp_ratio, ddof=1)) if np.sum(~np.isnan(ciqp_ratio))>1 else 0.0, "values": ciqp_ratio.tolist()},
        }
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Text tables
# ─────────────────────────────────────────────────────────────────────────────

def write_plaintext_table(agg, out_path):
    lines = []
    lines.append(f"{'='*88}")
    lines.append(f"  Seed-sweep summary")
    lines.append(f"{'='*88}")
    for loss, a in agg.items():
        disp = LOSS_DISPLAY_SHORT.get(loss, loss)
        lines.append("")
        lines.append(f"  {disp}  (n_seeds = {a['n_seeds']})")
        lines.append(f"  seeds: {a['seeds']}")
        lines.append(f"  {'─'*60}")
        lines.append(f"    ρ  MAE            : {a['rho_mae']['mean']:.4f} ± {a['rho_mae']['std']:.4f}")
        lines.append(f"    z  MAE            : {a['z_mae']['mean']:.4f} ± {a['z_mae']['std']:.4f}")
        lines.append(f"    ρ  Pearson r      : {a['rho_r']['mean']:.4f} ± {a['rho_r']['std']:.4f}")
        lines.append(f"    encoding fidelity : {a['enc_fid']['mean']:.4f} ± {a['enc_fid']['std']:.4f}")
        lines.append(f"    cIQP MAE / MC     : {a['ciqp_ratio']['mean']:.2f}× ± {a['ciqp_ratio']['std']:.2f}×")
        lines.append(f"    train time (s)    : {a['train_time']['mean']:.0f} ± {a['train_time']['std']:.0f}")
    lines.append(f"\n{'='*88}\n")
    s = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(s)
    print(s)


def write_latex_table(agg, out_path):
    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{Multi-seed reproducibility sweep at 64 qubits (B=8, L=4, 500 epochs, identical graph). Uncertainty is one standard deviation across seeds.}",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Loss & $\rho$ MAE & $z$ MAE & $\rho$ Pearson $r$ & cIQP/MC \\",
        r"    \midrule",
    ]
    for loss, a in agg.items():
        disp = LOSS_DISPLAY_SHORT.get(loss, loss).replace("_", r"\_")
        lines.append(
            f"    {disp} & "
            f"${a['rho_mae']['mean']:.4f}\\pm{a['rho_mae']['std']:.4f}$ & "
            f"${a['z_mae']['mean']:.4f}\\pm{a['z_mae']['std']:.4f}$ & "
            f"${a['rho_r']['mean']:.4f}\\pm{a['rho_r']['std']:.4f}$ & "
            f"${a['ciqp_ratio']['mean']:.2f}\\pm{a['ciqp_ratio']['std']:.2f}$ \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_seed_bar(runs, key, ylabel, out_dir, stem, title=None,
                        log_y=False, enc_floor=None, value_fmt="{:.4f}"):
    """Bar plot of per-seed final metric values with mean line and std band."""
    seeds = [r["seed"] for r in runs]
    vals = np.array([r["metrics"][key] for r in runs])
    color = ROOT_COLORS.get(runs[0]["loss"], "#c0392b")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(seeds))
    ax.bar(x, vals, 0.62, color=color, edgecolor="black", linewidth=1.2)
    mean = vals.mean()
    std  = vals.std(ddof=1) if len(vals) > 1 else 0.0
    ax.axhline(mean, color="black", linestyle="-", linewidth=1.4,
                 label=f"mean = {value_fmt.format(mean)} ± {value_fmt.format(std)}")
    ax.axhspan(mean - std, mean + std, color="black", alpha=0.08)
    if enc_floor is not None:
        ax.axhline(enc_floor, color="black", linestyle="--", linewidth=1.2,
                     label=f"encoding floor = {value_fmt.format(enc_floor)}")
    for xi, v in zip(x, vals):
        ax.text(xi, v + (0.02 * max(vals.max(), 1e-6) if not log_y else v * 0.03),
                 value_fmt.format(v), ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=11)
    ax.set_ylabel(ylabel)

    if log_y:
        ax.set_yscale("log")
    _add_minor_ticks(ax, logy=log_y)
    ax.legend(fontsize=10, loc="best")
    _save_both(fig, out_dir, stem)


def plot_ciqp_bar(runs, out_dir):
    seeds = [r["seed"] for r in runs]
    ratios = np.array([
        r["verification"]["ratio"] if r["verification"] else np.nan
        for r in runs
    ])
    if np.all(np.isnan(ratios)):
        print("  [skip] ciqp_mae_err: no verification.json in any run")
        return
    color = "#8e44ad"
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(seeds))
    ax.bar(x, ratios, 0.62, color=color, edgecolor="black", linewidth=1.2)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.3,
                 label="1× MC noise")
    ax.axhline(5.0, color="gray", linestyle=":", linewidth=1.0,
                 label="5× MC noise (PASS threshold)")
    for xi, v in zip(x, ratios):
        if not np.isnan(v):
            ax.text(xi, v + 0.05, f"{v:.2f}×", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=11)
    ax.set_ylabel("cIQP MAE / MC noise")

    _add_minor_ticks(ax)
    ax.legend(fontsize=10, loc="best")
    _save_both(fig, out_dir, "ciqp_mae_err")


def plot_band(runs, key, ylabel, out_dir, stem, log_y=False,
               enc_floor=None, baseline=None, title=None):
    """Mean-with-std-band trajectory across seeds for a training-history key."""
    # History arrays may have different lengths if some runs diverged; clip.
    Hs = [np.asarray(r["history"][key]) for r in runs]
    n = min(H.shape[0] for H in Hs)
    Hs = np.stack([H[:n] for H in Hs], axis=0)  # (n_seeds, n_epochs)
    mean = Hs.mean(axis=0)
    std  = Hs.std(axis=0, ddof=1) if Hs.shape[0] > 1 else np.zeros_like(mean)
    color = ROOT_COLORS.get(runs[0]["loss"], "#c0392b")
    fig, ax = plt.subplots(figsize=(8, 5))
    ep = np.arange(n)
    ax.plot(ep, mean, color=color, linewidth=1.8,
             label=f"{LOSS_DISPLAY_SHORT.get(runs[0]['loss'], runs[0]['loss'])} "
                     f"(n={len(runs)} seeds)")
    ax.fill_between(ep, mean - std, mean + std, color=color, alpha=0.20,
                      label="±1σ across seeds")
    if baseline is not None:
        bkey = key
        if bkey in baseline["history"]:
            bH = np.asarray(baseline["history"][bkey])[:n]
            bcolor = ROOT_COLORS.get(baseline["loss"], "#e67e22")
            ax.plot(ep, bH, color=bcolor, linewidth=1.5, linestyle="--",
                     label=f"baseline: {LOSS_DISPLAY_SHORT.get(baseline['loss'], baseline['loss'])} "
                             f"(seed {baseline['seed']})")
    if enc_floor is not None:
        ax.axhline(enc_floor, color="black", linestyle=":", linewidth=1.2,
                     label=f"encoding floor = {enc_floor:.3f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)

    if log_y:
        ax.set_yscale("log")
    _add_minor_ticks(ax, logy=log_y)
    ax.legend(fontsize=10, loc="best")
    _save_both(fig, out_dir, stem)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_root", type=str, help="directory containing sweep run folders")
    ap.add_argument("--style", type=str, default="ROOT",
                     choices=["ROOT", "CMS", "ATLAS", "LHCb", "ALICE"])
    ap.add_argument("--baseline", type=str, default=None,
                     help="optional reference single-seed run folder to overlay on band plots")
    ap.add_argument("--out-dir", type=str, default=None,
                     help="plots output folder (default: <sweep_root>/plots_sweep/)")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.sweep_root, "plots_sweep")

    set_root_style(args.style)

    runs = discover_runs(args.sweep_root)
    if not runs:
        print(f"no valid runs found under {args.sweep_root}")
        sys.exit(1)

    by_loss = {}
    for r in runs:
        by_loss.setdefault(r["loss"], []).append(r)
    # sort by seed within each loss
    for loss in by_loss:
        by_loss[loss].sort(key=lambda r: r["seed"])

    print(f"\n  Discovered {len(runs)} runs across {len(by_loss)} losses:")
    for loss, rs in by_loss.items():
        print(f"    {loss}: {[r['seed'] for r in rs]}")
    print()

    agg = aggregate(runs)
    # Save structured summary
    summary_path = os.path.join(args.sweep_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "sweep_root": args.sweep_root,
            "n_runs": len(runs),
            "by_loss": agg,
        }, f, indent=2)
    print(f"  summary.json → {summary_path}")

    # Text & LaTeX tables
    write_plaintext_table(agg, os.path.join(args.sweep_root, "summary_table.txt"))
    write_latex_table(agg, os.path.join(args.sweep_root, "summary_table.tex"))
    print(f"  summary_table.{{txt,tex}} → {args.sweep_root}/\n")

    # Optional baseline reference run
    baseline = None
    if args.baseline is not None:
        with open(os.path.join(args.baseline, "metrics.json")) as f:
            bm = json.load(f)
        with open(os.path.join(args.baseline, "history.json")) as f:
            bh = json.load(f)
        baseline = {"dir": args.baseline, "metrics": bm, "history": bh,
                     "loss": bm["loss"], "seed": bm.get("seed", -1)}
        print(f"  baseline overlay: {args.baseline} ({baseline['loss']})")

    # Plots per loss
    for loss, rs in by_loss.items():
        enc_floor = rs[0]["metrics"]["enc_fid"]
        print(f"\n  plotting loss={loss} ...")
        plot_per_seed_bar(
            rs, "rho_mae",
            ylabel=r"$\langle|\Delta\rho|\rangle$",

            out_dir=out_dir, stem=f"{loss}_rho_mae_err",
            log_y=True, enc_floor=enc_floor,
        )
        plot_per_seed_bar(
            rs, "z_mae",
            ylabel=r"$\langle|\Delta\langle Z_S\rangle|\rangle$",

            out_dir=out_dir, stem=f"{loss}_z_mae_err",
            log_y=False,
        )
        plot_per_seed_bar(
            rs, "rho_r",
            ylabel=r"Pearson $r$ ($\rho$)",

            out_dir=out_dir, stem=f"{loss}_rho_r_err",
            log_y=False, value_fmt="{:.4f}",
        )
        plot_ciqp_bar(rs, out_dir)

        plot_band(
            rs, "rho_err",
            ylabel=r"$\langle|\Delta\rho|\rangle$",

            out_dir=out_dir, stem=f"{loss}_rho_curves_band",
            enc_floor=enc_floor, baseline=baseline,
        )
        plot_band(
            rs, "z_err",
            ylabel=r"$\langle|\Delta\langle Z_S\rangle|\rangle$",

            out_dir=out_dir, stem=f"{loss}_z_curves_band",
            baseline=baseline,
        )
        plot_band(
            rs, "loss",
            ylabel="loss",

            out_dir=out_dir, stem=f"{loss}_loss_curves_band",
            log_y=True, baseline=baseline,
        )

    print(f"\n  All outputs under: {args.sweep_root}")
    print(f"  Plots:   {out_dir}")
    print(f"  Table:   {os.path.join(args.sweep_root, 'summary_table.txt')}\n")


if __name__ == "__main__":
    main()
