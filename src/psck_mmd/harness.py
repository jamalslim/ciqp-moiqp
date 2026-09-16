"""
psck_mmd.harness — Shared training/evaluation/persistence harness used by
the scripts/run_psck.py, scripts/run_lwmmd.py, scripts/run_corrmse.py
driver CLIs.

Centralises:
  - CLI argument parsing (common flags across all three losses)
  - data loading + graph construction
  - training invocation
  - evaluation
  - results persistence (params, metrics.json, history.json)

Each driver script only needs to:
    from psck_mmd.harness import run_driver
    run_driver(loss_kind="psck", default_kwargs={...})

Output directory layout:
    outputs/<loss>_B<bits>_<encoding>_L<L>_ep<epochs>_seed<seed>/
        metrics.json
        history.json
        params_<l>.npy      (L files)
        config.json
"""

from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np

from .data import load_calorimeter, ENCODING_TABLES
from .graph import build_er_graph, build_complete_graph
from .feature_obs import enumerate_feature_observables
from .train_moiqp import train_moiqp, evaluate_mixture


LOSS_DISPLAY = {
    "corrmse": "CorrMSE (L_z + λ L_ρ)",
    "lw_mmd":  "Liu–Wang MMD (heat kernel)",
    "psck":    "PSCK-MMD (this work)",
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI parser
# ─────────────────────────────────────────────────────────────────────────────

def make_arg_parser(loss_kind: str) -> argparse.ArgumentParser:
    """Build the CLI parser for driver scripts. Flags are uniform across the
    three driver scripts so they can all be invoked with the same arguments.
    """
    p = argparse.ArgumentParser(
        description=f"Train a MoIQP Born machine with {LOSS_DISPLAY[loss_kind]} loss.",
    )
    # Architecture
    p.add_argument("--bits", "-B", "--B", dest="bits", type=int, default=2,
                    help="number of bits per feature (B)")
    p.add_argument("--L", "-L", type=int, default=4,
                    help="number of MoIQP mixture components")
    p.add_argument("--D", type=int, default=8,
                    help="number of features (fixed for calorimeter: 8)")
    p.add_argument("--encoding", type=str, default="binary",
                    choices=sorted(ENCODING_TABLES),
                    help="bit encoding per feature: binary | gray | hadamard")
    p.add_argument("--graph", type=str, default="erdos_renyi",
                    choices=["erdos_renyi", "complete", ],
                    help="IQP gate graph type. 'hardware_native' uses the induced "
                         "subgraph of a device coupling map (requires --coupling-map) "
                         "so the deployed circuit needs no SWAP routing -> higher F.")
    p.add_argument("--avg-deg", type=float, default=6.0,
                    help="average degree for ER graph")
    # Optimization
    p.add_argument("--epochs", "-E", type=int, default=500,
                    help="Adam steps")
    p.add_argument("--mc-batch", "-M", type=int, default=4096,
                    help="MC batch size per forward pass")
    p.add_argument("--lr", type=float, default=0.02,
                    help="peak Adam learning rate")
    p.add_argument("--seed", type=int, default=42,
                    help="random seed (params init, MC latents)")

    # Loss hyperparameters
    if loss_kind == "corrmse":
        p.add_argument("--lambda-rho", type=float, default=40.0,
                        help="Lagrange multiplier on L_ρ")
    elif loss_kind == "psck":
        p.add_argument("--eta-psck", type=float, default=5.0,
                        help="η for PSCK (weight of Pearson-Jacobian correction)")
        p.add_argument("--heat-scale", type=float, default=1.0,
                        help="multiplier on the heat-kernel diagonal")
        p.add_argument("--biased", action="store_true",
                        help="use biased single-batch estimator (default: unbiased U-stat)")
    elif loss_kind == "lw_mmd":
        p.add_argument("--heat-scale", type=float, default=1.0,
                        help="multiplier on the heat-kernel diagonal")

    # I/O
    p.add_argument("--data", type=str, default="data/cal_shower_img_8q.npy",
                    help="path to raw calorimeter (N, D) .npy file")
    p.add_argument("--split", type=str, default=None,
                    help="split_indices.npz path -> split-correct training "
                         "(review item A1): quantile edges and ALL training "
                         "targets are fit on the training split only")
    p.add_argument("--t-restart", type=int, default=None,
                    help="cosine-restart period in epochs "
                         "(default: max(50, n_epochs//5))")
    p.add_argument("--out-root", type=str, default="outputs",
                    help="root folder under which results are saved "
                         "(ignored if --output is given)")
    p.add_argument("--output", "-o", type=str, default=None,
                    help="explicit output directory; overrides --out-root and "
                         "the auto-generated run-name")
    p.add_argument("--tag", type=str, default=None,
                    help="optional custom suffix appended to the run folder name")
    p.add_argument("--eval-M", type=int, default=50_000,
                    help="MC sample count at final evaluation")
    p.add_argument("--quiet", action="store_true",
                    help="suppress per-epoch training printouts")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Output folder naming
# ─────────────────────────────────────────────────────────────────────────────

def make_out_dir(args, loss_kind: str) -> str:
    if getattr(args, "output", None):
        os.makedirs(args.output, exist_ok=True)
        return args.output
    base = (
        f"{loss_kind}_B{args.bits}_{args.encoding}_L{args.L}"
        f"_ep{args.epochs}_seed{args.seed}"
    )
    if args.tag:
        base += f"_{args.tag}"
    out = os.path.join(args.out_root, base)
    os.makedirs(out, exist_ok=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────────────

def run_driver(loss_kind: str):
    """Entry point used by each scripts/run_<loss>.py driver."""
    parser = make_arg_parser(loss_kind)
    args = parser.parse_args()

    D = args.D
    n_feat = D * args.bits
    out_dir = make_out_dir(args, loss_kind)

    print(f"\n{'='*72}")
    print(f"  {LOSS_DISPLAY[loss_kind]}")
    print(f"  D={D}  B={args.bits}  n={n_feat}q  L={args.L}"
          f"  enc={args.encoding}  graph={args.graph}  "
          f"ep={args.epochs}  M={args.mc_batch}  seed={args.seed}")
    print(f"  out_dir = {out_dir}")
    print(f"{'='*72}\n")

    # Data
    floor_train = floor_test = None
    if args.split:
        # Split-correct path (review item A1): edges + training targets from
        # the training split ONLY. Mirrors the run_hcgt.py wiring.
        from .data import load_calorimeter_split
        _sp = load_calorimeter_split(args.data, bits=args.bits,
                                       encoding=args.encoding,
                                       split_path=args.split)
        binary, raw = _sp["binary_train"], _sp["raw_train"]
        floor_train = float(_sp["floor_train"])
        floor_test = float(_sp["floor_test"])
        print(f"  [split-correct] training on {binary.shape[0]} train samples; "
              f"floors: train={floor_train:.4f} test={floor_test:.4f}")
    else:
        print("  WARNING: full-sample encoding (NOT split-correct). "
              "Pass --split data/split_indices.npz for paper-grade runs.")
        binary, raw = load_calorimeter(args.data, bits=args.bits,
                                         encoding=args.encoding)
    obs, _ = enumerate_feature_observables(D, args.bits, max_weight=2)
    if args.graph == "erdos_renyi":
        gates, q2g = build_er_graph(n_feat, avg_deg=args.avg_deg,
                                      seed=args.seed + 1)
    else:
        gates, q2g = build_complete_graph(n_feat)
    # persist the realized graph so all downstream tools (run_sample, plot_images,
    # evaluation) reproduce the EXACT circuit and never reconstruct a different one
    json.dump({"gates": [list(g) for g in gates],
               "q2g": {str(k): list(v) for k, v in q2g.items()}},
              open(os.path.join(out_dir, "graph.json"), "w"))
    print(f"  data: {binary.shape}   gates: {len(gates)}   obs: {len(obs)}\n")

    # Loss-specific hyperparameters
    kw = dict(
        n_epochs=args.epochs, mc_batch=args.mc_batch, lr=args.lr,
        seed=args.seed, verbose=not args.quiet,
        T_restart=args.t_restart,
    )
    if loss_kind == "corrmse":
        kw["lambda_rho"] = args.lambda_rho
    elif loss_kind == "psck":
        kw["eta_psck"] = args.eta_psck
        kw["heat_scale"] = args.heat_scale
        kw["unbiased"] = not args.biased
    elif loss_kind == "lw_mmd":
        kw["heat_scale"] = args.heat_scale

    # Train
    t0 = time.time()
    params, hist = train_moiqp(
        binary, raw, gates, q2g, D, args.bits, args.L, obs,
        loss_kind=loss_kind, **kw,
    )
    train_time = time.time() - t0

    # Evaluate
    res = evaluate_mixture(
        params, gates, q2g, D, args.bits, obs, binary, raw,
        M_eval=args.eval_M, seed=args.seed + 10_000,
    )

    print(f"\n  {'─'*55}")
    print(f"  {LOSS_DISPLAY[loss_kind]}  —  results")
    print(f"  {'─'*55}")
    print(f"  z  MAE             : {res['z_mae']:.6f}")
    print(f"  ρ  MAE vs raw      : {res['rho_mae']:.6f}")
    print(f"  ρ  Pearson r       : {res['rho_r']:.6f}")
    print(f"  encoding fidelity  : {res['enc_fid']:.4f}")
    print(f"  train time         : {train_time:.0f} s")

    # Save
    for ell, p in enumerate(params):
        np.save(os.path.join(out_dir, f"params_{ell}.npy"), p)

    metrics = dict(
        loss=loss_kind,
        loss_display=LOSS_DISPLAY[loss_kind],
        D=D, B=args.bits, n_feat=n_feat, L=args.L,
        encoding=args.encoding, graph=args.graph,
        epochs=args.epochs, mc_batch=args.mc_batch, lr=args.lr,
        seed=args.seed,
        rho_mae=res["rho_mae"], rho_r=res["rho_r"],
        z_mae=res["z_mae"], enc_fid=res["enc_fid"],
        split=os.path.abspath(args.split) if args.split else None,
        encoding_floor_train=floor_train, encoding_floor_test=floor_test,
        t_restart=args.t_restart,
        train_time=train_time,
        final_loss=hist["loss"][-1],
    )
    if loss_kind == "corrmse":
        metrics["lambda_rho"] = args.lambda_rho
    elif loss_kind == "psck":
        metrics["eta_psck"] = args.eta_psck
        metrics["heat_scale"] = args.heat_scale
        metrics["unbiased"] = not args.biased
    elif loss_kind == "lw_mmd":
        metrics["heat_scale"] = args.heat_scale

    json.dump(metrics, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    json.dump(
        {"loss":   [float(x) for x in hist["loss"]],
         "z_err":  list(hist["z_err"]),
         "rho_err": list(hist["rho_err"])},
        open(os.path.join(out_dir, "history.json"), "w"),
    )
    json.dump(vars(args), open(os.path.join(out_dir, "config.json"), "w"),
              indent=2, default=str)
    print(f"\n  Saved under {out_dir}/\n")
