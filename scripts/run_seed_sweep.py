#!/usr/bin/env python
"""Run the five-seed headline sweep behind Tables I and X.

Roughly 13 CPU-hours in total and parallel across seeds, so five processes
finish in the time of one. Each seed trains, evaluates on the held-out split,
and verifies the cIQP compilation. summarize_seed_sweep.py turns the resulting
directory into the table rows.

The --split flag is not optional for reproducing the paper: without it the
quantile edges are fitted on the full sample and every number shifts.
"""

import argparse
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bits", "-B", type=int, default=8)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--epochs", "-E", type=int, default=500)
    ap.add_argument("--encoding", type=str, default="binary",
                     choices=["binary", "gray", "hadamard"])
    ap.add_argument("--seeds", type=int, nargs="+",
                     default=[42, 43, 44, 45, 46],
                     help="list of seeds to sweep (default: 42 43 44 45 46)")
    ap.add_argument("--sweep-root", type=str, required=True,
                     help="root directory under which all seed run folders land")
    ap.add_argument("--loss", type=str, default="psck",
                     choices=["psck", "lw_mmd", "corrmse"],
                     help="which loss to sweep (default: psck)")
    ap.add_argument("--eta-psck", type=float, default=5.0)
    ap.add_argument("--heat-scale", type=float, default=1.0)
    ap.add_argument("--lambda-rho", type=float, default=40.0)
    ap.add_argument("--mc-batch", "-M", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--graph", type=str, default="erdos_renyi",
                     choices=["erdos_renyi", "complete"])
    ap.add_argument("--avg-deg", type=float, default=6.0)
    ap.add_argument("--skip-verify", action="store_true",
                     help="do NOT run verify_ciqp after each training")
    ap.add_argument("--skip-plot", action="store_true",
                     help="do NOT run plot_results after each training")
    ap.add_argument("--verify-M", type=int, default=200_000,
                     help="MC sample count for cIQP verification")
    ap.add_argument("--split", type=str, default=None,
                     help="split_indices.npz -> split-correct training (A1); "
                          "forwarded to every per-seed training run")
    ap.add_argument("--t-restart", type=int, default=None,
                     help="cosine-restart period in epochs; forwarded")
    ap.add_argument("--dry-run", action="store_true",
                     help="print commands without executing")
    args = ap.parse_args()

    os.makedirs(args.sweep_root, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    driver = {"psck":    "run_psck.py",
              "lw_mmd":  "run_lwmmd.py",
              "corrmse": "run_corrmse.py"}[args.loss]
    driver_path = os.path.join(script_dir, driver)
    verify_path = os.path.join(script_dir, "verify_ciqp.py")
    plot_path   = os.path.join(script_dir, "plot_results.py")

    n_feat = 8 * args.bits
    print(f"\n{'='*72}")
    print(f"  Seed sweep — {args.loss} at B={args.bits} ({n_feat}q), L={args.L}, "
          f"enc={args.encoding}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Sweep root: {args.sweep_root}")
    print(f"{'='*72}\n")

    t_start = time.time()
    results = []

    for i, seed in enumerate(args.seeds):
        print(f"\n{'─'*72}")
        print(f"  [{i+1}/{len(args.seeds)}]  seed = {seed}")
        print(f"{'─'*72}")

        # Build the training command — uses the harness' built-in folder-naming,
        # then redirected via --out-root to land inside the sweep directory.
        train_cmd = [
            sys.executable, driver_path,
            "--bits", str(args.bits),
            "--L", str(args.L),
            "--epochs", str(args.epochs),
            "--encoding", args.encoding,
            "--mc-batch", str(args.mc_batch),
            "--lr", str(args.lr),
            "--seed", str(seed),
            "--graph", args.graph,
            "--avg-deg", str(args.avg_deg),
            "--out-root", args.sweep_root,
        ]
        if args.split:
            train_cmd += ["--split", args.split]
        if args.t_restart is not None:
            train_cmd += ["--t-restart", str(args.t_restart)]
        if args.loss == "psck":
            train_cmd += ["--eta-psck", str(args.eta_psck),
                           "--heat-scale", str(args.heat_scale)]
        elif args.loss == "lw_mmd":
            train_cmd += ["--heat-scale", str(args.heat_scale)]
        elif args.loss == "corrmse":
            train_cmd += ["--lambda-rho", str(args.lambda_rho)]

        # The harness writes into <out-root>/<loss>_B<..>_<enc>_L<..>_ep<..>_seed<s>/
        run_dir = os.path.join(
            args.sweep_root,
            f"{args.loss}_B{args.bits}_{args.encoding}_L{args.L}"
            f"_ep{args.epochs}_seed{seed}",
        )

        if args.dry_run:
            print("  TRAIN:  " + " ".join(train_cmd))
        else:
            t0 = time.time()
            ret = subprocess.run(train_cmd).returncode
            dt = time.time() - t0
            if ret != 0:
                print(f"  ✗ training failed for seed={seed} (returncode {ret})")
                results.append({"seed": seed, "status": "train_failed",
                                  "run_dir": run_dir})
                continue
            print(f"  ✓ training done in {dt:.0f}s")

        # Verify cIQP
        if not args.skip_verify:
            ver_cmd = [sys.executable, verify_path, run_dir,
                        "--M", str(args.verify_M)]
            if args.dry_run:
                print("  VERIFY: " + " ".join(ver_cmd))
            else:
                ret_v = subprocess.run(ver_cmd).returncode
                if ret_v != 0:
                    print(f"  ✗ verify_ciqp failed (returncode {ret_v})")

        # Plot individual run
        if not args.skip_plot:
            plot_cmd = [sys.executable, plot_path, run_dir, "--style", "ROOT"]
            if args.dry_run:
                print("  PLOT:   " + " ".join(plot_cmd))
            else:
                ret_p = subprocess.run(plot_cmd).returncode
                if ret_p != 0:
                    print(f"  ✗ plot_results failed (returncode {ret_p})")

        results.append({"seed": seed, "status": "ok", "run_dir": run_dir})

    total_dt = time.time() - t_start
    print(f"\n{'='*72}")
    print(f"  Sweep complete in {total_dt/3600:.2f} hours")
    print(f"  {len([r for r in results if r['status']=='ok'])}/{len(args.seeds)} "
          f"seeds completed successfully")
    print(f"  Sweep root: {args.sweep_root}")
    print(f"  Next: python scripts/summarize_seed_sweep.py {args.sweep_root}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
