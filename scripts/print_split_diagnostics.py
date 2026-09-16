#!/usr/bin/env python
"""Print train/test split summary and encoding-fidelity floors for a given B.

This is a quick diagnostic: shows the split sizes, the per-split encoding
floors (irreducible ρ-MAE from the bit encoding alone), and the per-feature
correlation-matrix Frobenius difference between train and test ρ.

Usage:
    python scripts/print_split_diagnostics.py --bits 2
    python scripts/print_split_diagnostics.py --bits 8 --encoding hadamard
    python scripts/print_split_diagnostics.py --bits 8 --regenerate-split

If split_indices.npz does not exist under data/, it is created with the
default split_seed = 20260420.
"""

import argparse
import os
import sys

# Allow running from repo root: scripts/ is a sibling of src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from psck_mmd.data_split import (
    ensure_split_exists, load_calorimeter_split,
    compute_encoding_fidelity_with_encoding, split_summary,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bits", "-B", type=int, default=8)
    ap.add_argument("--encoding", type=str, default="binary",
                     choices=["binary", "gray", "hadamard"])
    ap.add_argument("--data", type=str,
                     default="data/cal_shower_img_8q.npy")
    ap.add_argument("--split", type=str,
                     default="data/split_indices.npz")
    ap.add_argument("--regenerate-split", action="store_true",
                     help="force regeneration of the split (invalidates prior results)")
    args = ap.parse_args()

    ensure_split_exists(data_path=args.data, split_path=args.split,
                          force=args.regenerate_split)

    s = split_summary(data_path=args.data, split_path=args.split)
    print(f"\n{'='*72}")
    print(f"  Calorimeter dataset split")
    print(f"{'='*72}")
    print(f"  data path     : {args.data}")
    print(f"  split path    : {args.split}")
    print(f"  N total       : {s['n_total']}")
    print(f"  N train       : {s['n_train']}  ({(1 - s['test_fraction']) * 100:.0f}%)")
    print(f"  N test        : {s['n_test']}   ({s['test_fraction'] * 100:.0f}%)")
    print(f"  D features    : {s['n_features']}")
    print(f"  split seed    : {s['split_seed']}\n")

    print(f"  Loading with bits = {args.bits}, encoding = {args.encoding} ...")
    tr_bin, tr_raw, te_bin, te_raw = load_calorimeter_split(
        data_path=args.data, bits=args.bits, encoding=args.encoding,
        split_path=args.split,
    )

    print(f"\n{'='*72}")
    print(f"  Encoding-fidelity floors  (bits={args.bits}, encoding={args.encoding})")
    print(f"{'='*72}")
    res_tr = compute_encoding_fidelity_with_encoding(
        tr_bin, tr_raw, bits=args.bits, encoding=args.encoding,
    )
    res_te = compute_encoding_fidelity_with_encoding(
        te_bin, te_raw, bits=args.bits, encoding=args.encoding,
    )
    print(f"  TRAIN ρ MAE floor      : {res_tr['rho_mae_floor']:.6f}")
    print(f"  TEST  ρ MAE floor      : {res_te['rho_mae_floor']:.6f}")
    print(f"  TRAIN ρ Pearson r floor: {res_tr['rho_r_floor']:.6f}")
    print(f"  TEST  ρ Pearson r floor: {res_te['rho_r_floor']:.6f}\n")

    # Train-vs-test ρ matrix difference (sanity check on split balance)
    iu = np.triu_indices(s["n_features"], k=1)
    rho_tr_raw = np.corrcoef(tr_raw.T)
    rho_te_raw = np.corrcoef(te_raw.T)
    diff = rho_tr_raw[iu] - rho_te_raw[iu]
    print(f"  Train-vs-test raw-ρ MAE: {np.mean(np.abs(diff)):.6f}")
    print(f"  Train-vs-test raw-ρ r  : {np.corrcoef(rho_tr_raw[iu], rho_te_raw[iu])[0,1]:.6f}")
    print()


if __name__ == "__main__":
    main()
