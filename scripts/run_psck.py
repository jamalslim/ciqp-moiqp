#!/usr/bin/env python
"""Train a PSCK-MoIQP model.

This is the headline objective of the paper, Sec. V B. The loss is the
Walsh-diagonal MMD of Eq. 6 with the rank-P Pearson correction of Eq. 11,
evaluated by the Van den Nest estimator so that no quantum device is involved.

A headline seed is

    python scripts/run_psck.py --bits 8 --L 8 --epochs 1500 --mc-batch 4096 \
           --lr 0.02 --eta-psck 5.0 --seed 42 --split data/split_indices.npz \
           --output outputs/psck_B8_binary_L8_ep1500_seed42

and takes about 2.5 hours on one core. Reduce --bits to 3 and --epochs to 60
for a ten-second check that the install works. Run run_evaluation.py afterwards
to obtain the split-correct test numbers that Tables I and X report; the values
printed here are training-split only.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from psck_mmd.harness import run_driver

if __name__ == "__main__":
    run_driver(loss_kind="psck")
