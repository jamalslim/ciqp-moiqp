#!/usr/bin/env python
"""Train the Liu and Wang heat-kernel baseline.

Same pipeline as run_psck.py with the rank-P correction switched off, so the
comparison in Table I differs only in the kernel. Use identical --bits, --L,
--epochs, --seed and --split when comparing, or the difference is not
attributable to the objective.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from psck_mmd.harness import run_driver

if __name__ == "__main__":
    run_driver(loss_kind="lw_mmd")
