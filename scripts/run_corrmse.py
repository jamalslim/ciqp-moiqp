#!/usr/bin/env python
"""Train the explicit Pearson-MSE reference of Sec. V F.

This optimizes the correlation matrix directly rather than through a kernel on
the correlators. It is the reference that shows what happens when the
downstream functional is made the objective instead of being used to shape the
metric.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from psck_mmd.harness import run_driver

if __name__ == "__main__":
    run_driver(loss_kind="corrmse")
