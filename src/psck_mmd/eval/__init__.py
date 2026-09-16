"""psck_mmd.eval — Held-out evaluation, distributional metrics, and trainability diagnostics.

This subpackage groups everything that is computed AFTER training:
    data_split           — fixed train/test split with reproducible indices
                           (also re-exported at the top-level psck_mmd.data_split for
                           backwards compatibility with print_split_diagnostics.py)
    marginal_recovery    — Walsh–Hadamard inversion of intra-feature Z-correlators
    distributional_metrics — Wasserstein-1, KS, joint moment-MMD
    barren_plateau       — gradient variance scaling at random parameter inits
"""

from .marginal_recovery import (
    enumerate_intra_feature_observables,
    recover_feature_distributions,
    empirical_feature_distributions,
    moment_mmd_from_correlators,
)
from .distributional_metrics import *  # exports defined in that module's __all__
from .barren_plateau import *
