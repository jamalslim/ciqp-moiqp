"""psck_mmd — Pearson-Stabilized Correlation Kernel MMD for IQP Born machines.

Core modules:
    correlator   — Van den Nest Fourier Monte Carlo engine
    graph        — IQP graph constructors (complete, Erdős–Rényi)
    feature_obs  — enumeration of Z-observables for feature-structured data
    data         — calorimeter data loader with binary/Gray encoding
    corr_jacobian— analytic Pearson-correlation Jacobian ∂ρ/∂⟨Z_β⟩
    psck_kernel  — heat coefficients, K = diag(ω) + η Jᵀ J, biased/unbiased MMD²
    train_moiqp  — MoIQP trainer with three swappable losses
    ciqp         — Walsh–Hadamard / deferred-measurement cIQP compiler
    plotting     — ROOT-style figures via mplhep (optional import)

Public API:
    from psck_mmd import (
        build_graph, enumerate_feature_observables, load_calorimeter,
        ParityCache, sample_latents, forward_batch, backward_batch,
        pearson_value_and_jacobian, build_psck_kernel,
        psck_mmd2_and_grad, psck_mmd2_unbiased,
        train_moiqp, evaluate_mixture,
        build_ciqp_circuit,
    )
"""

from .correlator import (
    ParityCache, sample_latents, forward_batch, backward_batch, z_data_batch,
)
from .graph import (
    build_complete_graph, build_er_graph, build_graph,
    active_gates_for_observable, precompute_active_lists,
)
from .feature_obs import enumerate_feature_observables
from .data import load_calorimeter
from .corr_jacobian import pearson_value_and_jacobian
from .psck_kernel import (
    heat_kernel_coeffs, build_psck_kernel,
    psck_mmd2_and_grad, psck_mmd2_unbiased, psck_diagnostics,
)
from .train_moiqp import train_moiqp, evaluate_mixture
from .ciqp import build_ciqp_circuit, print_ciqp_info

from .plotting import (
    set_root_style, save_heatmap, save_corr_triplet_separate,
    save_marginals_separate, save_scatter, save_bar, save_image,
)

__version__ = "0.1.0"
