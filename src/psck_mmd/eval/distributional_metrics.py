"""
psck_mmd.eval.distributional_metrics — IQP-native distributional comparisons.

All metrics here are evaluated WITHOUT sampling from the model. They take
either:
  (a) per-feature exact level distributions (D, 2^B) recovered via
      marginal_recovery.recover_feature_distributions(...), or
  (b) low-order Z-correlator vectors estimated from the model and from
      held-out data via the Van den Nest oracle.

Provided metrics
----------------
Per-feature (closed form on the recovered distributions):
  * wasserstein1_per_feature(p_model, p_data)                → (D,) array
  * ks_per_feature(p_model, p_data)                          → (D,) array
  * tv_per_feature(p_model, p_data)                          → (D,) array
  * chi2_per_feature(p_model, p_data, eps=1e-6)              → (D,) array

Joint distributional (closed form on Z-correlator vectors):
  * moment_mmd(z_model, z_data, kernel_coeffs)               → scalar

Aggregators:
  * summarize_distribution_metrics(p_model, p_data)          → dict

Each per-feature metric is exact for the integer-level lattice {0,…,2^B−1}
and uses the convention that the underlying support is uniformly spaced on
that lattice (i.e., we do NOT decode levels back to physical-unit feature
values; we measure level-distribution agreement). This is the natural
metric for a discretely-encoded generative model. To convert to physical
units, multiply Wasserstein-1 by the per-feature average bin width.
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Per-feature distributional distances (closed form on (D, 2^B) arrays)
# ─────────────────────────────────────────────────────────────────────────────

def _check_pf(p_model: np.ndarray, p_data: np.ndarray):
    if p_model.shape != p_data.shape:
        raise ValueError(f"shape mismatch: {p_model.shape} vs {p_data.shape}")
    if p_model.ndim != 2:
        raise ValueError("expected (D, n_levels) per-feature distributions")
    # Allow tiny clipping/normalization slack
    s_m = p_model.sum(axis=1)
    s_d = p_data.sum(axis=1)
    if not np.allclose(s_m, 1.0, atol=1e-6) or not np.allclose(s_d, 1.0, atol=1e-6):
        raise ValueError(
            "per-feature rows must each sum to 1.0; got sums "
            f"model={s_m}, data={s_d}"
        )


def wasserstein1_per_feature(p_model: np.ndarray, p_data: np.ndarray) -> np.ndarray:
    """W₁(p_model[f], p_data[f]) for each feature f, treating support as the
    integer lattice {0, …, n_levels − 1}.

    Closed form on a 1-D ordered support: W₁ = Σ_v |F_model(v) − F_data(v)|
    summed over level boundaries. Equivalently, integral of |Δ CDF|.
    """
    _check_pf(p_model, p_data)
    D, n = p_model.shape
    cdf_m = np.cumsum(p_model, axis=1)
    cdf_d = np.cumsum(p_data, axis=1)
    # Sum of |ΔCDF| over the n-1 internal boundaries (last entry is 0 by
    # construction since both CDFs end at 1, so summing all n entries is fine).
    return np.abs(cdf_m - cdf_d).sum(axis=1)


def ks_per_feature(p_model: np.ndarray, p_data: np.ndarray) -> np.ndarray:
    """Kolmogorov–Smirnov per-feature: max |F_model(v) − F_data(v)|."""
    _check_pf(p_model, p_data)
    cdf_m = np.cumsum(p_model, axis=1)
    cdf_d = np.cumsum(p_data, axis=1)
    return np.abs(cdf_m - cdf_d).max(axis=1)


def tv_per_feature(p_model: np.ndarray, p_data: np.ndarray) -> np.ndarray:
    """Total-variation distance per feature: (1/2) Σ_v |p_model(v) − p_data(v)|."""
    _check_pf(p_model, p_data)
    return 0.5 * np.abs(p_model - p_data).sum(axis=1)


def chi2_per_feature(p_model: np.ndarray, p_data: np.ndarray,
                       eps: float = 1e-6) -> np.ndarray:
    """χ² distance per feature using `data` as reference. eps regularises
    bins where p_data ≈ 0 to avoid division-by-zero."""
    _check_pf(p_model, p_data)
    return ((p_model - p_data) ** 2 / (p_data + eps)).sum(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Joint distributional metrics on low-order Z-correlator vectors
# ─────────────────────────────────────────────────────────────────────────────

def moment_mmd(z_model: np.ndarray,
                z_data:  np.ndarray,
                kernel_coeffs: np.ndarray) -> float:
    """Closed-form MMD² on Z-correlator vectors with Fourier weights k̂_β.

        MMD² = Σ_β k̂_β · (z_model[β] − z_data[β])²

    Use this with the SAME kernel the model was trained on (heat or PSCK)
    to evaluate generalization on the training-objective distance.
    Use this with an INDEPENDENT kernel (e.g. uniform k̂_β = 1) to get an
    objective-agnostic moment-matching score.
    """
    if not (z_model.shape == z_data.shape == kernel_coeffs.shape):
        raise ValueError(
            f"shapes: z_model={z_model.shape}, z_data={z_data.shape}, "
            f"kc={kernel_coeffs.shape}"
        )
    if (kernel_coeffs < 0).any():
        raise ValueError("kernel coefficients must be non-negative")
    delta = z_model - z_data
    return float((kernel_coeffs * delta * delta).sum())


def z_vector_rbf_mmd2(*args, **kwargs):
    """REMOVED. This metric was degenerate.

    With model and data each represented by a single deterministic point
    z_model, z_data ∈ ℝ^K (the low-order Z-correlator vectors), the
    RBF MMD² reduces to 2(1 - exp(-‖δz‖² / 2σ²)). For any sensible
    bandwidth σ at K=2080 (B=8) the exponent is large, the kernel
    saturates to ~0, and the metric saturates to ~2 regardless of δz.
    The output carried no information beyond what moment_mmd() already
    reports, and was misleading.

    Use moment_mmd(z_model, z_data, kernel_coeffs) for kernel-aligned
    moment-domain distance, with the appropriate kernel_coeffs for the
    semantic you want (e.g. heat_kernel_coeffs from psck_kernel for
    bandwidth-mixed Liu-Wang style).
    """
    raise NotImplementedError(
        "z_vector_rbf_mmd2 was removed; it was a degenerate metric that "
        "saturated to 2 for any nonzero δz at K = O(2000). Use "
        "moment_mmd(z_model, z_data, kernel_coeffs) instead."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator: one-call summary on (D, 2^B) per-feature distributions
# ─────────────────────────────────────────────────────────────────────────────

def summarize_distribution_metrics(
    p_model: np.ndarray,
    p_data:  np.ndarray,
    bin_widths: np.ndarray | None = None,
) -> dict:
    """All per-feature metrics + their D-mean and D-max aggregates.

    `bin_widths` of shape (D,) optionally rescales W₁ from level-units
    to physical-feature-units. If None, returns level-unit W₁.
    """
    w1 = wasserstein1_per_feature(p_model, p_data)
    ks = ks_per_feature(p_model, p_data)
    tv = tv_per_feature(p_model, p_data)
    ch = chi2_per_feature(p_model, p_data)
    out = {
        "wasserstein1_per_feature_levels": w1.tolist(),
        "wasserstein1_mean_levels": float(w1.mean()),
        "wasserstein1_max_levels":  float(w1.max()),
        "ks_per_feature":           ks.tolist(),
        "ks_mean":                  float(ks.mean()),
        "ks_max":                   float(ks.max()),
        "tv_per_feature":           tv.tolist(),
        "tv_mean":                  float(tv.mean()),
        "tv_max":                   float(tv.max()),
        "chi2_per_feature":         ch.tolist(),
        "chi2_mean":                float(ch.mean()),
        "chi2_max":                 float(ch.max()),
    }
    if bin_widths is not None:
        bw = np.asarray(bin_widths, dtype=np.float64)
        if bw.shape != (p_model.shape[0],):
            raise ValueError(f"bin_widths must have shape ({p_model.shape[0]},)")
        out["wasserstein1_per_feature_physical"] = (w1 * bw).tolist()
        out["wasserstein1_mean_physical"] = float((w1 * bw).mean())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Audits
# ─────────────────────────────────────────────────────────────────────────────

def audit_metrics() -> bool:
    """Self-test: identical distributions give zero on all metrics; a known
    1-step shift gives W₁ = 1, KS = 1 (for a delta), TV = 1."""
    n = 8
    # Two delta distributions, p_model on level 3, p_data on level 4
    p_m = np.zeros((1, n)); p_m[0, 3] = 1.0
    p_d = np.zeros((1, n)); p_d[0, 4] = 1.0
    w1 = wasserstein1_per_feature(p_m, p_d)
    ks = ks_per_feature(p_m, p_d)
    tv = tv_per_feature(p_m, p_d)
    ok = bool(np.allclose(w1, 1.0) and np.allclose(ks, 1.0) and np.allclose(tv, 1.0))
    print(f"  metrics audit (delta-shift-by-1): "
          f"W1={float(w1[0]):.4f}, KS={float(ks[0]):.4f}, TV={float(tv[0]):.4f}  "
          f"{'✓' if ok else '✗'}")

    # Identical distributions → all zero
    p = np.array([[0.1, 0.2, 0.3, 0.4]])
    ok2 = bool(
        np.allclose(wasserstein1_per_feature(p, p), 0.0)
        and np.allclose(ks_per_feature(p, p), 0.0)
        and np.allclose(tv_per_feature(p, p), 0.0)
    )
    print(f"  metrics audit (identical)        : {'✓' if ok2 else '✗'}")

    # Moment-MMD: zero for identical correlator vectors, positive otherwise
    z = np.array([0.1, -0.2, 0.3])
    kc = np.array([1.0, 1.0, 1.0])
    ok3 = (moment_mmd(z, z, kc) == 0.0
           and moment_mmd(z, z + 0.01, kc) > 0.0)
    print(f"  metrics audit (moment_mmd)       : {'✓' if ok3 else '✗'}")

    return ok and ok2 and ok3


if __name__ == "__main__":
    audit_metrics()