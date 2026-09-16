"""
psck_mmd.data_split — Reproducible train/test split for the calorimeter dataset.

The dataset is N = 47,682 samples × D = 8 features. A single fixed 80/20 split
is created with a dedicated split-seed (independent of training seeds), saved
to data/split_indices.npz, and reused identically across every experiment.

This is the only place randomness enters the train/test boundary. All
training and evaluation in the package should call:

    train_binary, train_raw, test_binary, test_raw = load_calorimeter_split(
        path, bits, encoding,
    )

and never re-shuffle.

The split-indices file format:
    train_idx : (N_train,) int64
    test_idx  : (N_test,)  int64
    split_seed : int (for audit)
    n_total    : int (sanity check on dataset size)

Default split_seed = 20260420. Hardcoded; change only if you want to commit
to a new split (which invalidates all prior results).

Usage
-----
    from psck_mmd.data_split import load_calorimeter_split, ensure_split_exists

    # First-time use: creates split_indices.npz if missing
    ensure_split_exists(data_path="data/cal_shower_img_8q.npy",
                        split_path="data/split_indices.npz")

    # Standard load
    tr_bin, tr_raw, te_bin, te_raw = load_calorimeter_split(
        "data/cal_shower_img_8q.npy", bits=8, encoding="binary",
    )

Encoding-floor diagnostics: compute_encoding_fidelity(raw, bits, encoding)
returns the irreducible ρ-MAE incurred by the bit encoding alone, evaluated
on whichever sample set you pass in.
"""

from __future__ import annotations
import os
import numpy as np

# Reuse the existing loader for raw data + encoding mechanics.
# We import inside functions to avoid hard coupling at module import time.

DEFAULT_SPLIT_SEED = 20260420
DEFAULT_TEST_FRACTION = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# Split creation / loading
# ─────────────────────────────────────────────────────────────────────────────

def _make_split_indices(n_total: int,
                          test_fraction: float = DEFAULT_TEST_FRACTION,
                          split_seed: int = DEFAULT_SPLIT_SEED):
    """Deterministic permutation-based split. Identical across machines and
    numpy versions because we use a freshly-seeded Generator and a single
    permutation call."""
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n_total)
    n_test = int(round(test_fraction * n_total))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return train_idx, test_idx


def ensure_split_exists(data_path: str = "data/cal_shower_img_8q.npy",
                          split_path: str = "data/split_indices.npz",
                          test_fraction: float = DEFAULT_TEST_FRACTION,
                          split_seed: int = DEFAULT_SPLIT_SEED,
                          force: bool = False) -> str:
    """Create the split-indices file if missing. Returns the path.

    If `force=True`, regenerates and overwrites. Use only when intentionally
    invalidating prior split-dependent results.
    """
    if os.path.exists(split_path) and not force:
        # Sanity check: the existing split must agree with the current dataset size.
        d = np.load(data_path)
        existing = np.load(split_path)
        if int(existing["n_total"]) != int(d.shape[0]):
            raise RuntimeError(
                f"Existing split at {split_path} has n_total={int(existing['n_total'])} "
                f"but dataset at {data_path} has {d.shape[0]} rows. Refusing to "
                f"silently use a stale split. Pass force=True to regenerate."
            )
        return split_path

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data file not found: {data_path}")

    raw = np.load(data_path)
    n_total = int(raw.shape[0])
    train_idx, test_idx = _make_split_indices(
        n_total, test_fraction=test_fraction, split_seed=split_seed,
    )
    os.makedirs(os.path.dirname(split_path) or ".", exist_ok=True)
    np.savez(split_path,
             train_idx=train_idx,
             test_idx=test_idx,
             split_seed=np.int64(split_seed),
             n_total=np.int64(n_total),
             test_fraction=np.float64(test_fraction))
    print(f"  [data_split] created {split_path}: "
          f"N_train={len(train_idx)}, N_test={len(test_idx)} "
          f"(split_seed={split_seed})")
    return split_path


def load_split_indices(split_path: str = "data/split_indices.npz"):
    """Return (train_idx, test_idx) int arrays from the saved file."""
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"split file not found: {split_path}. "
            f"Call ensure_split_exists() first."
        )
    d = np.load(split_path)
    return d["train_idx"].astype(np.int64), d["test_idx"].astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Split-aware data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_calorimeter_split(
    data_path: str = "data/cal_shower_img_8q.npy",
    bits: int = 2,
    encoding: str = "binary",
    split_path: str = "data/split_indices.npz",
    auto_create_split: bool = True,
):
    """Load the calorimeter dataset and apply the fixed train/test split.

    Important: percentile bin edges (used for quantile encoding) are
    computed from the **training set only**, then applied to both train
    and test. This prevents test-set leakage into the encoding choices.

    Returns
    -------
    train_binary : (N_train, D*bits) int8
    train_raw    : (N_train, D)      float64
    test_binary  : (N_test,  D*bits) int8
    test_raw     : (N_test,  D)      float64
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data file not found: {data_path}")
    if auto_create_split:
        ensure_split_exists(data_path=data_path, split_path=split_path)
    train_idx, test_idx = load_split_indices(split_path)

    raw = np.load(data_path).astype(np.float64)
    train_raw = raw[train_idx]
    test_raw  = raw[test_idx]

    train_binary = _encode_quantile_with_train_bins(
        train_raw, train_raw, bits=bits, encoding=encoding,
    )
    test_binary = _encode_quantile_with_train_bins(
        test_raw, train_raw, bits=bits, encoding=encoding,
    )
    return train_binary, train_raw, test_binary, test_raw


def _encode_quantile_with_train_bins(samples_raw: np.ndarray,
                                       train_raw: np.ndarray,
                                       bits: int,
                                       encoding: str) -> np.ndarray:
    """Quantile-encode `samples_raw` using percentile bin edges fit on `train_raw`.

    This deliberately matches the encoding contract used by
    psck_mmd.data.load_calorimeter() but factors out the bin-fitting step so
    that test-set encoding uses the train-set's bin edges (no leakage).
    """
    # Lazy import to avoid circular imports at module load
    from psck_mmd.data import ENCODING_TABLES

    if encoding not in ENCODING_TABLES:
        raise ValueError(f"unknown encoding {encoding!r}")
    nb = 2 ** bits
    D = train_raw.shape[1]
    N = samples_raw.shape[0]

    # Bin edges from TRAIN ONLY
    pcts = np.percentile(
        train_raw, np.linspace(100.0 / nb, 100.0 - 100.0 / nb, nb - 1), axis=0,
    )
    bidx = np.zeros((N, D), dtype=int)
    for f in range(D):
        bidx[:, f] = np.digitize(samples_raw[:, f], pcts[:, f])

    table = ENCODING_TABLES[encoding](nb, bits)
    binary = np.zeros((N, D * bits), dtype=np.int8)
    for f in range(D):
        for k in range(bits):
            binary[:, f * bits + k] = table[bidx[:, f], k]
    return binary


# ─────────────────────────────────────────────────────────────────────────────
# Encoding-fidelity diagnostic, evaluated on either split
# ─────────────────────────────────────────────────────────────────────────────

def compute_encoding_fidelity(
    binary: np.ndarray,
    raw: np.ndarray,
    bits: int,
    D: int = 8,
):
    """Compute the irreducible ρ-MAE incurred by the encoding alone.

    Defined as: <|ρ_ij(raw) − ρ_ij(decoded(binary))|>_{i<j},
    where decoded(binary) reconstructs S_f = Σ_k 2^{B-1-k} b_{f,k} for binary
    encoding (or the appropriate map for other encodings — see notes).

    For binary encoding this is exact. For Gray and Hadamard encodings, this
    function decodes by mapping each B-bit pattern back to an integer level
    using the same per-encoding table.

    Returns a dict {rho_mae_floor, rho_pearson_r_floor, raw_corr, decoded_corr}.
    """
    from psck_mmd.data import ENCODING_TABLES

    nb = 2 ** bits
    N = binary.shape[0]

    # Reconstruct integer-level per feature from bits.
    # Build inverse lookup: bit-pattern -> level. For all three encodings the
    # encoding table is square 2^B × B, with each row a unique pattern.
    # We need to choose ONE encoding to do the reverse mapping, but at the
    # call site we're computing the floor of an EXISTING binary array — we
    # don't actually know which encoding produced it from this signature.
    # Solution: take encoding as an argument.
    raise NotImplementedError(
        "compute_encoding_fidelity requires the encoding name; use "
        "compute_encoding_fidelity_with_encoding(...) instead."
    )


def compute_encoding_fidelity_with_encoding(
    binary: np.ndarray,
    raw: np.ndarray,
    bits: int,
    encoding: str,
    D: int = 8,
):
    """As above, but with explicit `encoding` argument so we can invert
    the bit pattern back to integer levels.

    Returns dict with:
        rho_mae_floor       : <|ρ(raw) − ρ(decoded)|>_{i<j}
        rho_r_floor         : Pearson r between the two flattened ρ vectors
        decoded_levels      : (N, D) int per-sample feature levels
        raw_corr            : (D, D) Pearson ρ of raw
        decoded_corr        : (D, D) Pearson ρ of decoded levels
    """
    from psck_mmd.data import ENCODING_TABLES

    nb = 2 ** bits
    if encoding not in ENCODING_TABLES:
        raise ValueError(f"unknown encoding {encoding!r}")
    table = ENCODING_TABLES[encoding](nb, bits)   # (nb, bits) in {0,1}

    # Build inverse: pattern (tuple) -> level
    pattern_to_level: dict[tuple[int, ...], int] = {}
    for v in range(nb):
        key = tuple(int(x) for x in table[v])
        if key in pattern_to_level:
            # Duplicate row in encoding table — shouldn't happen for any
            # well-defined per-feature B-bit code where all 2^B levels exist.
            raise RuntimeError(
                f"Encoding {encoding!r} at B={bits} has duplicate row {v}; "
                f"inversion is not well defined."
            )
        pattern_to_level[key] = v

    N = binary.shape[0]
    decoded = np.zeros((N, D), dtype=np.int32)
    for f in range(D):
        bits_f = binary[:, f * bits:(f + 1) * bits]   # (N, bits)
        for n in range(N):
            decoded[n, f] = pattern_to_level[tuple(int(x) for x in bits_f[n])]

    raw_corr = np.corrcoef(raw.T)
    dec_corr = np.corrcoef(decoded.T.astype(np.float64))
    iu = np.triu_indices(D, k=1)
    diff = raw_corr[iu] - dec_corr[iu]
    rho_mae = float(np.mean(np.abs(diff)))
    rho_r = float(np.corrcoef(raw_corr[iu], dec_corr[iu])[0, 1])

    return {
        "rho_mae_floor": rho_mae,
        "rho_r_floor":   rho_r,
        "decoded_levels": decoded,
        "raw_corr":      raw_corr,
        "decoded_corr":  dec_corr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: header info for paper/table
# ─────────────────────────────────────────────────────────────────────────────

def split_summary(data_path: str = "data/cal_shower_img_8q.npy",
                    split_path: str = "data/split_indices.npz") -> dict:
    """Lightweight summary of the split — useful for the paper's data section."""
    d = np.load(split_path)
    raw = np.load(data_path)
    return {
        "n_total":      int(raw.shape[0]),
        "n_features":   int(raw.shape[1]),
        "n_train":      int(d["train_idx"].size),
        "n_test":       int(d["test_idx"].size),
        "split_seed":   int(d["split_seed"]),
        "test_fraction": float(d["test_fraction"]),
    }
