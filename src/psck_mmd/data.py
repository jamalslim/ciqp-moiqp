"""
psck_mmd.data — Calorimeter loader with binary / Gray / Hadamard encoding.

Encodings
---------
"binary" : natural binary code, S_f = Σ_k 2^{B-1-k} b_{f,k}
           — this is the encoding for which the Pearson Jacobian in
             corr_jacobian.py is analytic; use this for PSCK training.

"gray"   : reflected Gray code. Single-bit flips between adjacent levels.
           Pearson Jacobian is no longer bilinear in single-bit stats.

"hadamard" : NEW, original. Each feature level v ∈ {0,...,2^B-1} is mapped
             to row v of a ±1 Walsh–Hadamard matrix, then converted to bits
             via (+1 → 0, -1 → 1). The encoding spreads the level index
             across all B bits with maximal pairwise Hamming distance,
             giving IQP Born machines a more uniform gradient signal
             across bit planes. Theoretical analysis of this encoding
             w.r.t. the Pearson Jacobian is deferred; for now use it as
             a third baseline to probe encoding sensitivity.

             Sylvester recursion:
                H_1 = [[1]]
                H_{2n} = [[ H_n,  H_n],
                          [ H_n, -H_n]]
             Each row has exactly 2^{B-1} ones (for B ≥ 1), so the bit
             marginals are balanced.

Loader returns both the (N, D*bits) int8 encoded matrix and the raw (N,D)
float data for ground-truth correlation targets.
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Encoding tables
# ─────────────────────────────────────────────────────────────────────────────

def _binary_table(nb: int, bits: int) -> np.ndarray:
    table = np.zeros((nb, bits), dtype=np.int8)
    for v in range(nb):
        for k in range(bits):
            table[v, bits - 1 - k] = (v >> k) & 1
    return table


def _gray_table(nb: int, bits: int) -> np.ndarray:
    table = np.zeros((nb, bits), dtype=np.int8)
    for v in range(nb):
        g = v ^ (v >> 1)
        for k in range(bits):
            table[v, bits - 1 - k] = (g >> k) & 1
    return table


def _hadamard_table(nb: int, bits: int) -> np.ndarray:
    """WH encoding: level v -> row v of ±1 Hadamard matrix, bits = (1-H)//2."""
    N = 1 << bits
    H = np.array([[1]], dtype=np.int8)
    while H.shape[0] < N:
        H = np.block([[H, H], [H, -H]])
    table = ((1 - H) // 2).astype(np.int8)      # (N, N) in {0,1}
    return table[:nb, :bits].copy()


ENCODING_TABLES = {
    "binary":   _binary_table,
    "gray":     _gray_table,
    "hadamard": _hadamard_table,
}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_fallback(n_samples: int = 47682, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    mu = np.array([.0155, .0885, .2293, .3445, .3804, .3116, .1618, .0684])
    s = np.array([.0094, .0361, .0627, .0658, .0564, .0450, .0454, .0468])
    C = np.eye(8)
    for i in range(7): C[i, i+1] = C[i+1, i] = 0.7
    for i in range(6): C[i, i+2] = C[i+2, i] = 0.3
    L = np.linalg.cholesky(C).T
    raw = np.clip(rng.randn(n_samples, 8) @ L * s + mu, 3e-4, 0.5)
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

LAST_PROVENANCE: dict = {}


def load_calorimeter(path: str, bits: int = 2, encoding: str = "binary",
                     allow_synthetic: bool = False):
    """Load calorimeter data and quantile-encode each feature into `bits` bits."""
    if encoding not in ENCODING_TABLES:
        raise ValueError(
            f"unknown encoding {encoding!r}; choose one of {sorted(ENCODING_TABLES)}"
        )

    # v4.14 (review item B8): fail LOUDLY on missing/invalid data. The silent
    # path-walk + synthetic substitution is gone; synthetic data is opt-in only.
    import hashlib, os
    raw: np.ndarray | None = None
    if allow_synthetic and (path is None or not os.path.exists(path)):
        print("=" * 70)
        print("WARNING: SYNTHETIC DATA REQUESTED (allow_synthetic=True).")
        print("         No physics conclusions are valid from this run.")
        print("=" * 70)
        raw = _synthetic_fallback()
        LAST_PROVENANCE.update(path="SYNTHETIC", sha256=None, shape=raw.shape)
    else:
        try:
            d = np.load(path)
        except Exception as e:
            raise FileNotFoundError(
                f"load_calorimeter: cannot load {path!r} ({e}). "
                "Pass allow_synthetic=True ONLY for self-audit runs."
            ) from e
        if d.ndim != 2:
            raise ValueError(f"load_calorimeter: {path!r} has ndim={d.ndim}, expected 2.")
        raw = d
        with open(path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        LAST_PROVENANCE.update(path=os.path.abspath(path), sha256=sha, shape=raw.shape)
        print(f"[data] loaded {path}  shape={raw.shape}  sha256={sha[:16]}...")

    N, D = raw.shape
    nb = 2 ** bits
    pcts = np.percentile(
        raw, np.linspace(100.0 / nb, 100.0 - 100.0 / nb, nb - 1), axis=0
    )
    bidx = np.zeros((N, D), dtype=int)
    for f in range(D):
        bidx[:, f] = np.digitize(raw[:, f], pcts[:, f])

    table = ENCODING_TABLES[encoding](nb, bits)
    binary = np.zeros((N, D * bits), dtype=np.int8)
    for f in range(D):
        for k in range(bits):
            binary[:, f * bits + k] = table[bidx[:, f], k]
    return binary, raw


def load_calorimeter_split(path: str, bits: int = 2, encoding: str = "binary",
                           split_path: str = "data/split_indices.npz"):
    """Split-correct loader (review item A1).

    Quantile edges are fit on the TRAINING split only; both splits are
    encoded with those train-fit edges. Returns a dict with binary/raw
    arrays per split, the edges, per-feature train-median decode table,
    and the split-specific encoding floors.
    """
    binary_all, raw = load_calorimeter(path, bits=bits, encoding=encoding)
    sp = np.load(split_path)
    tr, te = sp["train_idx"], sp["test_idx"]
    assert len(np.intersect1d(tr, te)) == 0, "split indices overlap"
    nb = 2 ** bits
    Xtr, Xte = raw[tr], raw[te]
    pcts = np.percentile(Xtr, np.linspace(100.0 / nb, 100.0 - 100.0 / nb, nb - 1), axis=0)
    table = ENCODING_TABLES[encoding](nb, bits)

    def _enc(X):
        N, D = X.shape
        bidx = np.zeros((N, D), dtype=int)
        for f in range(D):
            bidx[:, f] = np.digitize(X[:, f], pcts[:, f])
        binary = np.zeros((N, D * bits), dtype=np.int8)
        for f in range(D):
            for k in range(bits):
                binary[:, f * bits + k] = table[bidx[:, f], k]
        return binary, bidx

    bin_tr, lev_tr = _enc(Xtr)
    bin_te, lev_te = _enc(Xte)
    medians = fit_decoder(Xtr, pcts)
    iu = np.triu_indices(raw.shape[1], 1)

    def _floor(lev, X):
        return float(np.mean(np.abs(np.corrcoef(lev.T.astype(float))[iu] - np.corrcoef(X.T)[iu])))

    return dict(binary_train=bin_tr, binary_test=bin_te, raw_train=Xtr, raw_test=Xte,
                levels_train=lev_tr, levels_test=lev_te, edges=pcts, decode_medians=medians,
                floor_train=_floor(lev_tr, Xtr), floor_test=_floor(lev_te, Xte),
                train_idx=tr, test_idx=te)


def fit_decoder(raw_train: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Level -> energy decode map (review item D5): per-feature train-split bin medians.

    Returns medians[level, feature]; empty bins fall back to the bin-edge midpoint.
    """
    N, D = raw_train.shape
    nb = edges.shape[0] + 1
    med = np.zeros((nb, D))
    for f in range(D):
        lev = np.digitize(raw_train[:, f], edges[:, f])
        lo = np.concatenate(([raw_train[:, f].min()], edges[:, f]))
        hi = np.concatenate((edges[:, f], [raw_train[:, f].max()]))
        for l in range(nb):
            v = raw_train[lev == l, f]
            med[l, f] = np.median(v) if len(v) else 0.5 * (lo[l] + hi[l])
    return med


def decode_levels_to_energy(levels: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Map integer levels (N, D) to calorimeter energies via the committed decoder."""
    N, D = levels.shape
    out = np.empty((N, D))
    for f in range(D):
        out[:, f] = medians[levels[:, f], f]
    return out
