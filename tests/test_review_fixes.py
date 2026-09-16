"""Regression tests for the review-fix patches (A1/A3/B5/B8/D5), v4.14."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from psck_mmd.data import (load_calorimeter, load_calorimeter_split,
                           decode_levels_to_energy, LAST_PROVENANCE)
from psck_mmd.ciqp import _pad_to_power_of_2
from psck_mmd.feature_obs import enumerate_feature_observables
from math import comb


def test_loader_fails_loudly_on_missing_file():
    try:
        load_calorimeter("does_not_exist.npy", bits=2)
    except FileNotFoundError:
        return
    raise AssertionError("silent fallback still active (B8)")


def test_loader_records_provenance():
    load_calorimeter("data/cal_shower_img_8q.npy", bits=2)
    assert LAST_PROVENANCE["shape"] == (47682, 8)
    assert len(LAST_PROVENANCE["sha256"]) == 64


def test_split_loader_floors_match_review():
    r = load_calorimeter_split("data/cal_shower_img_8q.npy", bits=8)
    assert abs(r["floor_train"] - 0.05154) < 2e-4, r["floor_train"]
    assert abs(r["floor_test"] - 0.05480) < 2e-4, r["floor_test"]
    assert len(np.intersect1d(r["train_idx"], r["test_idx"])) == 0


def test_decoder_roundtrip_within_bin():
    r = load_calorimeter_split("data/cal_shower_img_8q.npy", bits=8)
    dec = decode_levels_to_energy(r["levels_train"][:2000], r["decode_medians"])
    # decoded energy must land in the same bin as the original level
    for f in range(dec.shape[1]):
        lev2 = np.digitize(dec[:, f], r["edges"][:, f])
        assert np.mean(lev2 == r["levels_train"][:2000, f]) > 0.99


def test_padding_guard_raises_at_L3():
    params = [np.zeros(4), np.ones(4), np.ones(4)]
    try:
        _pad_to_power_of_2(params)
    except ValueError as e:
        assert "power of 2" in str(e)
        return
    raise AssertionError("non-power-of-two padding still allowed (A3)")


def test_padding_noop_at_L4():
    params = [np.ones(4) * k for k in range(4)]
    out, a = _pad_to_power_of_2(params)
    assert len(out) == 4 and a == 2


def test_weight3_is_intra_feature():
    D, B = 8, 8
    obs, _ = enumerate_feature_observables(D, B, max_weight=3)
    triples = [S for S in obs if len(S) == 3]
    assert len(triples) == D * comb(B, 3), (len(triples), D * comb(B, 3))
    for S in triples:
        feats = {q // B for q in S}
        assert len(feats) == 1, f"cross-feature triple {S} (B5 not fixed)"


if __name__ == "__main__":
    ok = fail = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); ok += 1; print("PASS", name)
            except Exception as e:
                fail += 1; print("FAIL", name, ":", e)
    print(f"\n{ok} passed, {fail} failed")
    sys.exit(1 if fail else 0)
