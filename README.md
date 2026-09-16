# PSCK-MoIQP

[![tests](https://github.com/jamalslim/ciqp-moiqp/actions/workflows/tests.yml/badge.svg)](https://github.com/jamalslim/ciqp-moiqp/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Simulation code and data for *An IQP Born Machine for Calorimeter Image Generation
at 64 Qubits with Compiled-IQP Deployment on Superconducting Hardware*.

Every number in Secs. III to V and VII comes from something in this archive.

## Install

    pip install numpy scipy matplotlib networkx
    export PYTHONPATH=$PWD/src

Python 3.10 or later. No compiled extensions, no GPU, no quantum SDK.

## Check the install

    python3 tests/test_core.py            # 7 audits, about 3 minutes
    python3 tests/test_review_fixes.py    # 7 regression tests

    python3 scripts/run_psck.py --bits 3 --L 4 --epochs 60 --mc-batch 512 \
            --seed 42 --output outputs/smoke

The last command trains a reduced model in about ten seconds.

## Layout

    src/psck_mmd/     the library
    scripts/          command-line drivers, one per task
    tests/            unit and regression tests
    data/             the encoded CLIC showers and the committed 80/20 split
    artifacts/        the trainability scan behind Tables IV and V
    outputs/          the five headline runs and the deployed model's parameters
    coupling_maps/    native heavy-hex connectivity

## Reproducing the paper

### Tables I and X, the headline sweep

    python3 scripts/run_seed_sweep.py --bits 8 --L 8 --epochs 1500 \
            --seeds 42 43 44 45 46 --split data/split_indices.npz \
            --out-root outputs/seed_sweep_B8_L8_split
    python3 scripts/summarize_seed_sweep.py outputs/seed_sweep_B8_L8_split

About 13 CPU-hours, parallel across seeds. The completed runs are already in
`outputs/seed_sweep_B8_L8_split`, so the summarizer works without retraining.
Expect MAE_rho 0.068 +/- 0.006 on the training split and 0.070 +/- 0.006 on the
held-out split.

Omitting `--split` refits the quantile edges on the full sample and shifts every
number. It is not a convenience flag.

### Proposition 1 and Table VII, the loss certificate

    python3 scripts/verify_certificate.py --sweep outputs/seed_sweep_B8_L8_split

Checks that the training loss bounds the downstream correlation error on all five
seeds. The bound on the linearised error is exact; the bound on the true error
carries a second-order remainder which the script reports alongside.

### Tables IV and V, Fig. 7, trainability

    python3 scripts/run_bp_scan.py --sigma 0.1 --K 200 --Mgrad 2048 \
            --out artifacts/bp_scan_results

### Sec. III B and App. A 6, the compilation

    python3 scripts/verify_ciqp.py --run <run_dir> -M 200000
    python3 scripts/audit_iqp_membership.py
    python3 scripts/audit_shallow_deploy.py

### Sec. VII A and App. A 5, treewidth

    python3 scripts/audit_simulability.py

Reports min-fill widths of 21 to 24 for the headline graphs and 3 for the
deployed one, and confirms that no deployment object exceeds the component-graph
width by more than a factor L.

## Scope

The classical side of the paper: the Van den Nest correlator estimator, the
Pearson-Stabilized Correlation Kernel, Mixture-of-IQP training, the exact
Walsh-Hadamard compilation of the mixture into one IQP circuit, and the structural
audits. Everything runs on CPU and nothing here requires or contacts a quantum
device.

The hardware campaign of Sec. VI is a separate programme and is not part of this
archive.

## A note on sampling

Training and evaluation are correlator-based throughout: the weight-<= 2 targets,
plus per-feature marginals recovered exactly by Walsh-Hadamard inversion. Nothing
here draws a sample from a trained model, because the objective never needs one.

Full n-bit samples are obtainable classically for every instance in this paper.
An IQP amplitude is an Ising partition function on the gate graph, costing
2^(w+1) with w the treewidth, and the marginal-free sampler of Bravyi, Gosset and
Liu, Phys. Rev. Lett. 128, 220503 (2022), turns amplitude evaluation into
sampling without forming a doubled network. The measured cost is 8.8e9 to 7.0e10
operations per sample for the headline graphs. The hardware deployment is
therefore a faithfulness demonstration against an available classical reference,
not access to an otherwise unreachable distribution.

## Data

`data/cal_shower_img_8q.npy` is the CLIC electromagnetic-calorimeter
electron-shower dataset, 47,682 events, integrated transversely to 8 longitudinal
depth bins. `data/split_indices.npz` is the committed 80/20 split, 38,146 train
and 9,536 test, created once and reused everywhere. Quantile edges and all
training targets are fitted on the training split only.

Source dataset: Zenodo 10.5281/zenodo.16027525.

## License

MIT. See LICENSE.

## Citation

If you use this code, please cite the paper and the dataset. Machine-readable
metadata is in `CITATION.cff`.

    J. Slim, S. Monaco, F. Rehm, D. Kruecker, and K. Borras,
    An IQP Born Machine for Calorimeter Image Generation at 64 Qubits with
    Compiled-IQP Deployment on Superconducting Hardware, arXiv:2605.27735 (2026).
