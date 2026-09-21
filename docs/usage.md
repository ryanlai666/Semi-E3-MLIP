# Setup and usage

A from-scratch PyTorch conservative E(3) scalar-vector GNN for semiconductor
industry metals and oxides. Uses PyTorch, NumPy, and Python standard-library I/O.
No ASE, pymatgen, e3nn, PyG, DGL, MACE, NequIP, MatGL, or pretrained MLIP weights
are used. Transitive PyTorch dependencies are installed but not used as a graph
implementation.

Read [dataset and architecture research](research.md) before interpreting
model coverage. This is a research baseline; a trained checkpoint is not by
itself evidence of reliable production MD.

The [expanded AIMD inventory and suitability audit](aimd_expansion.md)
documents the additional 2025–2026 device-material datasets, actual local counts,
duplicate leakage, DFT compatibility, and remaining coverage gaps. OMat ingestion
uses our own reader with the optional LMDB storage dependency (`.[data]`).

## Project and repository

Repository: [ryanlai666/Semi-E3-MLIP](https://github.com/ryanlai666/Semi-E3-MLIP)
(private). The Python import remains `semi_mlip`; both `semi-e3-mlip` and the
existing `semi-mlip` command are supported. Datasets and checkpoints remain local.

## Resume the focused Cu/Ti study

From the repository root, with no other focused training process running:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/continue_focused.ps1
```

This resumes the six architecture screens, runs the selected architectures at
300/900 training frames with three seeds, then freezes checkpoint selection
before protected test evaluation and short NVE checks. It stops on a failed
stage. After `reports/focused/frozen.json` exists, rerun only
`.\.venv\Scripts\python scripts/finish_focused.py --md` to finish evaluation;
the training scripts intentionally refuse further tuning of a frozen study.
See [the fixed protocol](focused_experiments.md) for split limitations.

## Setup on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python -m pip install numpy pytest
.\.venv\Scripts\python -m pip install -e . --no-deps --no-build-isolation
```

The local environment was tested with Python 3.13, CUDA 12.6 PyTorch, and an
RTX 3070 Laptop GPU (8 GB). CPU execution is supported with `--device cpu`.
The download/index choice can be changed for other hardware using the official
PyTorch installation instructions.

## Reproduce the data audit

```powershell
.\.venv\Scripts\python -m semi_mlip download
.\.venv\Scripts\python -m semi_mlip prepare
```

The downloader pins the dataset revision, validates its size and SHA-256, and
supports HTTP range resume. Approximately 2 GB raw storage is needed. `prepare`
streams the raw JSONL and writes normalized split JSONL plus `audit.json` and
rejection records. All atoms in a structure must satisfy the target chemistry.
Splitting is by indivisible parent/duplicate groups, stratified by chemical
system; the 80/10/10 fractions are targets, not exact quotas.

## Train and compare activations

```powershell
.\.venv\Scripts\python -m semi_mlip train --run runs/pilot --epochs 50
.\.venv\Scripts\python -m semi_mlip train --run runs/pilot --epochs 50 --resume runs/pilot/last.pt
.\.venv\Scripts\python -m semi_mlip ablate --run runs/ablation --epochs 50
```

The ablation runs SwiGLU, GeGLU, and SiLU with seeds 42/43/44. It uses the same
data selection and matched scalar-block parameter budgets. Read
`runs/ablation/comparison.json` for the selected activation, then train it:

```powershell
.\.venv\Scripts\python -m semi_mlip train --run runs/full --epochs 200 --max-train 0 --activation swiglu
.\.venv\Scripts\python -m semi_mlip evaluate --checkpoint runs/full/best.pt --output runs/full/test.json
```

Replace `swiglu` in the full-run command with the measured winner. `--max-train 0`
uses the complete retained training set. Test data are never used for model or
activation selection. Reports include per-chemical-system errors, high-force
errors, elemental-offset/zero-force baselines, and resource exclusions.

AdamW starts at 3e-6, linearly warms up to 3e-4 over 5% of optimizer updates,
then follows cosine decay to 3e-6. Weight decay is 1e-5, excluding biases and
normalization parameters. Four accumulated microbatches use atom/edge budgets
instead of a fixed structure count. Defaults: 256 atoms and 16,000 edges per
microbatch; no neighbors are truncated. Structures too large to fit a budget
are recorded in `excluded.json`. Nonfinite gradients fail explicitly.

Only the training split fits elemental offsets, coordination scale, and loss
normalizers. MSE losses on normalized energy/atom, force components, and stress
use weights 1:10:1. Missing stress labels contribute zero. Checkpoints include
optimizer, update-based schedule position, data signature, and RNG states.
Resume requires identical configuration and input data. Use
`--stop-after-epochs 1` to measure an epoch without changing the planned schedule.
CUDA scatter additions may be nondeterministic; exact CPU resume is tested.

## Prediction, relaxation, and MD

Input JSON contains `z`, `positions`, row-vector `cell`, and `pbc` arrays. Example:

```json
{"z":[14,14],"positions":[[0,0,0],[1.3575,1.3575,1.3575]],"cell":[[0,2.715,2.715],[2.715,0,2.715],[2.715,2.715,0]],"pbc":[true,true,true]}
```

```powershell
.\.venv\Scripts\python -m semi_mlip predict --checkpoint runs/full/best.pt --structure examples/silicon.json --output runs/prediction.json
.\.venv\Scripts\python -m semi_mlip relax --checkpoint runs/full/best.pt --structure examples/silicon.json --output runs/relaxation.json
.\.venv\Scripts\python -m semi_mlip md --checkpoint runs/full/best.pt --structure examples/silicon.json --ensemble nve --steps 1000 --dt 0.5 --output runs/nve.jsonl
```

Relaxation currently optimizes positions at fixed cell using FIRE; it reports
whether the requested force tolerance was reached. MD supports NVE velocity
Verlet and NVT BAOAB Langevin (`--ensemble nvt`). Time is fs; friction is fs^-1;
temperature is K. Predictions use eV, Angstrom, eV/Angstrom, and tensile-positive
eV/Angstrom^3. A model's `Calculator` refuses elements and chemical systems absent
from training. It cannot detect every out-of-distribution geometry.

Checkpoint files use PyTorch serialization: load only locally produced or trusted
checkpoints. Data arrays themselves use JSONL, not pickle.

## Verification

```powershell
.\.venv\Scripts\python -m pytest -q --basetemp=.test-tmp
```

Tests cover periodic images, triclinic geometry, rotations/reflections,
permutations, wrapping, extensivity, derivatives, activation double-backward,
stress conversion, grouped splitting, scheduler endpoints, resume, FIRE, and
MD timestep convergence. Run reports describe the actual measured training and
simulation results. Targets of 10 meV/atom and 0.1 eV/Angstrom are goals rather
than accuracy guarantees.

## Boundaries

Local scalar/vector features (l=0/l=1) with optional symmetric-traceless Cartesian
rank-2 channels (`ModelConfig(tensor_channels=16)`); no charge equilibration,
long-range Coulomb term, spin variable, or repulsive core.
Sparse hafnia/metals coverage, limited high-temperature data, and small test
sets constrain conclusions. Interfaces, reactive deposition, charged defects,
and melt/quench transferability are not validated. Neighbor enumeration targets
small training cells and short validation simulations, not large-scale MD.

## Device-material tuning and figures

`scripts/device_subset.py` selects predeclared elemental/oxide compositions while
preserving original parent-disjoint partitions (939/145/122 frames). Selection
does not use model errors. Coverage is uneven, including no Cu2O training frames.
`scripts/tune_device.py` and `scripts/tune_tensor.py` compare learning rates,
activation, descriptors, attention, tensor channels and width on validation only.
They record training metrics and use early stopping. The test set is not read
by either tuning script. See `reports/device_scope.json` for composition counts.

`scripts/paper_figures.py CHECKPOINT` exports backbone, comparison and
generalization plots as PNG, SVG and PDF, with the actual checkpoint settings.
`scripts/validate_and_render.py --checkpoint CHECKPOINT --data data/device/test.jsonl`
creates DFT/model force comparisons and model-only timestep diagnostics.

`scripts/import_aimd.py` audits/imports the separately downloaded 2026 CP2K/PBE
silicon archive; `scripts/compare_aimd.py CHECKPOINT` compares three external
300 K surface AIMD cases with our independent Langevin trajectories. This is
a short, cross-functional and surface-transfer diagnostic, not a matched-r2SCAN
accuracy benchmark. See [simulation protocol](simulation_validation.md).
