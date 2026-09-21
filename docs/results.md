# Measured results

Generated from versioned JSON reports. Checkpoint selection uses validation only.

## Multi-metal / metal-oxide model

Frozen checkpoint: `runs\device\expanded_tensor\best.pt`. SHA-256:
`f2286fe10cf6c9218a78f463d1b19beff301d255eec62c633135cabc84cb5bcc`.

| Partition | Frames | Energy MAE (meV/atom) | Force MAE (eV/A) |
| --- | --- | --- | --- |
| Expanded training | 3156 | 52.649 | 0.091 |
| MatPES validation | 145 | 103.966 | 0.231 |
| MatPES test | 122 | 92.189 | 0.304 |
| MP-ALOE oxide test | 279 | 72.259 | 0.228 |

All force MAEs are averages over Cartesian force components. Energy MAE is per atom.
Training metrics use the selected checkpoint, not the last epoch. The expanded
training set includes MatPES and MP-ALOE data; their held-out sets remain separate.
The broad model does not meet the original 10 meV/atom / 0.1 eV/A test goals.

![Material performance and coverage](assets/material_benchmark.png)

| System | Train / valid / test / oxide-test frames | Train force MAE | MatPES test force MAE | Oxide test force MAE |
| --- | --- | --- | --- | --- |
| Al | 11 / 1 / 1 / 0 | 0.030 | 0.102 | -- |
| Al oxides | 109 / 10 / 6 / 12 | 0.074 | 0.285 | 0.322 |
| Co | 25 / 3 / 2 / 0 | 0.072 | 0.027 | -- |
| Co oxides | 384 / 18 / 4 / 42 | 0.109 | 0.217 | 0.245 |
| Cu | 42 / 5 / 5 / 0 | 0.037 | 0.034 | -- |
| Cu oxides | 326 / 2 / 4 / 33 | 0.064 | 0.231 | 0.218 |
| Hf | 33 / 1 / 1 / 0 | 0.175 | 0.000 | -- |
| Hf oxides | 403 / 5 / 5 / 45 | 0.078 | 0.321 | 0.208 |
| Ru oxides | 219 / 2 / 0 / 21 | 0.121 | -- | 0.269 |
| Si oxides | 277 / 32 / 34 / 24 | 0.093 | 0.163 | 0.281 |
| Ta oxides | 29 / 7 / 7 / 3 | 0.129 | 0.878 | 0.396 |
| Ti oxides | 444 / 5 / 12 / 39 | 0.077 | 0.145 | 0.221 |
| W oxides | 192 / 14 / 5 / 18 | 0.115 | 0.174 | 0.231 |
| Zr oxides | 363 / 3 / 5 / 42 | 0.078 | 0.138 | 0.148 |
| Ru | 26 / 1 / 0 / 0 | 0.181 | -- | -- |
| Si | 124 / 16 / 15 / 0 | 0.100 | 0.211 | -- |
| Ta | 39 / 5 / 5 / 0 | 0.100 | 1.588 | -- |
| Ti | 22 / 3 / 2 / 0 | 0.102 | 0.068 | -- |
| W | 58 / 6 / 6 / 0 | 0.155 | 0.167 | -- |
| Zr | 30 / 6 / 3 / 0 | 0.091 | 0.004 | -- |

Errors are in eV/A. `--` means no test examples. Counts are frames, not independent
trajectories. Some systems have only one to five test structures. Ru and Ru oxides
lack MatPES test examples; Ru oxides have a separate MP-ALOE test partition.
A small elemental test error does not establish phase, defect, or temperature coverage.

![Force parity for predeclared examples](assets/force_parity.png)

Parity plots use the lexicographically first held-out ID in each chemical system;
they are illustrations, not substitutes for all-frame errors above.

## Dataset distributions

![Label and structure-size distributions](assets/dataset_distributions.png)

![Per-system counts and training distributions](assets/dataset_chemistry.png)

Energy, structure size and stress distributions weight each frame equally.
Force-magnitude curves weight atoms; component curves weight Cartesian components.
The energy residual subtracts elemental offsets fitted only to the training set.
Raw energies differ strongly by composition, so their pooled spread is not a
measure of structural diversity. The expanded training and MP-ALOE test sets
contain substantially stronger force/stress labels than the MatPES held-outs;
source and composition shifts must be considered alongside aggregate errors. ECDFs retain the full tails; the box plots show
median/IQR, 1.5-IQR whiskers, and outliers. Stress uses the Frobenius norm in eV/A^3;
unlabelled frames are excluded and counted in the numerical report.

[Exact quantiles, source counts, and input hashes](../reports/benchmark/dataset_distributions.json).

## Cu/Ti controlled temperature transfer

| Material | Architecture | Parameters | Warm validation force MAE (eV/A) |
| --- | --- | --- | --- |
| Cu | gated | 119,137 | 0.053 |
| Ti | gated | 119,137 | 0.176 |
| Cu | attention | 133,261 | 0.041 |
| Ti | attention | 133,261 | 0.184 |
| Cu | tensor | 146,785 | 0.025 |
| Ti | tensor | 146,785 | 0.145 |

Architecture screening uses 300 cold frames, 90 warm validation frames, seed 42,
and 60 epochs. Both materials selected the tensor model under the fixed 2% rule.
Architectures have unequal parameter counts. No test frames select the models.

Confirmation training in progress: 7/12 trial records. Final test results are not yet available.

Force MAE in eV/A; mean +/- sample SD across three initialization seeds.
Cold/warm train and test pools share source trajectories. Molten configurations
are a held-out temperature/trajectory shift. Seed variation is not independent-
dataset uncertainty. The 900-frame runs consume more optimizer updates than
300-frame runs, so this is an equal-epoch, not equal-compute, learning curve.

## Numerical stability

Focused NVE diagnostics are queued after training.

These are 100 fs model-only NVE checks at 300 K initial velocities. They measure
integration behavior, not agreement with DFT trajectories. A short trace and a
small energy drift do not establish long-time stability.

## Inference timing

Inference timing is queued after training and AIMD finish to avoid competing workloads.

Batch size one; five warmups and 30 timed calls per structure. Timings include
neighbor enumeration, transfers, energy, forces, and stress. CPU and CUDA use the
same checkpoint. This is a local small-cell benchmark, not a comparison against
MACE, NequIP, or other packages. Hardware and software are recorded in
[`inference.json`](../reports/benchmark/inference.json).

## External AIMD comparison

| Reference case | Force frames | Force MAE (eV/A) | Force RMSE (eV/A) | Completed seeds | Failed seeds |
| --- | --- | --- | --- | --- | --- |
| Si 110 elong0.500 | 21 | 0.193 | 0.269 | 3 | 0 |

Three 216-atom Si surface references at 300 K; 500 fs per reference and model run.
Each model case uses seeds 42/43/44, a 0.5 fs timestep, and Langevin dynamics.
Static force errors use every fifth stored reference frame (21 frames per case).
The reference uses CP2K/PBE and Nose-Hoover; this model uses r2SCAN training labels.
Initial velocities are unavailable, so trajectories start from the same geometry
with independent model velocities. These differences preclude a matched-Hamiltonian
accuracy claim or pointwise trajectory agreement claim.

![Si_110_elong0.500 AIMD and model dynamics](assets/aimd_Si_110_elong0.500.gif)

![Si_110_elong0.500 structure statistics](assets/aimd_Si_110_elong0.500_metrics.png)

Pair-distance density is normalized per atom and distance, not by bulk density,
because these cells contain vacuum. Displacement is COM-corrected displacement
from the initial frame, not a time-origin averaged diffusion MSD. No diffusion
coefficient is inferred from 500 fs.

## Reproducibility

- [Focused protocol](focused_experiments.md), [frozen checkpoint hashes](../reports/focused/frozen.json).
- [Broad-model selection](../reports/aimd_comparison/selection.json), [benchmark data hashes](../reports/benchmark/protocol.json).
- [AIMD protocol and hashes](../reports/aimd_comparison/protocol.json), [source manifest](../reports/aimd_manifest.json).
- [Material results](../reports/benchmark/materials.json), [raw focused results](../reports/focused/tests.json).
- [Data acquisition and limitations](aimd_expansion.md), [simulation protocol](simulation_validation.md).
