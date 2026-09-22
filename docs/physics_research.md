# Physics research and priority-system recovery

**Status:** the width experiment is still running. Physics changes are implemented and tested in an isolated [research branch](https://github.com/ryanlai666/Semi-E3-MLIP/tree/experiment/007-physics-stability), with retraining queued behind it. No improvement claim is made before the new fits finish. The priority systems are Ru, Ta, Ti, Ta-O and molten configurations.

## Evidence from this repository

| Issue | Observation | Consequence for the next experiment |
| --- | --- | --- |
| Ti molten blow-up | One previously identified extreme frame has minimum distance 1.779 A, versus 2.234 A minimum across the 900 cold training frames. Vector feature maxima grow about 31 → 29,800 → 1.70e19 across three blocks. DFT maximum force component is 8.46 eV/A; model output reaches 6.82e31 eV/A. | Test bounded equivariant feature scaling and representative hot data. This is a post-hoc mechanism diagnostic on one frame, not a general proof of causation. |
| Ta-O specialist | Only 29 training frames. All three seeds choose epoch 1 by validation force MAE, close to the zero-force baseline. In seed 42, validation energy MAE is 0.911 eV/atom. | Expand chemical environments and balance checkpoint selection. Later epochs reduce energy error but worsen force error, so changing selection alone is not a demonstrated solution. |
| Ru specialist | Seed 42 has training force MAE 0.213 eV/A at its selected checkpoint; cold test is similarly poor. | Investigate optimization, energy/force tradeoffs and representational capacity, not only extrapolation. |
| Ta specialist | Seed 42 cold force MAE is 0.047 eV/A but warm/molten are 0.261/0.610 eV/A. | Explicitly sample temperature-dependent environments; preserve the cold-only result as a historical transfer benchmark. |

[Ti trace and checkpoint hash](../reports/physics_research/ti_failure.json) | [Existing material metrics](material_study_metrics.md)

## What the model already contains

The neighbor graph enumerates periodic lattice images within a 5 A radius, including small and skewed cells, without a neighbor cap. Cell shifts remain inside the differentiable energy calculation, and strain derivatives give stress. Four message-passing blocks communicate beyond a single edge, but this is not an infinite-range Coulomb interaction. Widening rank-two tensor channels does not introduce arbitrary higher angular orders.

Current inputs do not include a system charge or explicit spin vectors. Optional Bader-population and absolute-moment prediction heads are auxiliary targets, not charge equilibration or spin dynamics; they are disabled in the current reported models. A learned positional vector feature is not an atomic magnetic moment.

## Literature and design decisions

### Transition-metal representation and data coverage

[TM23: Complexity of many-body interactions (2024)](https://www.nature.com/articles/s41524-024-01264-z) connects difficult transition-metal force learning with electronic structure and angular sensitivity, and compares angular resolution and cutoff choices. **Implication:** test richer angular/many-body representations after the width ablation, rather than assume a longer cutoff alone will solve Ti/Ta errors. Its results are not proof that our particular implementation has reached a representational limit.

[Origin of ML force-field errors across metal elements (2026)](https://www.nature.com/articles/s41524-026-01977-3) introduces Metal-43 and investigates electronic-structure-related error trends. We downloaded its [641.5 MB archive](https://doi.org/10.24435/materialscloud:hm-6z), verified the published checksum, and reserved it for a source-independent reference audit. DFT smearing, pseudopotentials, magnetism and energy references must be reconciled before comparison. We will not change electronic smearing merely to lower ML error or fit energy offsets on the final test set.

[MP-ALOE (2025)](https://www.nature.com/articles/s41524-025-01834-9) supplies compatible off-equilibrium r2SCAN candidates. Our earlier formula restriction retained only selected stoichiometries. The new Ta-O study removes that narrow formula restriction while excluding known Ta-O parent groups and exact duplicates. Hypothetical oxygen-rich compositions are retained as reference configurations, not asserted to be stable phases.

### Long-range electrostatics: preferred next physical extension for oxides

[Latent Ewald summation (LES, 2025)](https://www.nature.com/articles/s41524-025-01577-7) adds a reciprocal-space interaction to local learned features. [Machine learning of charges and long-range interactions from energies and forces (2025)](https://www.nature.com/articles/s41467-025-63852-x) investigates learning electrical behavior without prescribing partition-dependent charge labels. These provide a practical route for our energy/force datasets. The [2026 electrostatics perspective](https://arxiv.org/abs/2512.18029) discusses this design space, but is guidance rather than an independent benchmark of our materials.

**Proposed implementation, not yet implemented:** add a scalar latent-charge head and a differentiable periodic reciprocal-space energy, jointly trained with the local energy. Specify net-charge treatment and boundary conditions explicitly; converge reciprocal resolution and screening width; preserve lattice/stress derivatives. Distinguish latent channels from physical charges unless electrical observables validate that interpretation. Test cell replication, cell-size convergence, separated subsystems, and polar/oxide environments. Slabs require a deliberate electrostatic boundary treatment; 3D periodic Ewald plus arbitrary vacuum is not automatically an isolated-slab calculation. Metallic screening and oxide electrostatics should not be conflated.

### Charge equilibration: useful, but not the first automatic fix

[Learning non-local interactions with equivariant representations and charge equilibration (2025)](https://www.nature.com/articles/s41524-025-01790-4) motivates environment-dependent electronegativity/hardness models. [Pushing charge-equilibration potentials to their limits (2025)](https://www.nature.com/articles/s41524-025-01791-3) explicitly examines variable charge, fragmentation and external-field failure modes.

**Proposed route:** minimize a charge-dependent energy subject to total-charge conservation, using a well-conditioned hardness matrix and a differentiable constrained solve. Validate solve residuals and energy/force consistency. Add charged and charge-transfer configurations where the intended application requires them. Global equilibration can produce spurious long-distance charge transfer; dissociation and dielectric-response tests are required. Sparse Bader populations alone do not validate this behavior. Keep QEq as a separate ablation after the simpler long-range model.

### Spin: requires a different labeled problem

[SpinGNN++ / time-reversal equivariant potentials](https://arxiv.org/abs/2211.11403) treats magnetic symmetry explicitly. Recent **2026 preprints**, [equivariant many-body magnetic potentials](https://arxiv.org/abs/2604.08143) and [STEP](https://arxiv.org/abs/2607.17129), embed magnetic moments and learn spin-lattice interactions. These are promising research directions, not proof of applicability to our current labels.

A spin model needs axial magnetic moments, time-reversal behavior, and energies/forces across magnetic states; magnetic effective fields or torques would strengthen training. Spin-orbit coupling changes the relevant rotational constraints. TM23 labels in this repository are **non-spin-polarized**, so a spin-dependent model cannot be validated on them. Our stored absolute moments discard orientation/sign information. Do not invent spin labels for Ru/Ta/Ti or assume magnetism explains their current error. A future magnetic branch should begin with verified spin-resolved data, for example appropriate Co/oxide states.

### Repulsive core: a narrowly scoped safeguard

The [2025 comparison of repulsive pair potentials](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.111.032818) compares universal screening with more detailed quantum calculations. The [LAMMPS ZBL specification](https://docs.lammps.org/stable/pair_zbl.html) supplies a documented screened nuclear repulsion. Such a core is useful for extreme close approaches, but is not a replacement for molten DFT training.

The research branch implements an optional ZBL core with a **multiplicative quintic switch from 0.8 to 1.5 A**. These are provisional experiment parameters, not fitted physical boundaries; our switch differs from LAMMPS's additive switch. Pair energy is counted once despite directed periodic edges, and forces/stress are differentiated from the total energy. The learned residual is retrained with the core present. A residual network can still counteract a positive prior, so adding it does not guarantee global repulsion or stable MD. The diagnosed Ti frame has minimum distance 1.779 A, outside this core: this core alone cannot repair that example.

## Implemented experiment and queued retraining

The isolated branch adds smooth invariant vector/tensor norm scaling and bounded scalar gates, both opt-in. It preserves equivariance and energy-derived forces; the combined change is tested as a stabilization package, not attributed separately to either component. Tests cover reflection/rotation, force and stress finite differences, periodic replication, directed-pair counting, core monotonicity and smooth switching. A CPU training smoke test exercises force-loss second derivatives.

The queue contains **nine exploratory seed-42 pilots**, each with the same **2x shared architecture (1,356,513 parameters), 60 planned epochs**, and matched optimization within each comparison:

| Development data | Plain | Stabilized | Stabilized + core |
| --- | --- | --- | --- |
| Expanded shared r2SCAN, focused on Ta-O coverage | Yes | Yes | Yes |
| Ti cold/warm/molten PBE | Yes | Yes | No |
| Ru cold/warm/molten PBE | Yes | Yes | No |
| Ta cold/warm/molten PBE | Yes | Yes | No |

Ti/Ru/Ta each use 2,520 training frames and 180 validation frames. Original 100-frame cold/warm/molten tests remain unchanged, but are previously inspected, trajectory-correlated diagnostics. Including molten training makes the new test a within-temperature check, not cold-to-molten extrapolation. A new independent reference is still necessary.

The expanded shared set contains 4,995 training and 376 validation frames, including 1,839/231 new Ta-O train/validation frames, with **228 newly reserved Ta-O test frames**. These additions do not include new Ta2O5 parent groups; improvement on historical Ta2O5 is a transfer hypothesis, not guaranteed by the added count.

Checkpoint selection uses the equal-chemistry mean of `0.5 * (energy_MAE / 0.01 + force_MAE / 0.1)`. These scales are project screening preferences, not universal suitability thresholds. Training loss weights remain 1/10/1 for the shared r2SCAN fits and 1/10/0 for TM23. All checkpoints are frozen before test prediction. Each system, force tails, baseline comparisons and energy errors must be assessed separately; the scalar selection score is not an acceptance certificate.

The queue waits for the existing width queue to finish, avoiding simultaneous GPU training and preserving the width experiment. Single-seed pilots screen ideas; successful candidates still require convergence checks, additional seeds, representative independent tests, and MD/property validation. No production checkpoint is replaced automatically.

[Data manifest](../reports/physics_research/priority_data.json) | [Retraining protocol](../reports/physics_research/retraining_protocol.json) | [Progress snapshot](../reports/physics_research/retraining_progress.json) | [Loss explanation](../README.md#training-loss-and-why-we-use-it)
