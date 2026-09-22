# Model capacity and research priorities

This is a literature-informed experiment plan, not a claim that the proposed changes have solved the failures. Current checkpoints and test results remain frozen.

## Exact model sizes

Counts are learned parameters, excluding fixed elemental offsets and other buffers. Candidate counts are from constructing the current implementation, not trained results.

| Configuration | Scalar / vector / tensor channels | Blocks | Parameters |
| --- | --- | --- | --- |
| Current independent specialists | 48 / 24 / 8 | 3 | 146,785 |
| Current shared model | 64 / 32 / 16 | 4 | 354,545 |
| First width experiment | 96 / 48 / 24 | 4 | 774,537 |
| Larger width experiment | 128 / 64 / 32 | 4 | 1,356,513 |
| Depth-only candidate | 64 / 32 / 16 | 6 | 525,913 |

The shared model has 342,736 parameters in interaction blocks (96.7%), 7,616 in its embedding and 4,193 in the readout. Expanding scalar/vector/tensor channels in [ModelConfig](../semi_mlip/model.py) expands those interaction networks. Keep cutoff 5 A, radial basis 32 and attention heads 4 fixed for the first width comparison. Radial basis 64 gives only 375,537 parameters; eight attention heads gives 355,073. More heads is not a substantial capacity expansion. More tensor channels does not introduce angular orders above the implemented rank-two representation.

The 8 GB GPU must also store edge features and force-training derivative graphs; parameter memory alone is not a reliable batch-size estimate.

## What the existing evidence says

The shared model has training energy/force MAE of 52.65 meV/atom and 0.0906 eV/A; validation gives 103.97 meV/atom and 0.2312 eV/A. There is room to improve the energy fit, but the generalization gap also matters. The narrow cold-trained specialists can fit cold forces well and fail catastrophically on molten snapshots. That is not evidence that insufficient parameter count is the sole cause.

The existing expanded-attention to expanded-tensor comparison improves validation energy MAE from 119.73 to 103.97 meV/atom and force MAE from 0.2430 to 0.2312 eV/A, but force RMSE worsens from 0.6235 to 0.6650 eV/A. This changes representation as well as size, so it is not an isolated width-scaling experiment. A second tensor seed gives 114.10 meV/atom and 0.2359 eV/A. Tail errors and seed variation must remain visible. See [expanded trials](../reports/expanded_search.json).

## Literature mapped to experiments

| Primary research | Finding relevant here | Repository action proposed |
| --- | --- | --- |
| [TM23: Complexity of many-body interactions in transition metals (2024)](https://www.nature.com/articles/s41524-024-01264-z) | Explicitly studies cold-to-warm and cold-to-melt transfer using AIMD-derived metal configurations. | Preserve the original cold-only transfer benchmark; create a separate mixed-temperature experiment with new blocked validation/test partitions. |
| [Training data selection for accuracy and transferability (2022)](https://www.nature.com/articles/s41524-022-00872-x) | Diverse tungsten environments improve transferability across several model classes. | Compare equal-size narrow and diverse sets before attributing improvements to architecture. Sample coordination, neighbor distances, density and force/energy ranges, not just more adjacent frames. |
| [MP-ALOE (2025)](https://www.nature.com/articles/s41524-025-01834-9) | Provides nearly one million r2SCAN calculations emphasizing off-equilibrium environments and tests extreme conditions. | Expand the compatible shared r2SCAN pool with parent-disjoint elemental, oxide and alloy environments. Check calculation metadata; a functional name alone is not a complete compatibility audit. |
| [Automated discovery of a robust aluminum potential (2021)](https://www.nature.com/articles/s41467-021-21376-0) | Active learning samples nonequilibrium temperature-driven trajectories and validates liquid/crystal properties. | For Al, seek solid, strained and liquid references; use calibrated committee disagreement plus diversity to nominate new DFT calculations. Existing PBE data can support a separate benchmark, not silently join r2SCAN training. |
| [Forces are not Enough (2022/ICLR 2023)](https://arxiv.org/abs/2210.07237) | Force errors alone do not establish accurate molecular simulation behavior. | Pair parity and tail metrics with stability and structural observables; do not interpret an animation or a short surviving trajectory as validation. |
| [Learning Smooth and Expressive Interatomic Potentials / eSEN (2025)](https://arxiv.org/abs/2502.12147) | Examines numerical energy conservation and architecture choices affecting physical property predictions. | Test energy/force smoothness at cutoff crossings, finite-difference forces, and NVE timestep convergence in multiple regimes before depth expansion. |
| [EquiformerV2 (2023/ICLR 2024)](https://arxiv.org/abs/2306.12059) | Uses normalization and activation changes to support higher-degree equivariant representations. | Motivate controlled equivariant feature-scaling experiments; its architecture is different, so its results do not prove a fix for this model. |

## Ordered experiments and acceptance evidence

1. **Diagnose frozen failures first.** On existing cold and extreme molten examples, record minimum periodic distance, neighbor count, energy/force error, and per-block scalar/vector/tensor norms and gates. Compare float32 with float64, force finite differences and small coordinate perturbations. Inspect graph cutoffs and duplicate/overlapping geometries. Reproduce the failure before changing code. These are post-hoc diagnostics, not new blind tests.
2. **Test stability mechanisms independently.** Current scalar features have LayerNorm, whereas vector/tensor contractions and residual gates can grow without bounds. This is a code-level hypothesis, not an established cause. Test smooth invariant norm scaling and bounded scalar gates separately while preserving equivariance and forces as energy derivatives. Avoid force clipping, which can hide the failure and break energy consistency. A short-range repulsive prior is a separate hypothesis requiring compatible short-distance DFT calibration and smooth joining, not an automatic repair.
3. **Expand representative data.** Build a new versioned manifest with source/calculation provenance, parent or trajectory groups, geometry hashes, and temperature/density/composition coverage. Keep PBE TM23 development separate from the shared r2SCAN track. Do not split adjacent trajectory frames randomly. Reserve new independent groups before selecting examples. Previously examined alloy tests remain historical diagnostics; do not reuse them as training and claim unchanged held-out performance.
4. **Balance validation.** Record energy and force MAE/RMSE, per-frame tail errors, stress where present, and per-system macro averages. Current shared checkpoint selection by force MAE alone can sacrifice energy. Predeclare an energy/force Pareto rule or dimensionless score using application-specific scales, with per-system and catastrophic-error checks. Training loss weights and validation selection are different controls. The existing 10 meV/atom and 0.1 eV/A targets are project screening targets, not universal adequacy criteria.
5. **Run a controlled capacity/data matrix.** Compare current width and 96/48/24 width on both original and diverse data; use matched optimization schedules, three seeds, and report compute/memory. Start with four blocks. Compare wider 128/64/32 only if the first width experiment improves validation without worsening tails. Test depth separately after stability diagnosis. Fix splits and selection rules before runs; never pick the winning setting using test errors.
6. **Repair elemental test coverage.** The shared elemental Al and Hf tests have one frame each and Ru has none. First inventory unused r2SCAN parent groups; report actual availability rather than promise a frame count. Target multiple independent structures and thermodynamic regimes. If absent, request or generate compatible DFT labels. Existing Hf/Ru PBE TM23 specialists do not fill this r2SCAN evidence gap. Report parent counts and group-bootstrap uncertainty; many correlated frames cannot replace independent groups.
7. **Validate the intended use.** For molten dynamics, evaluate multiple trajectories/seeds, failure rates, NVE timestep convergence, and liquid structural statistics against compatible references. Diffusion requires adequate trajectory duration and uncertainty estimates. For phase diagrams, use consistently calculated relaxed compounds and elemental endpoints, then compare formation energies and hull membership. Finite-temperature phase boundaries require free energies; AIMD snapshot parity is insufficient.

## AIMD visualization and reproducibility

[New continuous-reference AIMD parity](aimd_parity.md) covers all stored Si frames and distinguishes force agreement from relative-energy agreement across PBE/r2SCAN. [TM23 parity](parity.md#independent-specialists) already covers cold/warm/molten reference snapshots.

The current recommendation is to test **coverage and numerical behavior alongside a moderate width increase**, rather than assume that a larger model will extrapolate correctly. No proposed architecture change or new training result is reported as completed here.
