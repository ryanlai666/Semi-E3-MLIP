# Independent material studies

[Train/test parity plots](parity.md) | [Detailed metrics](material_study_metrics.md) | [Model scope](#one-potential-or-separate-potentials) | [Metal/oxide coverage](#complete-metaloxide-coverage)

<!-- material-assessment:start -->
## Are these models good enough?

**Not yet for general predictive MD, molten phases, or metal/oxide interfaces.** W, Ta and Co show promising accuracy only within the sampled cold TM23 regime. Most transfer tests miss the project targets, and several show severe extrapolation failures. Short energy-conservation checks establish numerical behavior of the learned potential, not agreement with DFT.

The screening targets here are energy MAE <= **10 meV/atom** and force MAE <= **0.1 eV/A**, both required on a held-out partition. These are this project's working targets, not universal acceptance standards. Passing them does not validate a particular physical observable. Counts below require each seed to meet both targets; they do not hide failed seeds behind an average. Seed SD is fit variability, not a confidence interval over independent trajectories.

## One potential or separate potentials?

| Model family | Learned weights | Scope and role |
| --- | --- | --- |
| This page: 18 studies, 54 fits | Separate weights for every study and seed | Single-element or single metal-oxygen chemical-system specialists; three seeds are replicates, not an automatically combined ensemble |
| Earlier Cu/Ti study: 12 fits | Separate Cu and Ti weights, two data sizes, three seeds | Additional elemental temperature-transfer experiments; [results](results.md#cuti-controlled-temperature-transfer) |
| Shared `expanded_tensor` baseline | One checkpoint jointly trained on all 20 chemical systems | Ten elements and their ten oxygen-containing systems in one model; [shared results](results.md#multi-metal--metal-oxide-model) |

**Same hyperparameters do not mean shared learned parameters.** The specialists use the same 48 scalar / 24 vector / 8 tensor channels, three blocks, 24 radial functions, 5 A cutoff, four attention heads and SwiGLU. Each trains from scratch for 60 epochs with learning rate 0.001, pseudo-Huber loss, energy/force weights 1/10, zero stress-loss weight, and seeds 42/43/44. Checkpoints minimize validation force MAE. Dataset sizes, update counts, fitted elemental energy offsets, loss scales, neighbor normalization and final learned weights differ between systems. Thus equal epochs are not equal compute or identical optimization trajectories. These are fixed-protocol baselines, not individually tuned best models.

The shared baseline is a different experiment: 64/32/16 channels, four blocks, 32 radial functions, up to 200 epochs and a nonzero stress-loss weight. It jointly fits 3,156 r2SCAN configurations. Comparing its errors with the specialists is useful, but does not isolate parameter sharing from architecture, data volume or training budget.

**A specialist cannot represent the union of the systems.** An Al-O model has seen Al-O chemistry, not Cu-O, W-O or arbitrary mixtures. Do not switch between elemental and oxide checkpoints atom by atom or along a reaction: their independently fitted energy functions do not define a single consistent energy surface. The shared model can accept the listed elements together, but elemental/oxide coverage alone does not validate alloys, multication oxides, oxygen molecules, oxidation reactions or interfaces.

## Complete metal/oxide coverage

Cu and Ti were omitted from the follow-up table because their elemental models were completed in the earlier study. They were not omitted from the repository. Si is included as the semiconductor member of the paired coverage.

| Element | Elemental specialist | Oxide specialist on this page | Shared model includes both? |
| --- | --- | --- | --- |
| Al | r2scan_al | r2scan_al_o | Yes |
| Si | r2scan_si | r2scan_o_si | Yes |
| Cu | Earlier Cu/Ti study | r2scan_cu_o | Yes |
| Ti | Earlier Cu/Ti study | r2scan_o_ti | Yes |
| W | tm23_w | r2scan_o_w | Yes |
| Ta | tm23_ta | r2scan_o_ta | Yes |
| Co | tm23_co | r2scan_co_o | Yes |
| Ru | tm23_ru | r2scan_o_ru | Yes |
| Hf | tm23_hf | r2scan_hf_o | Yes |
| Zr | tm23_zr | r2scan_o_zr | Yes |

Coverage still has gaps: the shared model has **no elemental Ru held-out test** in this benchmark. Elemental Al and Hf each have only one shared-model test frame. The elemental TM23 results use PBE while the oxide specialists use r2SCAN; they are not a matched-fidelity metal/oxide pair experiment. There are no dedicated matched r2SCAN elemental specialists for Cu, Ti, W, Ta, Co, Ru, Hf or Zr in this follow-up, although all are included in the shared model.

## Elemental suitability

| Metal | Cold energy / force MAE (meV/atom; eV/A) | Seeds passing cold / warm / molten | Interpretation |
| --- | ---: | --- | --- |
| W | 0.76 / 0.0386 | 3/3 / 0/3 / 0/3 | Promising for sampled cold structures; warm transfer misses force target; molten predictions fail severely. |
| Ta | 0.98 / 0.0478 | 3/3 / 0/3 / 0/3 | Promising for sampled cold structures; warm and molten force errors remain too large. |
| Co | 1.59 / 0.0511 | 3/3 / 0/3 / 0/3 | Promising for sampled cold structures; warm results vary strongly by seed and molten failures are severe. |
| Ru | 14.72 / 0.1595 | 0/3 / 0/3 / 0/3 | Cold mean already misses both targets; warm and molten results do not support deployment. |
| Hf | 8.55 / 0.0823 | 2/3 / 0/3 / 0/3 | Cold mean meets targets, but only two seeds pass; warm/molten errors prevent a robust suitability claim. |
| Zr | 11.04 / 0.0677 | 1/3 / 0/3 / 0/3 | Cold force mean is low, but energy mean misses target; only one seed passes both. Transfer remains inadequate. |

Cold and warm TM23 tests share source trajectories with development data, so low cold errors are evidence of interpolation within this sampling, not independent phase/defect/trajectory generalization. None of these six metals passes both targets in any seed on the warm or molten test partitions.

The earlier 900-frame Cu/Ti fits have mean cold force MAEs of about 0.00564 and 0.0440 eV/A, respectively, but their molten means rise to about 3.35e4 and 3.68e27 eV/A. These are extrapolation failures, not acceptable MD errors.

**Al:** its specialist has just one test frame: about 124 meV/atom energy MAE despite a small 0.0012 eV/A force MAE. That is neither adequate energetic accuracy nor enough test diversity. **Si:** its 15-frame test gives about 462 meV/atom and 0.257 eV/A; both targets are missed. Neither supports a general elemental-potential claim.

## Oxide suitability and comparison with the shared model

**None of the ten oxide specialists meets both targets on any available held-out source partition in any seed.** The MP-ALOE table below compares the same held-out chemistry subsets. Specialist values are three-seed means; the shared value is one frozen checkpoint. Lower shared-model errors are observed results, not a controlled proof that sharing alone caused the improvement.

| System | MP-ALOE test frames | Specialist energy / force MAE | Shared energy / force MAE | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Al-O | 12 | 689.7 / 1.432 | 88.2 / 0.322 | Large energy errors; low MatPES force error does not transfer to off-equilibrium data. |
| O-Si | 24 | 305.0 / 0.471 | 47.3 / 0.281 | Both targets missed; not evidence for accurate silica or Si/SiO2 interfaces. |
| Cu-O | 33 | 232.6 / 0.266 | 83.5 / 0.218 | Both targets missed; does not validate Cu/CuO interfaces or oxidation. |
| O-Ti | 39 | 283.5 / 0.377 | 65.9 / 0.221 | Both targets missed; oxide/metal transfer and defects unvalidated. |
| O-W | 18 | 465.7 / 0.444 | 96.6 / 0.231 | Both targets missed; no evidence for reliable oxide thermodynamics or MD. |
| O-Ta | 3 | 391.3 / 1.163 | 82.6 / 0.396 | Especially poor forces; only three MP-ALOE test frames, so coverage is also weak. |
| Co-O | 42 | 164.3 / 0.387 | 102.9 / 0.245 | Both force and energy targets missed; reactive or phase-transfer use unvalidated. |
| O-Ru | 21 | 202.3 / 0.466 | 57.9 / 0.269 | Both targets missed; only MP-ALOE oxide test coverage, no MatPES oxide test. |
| Hf-O | 45 | 220.4 / 0.413 | 72.9 / 0.208 | Both targets missed; polymorph and defect energetics remain unvalidated. |
| O-Zr | 42 | 152.9 / 0.277 | 43.7 / 0.148 | Lower errors than several other oxides, but still fails both project targets. |

Energy units: meV/atom; force units: eV/A. The shared model has lower energy and force MAE for all 10 oxide subsets here, but it still fails the joint targets for every oxide. Al-O illustrates why force alone is misleading: its specialist MatPES force MAE is 0.0369 eV/A, while energy MAE is about 1,320 meV/atom and MP-ALOE force MAE rises to 1.432 eV/A.

## What is needed for one useful metal/oxide potential?

Start from a jointly trained, single energy model with compatible DFT labels. Add representative elemental, oxide, metal/oxide interface, vacancy, surface, strained and high-temperature environments for the intended application; binary end-member coverage is insufficient. Keep PBE and r2SCAN separate unless a documented fidelity-aware approach is introduced.

Use held-out parent structures and independent trajectories, report worst cases and force-error tails, and validate the quantities that matter: relative phase/formation energies, equations of state, elastic/stress response, defect or adsorption energies, and reaction barriers as applicable. Then compare stable MD observables with reference data over the required temperature and time range. The existing short Si AIMD comparison and 100 fs NVE checks do not validate all metals and oxides.

For this repository, the immediate research priorities are to diagnose the cold-to-molten extrapolation failures, improve the shared model using representative compatible data, balance energy and force validation, and expand the weak elemental Al/Hf/Ru tests. More runs with unchanged narrow training distributions alone would not establish suitability.

<!-- material-assessment:end -->

## Plots and detailed results

- [Train/test parity gallery: shared model overall, every system, and all specialists](parity.md).
- [Detailed train/validation/test tables, learning curves and distributions](material_study_metrics.md).
- [Frozen protocols, checkpoint hashes and raw metrics](../reports/material_studies).
