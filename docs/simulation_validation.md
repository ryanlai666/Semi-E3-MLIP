# AIMD validation protocol

The MatPES pilot is not an AIMD trajectory benchmark. Its DFT-labelled,
model-sampled configurations support static energy/force/stress comparisons.
Model-only timestep comparisons assess integration error, not DFT agreement.

## Reference-data audit

Prefer 2025+ studies for validation methods. Older datasets remain candidates
where recent archives do not contain chronological AIMD frames.

* [Cu/alumina benchmark (2026)](https://www.nature.com/articles/s41524-026-02043-8):
  compares pair distributions and mean-square displacements with AIMD at 800 K,
  using repeated MLIP trajectories with independent velocities. This informs our
  separation of statistical agreement from exact trajectory agreement.
* [Silicon/silica device study (2026)](https://www.nature.com/articles/s43246-026-01130-z):
  CP2K/PBE AIMD; public [archive](https://doi.org/10.6084/m9.figshare.29422061.v1)
  requires inspection for chronology and reference trajectories. PBE differs
  from the present r2SCAN training target. Raw data are also described as
  available on request; publication of a model is not publication of AIMD.
* [Hf/HfO2 dataset (2026)](https://www.nature.com/articles/s41524-026-01984-4):
  PBE/ONCV single-point calculations on sampled structures, not reference AIMD.
* [Silica r2SCAN archive (2026)](https://zenodo.org/records/20053085):
  active-learning MD samples with DFT relabelling; not automatically AIMD.
  Its 1000 eV cutoff and PAW settings require compatibility checks before mixing.
* [TM23 (2024)](https://www.nature.com/articles/s41524-024-01264-z):
  genuine metal AIMD at 0.25, 0.75 and 1.25 times melting temperature, with a
  vacancy. Archived training frames are sampled every 50 fs; high-fidelity
  labels use different k-point sampling from the driving AIMD. Sparse frames
  cannot establish high-frequency velocity correlations or reliably unwrap
  rapid diffusion without image information. PBE/r2SCAN comparisons must be
  labelled cross-functional transfer, not pure ML approximation error.

## Frozen-model comparison

1. Fix model and hyperparameters using training/validation only. Hold whole
   reference trajectories out; never randomly split neighboring MD frames.
2. Match composition, atom count, cell/density, temperature, ensemble and
   equilibration protocol. Match thermostat where possible; otherwise disclose
   the difference and avoid interpreting thermostat-dependent kinetics.
3. Report force/energy errors on reference frames by temperature and material.
   Absolute energies across different DFT settings are not directly comparable.
4. Run independent model trajectories at each available reference temperature.
   For matched-state early-time comparisons, use reference velocities if present.
   Otherwise show independent equilibrium trajectories explicitly.
5. Compare species-resolved RDFs, coordination distributions, energy/temperature
   distributions and, with adequate unwrapped sampling, time-origin-averaged MSD.
   Estimate diffusion only when a sustained diffusive regime exists. Use block
   statistics and multiple velocity seeds for uncertainty; short trajectories
   do not support precision diffusion or rare-event claims.
6. Separately test NVE energy drift and timestep convergence, including a smaller
   timestep. A thermostat can conceal integration heating.
7. Animate AIMD and model side by side with shared cell, colors, time scale and
   camera. Do not fabricate intermediate AIMD frames or relabel model MD as AIMD.
8. Additional strained-cell/vacancy/high-temperature tests are extrapolation
   checks unless corresponding DFT/AIMD references exist. Record failures and
   stop unstable runs; do not omit failed cases from aggregate reporting.

The initial working static goals (10 meV/atom and 0.1 eV/angstrom force MAE)
are engineering targets, not universal acceptance thresholds. Passing them does
not substitute for trajectory and application-specific validation.
