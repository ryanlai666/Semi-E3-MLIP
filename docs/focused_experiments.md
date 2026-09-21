# Focused experiments: protocol fixed before test evaluation

The first question is whether our original conservative GNN can learn one well-defined potential-energy surface and transfer to warmer configurations. Cu and Ti use the same TM23 methodology but different bonding. This is a controlled diagnostic, not a claim that their labels represent all device conditions.

## Data and splits

Each element is trained separately. Use 300 cold configurations first, then a nested 900-configuration set. Select by a deterministic hash of source ID, never by error, energy or force. Cold is 0.25 of the experimental melting temperature. Validation uses 90 configurations from the separate warm trajectory at 0.75 of melting. Published test frames remain protected: 100 cold, 100 warm, and 100 molten at 1.25 of melting per element. Molten data are excluded from all selection and fitting.

Cold/warm final tests share trajectories with their respective training/validation pools. This is stated explicitly; these are not independent-seed tests. The molten regime measures a distinct temperature/trajectory shift. No sampled frame sequence is treated as a verified chronological AIMD trajectory.

## Comparisons and stopping

Stage 1 compares gated message passing, four-head smooth attention, and the same attention with rank-two tensor features. All use scalar/vector widths 48/24, three interaction blocks, 24 radial functions, 5 Å cutoff, SwiGLU, AdamW (peak LR 0.001, weight decay 1e-5), 5% warmup and cosine decay. Initial screening: 60 epochs, seed 42, identical 300/90 splits and batch budgets. Tensor width is 8. Report parameter counts and runtime because these are feature ablations, not parameter-matched models.

Energy and force targets use training-only normalization. Smooth pseudo-Huber loss weights are energy 1 and force 10. No stress labels are used until source stress conventions are separately verified. No auxiliary chemical descriptors are added: element descriptors are constant within each single-element experiment and cannot supply information about local environments.

Choose the architecture with the lowest warm-validation force MAE for each material. If scores differ by no more than 2%, prefer fewer parameters. Stage 2 trains that architecture on 900 cold configurations with seeds 42/43/44 for 60 epochs. Repeat its 300-frame experiment at the same 60-epoch budget and seeds 42/43/44 to compare data scaling at equal epoch budgets (the 900-frame runs use more optimizer updates and compute; this is not a compute-matched comparison). Selection and checkpoints use validation only. Uncertainty across three initialization seeds is descriptive, not independent-dataset uncertainty.

Freeze configurations and selected checkpoint hashes before running the three final test regimes. Show energy MAE/atom, component-wise force MAE/RMSE, relative force error against a zero-force baseline, and train/validation gap. Report all seed outcomes, including failures. Avoid treating validation or test frames as independent observations for confidence intervals.

Then run short NVE stability checks at two time steps on the chosen model. Energy conservation is necessary but does not imply DFT accuracy. Do not label this an AIMD trajectory comparison: TM23 arrays contain high-fidelity relabelled snapshots and verified continuous trajectories/velocities are unavailable. Later case-by-case trajectory comparisons use the separate verified references with their DFT mismatch explicitly stated.

## How results guide the next experiment

- Low train and high warm error: expand temperature/phase coverage rather than just enlarge the network.
- High train and warm errors: diagnose optimization, force scales and model capacity using a training-only fit check.
- Improvement at 900 frames across seeds: continue a controlled learning curve before broad material mixing.
- Similar gated and attention performance: retain the simpler model for that material; attention is an ablation, not an assumed improvement.
- Good force accuracy but unstable/drifting MD: inspect smoothness, precision, neighbor lists and time-step convergence.

Si/Al/interface and oxide data remain source-separated. Their storage shards lack reliable run/temperature IDs, so a random frame split would overstate evidence. Further experiments require defensible group construction or an explicitly weaker split claim.

This follows the emphasis on shifted-condition validation in the [2025 practical MLIP guide](https://ceder.berkeley.edu/publications/2025_Ryan_MLP-guide.pdf) and on smooth conservative dynamics in [Fu et al., ICML 2025](https://proceedings.mlr.press/v267/fu25h.html). We use the experimental principles, not those authors' implementations or trained weights.
