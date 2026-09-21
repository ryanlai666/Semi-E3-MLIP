# Losses and chemical information

## Implemented supervised objective

For structure b with N_b atoms, the energy residual is `(E_pred-E_DFT)/N_b`.
Force-component penalties are averaged over atoms/components within each
structure, then over structures. Stress uses the tensile-positive 3x3 tensor;
missing stress labels have zero contribution. Off-diagonal entries are counted
twice in this full-tensor Frobenius-style loss, consistently across runs.

Training-only scales are: standard deviation of per-atom energy residuals after
elemental-offset fitting, RMS force component, and RMS stress component. Floors
are 0.05 eV/atom, 0.1 eV/Angstrom and 0.001 eV/Angstrom^3. Default weights are
1:10:1 for the normalized energy, force, and stress penalties. These weights are
explicit starting choices, not an assertion of universally optimal weighting.
[Adaptive-loss research](https://arxiv.org/abs/2403.18122) shows that weighting
depends on dataset and model. We keep fixed weights for interpretable pilots.

MSE remains available. `--loss pseudo_huber` uses the smooth robust penalty

`rho(x) = 2 delta^2 (sqrt(1 + (x/delta)^2) - 1)`, with delta=1 in normalized units.

The implementation uses an algebraically equivalent form stable near zero.
Small residuals receive approximately MSE penalty; large ones receive
approximately linear penalty. This changes optimization, not the smoothness
of the inference energy. We retain high-force data and report high-force errors
separately; robust loss does not justify silently discarding difficult frames.

[CHGNet](https://www.nature.com/articles/s42256-023-00716-3) and
[MACE-MP](https://doi.org/10.1063/5.0297006) provide primary examples of robust
energy/force/stress losses. Our smooth pseudo-Huber option is a design choice,
not a reproduction of their exact loss settings. Normalized delta=1 is not
equivalent to delta=0.1 in physical units.

## Optional elemental priors

`--chemical-descriptors` adds four fixed descriptors to the learned element
embedding through a learned linear projection: scaled atomic number, period,
IUPAC group, and a d-block indicator. These are known at inference, invariant,
and geometry-independent. They do not encode an assigned oxidation state.

[SpookyNet](https://www.nature.com/articles/s41467-021-27504-0) combines fixed
electronic-configuration descriptors with learned element embeddings, providing
evidence for chemical inductive biases. Our smaller period/group prior is not
the same descriptor and must earn its inclusion by a validation comparison.
It is redundant with element identity in an information-theoretic sense; its
purpose is an optimization/regularization prior in sparse-data settings.

Electronegativity and covalent radii are not currently included: choices of
scale and coordination/oxidation-dependent radius need an explicit source and
separate validation. More descriptors are not automatically better.

## Optional DFT auxiliary supervision

The selected raw subset has Bader population and Bader magnetic-moment arrays
in 3,866 of 4,248 eligible rows before geometric deduplication. DDEC6 fields were
not found under the audited names in the selected JSONL, despite release-level
descriptions mentioning DDEC6 additions. See `reports/auxiliary_label_audit.json`.

`--auxiliary-weight 0.05` adds two scalar output heads and masked supervision:

- `bader_population`: raw `bader_charges` values, preserved as population-like
  labels. These are **not treated as net ionic charges**. For example, elemental
  Ta has values around 11. We do not subtract nuclear Z or assume a POTCAR ZVAL.
- `bader_abs_magmom`: absolute Bader moments; spin-sign ambiguity is removed,
  while missing labels remain masked rather than being filled with zero.

Training-only per-element means are subtracted and residuals scaled. Scalar
heads regularize shared node features. The energy never reads the reference
auxiliary arrays; the same species-and-geometry inputs work in MD. Unit tests
change the labels without changing predictions to protect against leakage.
The optional auxiliary outputs are normalized residuals, not a public physical
charge-prediction interface. They are used to test whether this supervision
improves energy/force/stress validation errors.

CHGNet demonstrates magnetic-moment supervision as a proxy for electronic
environments, but a moment is not a unique oxidation-state assignment. Likewise,
predicting a Bader population does not implement charge equilibration or
long-range electrostatics. Those require a different energy model; see
[fourth-generation potentials](https://doi.org/10.1038/s41467-020-20427-2).

## Why RDKit is not a baseline dependency

[RDKit descriptors](https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html)
operate on molecular objects. Many require bond connectivity, aromaticity,
valence, or molecular fragment definitions. For periodic metallic/oxide MD,
those assignments can be ambiguous or discontinuous as atoms move. Global
molecular fingerprints also do not supply the differentiable local geometry
needed for conservative atomic forces. This is our applicability assessment,
not a claim that RDKit cannot be useful for molecular adsorbates in a later task.

## Additional data after the small proof of concept

The present dataset is enough to test code, learning, and numerical consistency,
but not enough to establish broad MD transferability. Prioritize actual
high-temperature, vacancy and strained configurations for poorly covered systems.
[TM23](https://archive.materialscloud.org/record/2024.48) provides a published
transition-metal AIMD benchmark (~893 MiB archive), with discussion of angular
complexity in [the associated paper](https://www.nature.com/articles/s41524-024-01264-z).
OMat24 and specialized oxide archives remain additional candidates. Their DFT
functional, pseudopotentials, smearing, magnetism and label conventions must be
audited before combining or training a separate target. We have not downloaded
or incorporated TM23 into the r2SCAN model.
