# Dataset and message-passing research

Research checked against primary publications, dataset records, and publisher
examples on 2026-09-20. Dataset facts below distinguish download verification
from literature-only assessment. No third-party potential weights or MLIP code
are used by this project.

## Dataset decision

The initial target is neutral, periodic semiconductor-industry metals and oxides:
Cu, Al, W, Ti, Ta, Co, Ru, Si, Hf, Zr, O. Data include elemental structures and
oxygen-containing compounds made entirely from these elements. Oxygen-free
multielement alloys are deferred. This is not a semiconductor band-structure
model, an oxidation-state predictor, or a general process-chemistry potential.

| Dataset | Labels and sampling | Access/licence evidence | Decision |
|---|---|---|---|
| [MatPES](https://huggingface.co/datasets/materialyze/matpes) | Total energy, forces, stress; equilibrium and DFT-labelled configurations selected from model-driven 300 K MD; separate PBE/r2SCAN targets | Official JSONL download; dataset card states BSD-3-Clause | Selected r2SCAN 2025.2; fully downloaded and SHA-256 verified |
| [MPtrj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842) | 1,580,395 structures; GGA/GGA+U relaxation/static data, corrected and uncorrected energies, forces, stresses | 11.35 GB published archive; MIT on dataset record | Candidate for a separate PBE-family model; not mixed into r2SCAN |
| [OMat24](https://huggingface.co/datasets/facebook/OMAT24) | Non-equilibrium, relaxation, and AIMD-derived PBE/PBE+U configurations; E/F/stress | CC BY 4.0; full training table 100,824,585 entries; official 1,009,850-frame training subsplit | Future high-temperature coverage; requires LMDB reader and DFT compatibility audit |
| [Alexandria](https://alexandria.icams.rub.de/datasets.html) | Downloadable optimization paths; multiple functionals and dimensionalities | Publisher download index verified; per-archive licence and schema not yet audited | Future candidate; check overlap with OMat/sAlex and MP-derived sources |
| [Silica reference data](https://doi.org/10.5281/zenodo.6353684) | Specialized crystalline/disordered silica, final SCAN reference calculations | Public archive linked by [authors](https://www.nature.com/articles/s41524-022-00768-w); licence/schema not locally audited | Separate SCAN target; useful for future silica-focused work |
| [Unified Si/O data](https://gitlab.com/Kazongogit/MTPu) | Silicon, oxygen, silica configurations | Public repository linked by [authors](https://www.nature.com/articles/s41524-024-01390-8); licence/DFT settings not audited | Candidate for Si/SiO2 coverage; no claim of interface transferability without inspection |
| [Hf/HfO2](https://zenodo.org/records/18489916) | Specialized hafnium/hafnia training data | Published 22.9 MB Hf and 36.9 MB HfO2 archives; licence/labels/settings not locally audited | Important gap-filling candidate, but must match or separate DFT target |
| [Alumina](https://zenodo.org/records/13850687) | Published final train.xyz with 3,335 structures | Public dataset record; functional, units and licence require archive audit | Candidate for crystalline/disordered alumina validation |

MatPES is not AIMD: its generating MD used an existing potential, followed by
independent DFT single-point labels. Reusing those published DFT labels does not
require importing or using that potential in our code. The [MatPES paper](https://arxiv.org/abs/2503.04070)
explains the workflow and the limitations of mixing corrected GGA/GGA+U energies
with otherwise uncorrected forces/stresses. No model-driven sampling establishes
coverage of reactions, melt/quench, vacancies, surfaces, or interfaces by itself.

### Pinned source and measured coverage

- Revision: `47d2cc020cf913b5a48a3480136a128dddc0a92c`.
- File: `MatPES-R2SCAN-2025.2.jsonl`, 1,946,162,981 bytes.
- SHA-256: `cce36109689d75446b720d21a11faf3ed980df1a32ce6387237629774a770606`.
- 386,544 rows scanned; 4,248 eligible; three duplicate geometries removed;
  4,245 retained across 61 chemical systems (see generated audit for exact count).
- Grouped split: 3,430 train / 426 validation / 389 test. Frame fractions are
  approximate because parent trajectories are indivisible. One-parent systems
  have training data only; two-parent systems have train/validation only.
- Geometry fingerprint handles translation, atom permutation, global rotation,
  and periodic wrapping in the same lattice basis. It is not a general crystal
  equivalence solver. Parent grouping is the main correlated-frame safeguard.

| Chemical system | Frames | Parent materials | Train / valid / test |
|---|---:|---:|---:|
| Si | 155 | 27 | 124 / 16 / 15 |
| Si/O | 402 | 83 | 322 / 40 / 40 |
| Al/O | 285 | 33 | 228 / 29 / 28 |
| Hf/O | 53 | 7 | 43 / 5 / 5 |
| Zr/O | 105 | 21 | 84 / 11 / 10 |
| Cu | 52 | 8 | 42 / 5 / 5 |
| Al | 13 | 5 | 11 / 1 / 1 |
| Ru/O | 63 | 4 | 60 / 2 / 1 |

These counts demonstrate why a universal-material claim would be inappropriate.
Sparse systems remain unvalidated even if they participate in fitting.

### Stress and energy conventions

The official [MatPES training example](https://github.com/materialyzeai/matpes/blob/main/assets/Training%20a%20MatPES%20model.md)
uses ASE-order Voigt `(xx, yy, zz, yz, xz, xy)` and converts raw stresses by
`-0.1` to GPa. We independently implement that ordering and convert further by
`1/160.21766208` to tensile-positive eV/Angstrom^3. Total energies are eV, not
formation energies or per-atom labels. Forces are eV/Angstrom. Elemental offsets
are fitted on training data only and are explicitly not isolated-atom energies.

## Message passing: research and implementation decision

[PaiNN](https://proceedings.mlr.press/v139/schutt21a.html) provides evidence that
scalar/vector representations can efficiently carry directional information.
[TensorNet](https://arxiv.org/abs/2306.06482) uses Cartesian rank-2 tensors;
[MACE](https://arxiv.org/abs/2206.07697) uses higher-body-order equivariant
interactions. These have richer representation choices than our l=0/l=1 baseline.
We cite the ideas rather than calling their implementations or copying weights.

The [Multi-ACE design-space study](https://doi.org/10.1038/s42256-024-00956-x)
separates angular resolution, correlation/body order, depth, chemistry coupling,
and nonlinearities. Adding layers is not equivalent to systematically adding
angular irreducible representations or a complete many-body basis. Our nonlinear
network has implicit many-body interactions; we do not assign it a finite ACE
body order or claim complete angular expressiveness.

[eSEN](https://arxiv.org/abs/2502.12147) emphasizes conservative forces, smooth
energy derivatives, and actual energy-conservation checks. Implemented lessons:

1. Differentiate a scalar extensive energy for forces and homogeneous-strain stress.
2. Use all images within the distance cutoff, not top-k neighbors.
3. Multiply the entire edge message by a smooth envelope, including radial biases.
4. Use a fixed training-derived coordination scale, never an instantaneous degree
   denominator that changes discontinuously when neighbors enter or leave.
5. Normalize scalar node features only; do not apply arbitrary nonlinearities to
   vector components or mix statistics across atoms in a batch.
6. Test forces, stress, cutoff crossings, permutation symmetry, reflection symmetry,
   extensivity, and MD timestep convergence, beyond static prediction errors.

The compact backbone uses learned radial filters, source scalar features, a
central-node gate, and vector channels. For each directed periodic edge i <- j:

```
d_ij = r_j + shift_ij @ cell - r_i
u_ij = d_ij / |d_ij|
coeff_ij = cutoff(|d_ij|) * radial(RBF(|d_ij|))
           * source_MLP(s_j) * 2 sigmoid(target_linear(s_i))
m_scalar = coeff_scalar
m_vector = coeff_vector * W(v_j) + coeff_direction * u_ij
```

Messages sum at the receiving atom and are scaled by the square root of the
training mean neighbor count. On-node updates contract channel-mixed vectors to
invariant dot products, feed those plus scalar features through a scalar MLP,
and use scalar gates to update vectors. Four blocks use residual connections.
Scalar energy readouts sum over atoms. Reflections are preserved because features
are true scalars and polar vectors and no inconsistent pseudovector paths exist.

The effective message-passing receptive field is larger than one edge cutoff;
this does not replace explicit electrostatics. The Python neighbor builder uses
quadratic pair enumeration per periodic image. It is correctness-oriented for
small cells, not a production million-atom spatial-bin implementation.

[SwiGLU/GeGLU results](https://arxiv.org/abs/2002.05202) are from Transformers,
not evidence of an MLIP-specific winner. We compare SwiGLU, GeGLU, and SiLU under
matched scalar-MLP parameter budgets, the same structures/splits and three seeds.
Only validation force MAE and runtime determine the activation choice.

[UMA](https://arxiv.org/abs/2506.23971) demonstrates much larger universal-model
training, and [FlashTP](https://proceedings.mlr.press/v267/lee25l.html) addresses
tensor-product memory/computation. Neither justifies introducing large-model
complexity before benchmarking this 8 GB GPU and small chemistry subset. A later
rank-2 or explicit higher-order extension needs its own controlled accuracy,
memory, speed and MD comparison. No universal best backbone is asserted here.
# Expanded device-material experiment

[MP-ALOE (2025)](https://www.nature.com/articles/s41524-025-01834-9)
is directly compatible with MatPES r2SCAN according to its authors. The verified
public `MP_ALOE_data.jsonl.gz` archive contains 909,792 records, is 675,635,633
bytes, and has MD5 `0cac41b76cdc936848d360a14bdc9bf2`. It is CC BY 4.0.
Only data were used; pretrained models and external model packages were not used.

The device-composition selection yields 2,775 new non-MP frames, partitioned as
2,217 training, 279 development, and 279 test records. Entire composition/prototype
families stay together. Excluding MP-derived ALOE entries follows the authors'
merging recommendation. Original MatPES validation/test partitions are retained.
The combined training set has 3,156 frames. Its additional oxide coverage helps
address scarcity, but does not provide surface or chronological AIMD training.
Independent families may still contain structurally similar polymorphs; exact
geometry hashing is not an arbitrary-supercell structure matcher.
