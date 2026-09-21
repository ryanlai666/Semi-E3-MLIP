# AIMD data expansion audit — 21 September 2026

The original three Si trajectories were insufficient. We have now downloaded and audited substantially more relevant data. The collection is sufficient to begin meaningful, source-specific learning-curve experiments for Si, Al, Cu and several refractory metals, with oxide and interface studies alongside them. It does **not** establish that one universal device-material model is adequately trained or validated.

## Acquired data, not just search results

Counts below are measured from the downloaded files. They count configurations, not independent simulations. Related frames remain correlated. The JSON audits in `reports/` provide the reproducible counts and limitations.

| Source | Actual local inventory | Classification and intended use |
|---|---:|---|
| [Al/Si interface study, 2025](https://doi.org/10.1063/5.0243641), [public data](https://github.com/krutarth24/Al-Si-DeePMD-NNP) | 52,608: Si 18,804; Al 15,804; Al/Si 18,000 | AIMD-derived PBE labels. Paper covers 50, 100, 200, 300, 400, 500 K and ramp tests. Raw arrays lack per-frame temperature/run IDs; storage shards are not independent trajectories. |
| [TM23, 2024](https://www.nature.com/articles/s41524-024-01264-z), [data](https://archive.materialscloud.org/record/2024.48) | 24,000 selected: 3,000 each of Cu, W, Ti, Ta, Co, Ru, Hf, Zr | AIMD snapshots relabelled with denser k meshes. Three temperature regimes, vacancy-containing cells. Published train/test: 21,600/2,400. Retained despite older date because of relevant metal dynamics. |
| [Si/SiO₂ device study, 2026](https://doi.org/10.1038/s43246-026-01130-z), [data](https://doi.org/10.6084/m9.figshare.29422061.v1) | 12 Si trajectories ×101 frames; 14 verified SiO₂ trajectories ×101 frames | Actual ordered CP2K PBE AIMD references at 300 K, 500 fs each. All Si frames have forces; only 84 of 1,414 silica frames have matched force labels. One silica trajectory rejected because only two positions accompany 101 energies. |
| [Hafnia study, 2025 preprint / 2026 journal](https://arxiv.org/abs/2511.09976), [data](https://github.com/nmdl-mizo/HfO2_data_and_model) | 3,478 structures; 6 separate BEC-labelled structures | **On-the-fly DFT-selected snapshots**, not continuous AIMD. PBEsol, six phases, reported 1–4,000 K. No per-frame phase/temperature IDs in XYZ. BEC data remain separate. |
| [Al₂O₃/ZrO₂, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12435261/), [data](https://doi.org/10.6084/m9.figshare.29923850) | Al₂O₃ file 2,190; ZrO₂ file 1,875 | Mixed PBE labels: AIMD samples, model-sampled and perturbed structures. Each file has 1,300 structures at the AIMD cell size (80/96 atoms); this is **not sufficient evidence to tag each frame as AIMD**. Paper covers 300–5,000 K. Pure-O entries are also present. |
| [UniAlCu, 2025](https://zenodo.org/records/15865800), [2026 benchmark](https://www.nature.com/articles/s41524-026-02043-8) | 147,576 records; 138,968 distinct exact geometries | Mixed DFT-labelled Cu, alumina and Cu/alumina snapshots. Published paper count differs from actual files. Contains 1,580 exact geometries shared across train/validation: **do not use the supplied split unmodified**. No per-frame AIMD identity/chronology. |
| [OMat24 official subset released December 2025](https://huggingface.co/datasets/facebook/OMAT24) | Scanned 1,171,309 train/validation/test records; 129 match our strict device scope, of which 84 are AIMD samples | PBE(+U), 1,000/3,000 K, NVT/NPT. Parent IDs retained. The random million-record subset has low yield for exact common-device compositions. It is not a million relevant device frames. |
| [Silica mantle study, 2025 data / 2026 paper](https://doi.org/10.1029/2025GL119400), [data](https://zenodo.org/records/17049935) | 20,353 entries in 191 NPY sets | Mixed AIMD/enhanced-sampling PBEsol relabels. Predominantly extreme-pressure silica. Parked outside the device baseline; a bigger count does not repair ambient-pressure coverage. |
| [Al–Si nucleation, 2025 data](https://doi.org/10.24435/materialscloud:3h-sc), [paper](https://arxiv.org/abs/2506.08818) | 36,000: original train 32,281 / test 3,719 | Genuine AIMD-derived pure Al/Si and alloy configurations, solids/undercooled liquids. **LDA**, not PBE. Source path and structure numbers are present, but relative paths can collide across compositions. Kept as a separate fidelity; text-unit conversion still requires confirmation. |

## Why OMol25 is not the main source

[OMol25](https://fair-chem.github.io/omol25/) is a large molecular dataset, including metal complexes and electrolytes, calculated with wB97M-V/def2-TZVPD in ORCA. It is not a periodic bulk metal/oxide AIMD dataset. Its charge/spin and energy conventions would introduce a different learning problem. OMat24 is the relevant Meta materials source. We used its data format directly with our own reader and the LMDB storage library; no FAIRChem, ASE, PyG, MACE, NequIP or DeePMD model implementation was installed or used.

OMat's AIMD runs are only 50 ×2 fs and its 1M subset samples isolated records. That helps force learning, but does not provide long continuous reference dynamics. [The current OMat paper](https://arxiv.org/abs/2410.12771) also describes PAW differences, PBE+U, and missing surface/defect coverage. Those limitations motivate the focused sources above.

## How these data will be used

1. Preserve r2SCAN, PBE, PBE+U, PBEsol, LDA, nonspin/spin and PAW settings as separate label families. Use source-specific models first; consider a shared backbone with explicit fidelity conditioning only after controlled comparisons. Element offsets alone cannot remove functional-dependent forces.
2. Keep original test labels protected. Group by parent or complete AIMD run, and use temperature/phase/source holdouts. Never assume a `set.000` shard is an independent run. Unknown trajectory metadata is marked unknown, not guessed from file order.
3. Remove exact duplicates and inspect near-duplicate structures before training. UniAlCu demonstrates that published partitions can leak. TM23's train/test share a trajectory; whole-temperature transfer tests are additional necessary checks, not independent-seed validation.
4. Begin with learning curves at increasing decorrelated training sizes and at least three seeds. Reserve final tests before tuning. Track per-material force and energy errors, high-force tails, and source transfer rather than only an aggregate MAE.
5. Use chronological references for structure distributions and MD comparisons; use sampled configurations for energy/force tests. Positions without velocities cannot establish a reference kinetic temperature. Short references cannot establish diffusion or conductivity. Match functional, ensemble, cell and temperature when interpreting model-versus-AIMD differences.

## What “enough” means here

**Enough to stop collecting indiscriminately and start the next substantial training experiments:** bulk Si/Al plus their interface, three thermal regimes for eight metals, and substantial oxide configuration coverage. More than a hundred million unrelated molecular structures would be less useful than these focused sets.

**Not enough to claim universal device coverage:** Ta₂O₅, TiO₂ defect dynamics, WO₃, RuO₂, doped hafnia, realistic interfaces and independent long oxide trajectories remain gaps. More data should be selected in response to a measured validation or dynamics failure. Our aspirational energy/force targets (10 meV/atom and 0.1 eV/Å) remain targets, not results or universal accuracy standards.

The collection is not automatically merged into the current r2SCAN training directory. New-source importers/audits explicitly mark unassigned partitions and missing chronology. Training performance on this new corpus has not yet been established.

## Reproduction

Run with the project virtual environment. `pip install -e .[data]` adds only LMDB for the storage reader.

```powershell
.venv\Scripts\python scripts/import_tm23.py
.venv\Scripts\python scripts/import_omat.py
.venv\Scripts\python scripts/audit_oxide_data.py
.venv\Scripts\python scripts/audit_unialcu.py
.venv\Scripts\python scripts/download_alsi_interface.py
.venv\Scripts\python scripts/audit_alsi_interface.py
.venv\Scripts\python scripts/audit_alsi_alloy.py
.venv\Scripts\python scripts/import_aimd.py --all-si-surfaces
.venv\Scripts\python scripts/import_silica_reference.py
```

Source metadata, Git hashes, archive checksums and audit reports are in `reports/`. Original files are in `data/raw/`; normalized source-separated records are in `data/source_isolated/`. Do not run external scripts or load bundled third-party potentials from the archives.
