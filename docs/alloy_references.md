# Replacement alloy references

The Al/Si interface snapshot collection is no longer a recommended alloy or phase-diagram benchmark for this project. This decision concerns its suitability for the requested study, not a claim that its DFT labels are wrong. The existing external Si surface AIMD plots remain separately labeled historical transfer diagnostics.

## Downloaded: UNEP-v1 external alloy DFT tests

The [primary paper](https://www.nature.com/articles/s41467-024-54554-x) explicitly links [training and test data](https://doi.org/10.5281/zenodo.11533864), including extended-XYZ energy/force labels. We downloaded the **11.36 MB test archive**, README and INCAR and verified their published MD5 checksums. No pretrained NEP weights were downloaded or used.

Our schema audit found **4,849 frames**, of which **1,283 use only elements supported by the current shared model**. Relevant subsets include Al-Cu (124), Ti-Zr (124), Al-Cu-Zr (131), Cu-Ti-Zr (127), plus elemental and other supported alloy records. Counts are chemistry totals across source files, not numbers of independent trajectories. There is no Al-Si subset in this archive.

The supplied INCAR specifies PBE, 600 eV cutoff, KSPACING 0.2 and EDIFF 1e-6. Named subsets cover heating, compression, stretching and equation-of-state configurations. They offer explicit stress/deformation coverage missing from the rejected interface-only choice. These filenames do not establish continuous AIMD: source sampling and temporal provenance still need verification.

**Status:** downloaded and schema-audited, not yet evaluated or used for training. For the current r2SCAN checkpoint this would be a PBE transfer benchmark; it must not be silently merged with r2SCAN training. Before evaluation, freeze supported subsets and duplicate checks, confirm units and energy references, and report all eligible errors. [Audit and complete file inventory](../reports/unep_reference_audit.json).

## Verified access: AFLOW relaxed Al-Si structures

AFLOW serves relaxed structures with calculation metadata, energies and formation enthalpies. We downloaded metadata and relaxed CONTCAR files for [Al4Si8, entry 281](https://aflowlib.duke.edu/AFLOWDATA/LIB2_RAW/AlSi/281/) and [Al8Si4, entry 282](https://aflowlib.duke.edu/AFLOWDATA/LIB2_RAW/AlSi/282/). Both identify PAW/PBE. This is more directly relevant to solid phase-energy checks than an interface trajectory collection. See the [official API documentation](https://aflowlib.org/documentation/) and [local access audit](../reports/aflow_alsi_reference_audit.json).

**Status:** two access/provenance samples only. These do not constitute a converged hull. A defensible benchmark still needs a systematic composition/structure inventory, matched relaxed elemental endpoints, convergence checks, and consistent energy references. PBE versus r2SCAN remains a separate source of error. Raw AFLOW data stay local and are subject to the source's terms; they are not relicensed with the repository code.

## Investigated but not adopted

- [Kyoto PolyMLP Al-Si](https://cms.mtl.kyoto-u.ac.jp/seko/mlp-repository/alloy2/Al-Si-2022-06-12/info.html) reports 36,924 prototype-derived structures, but its [binary DFT download page](https://cms.mtl.kyoto-u.ac.jp/seko/mlp-repository/datasets/alloy2/datasets.html) says "Coming soon." Model downloads are not a substitute for independently accessible DFT labels.
- [Al-Si nucleation data](https://doi.org/10.24435/materialscloud:3h-sc) remain a relevant liquid-alloy candidate, but the local audit has not resolved the text-unit declaration; labels use LDA and source-family overlap also needs handling. Do not promote it as a matched r2SCAN reference.

These new references do not alter the frozen [MP-ALOE alloy results](alloy_validation.md) or the [1x/2x/4x width study](width_scaling.md). The latter deliberately holds data fixed so a size effect can be measured.
