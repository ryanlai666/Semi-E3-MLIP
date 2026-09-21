# MP-ALOE source and expansion audit

The user-suggested archive is already included in this project. The local file is
MP-ALOE v2 (Figshare article 29452190, file 57488704), 675,635,633 bytes, verified
against MD5 `0cac41b76cdc936848d360a14bdc9bf2` on 2026-09-21.

Source: [MP-ALOE v2](https://doi.org/10.6084/m9.figshare.29452190.v2), Kuner et al.
(2025), CC BY 4.0. The dataset contains DFT-labelled off-equilibrium structures;
it is not a collection of verified continuous AIMD trajectories. The archive's
pretrained MACE model files are not used by Semi-E3-MLIP.

## What is used now

The full 909,792-frame distribution was scanned. The current predeclared common-
oxide scope yields 2,775 retained candidate frames: 2,217 training, 279 auxiliary
validation, and 279 test frames, grouped by formula and prototype. The expanded
training set combines these 2,217 frames with 939 MatPES device frames, giving
3,156 training configurations. Model selection retains the original 145-frame
MatPES validation set. The 279-frame ALOE validation set remains separately
reserved and is not the model-selection partition in the reported benchmark.

MP-derived entries are excluded to reduce source leakage, and exact geometry
fingerprints are checked against all baseline partitions. Near-duplicate or
cross-prototype similarity is still a limitation.

## What remains available

The audit found 44,271 raw frames using only supported elements. It excludes
1,953 MP-derived entries, leaving 2,775 frames in the current formula scope and
39,543 raw candidates outside that scope. These include other oxides such as
WO, AlO, TaO, RuO, TiO, HfO, and ZrO, plus alloy/silicide systems.

These are metadata-screened candidates, not 39,543 additional verified training
examples. A new expansion experiment must convert labels, validate structures,
remove geometric duplicates, and preserve formula/prototype groups. Existing
validation and test groups must not be recycled into training. Additional
stoichiometries may improve diversity but do not establish coverage of realistic
surfaces, interfaces, defects, or finite-temperature dynamics.

The current benchmark remains frozen. A future expansion should use its own
numbered experiment branch and predeclared validation protocol.

## Reproduce

```powershell
python scripts/download_mpaloe.py
python scripts/audit_mpaloe_expansion.py
# After preparing the baseline MatPES data:
python scripts/device_subset.py
python scripts/prepare_mpaloe.py
```

The downloader verifies an existing archive rather than downloading it twice.
It supports resuming partial downloads and refuses an unexpected dataset version
or checksum. It downloads the dataset only, not pretrained potentials.

[Download verification](../reports/mpaloe_download.json) ?
[Import and split counts](../reports/mpaloe_merge.json) ?
[Unused-chemistry audit](../reports/mpaloe_expansion_audit.json)
