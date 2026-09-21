# Next material studies

The follow-up scope is W, Ta, Co, Ru, Hf, Zr, elemental Al and Si, and ten oxide chemical systems. Each gets a separate model and three initialization seeds. The fixed architecture and training budget are recorded before fitting in `reports/material_studies/protocol.json`.

- TM23: 900 cold training frames, 90 warm validation frames, official cold/warm/molten tests; PBE labels stay separate from r2SCAN.
- Al/Si and oxides: retain the existing MatPES/MP-ALOE parent partitions. Previously reserved ALOE validation joins MatPES validation for these new studies. Report MatPES and ALOE tests separately.
- Architecture: the existing tensor configuration, fixed 60-epoch budget, seeds 42/43/44. Select the checkpoint using validation force MAE only. Freeze all 54 checkpoint hashes before test evaluation.
- Al's current 11 training frames and single test frame make its result exploratory. Other imported Al sources require additional unit/provenance/split work before use; their presence is not evidence of validated training coverage.
- These r2SCAN tests were already reported for the broad model. This is a fixed follow-up comparison, not a new blind benchmark. No test-driven hyperparameter search is planned.

Preparation checks IDs, exact geometry overlap and, for r2SCAN, parent overlap. Source trajectories in TM23 remain correlated within temperature regimes. Initialization-seed variation does not measure dataset uncertainty.

Run `python scripts/material_studies.py --prepare-only` to audit and freeze the input protocol. Run `python scripts/material_studies.py` to resume training, freeze checkpoints, evaluate and generate the study report. The current Cu/Ti release and its measurements finish first; run this next round serially on the GPU after publishing that milestone.

The frozen Cu/Ti study revealed severe molten-regime failures. The fixed tensor follow-up measures whether that limitation extends to other systems; it does not claim improved stability or universal transfer. Every seed, including extreme or failed predictions, must remain visible.
