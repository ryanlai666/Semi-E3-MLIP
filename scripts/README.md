# Experiment and reporting scripts

Run from the repository root after installing the package. Datasets/checkpoints
are local prerequisites; the repository intentionally excludes their binaries.

| Stage | Entry point | Outputs |
| --- | --- | --- |
| Baseline dataset | `python -m semi_mlip download` and `prepare` | Local raw/processed data and audits |
| MP-ALOE acquisition | `download_mpaloe.py`, `audit_mpaloe_expansion.py` | Verified v2 archive and unused-chemistry inventory |
| Device scope | `device_subset.py`, `prepare_mpaloe.py` | Fixed device and expanded partitions |
| Focused data | `import_tm23.py`, `prepare_focused.py` | Cu/Ti cold training and protected temperature regimes |
| Focused pipeline | `continue_focused.ps1` | Screening, three-seed confirmation, frozen tests, NVE |
| Fresh baseline reproduction | `reproduce_baseline.py --run runs/reproduction/broad` | Same fixed configuration, new checkpoint and local evaluations |
| Remaining existing pilots | `resume_pilots.py` | Exact saved-config resumption; no new variants |
| Material benchmark | `benchmark_materials.py --device cpu` | 20-system train/validation/test metrics and figures |
| External AIMD | `compare_aimd.py CHECKPOINT --device cuda` | Three-seed Si surface comparison and animations |
| Dataset distributions | `dataset_distributions.py` | Split-aware energy/force/stress distributions, chemistry coverage, quantiles |
| Inference timing | `benchmark_inference.py` | End-to-end CPU/CUDA small-cell timings |
| Model diagrams and curves | `paper_figures.py CHECKPOINT` | PNG/SVG/PDF architecture and learning figures |
| Metal/oxide force and MD visuals | `validate_and_render.py --checkpoint CHECKPOINT --data data/device/test.jsonl --device cuda` | Static DFT force comparisons and model-only MD |
| README and result tables | `build_project_report.py` | Documentation generated from completed reports |

The current broad benchmark and AIMD study use the immutable checkpoint selection
in `reports/aimd_comparison/selection.json`. Do not replace that checkpoint after
reading external results. Protocol files hash inputs; changed studies should use
new output locations and an experiment branch.

`finish_project.ps1` is a local continuation supervisor. Its optional process IDs
refer to this machine's current jobs; omit them only when prerequisites are done.
It stops on failed stages. It does not commit or push automatically.

`build_project_report.py --allow-partial` builds an explicitly unfinished progress
report. The default requires final artifacts. Once a focused study is frozen,
rerun `finish_focused.py --md` for evaluation; do not restart its tuning stage.

The `audit_*`, `import_*`, and `download_*` scripts preserve source/fidelity
boundaries. See the data audit before combining source archives. The `tune_*`
scripts include historical planned variants; not every variant was started.
