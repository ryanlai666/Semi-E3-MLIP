# Reproducibility artifacts

| Location | Contents |
| --- | --- |
| `benchmark/` | Broad material metrics, source hashes, pilot completion and inference timing |
| `focused/` | Cu/Ti architecture selection, confirmation seeds, frozen checkpoints, tests, NVE |
| `aimd_comparison/` | Frozen model selection, reference hashes, static force errors, trajectory statistics |
| Root-level `*_audit.json`, `*_manifest.json` | Downloaded-source inventory, conversions, and split provenance |
| `device_search.json`, `tensor_search.json`, `expanded_search.json` | Validation-only pilot history, including interrupted runs |

Curated visual assets live in `docs/assets/`. Runtime logs, checkpoint binaries,
raw datasets, and generated trajectories remain local. `protocol_cpu_aborted.json`
records an abandoned CPU execution attempt, not an additional benchmark result.
JSON metric names specify units; force MAE is component-wise, energy MAE per atom.
