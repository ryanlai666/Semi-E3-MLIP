"""Build repository documentation directly from measured, immutable benchmark artifacts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'docs/assets'


def read(path,default=None):
    p=ROOT/path
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default


def number(value,scale=1):
    return f'{value*scale:.3f}' if value is not None else '--'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def copy_asset(source,name=None):
    source=ROOT/source
    if source.exists():
        target=ASSETS/(name or source.name)
        shutil.copy2(source,target)
        return target.name
    return None


def focused_table(study,tests,key="force_mae_eV_A",scale=1):
    rows=[]
    for metal in ('cu','ti'):
        for count in (300,900):
            trials=[t for t in study['trials'] if t['metal']==metal and t['train_frames']==count]
            values=[]
            for split in ('train','validation','cold','warm','melt'):
                scores=[scale*tests[t['name']][split]['overall'][key] for t in trials]
                values.append(f'{np.mean(scores):.3f} +/- {np.std(scores,ddof=1):.3f}')
            rows.append([metal.title(),count,*values])
    return table(['Material','Train frames','Train','Validation','Cold test','Warm test','Molten test'],rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    required=['reports/focused/summary.json','reports/focused/md.json','reports/aimd_comparison/metrics.json',
              'reports/benchmark/complete.json','reports/benchmark/pilot_completion.json',
              'reports/benchmark/inference.json','reports/simulation_checks.json',
              'reports/benchmark/dataset_distributions.json']
    missing=[p for p in required if not (ROOT/p).exists()]
    focused_state=read('reports/focused/confirm.json',{})
    if not focused_state.get('complete') or len(focused_state.get('trials',[]))!=12:
        missing.append('focused confirmation incomplete')
    aimd_state=read('reports/aimd_comparison/metrics.json',{'cases':{}})
    expected_cases=read('reports/aimd_manifest.json')['cases']
    if set(aimd_state['cases'])!=set(expected_cases) or any(
        len(c.get('model_simulations',[]))+len(c.get('failed_runs',[]))!=3 for c in aimd_state['cases'].values()):
        missing.append('AIMD cases or seeds incomplete')
    if not read('reports/benchmark/pilot_completion.json',{}).get('complete'):
        missing.append('existing pilot completion incomplete')
    if missing and not args.allow_partial:raise RuntimeError('Unfinished artifacts: '+', '.join(missing))
    if not args.allow_partial:
        from plot_aimd_results import main as plot_aimd
        plot_aimd()
    ASSETS.mkdir(parents=True,exist_ok=True)
    for name in ('backbone','generalization','hyperparameter_comparison'):
        for ext in ('png','svg','pdf'):copy_asset(f'reports/figures/{name}.{ext}')
    for name in ('architecture_comparison','temperature_transfer'):
        for ext in ('png','svg','pdf'):copy_asset(f'reports/focused/{name}.{ext}')
    aimd=read('reports/aimd_comparison/metrics.json',{'cases':{}})
    for name in aimd['cases']:
        for ext in ('gif','png'):copy_asset(f'reports/aimd_comparison/{name}.{ext}','aimd_'+name+'.'+ext)
        for ext in ('png','svg','pdf'):
            copy_asset(f'reports/aimd_comparison/{name}_metrics.{ext}','aimd_'+name+'_metrics.'+ext)
    for name in ('silicon','copper','silica','alumina','hafnia','titania'):
        for ext in ('gif','png'):copy_asset(f'reports/visualizations/{name}.{ext}','forces_'+name+'.'+ext)
    benchmark={s:read(f'reports/benchmark/{s}.json') for s in ('train','validation','test','oxide_test')}
    overview=table(['Partition','Frames','Energy MAE (meV/atom)','Force MAE (eV/A)'],[
        [label,benchmark[s]['overall']['frames'],number(benchmark[s]['overall']['energy_mae_eV_atom'],1000),
         number(benchmark[s]['overall']['force_mae_eV_A'])]
        for s,label in [('train','Expanded training'),('validation','MatPES validation'),('test','MatPES test'),('oxide_test','MP-ALOE oxide test')]])
    coverage=read('reports/benchmark/materials.json')['systems']
    materials=table(['System','Train / valid / test / oxide-test frames','Train force MAE','MatPES test force MAE','Oxide test force MAE'],[
        [r['label'],' / '.join(str((r[s] or {}).get('frames',0)) for s in ('train','validation','test','oxide_test')),
         number(r['train']['force_mae_eV_A']),number((r['test'] or {}).get('force_mae_eV_A')),
         number((r['oxide_test'] or {}).get('force_mae_eV_A'))] for r in coverage])
    screen=read('reports/focused/screen.json')
    screening=table(['Material','Architecture','Parameters','Warm validation force MAE (eV/A)'],[
        [t['metal'].title(),t['architecture'],f"{t['parameters']:,}",number(t['validation']['force_mae_eV_A'])] for t in screen['trials']])
    study=read('reports/focused/confirm.json',{'complete':False,'trials':[]})
    tests=read('reports/focused/tests.json')
    focused=focused_table(study,tests) if tests else f"Confirmation training in progress: {len(study['trials'])}/12 trial records. Final test results are not yet available."
    focused_energy=focused_table(study,tests,'energy_mae_eV_atom',1000) if tests else 'Energy tables will be generated after checkpoint freezing.'
    aimd_rows=[]
    for name,case in aimd['cases'].items():
        aimd_rows.append([name.replace('_',' '),case.get('force_frames',0),number(case.get('force_mae_eV_A')),
                          number(case.get('force_rmse_eV_A')),len(case.get('model_simulations',[])),len(case['failed_runs'])])
    aimd_table=table(['Reference case','Force frames','Force MAE (eV/A)','Force RMSE (eV/A)','Completed seeds','Failed seeds'],aimd_rows) if aimd_rows else 'AIMD comparison is running; no completed case metrics yet.'
    md=read('reports/focused/md.json',{'cases':{}})
    md_rows=[]
    for name,case in md['cases'].items():
        md_rows.append([name,'failed' if case.get('failed') else number(case['energy_range_eV_atom'],1000),
                        '--' if case.get('failed') else number(case['energy_slope_eV_atom_ps'],1000)])
    md_table=table(['Model / timestep','Energy range (meV/atom)','Fitted drift (meV/atom/ps)'],md_rows) if md_rows else 'Focused NVE diagnostics are queued after training.'
    inference=read('reports/benchmark/inference.json')
    perf=table(['Device','System','Atoms','Median (ms)','p10 / p90 (ms)'],[
        [r['device'],r['system'],r['atoms'],number(r['median_ms']),f"{r['p10_ms']:.2f} / {r['p90_ms']:.2f}"] for r in inference['cases']]) if inference else 'Inference timing is queued after training and AIMD finish to avoid competing workloads.'
    selection=read('reports/aimd_comparison/selection.json')
    aimd_visual='\n'.join(f'![{name} AIMD and model dynamics](assets/aimd_{name}.gif)\n\n![{name} structure statistics](assets/aimd_{name}_metrics.png)' for name,c in aimd['cases'].items() if c.get('model_simulations'))
    results=f'''# Measured results

Generated from versioned JSON reports. Checkpoint selection uses validation only.

## Multi-metal / metal-oxide model

Frozen checkpoint: `{selection['checkpoint']}`. SHA-256:
`{selection['checkpoint_sha256']}`.

{overview}

All force MAEs are averages over Cartesian force components. Energy MAE is per atom.
Training metrics use the selected checkpoint, not the last epoch. The expanded
training set includes MatPES and MP-ALOE data; their held-out sets remain separate.
The broad model does not meet the original 10 meV/atom / 0.1 eV/A test goals.

![Material performance and coverage](assets/material_benchmark.png)

{materials}

Errors are in eV/A. `--` means no test examples. Counts are frames, not independent
trajectories. Some systems have only one to five test structures. Ru and Ru oxides
lack MatPES test examples; Ru oxides have a separate MP-ALOE test partition.
A small elemental test error does not establish phase, defect, or temperature coverage.

![Force parity for predeclared examples](assets/force_parity.png)

Parity plots use the lexicographically first held-out ID in each chemical system;
they are illustrations, not substitutes for all-frame errors above.

## Dataset distributions

![Label and structure-size distributions](assets/dataset_distributions.png)

![Per-system counts and training distributions](assets/dataset_chemistry.png)

Energy, structure size and stress distributions weight each frame equally.
Force-magnitude curves weight atoms; component curves weight Cartesian components.
The energy residual subtracts elemental offsets fitted only to the training set.
Raw energies differ strongly by composition, so their pooled spread is not a
measure of structural diversity. The expanded training and MP-ALOE test sets
contain substantially stronger force/stress labels than the MatPES held-outs;
source and composition shifts must be considered alongside aggregate errors. ECDFs retain the full tails; the box plots show
median/IQR, 1.5-IQR whiskers, and outliers. Stress uses the Frobenius norm in eV/A^3;
unlabelled frames are excluded and counted in the numerical report.

[Exact quantiles, source counts, and input hashes](../reports/benchmark/dataset_distributions.json).

## Cu/Ti controlled temperature transfer

{screening}

Architecture screening uses 300 cold frames, 90 warm validation frames, seed 42,
and 60 epochs. Both materials selected the tensor model under the fixed 2% rule.
Architectures have unequal parameter counts. No test frames select the models.

{focused}

Force MAE in eV/A; mean +/- sample SD across three initialization seeds.

Selected-checkpoint energy MAE (meV/atom):

{focused_energy}

Cold/warm train and test pools share source trajectories. Molten configurations
are a held-out temperature/trajectory shift. Seed variation is not independent-
dataset uncertainty. The 900-frame runs consume more optimizer updates than
300-frame runs, so this is an equal-epoch, not equal-compute, learning curve.

## Numerical stability

{md_table}

These are 100 fs model-only NVE checks at 300 K initial velocities. They measure
integration behavior, not agreement with DFT trajectories. A short trace and a
small energy drift do not establish long-time stability.

## Inference timing

{perf}

Batch size one; five warmups and 30 timed calls per structure. Timings include
neighbor enumeration, transfers, energy, forces, and stress. CPU and CUDA use the
same checkpoint. This is a local small-cell benchmark, not a comparison against
MACE, NequIP, or other packages. Hardware and software are recorded in
[`inference.json`](../reports/benchmark/inference.json).

## External AIMD comparison

{aimd_table}

Three 216-atom Si surface references at 300 K; 500 fs per reference and model run.
Each model case uses seeds 42/43/44, a 0.5 fs timestep, and Langevin dynamics.
Static force errors use every fifth stored reference frame (21 frames per case).
The reference uses CP2K/PBE and Nose-Hoover; this model uses r2SCAN training labels.
Initial velocities are unavailable, so trajectories start from the same geometry
with independent model velocities. These differences preclude a matched-Hamiltonian
accuracy claim or pointwise trajectory agreement claim.

{aimd_visual}

Pair-distance density is normalized per atom and distance, not by bulk density,
because these cells contain vacuum. Displacement is COM-corrected displacement
from the initial frame, not a time-origin averaged diffusion MSD. No diffusion
coefficient is inferred from 500 fs.

## Reproducibility

- [Focused protocol](focused_experiments.md), [frozen checkpoint hashes](../reports/focused/frozen.json).
- [Broad-model selection](../reports/aimd_comparison/selection.json), [benchmark data hashes](../reports/benchmark/protocol.json).
- [AIMD protocol and hashes](../reports/aimd_comparison/protocol.json), [source manifest](../reports/aimd_manifest.json).
- [Material results](../reports/benchmark/materials.json), [raw focused results](../reports/focused/tests.json).
- [Data acquisition and limitations](aimd_expansion.md), [simulation protocol](simulation_validation.md).
'''
    (ROOT/'docs/results.md').write_text(results,encoding='utf-8')
    focus_image='![Controlled temperature transfer](docs/assets/temperature_transfer.png)' if (ASSETS/'temperature_transfer.png').exists() else ''
    preview=next((name for name,c in aimd['cases'].items() if c.get('model_simulations')),None)
    aimd_preview=f'![AIMD reference and independent model dynamics](docs/assets/aimd_{preview}.gif)' if preview else ''
    state='Training and evaluation complete for the recorded study.' if not missing else 'Results are being completed; finished benchmarks below are measured, and pending stages are labeled.'
    readme=f'''<div align="center">

# Semi-E3-MLIP

**Conservative E(3)-equivariant interatomic potentials for metals and metal oxides**

PyTorch from scratch · Energy-derived forces · Periodic structures · Reproducible benchmarks

[Results](docs/results.md) · [Quick start](#quick-start) · [Model & physics](docs/attention.md) · [Data](docs/aimd_expansion.md) · [Reproduce](docs/usage.md)

</div>

![Implemented E(3) model architecture](docs/assets/backbone.png)

Semi-E3-MLIP is a research codebase for learning potential-energy surfaces across
semiconductor-relevant materials. It implements scalar, vector, and optional
rank-two tensor features; gated or smooth-attention message passing; and
conservative forces and stress from a learned total energy. It uses PyTorch
without ASE, e3nn, PyG, or pretrained MLIP weights.

**Status:** {state} The repository remains private.
The broad model is a research baseline, not a validated production MD potential.

## Materials at a glance

| Family | Included chemical systems |
| --- | --- |
| Elemental metals and silicon | Al, Co, Cu, Hf, Ru, Si, Ta, Ti, W, Zr |
| Oxides | Al-O, Co-O, Cu-O, Hf-O, Ru-O, Si-O, Ta-O, Ti-O, W-O, Zr-O |
| Controlled temperature-transfer study | Separate Cu and Ti models, 300/900 cold frames, three seeds |
| Verified continuous AIMD comparison | Three Si surface trajectories, 216 atoms, 300 K, 500 fs |

The broad training set contains **3,156 configurations across 20 chemical systems**.
Coverage varies greatly by material and does not imply arbitrary alloy/interface
support. Per-system counts, errors, and missing test coverage are explicit in
[the material results](docs/results.md#multi-metal--metal-oxide-model).

## Dataset distributions

![Energy, force, stress, and structure-size distributions](docs/assets/dataset_distributions.png)

The partitions are shown separately. Composition-adjusted energies subtract
training-only elemental offsets; force distributions retain their high-force tails.
[Per-system distributions, weighting, and numerical quantiles](docs/results.md#dataset-distributions).

## Measured train and test results

The same frozen, validation-selected multi-material checkpoint is used throughout
this table and the external silicon comparison.

{overview}

The test errors remain above the original 10 meV/atom and 0.1 eV/A targets.
Ta-containing test configurations are a prominent failure case; aggregate
performance must not be interpreted as uniform accuracy across materials.

![Per-material accuracy and data coverage](docs/assets/material_benchmark.png)

### Controlled Cu/Ti study

{focused}

Force MAE in eV/A; mean +/- SD across three initialization seeds when complete.
Architecture selection uses warm validation only. See the [fixed experimental
protocol](docs/focused_experiments.md) for temperature definitions and correlated-frame limitations.

{focus_image}

### AIMD versus model inference

{aimd_table}

{aimd_preview}

The animation compares a real **CP2K/PBE AIMD reference** with independent
**r2SCAN-trained model Langevin dynamics**. It shows cross-functional and surface
transfer, with different thermostats and unavailable reference velocities.
It is not a pointwise trajectory accuracy claim. [All cases, metrics, and
structural statistics](docs/results.md#external-aimd-comparison).

## Metal and oxide visualizations

![DFT and model forces on held-out metals and oxides](docs/assets/metal_oxide_forces.png)

Each pair uses the same held-out geometry, camera, and force-arrow scale.
Examples are selected by ID rather than error. These are static DFT comparisons;
they are not presented as continuous AIMD. [Force parity and complete metrics](docs/results.md).

## Quick start

Tested locally on Windows, Python 3.13, and an RTX 3070 Laptop GPU (8 GB).
CPU inference is also supported. Install a PyTorch build suited to your machine.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python -m pip install torch numpy
.\\.venv\\Scripts\\python -m pip install -e ".[test,visualize]"
.\\.venv\\Scripts\\python -m pytest -q
```

```powershell
# Download and prepare the baseline data
.\\.venv\\Scripts\\python -m semi_mlip download
.\\.venv\\Scripts\\python -m semi_mlip prepare

# Train; checkpoints and data remain local
.\\.venv\\Scripts\\python -m semi_mlip train --run runs/pilot --epochs 50

# Predict energy, forces, and stress from a local checkpoint
.\\.venv\\Scripts\\python -m semi_mlip predict --checkpoint runs/pilot/best.pt --structure examples/silicon.json --output runs/prediction.json
```

The import remains `semi_mlip`; command aliases are `semi-e3-mlip` and `semi-mlip`.
[Detailed setup, checkpoint resumption, relaxation, and MD](docs/usage.md).
The MP-ALOE v2 archive is already used in the expanded training set. Reproduce its
checksum-verified acquisition with `python scripts/download_mpaloe.py`, then use
`device_subset.py` and `prepare_mpaloe.py` after baseline preparation. The source
is [MP-ALOE on Figshare](https://doi.org/10.6084/m9.figshare.29452190.v2), licensed
CC BY 4.0; see the [import audit](reports/mpaloe_merge.json) for grouping and counts.
Pretrained source-model weights are not used. [MP-ALOE coverage and expansion audit](docs/mpaloe.md).

Raw datasets and checkpoint binaries are excluded from Git; cloning the repository
does not download trained weights. Dataset acquisition scripts and source manifests
are versioned separately from the source archives.

## Repository guide

| Path | Purpose |
| --- | --- |
| `semi_mlip/` | Model, periodic graphs, training, inference, relaxation, and MD |
| `scripts/` | Data audits, experiment runners, benchmarks, and report generation |
| `tests/` | Symmetries, derivatives, splitting, resume, integrators, and analysis |
| `docs/` | Physics, protocols, data provenance, results, and curated figures |
| `reports/` | Machine-readable audits, metrics, and checkpoint/data hashes |
| `examples/` | Minimal structure inputs |
| `data/`, `runs/` | Local-only datasets, checkpoints, histories, and trajectories |

[Script guide](scripts/README.md) · [Documentation index](docs/README.md) · [Experiment versioning](docs/versioning.md)

## Scientific boundaries

- E(3) equivariance, conservative forces, and energy conservation are implementation
  properties; they do not guarantee DFT accuracy or chemical transferability.
- The current backbone has no long-range electrostatics, charge equilibration,
  spin degrees of freedom, or repulsive core. It is intended for small-cell research.
- Test subsets are small and uneven. Unknown chemical systems are rejected, but
  familiar elements do not make a new geometry in distribution.
- DFT functional, trajectory, temperature, and source differences are preserved in
  the reports. Short model-only MD checks and external AIMD comparisons are separate.
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    manifest={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'complete':not missing,
              'pending_artifacts':missing,'checkpoint_sha256':selection['checkpoint_sha256'],
              'materials':len(coverage),'aimd_completed_cases':len(aimd['cases'])}
    (ROOT/'reports/benchmark/documentation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
