"""Add reproducible suitability assessments without retraining or changing frozen metrics."""
import json
import hashlib
from pathlib import Path

START='<!-- material-assessment:start -->'
END='<!-- material-assessment:end -->'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def update_report():
    summary=read('reports/material_studies/summary.json')
    results=read('reports/material_studies/results.json')
    protocol=read('reports/material_studies/protocol.json')
    broad={r['system']:r for r in read('reports/benchmark/materials.json')['systems']}
    lines=[START,
        '## Are these models good enough?', '',
        '**Not yet for general predictive MD, molten phases, or metal/oxide interfaces.** W, Ta and Co show promising accuracy only within the sampled cold TM23 regime. Most transfer tests miss the project targets, and several show severe extrapolation failures. Short energy-conservation checks establish numerical behavior of the learned potential, not agreement with DFT.', '',
        'The screening targets here are energy MAE <= **10 meV/atom** and force MAE <= **0.1 eV/A**, both required on a held-out partition. These are this project\'s working targets, not universal acceptance standards. Passing them does not validate a particular physical observable. Counts below require each seed to meet both targets; they do not hide failed seeds behind an average. Seed SD is fit variability, not a confidence interval over independent trajectories.', '',
        '## One potential or separate potentials?', '',
        '| Model family | Learned weights | Scope and role |',
        '| --- | --- | --- |',
        '| This page: 18 studies, 54 fits | Separate weights for every study and seed | Single-element or single metal-oxygen chemical-system specialists; three seeds are replicates, not an automatically combined ensemble |',
        '| Earlier Cu/Ti study: 12 fits | Separate Cu and Ti weights, two data sizes, three seeds | Additional elemental temperature-transfer experiments; [results](results.md#cuti-controlled-temperature-transfer) |',
        '| Shared `expanded_tensor` baseline | One checkpoint jointly trained on all 20 chemical systems | Ten elements and their ten oxygen-containing systems in one model; [shared results](results.md#multi-metal--metal-oxide-model) |', '',
        '**Same hyperparameters do not mean shared learned parameters.** The specialists use the same 48 scalar / 24 vector / 8 tensor channels, three blocks, 24 radial functions, 5 A cutoff, four attention heads and SwiGLU. Each trains from scratch for 60 epochs with learning rate 0.001, pseudo-Huber loss, energy/force weights 1/10, zero stress-loss weight, and seeds 42/43/44. Checkpoints minimize validation force MAE. Dataset sizes, update counts, fitted elemental energy offsets, loss scales, neighbor normalization and final learned weights differ between systems. Thus equal epochs are not equal compute or identical optimization trajectories. These are fixed-protocol baselines, not individually tuned best models.', '',
        'The shared baseline is a different experiment: 64/32/16 channels, four blocks, 32 radial functions, up to 200 epochs and a nonzero stress-loss weight. It jointly fits 3,156 r2SCAN configurations. Comparing its errors with the specialists is useful, but does not isolate parameter sharing from architecture, data volume or training budget.', '',
        '**A specialist cannot represent the union of the systems.** An Al-O model has seen Al-O chemistry, not Cu-O, W-O or arbitrary mixtures. Do not switch between elemental and oxide checkpoints atom by atom or along a reaction: their independently fitted energy functions do not define a single consistent energy surface. The shared model can accept the listed elements together, but elemental/oxide coverage alone does not validate alloys, multication oxides, oxygen molecules, oxidation reactions or interfaces.', '',
        '## Complete metal/oxide coverage', '',
        'Cu and Ti were omitted from the follow-up table because their elemental models were completed in the earlier study. They were not omitted from the repository. Si is included as the semiconductor member of the paired coverage.', '',
        '| Element | Elemental specialist | Oxide specialist on this page | Shared model includes both? |',
        '| --- | --- | --- | --- |']
    mapping=[('Al','r2scan_al','r2scan_al_o','Al-O'),('Si','r2scan_si','r2scan_o_si','O-Si'),
        ('Cu','Earlier Cu/Ti study','r2scan_cu_o','Cu-O'),('Ti','Earlier Cu/Ti study','r2scan_o_ti','O-Ti'),
        ('W','tm23_w','r2scan_o_w','O-W'),('Ta','tm23_ta','r2scan_o_ta','O-Ta'),
        ('Co','tm23_co','r2scan_co_o','Co-O'),('Ru','tm23_ru','r2scan_o_ru','O-Ru'),
        ('Hf','tm23_hf','r2scan_hf_o','Hf-O'),('Zr','tm23_zr','r2scan_o_zr','O-Zr')]
    for element,metal,oxide,chem in mapping:
        assert element in broad and chem in broad and oxide in summary
        lines.append(f'| {element} | {metal} | {oxide} | Yes |')
    lines += ['', 'Coverage still has gaps: the shared model has **no elemental Ru held-out test** in this benchmark. Elemental Al and Hf each have only one shared-model test frame. The elemental TM23 results use PBE while the oxide specialists use r2SCAN; they are not a matched-fidelity metal/oxide pair experiment. There are no dedicated matched r2SCAN elemental specialists for Cu, Ti, W, Ta, Co, Ru, Hf or Zr in this follow-up, although all are included in the shared model.', '',
        '## Elemental suitability', '',
        '| Metal | Cold energy / force MAE (meV/atom; eV/A) | Seeds passing cold / warm / molten | Interpretation |',
        '| --- | ---: | --- | --- |']
    verdicts={'w':'Promising for sampled cold structures; warm transfer misses force target; molten predictions fail severely.',
        'ta':'Promising for sampled cold structures; warm and molten force errors remain too large.',
        'co':'Promising for sampled cold structures; warm results vary strongly by seed and molten failures are severe.',
        'ru':'Cold mean already misses both targets; warm and molten results do not support deployment.',
        'hf':'Cold mean meets targets, but only two seeds pass; warm/molten errors prevent a robust suitability claim.',
        'zr':'Cold force mean is low, but energy mean misses target; only one seed passes both. Transfer remains inadequate.'}
    assessment={}
    for metal in ('w','ta','co','ru','hf','zr'):
        name='tm23_'+metal
        trials=[v for v in results.values() if v['study']==name]
        counts={split:sum(v['splits'][split]['energy_mae_eV_atom']<=.01 and v['splits'][split]['force_mae_eV_A']<=.1 for v in trials)
                for split in ('test_cold','test_warm','test_melt')}
        cold=summary[name]['test_cold']; e=cold['energy_mae_eV_atom']['mean']*1000; f=cold['force_mae_eV_A']['mean']
        lines.append(f'| {metal.title()} | {e:.2f} / {f:.4f} | '+ ' / '.join(f'{n}/3' for n in counts.values())+f' | {verdicts[metal]} |')
        assessment[name]={'passing_seeds':counts,'interpretation':verdicts[metal]}
    lines += ['', 'Cold and warm TM23 tests share source trajectories with development data, so low cold errors are evidence of interpolation within this sampling, not independent phase/defect/trajectory generalization. None of these six metals passes both targets in any seed on the warm or molten test partitions.', '',
        'The earlier 900-frame Cu/Ti fits have mean cold force MAEs of about 0.00564 and 0.0440 eV/A, respectively, but their molten means rise to about 3.35e4 and 3.68e27 eV/A. These are extrapolation failures, not acceptable MD errors.', '',
        '**Al:** its specialist has just one test frame: about 124 meV/atom energy MAE despite a small 0.0012 eV/A force MAE. That is neither adequate energetic accuracy nor enough test diversity. **Si:** its 15-frame test gives about 462 meV/atom and 0.257 eV/A; both targets are missed. Neither supports a general elemental-potential claim.', '',
        '## Oxide suitability and comparison with the shared model', '',
        '**None of the ten oxide specialists meets both targets on any available held-out source partition in any seed.** The MP-ALOE table below compares the same held-out chemistry subsets. Specialist values are three-seed means; the shared value is one frozen checkpoint. Lower shared-model errors are observed results, not a controlled proof that sharing alone caused the improvement.', '',
        '| System | MP-ALOE test frames | Specialist energy / force MAE | Shared energy / force MAE | Interpretation |',
        '| --- | ---: | ---: | ---: | --- |']
    oxide_notes={'Al-O':'Large energy errors; low MatPES force error does not transfer to off-equilibrium data.',
        'Co-O':'Both force and energy targets missed; reactive or phase-transfer use unvalidated.',
        'Cu-O':'Both targets missed; does not validate Cu/CuO interfaces or oxidation.',
        'Hf-O':'Both targets missed; polymorph and defect energetics remain unvalidated.',
        'O-Ru':'Both targets missed; only MP-ALOE oxide test coverage, no MatPES oxide test.',
        'O-Si':'Both targets missed; not evidence for accurate silica or Si/SiO2 interfaces.',
        'O-Ta':'Especially poor forces; only three MP-ALOE test frames, so coverage is also weak.',
        'O-Ti':'Both targets missed; oxide/metal transfer and defects unvalidated.',
        'O-W':'Both targets missed; no evidence for reliable oxide thermodynamics or MD.',
        'O-Zr':'Lower errors than several other oxides, but still fails both project targets.'}
    improved=0
    for _,_,name,chem in mapping:
        a=summary[name]['aloe_test'];b=broad[chem]['oxide_test'];n=protocol['studies'][name]['partitions']['aloe_test']['frames']
        assert n==b['frames']
        e=a['energy_mae_eV_atom']['mean']*1000;f=a['force_mae_eV_A']['mean']
        improved+=b['energy_mae_eV_atom']<e/1000 and b['force_mae_eV_A']<f
        lines.append(f'| {chem} | {n} | {e:.1f} / {f:.3f} | {b["energy_mae_eV_atom"]*1000:.1f} / {b["force_mae_eV_A"]:.3f} | {oxide_notes[chem]} |')
    lines += ['', 'Energy units: meV/atom; force units: eV/A. The shared model has lower energy and force MAE for all '+str(improved)+' oxide subsets here, but it still fails the joint targets for every oxide. Al-O illustrates why force alone is misleading: its specialist MatPES force MAE is 0.0369 eV/A, while energy MAE is about 1,320 meV/atom and MP-ALOE force MAE rises to 1.432 eV/A.', '',
        '## What is needed for one useful metal/oxide potential?', '',
        'Start from a jointly trained, single energy model with compatible DFT labels. Add representative elemental, oxide, metal/oxide interface, vacancy, surface, strained and high-temperature environments for the intended application; binary end-member coverage is insufficient. Keep PBE and r2SCAN separate unless a documented fidelity-aware approach is introduced.', '',
        'Use held-out parent structures and independent trajectories, report worst cases and force-error tails, and validate the quantities that matter: relative phase/formation energies, equations of state, elastic/stress response, defect or adsorption energies, and reaction barriers as applicable. Then compare stable MD observables with reference data over the required temperature and time range. The existing short Si AIMD comparison and 100 fs NVE checks do not validate all metals and oxides.', '',
        'For this repository, the immediate research priorities are to diagnose the cold-to-molten extrapolation failures, improve the shared model using representative compatible data, balance energy and force validation, and expand the weak elemental Al/Hf/Ru tests. More runs with unchanged narrow training distributions alone would not establish suitability.', '', END]
    p=Path('docs/material_studies.md');text=p.read_text(encoding='utf-8')
    if START in text:
        before,rest=text.split(START,1);_,after=rest.split(END,1);text=before+'\n'.join(lines)+after
    else:
        head,body=text.split('\n',1);text=head+'\n\n'+'\n'.join(lines)+'\n\n## Detailed metrics\n'+body
        text=text.replace('## Interpretation\n','## Protocol limitations\n')
    if '## Detailed metrics\n' in text:
        text,details=text.split('## Detailed metrics\n',1)
        Path('docs/material_study_metrics.md').write_text('# Detailed specialist metrics\n\n[Suitability and coverage](material_studies.md) | [Parity gallery](parity.md)\n\n'+details,encoding='utf-8')
    text=text.rstrip()+'\n\n## Plots and detailed results\n\n- [Train/test parity gallery: shared model overall, every system, and all specialists](parity.md).\n- [Detailed train/validation/test tables, learning curves and distributions](material_study_metrics.md).\n- [Frozen protocols, checkpoint hashes and raw metrics](../reports/material_studies).\n' if '## Plots and detailed results' not in text else text
    navigation='[Train/test parity plots](parity.md) | [Detailed metrics](material_study_metrics.md) | [Model scope](#one-potential-or-separate-potentials) | [Metal/oxide coverage](#complete-metaloxide-coverage)'
    if navigation not in text:text=text.replace('# Independent material studies\n','# Independent material studies\n\n'+navigation+'\n',1)
    if Path('reports/alloy_validation/summary.json').exists() and '## Alloy transfer follow-up' not in text:
        text+='\n## Alloy transfer follow-up\n\nThe frozen shared model has now been checked on 2,421 previously unused r2SCAN binary configurations across Al-Si, Al-Ti, Cu-Zr, Hf-Zr, Ta-W and Ti-W. None meets both project targets. See [alloy errors, parity plots and phase-stability limits](alloy_validation.md). These are off-equilibrium DFT snapshots, not a validated equilibrium phase diagram.\n'
    p.write_text(text,encoding='utf-8')
    Path('reports/material_studies/suitability.json').write_text(json.dumps({'thresholds':{'energy_mae_eV_atom':.01,'force_mae_eV_A':.1},
        'purpose':'Project screening targets; not a production-MD certification','elemental':assessment,'oxide_subsets_shared_better_both_metrics':int(improved),
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('reports/material_studies/results.json'),Path('reports/material_studies/summary.json'),Path('reports/benchmark/materials.json')]},
        'specialists_share_weights':False,'shared_baseline_covers_systems':len(broad)},indent=2)+'\n')
    print('Updated measured suitability assessment and complete paired coverage.')

if __name__=='__main__':update_report()
