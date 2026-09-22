"""Independent, predeclared material studies after the frozen Cu/Ti release."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from semi_mlip.data import read_jsonl, write_json
from semi_mlip.train import TrainConfig, train, evaluate_checkpoint
from run_focused import model_config

ROOT = Path('reports/material_studies')
DATA = Path('data/material_studies')
METALS = ('W', 'Ta', 'Co', 'Ru', 'Hf', 'Zr')
CHEMISTRIES = ('Al', 'Si', 'Al-O', 'Co-O', 'Cu-O', 'Hf-O', 'O-Ru', 'O-Si', 'O-Ta', 'O-Ti', 'O-W', 'O-Zr')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def choose(rows, count):
    return sorted(rows, key=lambda r: hashlib.sha256(r['id'].encode()).digest())[:count]

def immutable(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8-sig')) != value:
            raise ValueError(f'Frozen protocol differs: {path}')
    else:
        write_json(path, value)

def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    studies = []
    for metal in METALS:
        source = Path('data/source_isolated/tm23')
        parts = {'train': choose(list(read_jsonl(source/f'{metal}_cold_train.jsonl')), 900),
                 'valid': choose(list(read_jsonl(source/f'{metal}_warm_train.jsonl')), 90)}
        for regime in ('cold', 'warm', 'melt'):
            parts['test_'+regime] = list(read_jsonl(source/f'{metal}_{regime}_test.jsonl'))
        studies.append((f'tm23_{metal.lower()}', metal, 'TM23 PBE', parts))
    expanded = {s: list(read_jsonl(f'data/device_expanded/{s}.jsonl'))
                for s in ('train', 'valid', 'aloe_valid', 'test', 'aloe_test')}
    for chemistry in CHEMISTRIES:
        parts = {s: [r for r in rows if r['chemsys']==chemistry] for s, rows in expanded.items()}
        # Same parent partitions; previously reserved ALOE development data only.
        parts['valid'] += parts.pop('aloe_valid')
        parts = {s: rows for s, rows in parts.items() if rows}
        studies.append((f'r2scan_{chemistry.lower().replace("-", "_")}', chemistry, 'MatPES / MP-ALOE r2SCAN', parts))
    manifest = {}
    for name, chemistry, source, parts in studies:
        if not parts.get('train') or not parts.get('valid'):
            raise ValueError(f'Missing development split: {name}')
        folder = DATA/name
        folder.mkdir(parents=True, exist_ok=True)
        # Audit IDs and exact same-basis geometry without selecting by labels.
        ids = {s: {r['id'] for r in rows} for s, rows in parts.items()}
        geometry = {s: {hashlib.sha256(json.dumps([r['z'], r['cell'], r['positions']]).encode()).hexdigest()
                        for r in rows} for s, rows in parts.items()}
        for a in parts:
            for b in parts:
                if a >= b: continue
                if ids[a] & ids[b] or geometry[a] & geometry[b]:
                    raise ValueError(f'Cross-split exact duplicate: {name} {a} {b}')
                if name.startswith('r2scan') and {r['group'] for r in parts[a]} & {r['group'] for r in parts[b]}:
                    raise ValueError(f'Cross-split parent: {name} {a} {b}')
        paths = {}
        for split, rows in parts.items():
            p = folder/f'{split}.jsonl'
            text = ''.join(json.dumps({**r, 'split':split})+'\n' for r in rows)
            if p.exists() and p.read_text() != text:
                raise ValueError(f'Existing dataset changed: {p}')
            if not p.exists(): p.write_text(text)
            paths[split] = {'path':p.as_posix(), 'sha256':digest(p), 'frames':len(rows),
                            'groups':len({r['group'] for r in rows})}
        manifest[name] = {'chemistry':chemistry, 'label_family':source, 'partitions':paths}
    cfg = TrainConfig(epochs=60, max_train=0, atom_budget=1024, edge_budget=64000,
                      accumulation=1, lr=.001, loss='pseudo_huber', stress_weight=0, monitor_train_every=10)
    protocol = {'studies':manifest, 'seeds':[42,43,44], 'train_config':asdict(cfg),
                'model_config':asdict(model_config('tensor')),
                'selection':'Fixed tensor architecture; per-seed checkpoint selected by validation force MAE using existing trainer.',
                'test_policy':'Train all specified runs and freeze checkpoint hashes before final test evaluation. No test-driven tuning.',
                'limitations':['TM23 cold/warm tests share source trajectories with development data; molten is temperature transfer.',
                    'r2SCAN test labels were previously reported for the broad baseline; these are fixed follow-up comparisons, not a fresh blind benchmark.',
                    'Al has only 11 training frames and one MatPES test frame; exploratory only.',
                    'Exact geometry and parent checks do not prove structural independence.',
                    'Three initialization seeds measure fit variability, not independent dataset replicates.',
                    'DFT families are trained separately; oxide chemical systems may include multiple stoichiometries.']}
    immutable(ROOT/'protocol.json', protocol)
    print(json.dumps({'studies':len(manifest), 'runs':len(manifest)*3, 'prepared':True}), flush=True)
    return protocol

def fit(protocol):
    if (ROOT/'frozen.json').exists(): return
    trials = []
    for name, study in protocol['studies'].items():
        for seed in protocol['seeds']:
            folder = Path('runs/material_studies')/f'{name}_s{seed}'
            cfg = TrainConfig(**{**protocol['train_config'], 'seed':seed})
            for partition in study['partitions'].values():
                if digest(partition['path']) != partition['sha256']: raise ValueError('Input hash changed')
            done = folder/'study_complete.json'
            if done.exists():
                trial = json.loads(done.read_text())
                if digest(trial['checkpoint']) != trial['sha256']: raise ValueError('Completed checkpoint changed')
            else:
                ckpt = train(DATA/name, folder, cfg, model_config('tensor'), folder/'last.pt' if (folder/'last.pt').exists() else None)
                excluded = json.loads((folder/'excluded.json').read_text())
                if any(excluded.values()): raise ValueError(f'Excluded development records: {name}')
                trial = {'study':name, 'seed':seed, 'checkpoint':str(ckpt), 'sha256':digest(ckpt)}
                write_json(done, trial)
            trials.append(trial)
            write_json(ROOT/'progress.json', {'complete':False,'completed':len(trials),'expected':54,'trials':trials})
    immutable(ROOT/'frozen.json', {'protocol_sha256':digest(ROOT/'protocol.json'), 'trials':trials})

def evaluate(protocol):
    frozen = json.loads((ROOT/'frozen.json').read_text())
    if frozen['protocol_sha256'] != digest(ROOT/'protocol.json'): raise ValueError('Protocol changed')
    results = {}
    for trial in frozen['trials']:
        if digest(trial['checkpoint']) != trial['sha256']: raise ValueError('Frozen checkpoint changed')
        name = trial['study']; key = f'{name}_s{trial["seed"]}'
        results[key] = {'study':name,'seed':trial['seed'],'splits':{}}
        for split, part in protocol['studies'][name]['partitions'].items():
            if digest(part['path']) != part['sha256']: raise ValueError('Dataset changed')
            output = ROOT/'evaluations'/f'{key}_{split}.json'
            metric = json.loads(output.read_text()) if output.exists() else evaluate_checkpoint(
                trial['checkpoint'], part['path'], output, atom_budget=1024, edge_budget=64000)
            if metric['excluded']: raise ValueError(f'Excluded evaluation records: {key} {split}')
            results[key]['splits'][split] = metric['overall']
    write_json(ROOT/'results.json', results)
    write_json(ROOT/'progress.json', {'complete':True,'completed':len(frozen['trials']),'expected':54,'trials':frozen['trials']})
    return results

def report(protocol, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assets = Path('docs/assets/material_studies'); assets.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, study in protocol['studies'].items():
        summary[name] = {}
        for split in study['partitions']:
            rows = [v['splits'][split] for v in results.values() if v['study']==name]
            summary[name][split] = {metric:{'mean':float(np.mean([r[metric] for r in rows])),
                'seed_std':float(np.std([r[metric] for r in rows],ddof=1))}
                for metric in ('energy_mae_eV_atom','force_mae_eV_A','force_rmse_eV_A')}
    write_json(ROOT/'summary.json', summary)
    lines = ['# Independent material studies','',
             'Fixed tensor models, 60 epochs, three initialization seeds per system. All values are mean +/- seed SD.', '',
             'These follow-up studies preserve source partitions and DFT settings. Al remains exploratory because its test has one frame.', '',
             '| Study | Split | Frames | Energy MAE (meV/atom) | Force MAE (eV/A) |',
             '| --- | --- | ---: | ---: | ---: |']
    for name, splits in summary.items():
        for split, metrics in splits.items():
            e,f=metrics['energy_mae_eV_atom'],metrics['force_mae_eV_A']
            lines.append(f'| {name} | {split} | {protocol["studies"][name]["partitions"][split]["frames"]} | {1000*e["mean"]:.2f} +/- {1000*e["seed_std"]:.2f} | {f["mean"]:.4f} +/- {f["seed_std"]:.4f} |')
    fig, ax = plt.subplots(figsize=(10,12),layout='constrained')
    labels=[]; means=[]; stds=[]; individual=[]
    for name, splits in summary.items():
        for split, metrics in splits.items():
            if split in ('train','valid'): continue
            labels.append(name.replace('r2scan_','').replace('tm23_','')+' / '+split)
            individual.append([v['splits'][split]['force_mae_eV_A'] for v in results.values() if v['study']==name])
            means.append(metrics['force_mae_eV_A']['mean']); stds.append(metrics['force_mae_eV_A']['seed_std'])
    y=np.arange(len(labels)); ax.scatter(means,y,color='#328c70',label='Three-seed mean')
    for i,scores in enumerate(individual):
        ax.scatter(scores,i+np.linspace(-.12,.12,len(scores)),marker='x',color='#3b6c96',s=22)
    ax.set_xscale('log'); ax.legend(frameon=False)
    ax.set(yticks=y,yticklabels=labels,xlabel='Force MAE (eV/A)',title='Final tests: means (dots), all seeds (crosses); log scale')
    ax.tick_params(axis='y',labelsize=7); ax.invert_yaxis(); ax.grid(axis='x',alpha=.2)
    for ext in ('png','svg','pdf'): fig.savefig(assets/f'test_forces.{ext}',dpi=180)
    plt.close(fig)
    for p in assets.glob('*.svg'):
        p.write_text('\n'.join(line.rstrip() for line in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    fig, axes = plt.subplots(6,3,figsize=(13,16),layout='constrained')
    distribution = {}
    for ax, (name, study) in zip(axes.flat, protocol['studies'].items()):
        distribution[name] = {}
        for split, part in study['partitions'].items():
            rows = list(read_jsonl(part['path']))
            forces = np.concatenate([np.linalg.norm(np.asarray(r['forces']),axis=1) for r in rows])
            energy = np.array([r['energy']/len(r['z']) for r in rows])
            ordered = np.sort(forces)
            ax.step(ordered,np.arange(1,len(ordered)+1)/len(ordered),where='post',label=split,linewidth=1)
            distribution[name][split] = {'force_magnitude_eV_A_quantiles':np.quantile(forces,[0,.5,.9,.99,1]).tolist(),
                'raw_energy_eV_atom_quantiles':np.quantile(energy,[0,.5,.9,.99,1]).tolist(),
                'frames':len(rows),'atoms':len(forces)}
        ax.set(title=name.replace('r2scan_','').replace('tm23_',''),xlabel='Force magnitude (eV/A)',ylabel='Atom-weighted ECDF')
        ax.set_xlim(left=0); ax.grid(alpha=.15); ax.legend(fontsize=6,frameon=False)
    fig.suptitle('Material study distributions: all partitions, no tail clipping')
    for ext in ('png','svg','pdf'): fig.savefig(assets/f'distributions.{ext}',dpi=160)
    plt.close(fig)
    write_json(ROOT/'distributions.json',distribution)
    fig, axes = plt.subplots(6,3,figsize=(13,15),layout='constrained')
    for ax, name in zip(axes.flat,protocol['studies']):
        for seed in protocol['seeds']:
            history = list(read_jsonl(Path('runs/material_studies')/f'{name}_s{seed}'/'history.jsonl'))
            ax.plot([h['epoch'] for h in history],[h['valid']['force_mae_eV_A'] for h in history],label=f'seed {seed}')
        ax.set(title=name.replace('r2scan_','').replace('tm23_',''),xlabel='Epoch',ylabel='Validation force MAE (eV/A)',yscale='log')
        ax.grid(alpha=.15);ax.legend(fontsize=6,frameon=False)
    for ext in ('png','svg','pdf'): fig.savefig(assets/f'learning_curves.{ext}',dpi=160)
    plt.close(fig)
    for p in assets.glob('*.svg'):
        p.write_text('\n'.join(line.rstrip() for line in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    lines += ['', '![Final test errors](assets/material_studies/test_forces.png)', '',
        '![Validation learning curves](assets/material_studies/learning_curves.png)', '',
        '![Force distributions](assets/material_studies/distributions.png)', '', '## Interpretation', '']
    lines += ['- '+v for v in protocol['limitations']]
    lines += ['', 'Protocol, hashes, full metrics, and seed summaries: [reports/material_studies](../reports/material_studies).']
    Path('docs/material_studies.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    from interpret_material_studies import update_report
    update_report()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare-only',action='store_true'); args=parser.parse_args()
    protocol=prepare()
    if args.prepare_only:return
    fit(protocol); results=evaluate(protocol); report(protocol,results)

if __name__=='__main__':main()
