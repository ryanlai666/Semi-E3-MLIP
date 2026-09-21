"""Freeze completed focused trials, evaluate protected tests, and plot results."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from semi_mlip.data import read_jsonl, write_json
from semi_mlip.train import evaluate_checkpoint
from semi_mlip.simulate import Calculator, md


def freeze(trials, output):
    records = []
    for trial in trials:
        p = Path(trial['checkpoint'])
        records.append({'name':trial['name'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                        'checkpoint':str(p),'metal':trial['metal'],'seed':trial['seed'],
                        'train_frames':trial['train_frames'],'architecture':trial['architecture']})
    frozen = {'checkpoints':records,'selection_source':'warm validation only',
              'test_regimes':['cold','warm','melt'],'nve_seed':43,
              'note':'All configurations frozen before any final test evaluation.'}
    target = output/'frozen.json'
    if target.exists():
        if json.loads(target.read_text()) != frozen:
            raise ValueError('Frozen experiment changed; do not silently retune against final tests')
    else:
        write_json(target,frozen)


def figures(screen, trials, results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'gated':'#3b6c96','attention':'#d98a2b','tensor':'#328c70'}
    fig, axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,metal in zip(axes,('cu','ti')):
        for trial in screen['trials']:
            if trial['metal']!=metal:continue
            h=list(read_jsonl(Path(trial['checkpoint']).parent/'history.jsonl'))
            ax.plot([r['epoch'] for r in h],[r['valid']['force_mae_eV_A'] for r in h],
                    label=trial['architecture'],color=colors[trial['architecture']])
        ax.set(title=f'{metal.title()}: cold training → warm validation',xlabel='Epoch',ylabel='Force MAE (eV/Å)',yscale='log')
        ax.grid(alpha=.15);ax.legend(frameon=False)
    fig.suptitle('Architecture comparison • 300 cold configurations • seed 42')
    for ext in ('png','svg','pdf'):fig.savefig(output/f'architecture_comparison.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    summaries={}
    for ax,metal in zip(axes,('cu','ti')):
        summaries[metal]={}
        for count,shift,color in [(300,-.15,'#3b6c96'),(900,.15,'#d98a2b')]:
            means=[];stds=[]
            for regime in ('cold','warm','melt'):
                scores=[results[t['name']][regime]['overall']['force_mae_eV_A'] for t in trials
                        if t['metal']==metal and t['train_frames']==count]
                energies=[results[t['name']][regime]['overall']['energy_mae_eV_atom'] for t in trials
                        if t['metal']==metal and t['train_frames']==count]
                means.append(np.mean(scores));stds.append(np.std(scores,ddof=1))
                summaries[metal][f'n{count}_{regime}']={'force_mae_mean':float(np.mean(scores)),
                    'force_mae_seed_std':float(np.std(scores,ddof=1)), 'energy_mae_mean':float(np.mean(energies)),
                    'energy_mae_seed_std':float(np.std(energies,ddof=1)), 'seeds':len(scores)}
            ax.errorbar(np.arange(3)+shift,means,yerr=stds,fmt='o-',color=color,capsize=4,label=f'{count} train frames')
        ax.set(xticks=np.arange(3),xticklabels=['Cold','Warm','Molten'],ylabel='Force MAE (eV/Å)',title=metal.title())
        ax.grid(axis='y',alpha=.15);ax.legend(frameon=False)
    fig.suptitle('Protected test regimes • mean ± SD across 3 initialization seeds')
    for ext in ('png','svg','pdf'):fig.savefig(output/f'temperature_transfer.{ext}',dpi=180)
    plt.close(fig)
    write_json(output/'summary.json',summaries)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--md',action='store_true');args=parser.parse_args()
    output=Path('reports/focused')
    screen=json.loads((output/'screen.json').read_text());study=json.loads((output/'confirm.json').read_text())
    assert screen['complete'] and study['complete']
    trials=study['trials'];assert len(trials)==12
    freeze(trials,output)
    results={}
    for trial in trials:
        results[trial['name']]={}
        for regime in ('cold','warm','melt'):
            path=output/'tests'/f"{trial['name']}_{regime}.json"
            if path.exists(): metric=json.loads(path.read_text())
            else: metric=evaluate_checkpoint(trial['checkpoint'],
                f"data/focused/{trial['metal']}_cold_{trial['train_frames']}/test_{regime}.jsonl",
                path,atom_budget=1024,edge_budget=64000)
            assert not metric['excluded']
            results[trial['name']][regime]=metric
    write_json(output/'tests.json',results);figures(screen,trials,results,output)
    if args.md:
        diagnostics={}
        # Fixed seed 43, not the seed with the most favorable test metric.
        for trial in trials:
            if trial['seed']!=43 or trial['train_frames']!=900:continue
            row=next(read_jsonl(f"data/focused/{trial['metal']}_cold_900/valid.jsonl"))
            calc=Calculator(trial['checkpoint'])
            for dt in (.5,.25):
                name=f"{trial['metal']}_dt{dt}"
                try:
                    diagnostics[name]=md(row,calc,output/'md'/f'{name}.jsonl',steps=round(100/dt),
                        dt=dt,temperature=300,ensemble='nve',seed=43)
                except (ValueError,FloatingPointError,RuntimeError) as exc:
                    diagnostics[name]={'failed':True,'reason':str(exc)}
                write_json(output/'md.json',{'cases':diagnostics,
                    'interpretation':'100 fs model-only NVE check at 300 K initial velocities; not an AIMD trajectory comparison'})


if __name__=='__main__':main()
