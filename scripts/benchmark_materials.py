"""Frozen multi-material benchmark; exports metrics and publication-quality figures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from semi_mlip.data import read_jsonl, write_json
from semi_mlip.train import evaluate_checkpoint
from semi_mlip.simulate import Calculator
from semi_mlip.visualize import plotting, draw_structure

LABELS = {'Al-O':'Al oxides','Co-O':'Co oxides','Cu-O':'Cu oxides','Hf-O':'Hf oxides',
          'O-Ru':'Ru oxides','O-Si':'Si oxides','O-Ta':'Ta oxides','O-Ti':'Ti oxides',
          'O-W':'W oxides','O-Zr':'Zr oxides'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');args=parser.parse_args()
    selection=json.loads(Path('reports/aimd_comparison/selection.json').read_text())
    checkpoint=Path(selection['checkpoint'].replace('\\','/'));out=Path('reports/benchmark');out.mkdir(exist_ok=True)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=selection['checkpoint_sha256']:
        raise ValueError('Frozen benchmark checkpoint changed')
    files={'train':Path('data/device_expanded/train.jsonl'),
           'validation':Path('data/device_expanded/valid.jsonl'),
           'test':Path('data/device/test.jsonl'), 'oxide_test':Path('data/device_expanded/aloe_test.jsonl')}
    protocol={'checkpoint':str(checkpoint),'checkpoint_sha256':selection['checkpoint_sha256'],
              'datasets':{s:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for s,p in files.items()},
              'atom_budget':1024,'edge_budget':64000,'device':args.device,'used_for_selection':False}
    lock=out/'protocol.json'
    if lock.exists() and json.loads(lock.read_text())!=protocol:raise ValueError('Benchmark protocol changed')
    write_json(lock,protocol)
    metrics={}
    for split,path in files.items():
        target=out/f'{split}.json'
        if target.exists():metrics[split]=json.loads(target.read_text())
        else:metrics[split]=evaluate_checkpoint(checkpoint,path,target,device=args.device,atom_budget=1024,edge_budget=64000)
        print(split,json.dumps(metrics[split]['overall']),flush=True)
    systems=sorted(k for k in metrics['train'] if k not in ('overall','excluded') and not k.startswith('formula:'))
    table=[]
    for system in systems:
        table.append({'system':system,'label':LABELS.get(system,system),
                      **{split:metrics[split].get(system) for split in files}})
    write_json(out/'materials.json',{'systems':table,'checkpoint':str(checkpoint),
        'note':'Static DFT-labelled frames. MatPES and MP-ALOE held-outs are reported separately; small per-system samples limit conclusions.'})
    assets=Path('docs/assets');assets.mkdir(parents=True,exist_ok=True)
    plt=plotting();plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(14,8),layout='constrained')
    x=np.arange(len(systems));colors={'train':'#38748c','validation':'#dd9b49','test':'#805eac','oxide_test':'#429577'}
    for split,shift in [('train',-.27),('validation',-.09),('test',.09),('oxide_test',.27)]:
        values=[metrics[split].get(s,{}).get('force_mae_eV_A',np.nan) for s in systems]
        counts=[metrics[split].get(s,{}).get('frames',0) for s in systems]
        axes[0].bar(x+shift,values,.18,label=split.replace('_',' ').title(),color=colors[split])
        axes[1].bar(x+shift,counts,.18,color=colors[split])
    axes[0].set(ylabel='Force MAE (eV / A)',title='Frozen multi-material model | force error by chemical system')
    axes[0].legend(ncol=4,frameon=False);axes[0].grid(axis='y',alpha=.15)
    axes[1].set(ylabel='Frames',title='Coverage: missing bars mean no examples, not zero error',yscale='symlog')
    for ax in axes:ax.set_xticks(x,[LABELS.get(s,s) for s in systems],rotation=35,ha='right')
    for ext in ('png','svg','pdf'):fig.savefig(assets/f'material_benchmark.{ext}',dpi=170)
    plt.close(fig)
    calc=Calculator(checkpoint,args.device)
    rows=list(read_jsonl(files['test']))
    examples=[]
    # One deterministic frame per chemical system; do not select visually favorable cases.
    for system in systems:
        candidates=sorted([r for r in rows if r['chemsys']==system],key=lambda r:r['id'])
        if candidates:examples.append(candidates[0])
    pairs=[];all_ref=[];all_pred=[]
    for row in examples:
        prediction=calc(row);ref=np.asarray(row['forces']);pred=prediction['forces']
        pairs.append((row,ref,pred));all_ref.extend(ref.ravel());all_pred.extend(pred.ravel())
    write_json(out/'visual_examples.json',{'selection':'Lexicographically first test ID per chemical system',
        'examples':[{'id':r['id'],'system':r['chemsys'],'force_mae_eV_A':float(np.abs(a-b).mean())} for r,a,b in pairs]})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for row,ref,pred in pairs:
        axes[0].scatter(ref.ravel(),pred.ravel(),s=9,alpha=.55,label=row['chemsys'])
    bound=max(np.max(np.abs(all_ref)),np.max(np.abs(all_pred)))
    axes[0].plot([-bound,bound],[-bound,bound],color='#34445a',ls='--',lw=1)
    axes[0].set(xlabel='DFT force component (eV / A)',ylabel='Model force component (eV / A)',title='One fixed example per test chemical system')
    axes[1].hist(np.asarray(all_pred)-np.asarray(all_ref),bins=60,color='#38748c')
    axes[1].set(xlabel='Model - DFT force component (eV / A)',ylabel='Components',title='Same predeclared examples')
    for ext in ('png','svg','pdf'):fig.savefig(assets/f'force_parity.{ext}',dpi=170)
    plt.close(fig)
    chosen=['Cu','Ti','W','O-Si','Al-O','Hf-O','O-Ti','Cu-O']
    selected=[p for system in chosen for p in pairs if p[0]['chemsys']==system]
    fig=plt.figure(figsize=(16,3.8*((len(selected)+1)//2)),layout='constrained')
    for i,(row,ref,pred) in enumerate(selected):
        scale=1.5/max(float(np.linalg.norm(ref,axis=1).max()),float(np.linalg.norm(pred,axis=1).max()),.1)
        for j,(force,label) in enumerate([(ref,'DFT reference'),(pred,'Semi-E3-MLIP')]):
            ax=fig.add_subplot((len(selected)+1)//2,4,(i//2)*4+(i%2)*2+j+1,projection='3d')
            draw_structure(ax,row,force,scale,title=f"{row.get('formula',row['chemsys'])} | {label}")
        fig.axes[-1].text2D(.02,.02,f"Force MAE {np.abs(ref-pred).mean():.3f} eV/A",transform=fig.axes[-1].transAxes,fontsize=9)
    fig.suptitle('Held-out metal and oxide structures | same geometry and force scale per pair',fontsize=15)
    fig.savefig(assets/'metal_oxide_forces.png',dpi=120);plt.close(fig)
    write_json(out/'complete.json',{'complete':True,'checkpoint_sha256':selection['checkpoint_sha256'],'systems':len(systems)})


if __name__=='__main__':main()
