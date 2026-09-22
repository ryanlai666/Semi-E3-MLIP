"""Full-frame train/test parity galleries for frozen shared and specialist models."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from semi_mlip.train import load_potential, load_graphs, make_batches
from semi_mlip.graph import collate

OUT=Path('docs/assets/parity')
CACHE=Path('runs/parity_cache')
REPORT=Path('reports/parity')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def predict(checkpoint, expected_hash, file):
    checkpoint=Path(str(checkpoint).replace('\\','/'));file=Path(file)
    assert sha(checkpoint)==expected_hash,'Frozen checkpoint changed'
    key=hashlib.sha256((expected_hash+sha(file)+'parity-v1-cuda-stress').encode()).hexdigest()
    target=CACHE/(key+'.npz')
    if target.exists():
        with np.load(target,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    model,_=load_potential(checkpoint,'cuda' if torch.cuda.is_available() else 'cpu');model.eval()
    device=next(model.parameters()).device
    rows,graphs,excluded=load_graphs(file,model.config.cutoff,1024,64000)
    assert not excluded
    values={k:[] for k in ('er','ep','fr','fp','ec','fc','id')}
    for indices in make_batches(rows,graphs,1024,64000):
        batch_rows=[rows[i] for i in indices]
        prediction=model(collate(batch_rows,[graphs[i] for i in indices],device))
        energies=prediction['energy'].detach().cpu().numpy().astype(float)
        forces=prediction['forces'].detach().cpu().numpy().astype(float)
        offset=0
        for row,energy in zip(batch_rows,energies):
            n=len(row['z']);baseline=sum(float(model.offsets[z]) for z in row['z'])
            values['er'].append((row['energy']-baseline)/n);values['ep'].append((energy-baseline)/n)
            values['fr'].extend(np.asarray(row['forces']).ravel());values['fp'].extend(forces[offset:offset+n].ravel())
            values['ec'].append(row['chemsys']);values['fc'].extend([row['chemsys']]*(3*n));values['id'].append(row['id']);offset+=n
    values={k:np.asarray(v) for k,v in values.items()}
    np.savez_compressed(target,**values)
    return values

def joined(items):return {k:np.concatenate([d[k] for d in items]) for k in items[0]}

def render(name,title,partitions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(partitions),2,figsize=(10,3.8*len(partitions)),squeeze=False,layout='constrained')
    metrics={}
    for i,(split,d) in enumerate(partitions.items()):
        metrics[split]={}
        for j,(ref,pred,label) in enumerate([('er','ep','Offset-adjusted energy (eV/atom)'),('fr','fp','Force component (eV/A)')]):
            ax=axes[i,j];x=d[ref];y=d[pred]
            if not len(x):ax.text(.5,.5,'No held-out records\nNot a zero-error result',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off();continue
            assert np.isfinite(x).all() and np.isfinite(y).all(),f'Nonfinite prediction: {name} {split}'
            mae=float(np.abs(x-y).mean());rmse=float(np.sqrt(np.mean((x-y)**2)))
            metrics[split][ref]={'points':len(x),'mae':mae,'rmse':rmse}
            ax.scatter(x,y,s=1.7 if len(x)<5000 else 1.3,alpha=.55 if len(x)<5000 else .25 if len(x)<50000 else .16,color='#237a8b' if i==0 else '#bb6945',rasterized=True)
            lo=float(min(x.min(),y.min()));hi=float(max(x.max(),y.max()));span=max(hi-lo,1e-5);lo-=.04*span;hi+=.04*span
            transformed=max(abs(lo),abs(hi))>max(100.,100*np.quantile(np.abs(x),.99))
            if transformed:
                threshold=max(float(np.quantile(np.abs(x),.9)),.01)
                ax.set_xscale('symlog',linthresh=threshold);ax.set_yscale('symlog',linthresh=threshold)
                # Multiplicative padding avoids huge artificial negative range for one-sided failures.
                lo=min(float(x.min()),float(y.min()),0)*1.2;hi=max(float(x.max()),float(y.max()),0)*1.2
                from matplotlib.ticker import SymmetricalLogLocator, FixedLocator
                for axis in (ax.xaxis,ax.yaxis):
                    locator=SymmetricalLogLocator(linthresh=threshold,base=10)
                    locator.set_params(numticks=7)
                    ticks=[v for v in locator.tick_values(lo,hi) if lo<=v<=hi and (v==0 or abs(v)>=10*threshold)]
                    axis.set_major_locator(FixedLocator(ticks))
                    axis.set_minor_locator(plt.NullLocator())
            ax.plot([lo,hi],[lo,hi],'--',color='#444444',lw=1);ax.set_xlim(lo,hi);ax.set_ylim(lo,hi)
            ax.set_aspect('equal',adjustable='box');ax.set_xlabel('DFT '+label);ax.set_ylabel('Predicted '+label)
            display={'train':'Train','test':'MatPES test','aloe_test':'MP-ALOE test','test_cold':'Cold test','test_warm':'Warm test','test_melt':'Molten test'}.get(split,split)
            ax.set_title(f'{display} | {len(x):,} points'+(' | symmetric-log axes' if transformed else ''),fontsize=10)
            ax.text(.03,.97,f'MAE {mae:.3g}\nRMSE {rmse:.3g}',transform=ax.transAxes,va='top',fontsize=9,
                    bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'});ax.grid(alpha=.12)
    fig.suptitle(title,fontsize=12)
    for ext in ('png','pdf'):fig.savefig(OUT/(name+'.'+ext),dpi=135)
    plt.close(fig)
    return metrics

def compare_metrics(d,expected):
    for x,y,key in [('er','ep','energy_mae_eV_atom'),('fr','fp','force_mae_eV_A')]:
        actual=float(np.mean(np.abs(d[x]-d[y])))
        assert np.isclose(actual,expected[key],rtol=.02,atol=2e-5),(key,actual,expected[key])

def main():
    for p in (OUT,CACHE,REPORT):p.mkdir(parents=True,exist_ok=True)
    manifest={'selection':'All frames and all force components; no favorable-example selection or tail clipping.',
        'energy':'Subtract the checkpoint training-fitted elemental offset from both reference and prediction; parity errors are unchanged.',
        'specialists':'All three initialization seeds pooled for visualization, not prediction averaging or a deployable ensemble.',
        'metrics':'Energy is frame-weighted; force is component-weighted. Specialist points repeat each reference once per seed.',
        'shared':{},'specialists':{}}
    selection=read('reports/aimd_comparison/selection.json');files={'Train':'data/device_expanded/train.jsonl','MatPES test':'data/device/test.jsonl','MP-ALOE test':'data/device_expanded/aloe_test.jsonl'}
    shared={s:predict(selection['checkpoint'],selection['checkpoint_sha256'],p) for s,p in files.items()}
    for s,old in [('Train','train'),('MatPES test','test'),('MP-ALOE test','oxide_test')]:compare_metrics(shared[s],read(f'reports/benchmark/{old}.json')['overall'])
    manifest['shared']['overall']=render('shared_overall','Shared model | all 20 systems | one frozen checkpoint',shared)
    for chemistry in sorted(set(shared['Train']['ec'])):
        parts={s:{k:v[np.isin(d['fc'] if k in ('fr','fp','fc') else d['ec'],[chemistry])] for k,v in d.items()} for s,d in shared.items()}
        name='shared_'+chemistry.replace('-','_').lower();manifest['shared'][chemistry]=render(name,f'Shared model | {chemistry} | one frozen checkpoint',parts)
        print('Rendered',name,flush=True)
    frozen=read('reports/material_studies/frozen.json');protocol=read('reports/material_studies/protocol.json');results=read('reports/material_studies/results.json')
    studies={name:{'trials':[t for t in frozen['trials'] if t['study']==name],
        'files':{s:p['path'] for s,p in info['partitions'].items() if s!='valid'}} for name,info in protocol['studies'].items()}
    focused=read('reports/focused/frozen.json');focused_results=read('reports/focused/tests.json')
    for metal in ('cu','ti'):
        for count in (300,900):
            studies[f'focused_{metal}_n{count}']={'trials':[t for t in focused['checkpoints'] if t['metal']==metal and t['train_frames']==count],
                'files':{s:f'data/focused/{metal}_cold_{count}/{s}.jsonl' for s in ('train','test_cold','test_warm','test_melt')}}
    for name,study in studies.items():
        parts={}
        for split,file in study['files'].items():
            arrays=[]
            for t in study['trials']:
                d=predict(t['checkpoint'],t['sha256'],file)
                expected=(focused_results[t['name']][split.replace('test_','')]['overall'] if name.startswith('focused_')
                          else results[f'{name}_s{t["seed"]}']['splits'][split])
                compare_metrics(d,expected);arrays.append(d)
            parts[split]=joined(arrays)
        manifest['specialists'][name]=render('specialist_'+name,f'{name} | three independent fits pooled, not an ensemble',parts)
        print('Rendered specialist',name,flush=True)
    manifest['checkpoint_hashes']={'shared':selection['checkpoint_sha256'],'specialists':[t['sha256'] for t in frozen['trials']],
        'focused':[t['sha256'] for t in focused['checkpoints']]}
    manifest['source_manifest_sha256']={str(p):sha(p) for p in [Path('reports/aimd_comparison/selection.json'),Path('reports/material_studies/frozen.json'),Path('reports/material_studies/protocol.json'),Path('reports/focused/frozen.json')]}
    manifest['metric_verification']={'all_passed':True,'relative_tolerance':.02,'absolute_tolerance':2e-5,'quantities':['energy_mae_eV_atom','force_mae_eV_A']}
    manifest['torch_version']=torch.__version__
    manifest['complete']=True
    (REPORT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Train and test parity plots','','[Back to suitability and coverage](material_studies.md) | [Shared-model results](results.md)','','## How to read these plots','',
        'Each figure separates training from held-out partitions and plots predicted against DFT energy and force. The dashed line is exact agreement. Every frame and force component is included; extreme predictions remain visible. Symmetric-log axes are explicitly labeled when needed to show severe failures. PNG previews and PDF downloads contain the same data.', '',
        'Energies subtract training-fitted elemental offsets from both axes; this makes different compositions comparable without changing errors. Force panels use signed Cartesian components, not magnitudes. MAE and RMSE use all plotted points. Narrow or nearly zero-force test sets can look deceptively good; consult counts and the suitability assessment.', '',
        '**Shared model:** one set of learned weights. **Specialists:** all three separate seeds are pooled in each panel, so reference points repeat three times. Predictions are not averaged; this is not one combined potential. PBE TM23 and r2SCAN specialist results are never pooled into a single overall score.', '',
        '## Shared model: overall','','![Shared model overall parity](assets/parity/shared_overall.png)','','[Download overall PDF](assets/parity/shared_overall.pdf)','','## Shared model: each system','','| Element | Elemental parity | Oxide parity |','| --- | --- | --- |']
    for element,oxide in [('Al','Al-O'),('Si','O-Si'),('Cu','Cu-O'),('Ti','O-Ti'),('W','O-W'),('Ta','O-Ta'),('Co','Co-O'),('Ru','O-Ru'),('Hf','Hf-O'),('Zr','O-Zr')]:
        a='shared_'+element.lower();b='shared_'+oxide.replace('-','_').lower()
        lines.append(f'| {element} | [PNG](assets/parity/{a}.png) / [PDF](assets/parity/{a}.pdf) | [PNG](assets/parity/{b}.png) / [PDF](assets/parity/{b}.pdf) |')
    lines+=['','## Independent specialists','','The paired layout below is a navigation aid, not a claim that elemental and oxide checkpoints form one compatible potential. TM23/Cu-Ti specialists use PBE; these oxide and Al/Si specialists use r2SCAN.','','| Element | Elemental specialist parity | Oxide specialist parity |','| --- | --- | --- |']
    def link(name,label=''):
        return f'{label} [PNG](assets/parity/specialist_{name}.png) / [PDF](assets/parity/specialist_{name}.pdf)'
    for element,metal,oxide in [('Al','r2scan_al','r2scan_al_o'),('Si','r2scan_si','r2scan_o_si'),
        ('Cu',None,'r2scan_cu_o'),('Ti',None,'r2scan_o_ti'),('W','tm23_w','r2scan_o_w'),('Ta','tm23_ta','r2scan_o_ta'),
        ('Co','tm23_co','r2scan_co_o'),('Ru','tm23_ru','r2scan_o_ru'),('Hf','tm23_hf','r2scan_hf_o'),('Zr','tm23_zr','r2scan_o_zr')]:
        elemental=link(metal) if metal else link(f'focused_{element.lower()}_n300','300 frames:')+'; '+link(f'focused_{element.lower()}_n900','900 frames:')
        lines.append(f'| {element} | {elemental} | {link(oxide)} |')
    lines+=['','No elemental Ru shared-model test exists; its test panels explicitly show no records. Missing source partitions are not zero error.', '',
        '[Numerical parity metrics, counts, checkpoint hashes and plotting conventions](../reports/parity/manifest.json).']
    Path('docs/parity.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
