"""Describe label distributions; never changes splits, fitting, or model selection."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from semi_mlip.data import read_jsonl, write_json
from semi_mlip.visualize import plotting

SPLITS={'train':'data/device_expanded/train.jsonl','validation':'data/device_expanded/valid.jsonl',
        'test':'data/device/test.jsonl','oxide_test':'data/device_expanded/aloe_test.jsonl'}
COLORS={'train':'#38748c','validation':'#dd9b49','test':'#805eac','oxide_test':'#429577'}


def describe(values):
    a=np.asarray(values,float)
    if not len(a):return {'count':0}
    if not np.isfinite(a).all():raise ValueError('Nonfinite dataset labels')
    return {'count':int(a.size),'min':float(a.min()),'p01':float(np.quantile(a,.01)),
            'p25':float(np.quantile(a,.25)),'median':float(np.median(a)),
            'p75':float(np.quantile(a,.75)),'p99':float(np.quantile(a,.99)),
            'max':float(a.max()),'mean':float(a.mean()),'std':float(a.std())}


def ecdf(ax,values,label,color):
    x=np.sort(values)
    if len(x):ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=label,color=color,lw=1.7)


def main():
    config=json.loads(Path('runs/device/expanded_tensor/config.json').read_text())
    offsets={int(k):v for k,v in config['statistics']['offsets'].items()}
    records={s:list(read_jsonl(p)) for s,p in SPLITS.items()}
    arrays={};summary={}
    for split,rows in records.items():
        energy=np.array([r['energy']/len(r['z']) for r in rows])
        residual=np.array([(r['energy']-sum(offsets[z] for z in r['z']))/len(r['z']) for r in rows])
        forces=np.concatenate([np.asarray(r['forces']) for r in rows])
        stress=np.array([np.linalg.norm(np.asarray(r['stress'])) for r in rows if r.get('stress') is not None])
        arrays[split]={'energy_eV_atom':energy,'offset_energy_eV_atom':residual,
            'force_component_eV_A':forces.ravel(),'force_magnitude_eV_A':np.linalg.norm(forces,axis=1),
            'stress_frobenius_eV_A3':stress,'atoms_per_frame':np.array([len(r['z']) for r in rows]),
            'volume_A3_atom':np.array([abs(np.linalg.det(r['cell']))/len(r['z']) for r in rows])}
        summary[split]={'frames':len(rows),'systems':dict(Counter(r['chemsys'] for r in rows)),
            'sources':dict(Counter(r['source'] for r in rows)),
            'stress_missing_frames':sum(r.get('stress') is None for r in rows),
            'distributions':{k:describe(v) for k,v in arrays[split].items()}}
    out=Path('docs/assets');out.mkdir(parents=True,exist_ok=True)
    plt=plotting();plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
    bins=np.linspace(min(a['energy_eV_atom'].min() for a in arrays.values()),max(a['energy_eV_atom'].max() for a in arrays.values()),70)
    for split,a in arrays.items():
        label=split.replace('_',' ').title();color=COLORS[split]
        axes[0,0].hist(a['energy_eV_atom'],bins=bins,density=True,histtype='step',lw=1.8,label=label,color=color)
        ecdf(axes[0,1],a['offset_energy_eV_atom'],label,color)
        ecdf(axes[0,2],a['force_magnitude_eV_A'],label,color)
        ecdf(axes[1,0],a['force_component_eV_A'],label,color)
        ecdf(axes[1,1],a['stress_frobenius_eV_A3'],label,color)
        ecdf(axes[1,2],a['atoms_per_frame'],label,color)
    axes[0,0].set(title='Energy per atom | composition-dependent',xlabel='DFT energy (eV / atom)',ylabel='Probability density')
    axes[0,1].set(title='Energy after training-only elemental offsets',xlabel='Residual energy (eV / atom)',ylabel='Cumulative fraction of frames',xscale='symlog');axes[0,1].set_xscale('symlog',linthresh=.1)
    axes[0,2].set(title='Force magnitude | atom-weighted',xlabel='Force magnitude (eV / A)',ylabel='Cumulative fraction of atoms');axes[0,2].set_xscale('symlog',linthresh=.01)
    axes[1,0].set(title='Signed force components',xlabel='Force component (eV / A)',ylabel='Cumulative fraction of components');axes[1,0].set_xscale('symlog',linthresh=.01)
    axes[1,1].set(title='Stress Frobenius norm | labelled frames',xlabel='Stress norm (eV / A^3)',ylabel='Cumulative fraction of labelled frames');axes[1,1].set_xscale('symlog',linthresh=.001)
    axes[1,2].set(title='Structure sizes | frame-weighted',xlabel='Atoms per frame',ylabel='Cumulative fraction of frames',xscale='log')
    for ax in axes.ravel():ax.grid(alpha=.15)
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('Dataset distributions | fixed metal / oxide partitions',fontsize=17,weight='bold')
    for ext in ('png','svg','pdf'):fig.savefig(out/f'dataset_distributions.{ext}',dpi=170)
    plt.close(fig)
    systems=sorted(summary['train']['systems']);names=list(SPLITS)
    counts=np.array([[summary[s]['systems'].get(k,0) for s in names] for k in systems])
    fig,axes=plt.subplots(1,3,figsize=(16,10),layout='constrained',gridspec_kw={'width_ratios':[1,1.4,1.4]})
    axes[0].imshow(np.log1p(counts),aspect='auto',cmap='Blues')
    axes[0].set_xticks(range(4),['Train','Valid','Test','Oxide test'],rotation=30,ha='right')
    axes[0].set_yticks(range(len(systems)),systems);axes[0].set_title('Frame counts | log color scale')
    for i in range(len(systems)):
        for j in range(4):axes[0].text(j,i,str(counts[i,j]),ha='center',va='center',fontsize=9,color='white' if counts[i,j]>50 else '#172c44')
    for ax,key,title in [(axes[1],'energy_eV_atom','Training energy per atom (eV)'),(axes[2],'force_magnitude_eV_A','Training force magnitude (eV / A)')]:
        values=[]
        for system in systems:
            rows=[r for r in records['train'] if r['chemsys']==system]
            values.append(np.array([r['energy']/len(r['z']) for r in rows]) if key=='energy_eV_atom' else np.concatenate([np.linalg.norm(r['forces'],axis=1) for r in rows]))
        box=ax.boxplot(values,orientation='horizontal',tick_labels=systems,patch_artist=True,
                       flierprops={'markersize':1.5,'alpha':.3},widths=.65)
        for patch in box['boxes']:patch.set(facecolor='#b8d5df',edgecolor='#38748c')
        ax.invert_yaxis();ax.set_title(title);ax.grid(axis='x',alpha=.15)
        if key=='force_magnitude_eV_A':
            ax.set_xscale('symlog',linthresh=.01);ax.set_xlim(left=0)
    fig.suptitle('Chemical coverage and within-system spread | boxes: median and IQR; whiskers: 1.5 IQR',fontsize=15,weight='bold')
    for ext in ('png','svg','pdf'):fig.savefig(out/f'dataset_chemistry.{ext}',dpi=170)
    plt.close(fig)
    write_json('reports/benchmark/dataset_distributions.json',{'partitions':summary,
        'dataset_sha256':{s:hashlib.sha256(Path(p).read_bytes()).hexdigest() for s,p in SPLITS.items()},
        'offsets_source':'runs/device/expanded_tensor/config.json; fitted on training only',
        'element_offsets_eV':offsets,'partitions_modified':False,
        'interpretation':'Energy/size/stress distributions weight frames equally; force magnitudes weight atoms and components weight Cartesian components. Full ECDF ranges retain tails. Energy offsets fitted to training only; no test normalization fit.'})
    print('Dataset distribution figures and numerical quantiles written.',flush=True)


if __name__=='__main__':main()
