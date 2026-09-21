"""Describe frozen TM23 temperature regimes, separately from r2SCAN device data."""
import hashlib
import json
from pathlib import Path
import numpy as np
from semi_mlip.data import read_jsonl,write_json
from semi_mlip.visualize import plotting
from dataset_distributions import describe,ecdf


def main():
    frozen=Path('reports/focused/frozen.json')
    if not frozen.exists():raise ValueError('Freeze focused selection before describing protected test labels')
    splits={'train':'train','validation':'valid','cold test':'test_cold','warm test':'test_warm','molten test':'test_melt'}
    colors=['#38748c','#dd9b49','#805eac','#429577','#b65662']
    plt=plotting();fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained');report={}
    for row,metal in enumerate(('cu','ti')):
        folder=Path(f'data/focused/{metal}_cold_900')
        records={name:list(read_jsonl(folder/(file+'.jsonl'))) for name,file in splits.items()}
        origin=float(np.mean([r['energy']/len(r['z']) for r in records['train']]))
        report[metal]={'training_mean_energy_eV_atom':origin,'partitions':{},'file_sha256':{}}
        for (name,rows),color in zip(records.items(),colors):
            energies=np.array([r['energy']/len(r['z'])-origin for r in rows])
            forces=np.concatenate([np.linalg.norm(r['forces'],axis=1) for r in rows])
            ecdf(axes[row,0],energies,name.title(),color);ecdf(axes[row,1],forces,name.title(),color)
            report[metal]['partitions'][name]={'frames':len(rows),'relative_energy_eV_atom':describe(energies),'force_magnitude_eV_A':describe(forces)}
            report[metal]['file_sha256'][name]=hashlib.sha256((folder/(splits[name]+'.jsonl')).read_bytes()).hexdigest()
        axes[row,0].set(title=f'{metal.title()} | energy relative to cold training mean',xlabel='Relative energy (eV / atom)',ylabel='Cumulative fraction of frames')
        axes[row,1].set(title=f'{metal.title()} | force magnitudes',xlabel='Force magnitude (eV / A)',ylabel='Cumulative fraction of atoms');axes[row,1].set_xscale('symlog',linthresh=.01)
        axes[row,2].bar(range(5),[len(rows) for rows in records.values()],color=colors)
        axes[row,2].set_xticks(range(5),[s.title() for s in records],rotation=30,ha='right')
        axes[row,2].set(title=f'{metal.title()} | 900-frame study partitions',ylabel='Frames')
        axes[row,0].legend(frameon=False,fontsize=8)
        for ax in axes[row]:ax.grid(axis='y',alpha=.15)
    fig.suptitle('Focused temperature-transfer data | TM23 PBE labels | separate from the r2SCAN material benchmark',fontsize=15,weight='bold')
    out=Path('docs/assets')
    for ext in ('png','svg','pdf'):fig.savefig(out/f'focused_dataset_distributions.{ext}',dpi=170)
    plt.close(fig)
    write_json('reports/focused/dataset_distributions.json',{'materials':report,'frozen_selection_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),
        'note':'900 cold training frames; 300-frame study uses a nested subset. Warm validation and warm test share a trajectory; molten test is a held-out temperature. No independent trajectory replicate claim.'})


if __name__=='__main__':main()
