"""Plot recorded model-only NVE energy traces; never labels them as AIMD."""
import json
from pathlib import Path
import numpy as np
from semi_mlip.data import read_jsonl
from semi_mlip.visualize import plotting


def curve(ax,path,label,color):
    rows=list(read_jsonl(path));energy=np.array([r['energy_eV']+r['kinetic_eV'] for r in rows])
    ax.plot([r['time_fs'] for r in rows],(energy-energy[0])/len(rows[0]['z'])*1000,label=label,color=color,lw=1.2)


def main():
    plt=plotting();plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
    checks=json.loads(Path('reports/simulation_checks.json').read_text())
    names=[n for n,c in checks.items() if 'failed' not in c]
    fig,axes=plt.subplots(2,3,figsize=(14,7.5),layout='constrained')
    for ax,name in zip(axes.ravel(),names):
        for suffix,label,color in [('05','0.5 fs','#d78a36'),('025','0.25 fs','#38748c')]:
            curve(ax,Path('runs/simulations')/name/f'nve_{suffix}.jsonl',label,color)
        ax.set(title=name.title(),xlabel='Time (fs)',ylabel='Total energy change (meV / atom)');ax.legend(frameon=False);ax.grid(alpha=.15)
    for ax in axes.ravel()[len(names):]:ax.set_visible(False)
    fig.suptitle('Model-only numerical stability | 100 fs NVE | identical initial states at two timesteps',fontsize=15,weight='bold')
    for ext in ('png','svg','pdf'):fig.savefig(Path('docs/assets')/f'material_nve.{ext}',dpi=170)
    plt.close(fig)
    if Path('reports/focused/md.json').exists():
        cases=json.loads(Path('reports/focused/md.json').read_text())['cases']
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for ax,metal in zip(axes,('cu','ti')):
            for dt,color in [(.5,'#d78a36'),(.25,'#38748c')]:
                name=f'{metal}_dt{dt}'
                if name in cases and not cases[name].get('failed',False):
                    curve(ax,Path('reports/focused/md')/f'{name}.jsonl',f'{dt} fs',color)
            ax.set(title=metal.title()+' | 900 frames, seed 43',xlabel='Time (fs)',ylabel='Total energy change (meV / atom)');ax.legend(frameon=False);ax.grid(alpha=.15)
        fig.suptitle('Focused models | 100 fs NVE numerical check | not a DFT trajectory comparison',fontsize=13)
        for ext in ('png','svg','pdf'):fig.savefig(Path('docs/assets')/f'focused_nve.{ext}',dpi=170)
        plt.close(fig)


if __name__=='__main__':main()
