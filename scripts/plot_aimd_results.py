"""Render cached AIMD statistics without rerunning dynamics or changing metrics."""
import json
from pathlib import Path
import numpy as np
from semi_mlip.data import read_jsonl
from semi_mlip.visualize import plotting
from compare_aimd import displacement,pair_histogram


def main():
    out=Path('reports/aimd_comparison');summary=json.loads((out/'metrics.json').read_text())
    plt=plotting();plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
    for name,case in summary['cases'].items():
        if not case.get('model_simulations'):continue
        ref=list(read_jsonl(case['reference']['path']))
        traces=[]
        for report in case['model_simulations']:
            p=Path('runs/aimd_comparison')/name/f"seed{report['seed']}.jsonl"
            traces.append([row for i,row in enumerate(read_jsonl(p)) if i%10==0])
        bins=np.linspace(0,5,101);centers=(bins[1:]+bins[:-1])/2
        hist=np.array([pair_histogram(t[20:],bins) for t in traces])
        disp=np.array([displacement(t) for t in traces]);times=np.array([r['time_fs'] for r in ref])
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
        for ax,x,reference,values in [(axes[0],centers,pair_histogram(ref[20:],bins),hist),
                                       (axes[1],times,displacement(ref),disp)]:
            ax.plot(x,reference,color='#38748c',label='CP2K/PBE AIMD',lw=1.8)
            ax.plot(x,values.mean(0),color='#d78a36',label='Semi-E3-MLIP mean',lw=1.8)
            ax.fill_between(x,values.min(0),values.max(0),color='#d78a36',alpha=.18,label=f'{len(traces)}-seed range')
            ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.13)
        axes[0].set(xlabel='Pair distance (Angstrom)',ylabel='Neighbors / atom / Angstrom',title='Slab pair-distance density | 100-500 fs')
        axes[1].set(xlabel='Time (fs)',ylabel='Initial-frame displacement (Angstrom^2)',title='COM-corrected displacement | not diffusion')
        fig.suptitle(name.replace('_',' ')+' | 300 K | PBE reference / r2SCAN-trained model',fontsize=13)
        for ext in ('png','svg','pdf'):fig.savefig(out/f'{name}_metrics.{ext}',dpi=170)
        plt.close(fig)
    print('Cached AIMD statistics rendered; trajectories and metrics unchanged.',flush=True)


if __name__=='__main__':main()
