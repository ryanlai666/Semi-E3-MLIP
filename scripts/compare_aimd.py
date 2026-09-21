"""External AIMD transfer diagnostic; never used for selecting hyperparameters."""
import argparse
import json
from pathlib import Path
import numpy as np
from semi_mlip.data import read_jsonl, write_json
from semi_mlip.simulate import Calculator, md
from semi_mlip.visualize import plotting, draw_structure

def displacement(rows):
    # Source cells are orthogonal and constant. Minimum-image incremental
    # unwrapping is appropriate only when each sampled displacement < L/2.
    cells=np.array([r['cell'] for r in rows])
    if not np.allclose(cells,cells[0]) or not np.allclose(cells[0],np.diag(np.diag(cells[0])),atol=1e-6):
        raise ValueError('This diagnostic requires a fixed orthogonal cell')
    x=np.array([r['positions'] for r in rows]); delta=np.diff(x,axis=0)
    lengths=np.diag(cells[0]); delta-=np.round(delta/lengths)*lengths
    unwrapped=np.concatenate([x[:1],x[:1]+np.cumsum(delta,axis=0)])
    d=unwrapped-unwrapped[:1]
    d-=d.mean(axis=1,keepdims=True)
    # Initial-time displacement, not time-origin-averaged transport MSD.
    return np.mean(np.sum(d*d,axis=2),axis=1)

def pair_histogram(rows,bins):
    total=np.zeros(len(bins)-1)
    for row in rows:
        x=np.array(row['positions']); cell=np.array(row['cell'])
        if not np.allclose(cell,np.diag(np.diag(cell)),atol=1e-6):
            raise ValueError('Orthogonal cell required')
        d=x[:,None,:]-x[None,:,:]; length=np.diag(cell)
        d-=np.round(d/length)*length
        distances=np.linalg.norm(d,axis=-1)[np.triu_indices(len(x),1)]
        total+=np.histogram(distances,bins)[0]*2/len(x)
    # Slabs contain vacuum: bulk-density normalization would distort RDF.
    return total/len(rows)/np.diff(bins)

def render(reference,predicted,path,title):
    from matplotlib.animation import FuncAnimation,PillowWriter
    plt=plotting(); fig=plt.figure(figsize=(12,6),dpi=100)
    axes=[fig.add_subplot(1,2,k+1,projection='3d') for k in range(2)]
    fig.suptitle(title+' | 300 K, external transfer check',fontsize=14)
    footer=fig.text(.5,.04,'',ha='center',fontsize=9)
    lookup={round(r['time_fs'],6):r for r in predicted}
    indices=np.linspace(0,len(reference)-1,41).astype(int)
    def update(k):
        row=reference[indices[k]]; other=lookup[round(row['time_fs'],6)]
        for ax,r,label in zip(axes,(row,other),('Reference: CP2K/PBE AIMD','Our r2SCAN GNN: Langevin MD')):
            ax.clear();draw_structure(ax,r,title=label)
        footer.set_text(f"Time {row['time_fs']:.0f} fs | Same initial geometry; independent velocities and different thermostats\n"
                        'Surface and DFT-functional transfer; visual agreement alone is not validation.')
    update(0);fig.savefig(path.with_suffix('.png'),dpi=140)
    FuncAnimation(fig,update,frames=len(indices),interval=100).save(path,writer=PillowWriter(fps=10));plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--device',default='cpu')
    a=p.parse_args(); calculate=Calculator(a.checkpoint,a.device)
    manifest=json.loads(Path('reports/aimd_manifest.json').read_text())
    out=Path('reports/aimd_comparison');out.mkdir(exist_ok=True)
    summary={'checkpoint':a.checkpoint,'used_for_selection':False,'cases':{},
             'limitations':'500 fs; surface transfer; PBE versus r2SCAN; different thermostat and initial velocities; no diffusion claim'}
    for name,meta in manifest['cases'].items():
        ref=list(read_jsonl(meta['path']));errors=[]
        for row in ref[::5]:
            result=calculate(row);errors.extend((result['forces']-np.array(row['forces'])).ravel().tolist())
        traces=[]; reports=[]; failures=[]
        for seed in (42,43,44):
            path=Path('runs/aimd_comparison')/name/f'seed{seed}.jsonl'
            try:
                reports.append(md(ref[0],calculate,path,steps=1000,dt=.5,temperature=300,ensemble='nvt',gamma=.005,seed=seed))
            except (ValueError,RuntimeError,FloatingPointError) as exc:
                failures.append({'seed':seed,'error':str(exc)})
                continue
            traces.append(list(read_jsonl(path))[::10])
        if not traces:
            summary['cases'][name]={'failed_runs':failures,'reference':meta}
            write_json(out/'metrics.json',summary)
            continue
        bins=np.linspace(0,5,101);centers=(bins[1:]+bins[:-1])/2
        refhist=pair_histogram(ref[20:],bins)
        hists=np.array([pair_histogram(t[20:],bins) for t in traces])
        displacements=np.array([displacement(t) for t in traces])
        plt=plotting();fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        axes[0].plot(centers,refhist,label='PBE AIMD');axes[0].plot(centers,hists.mean(0),label='Our GNN mean')
        axes[0].fill_between(centers,hists.min(0),hists.max(0),alpha=.2,label=f'{len(traces)} completed seed range')
        axes[0].set(xlabel='Pair distance (Å)',ylabel='Neighbors / atom / Å',title='Slab pair-distance density (100–500 fs)')
        times=np.array([r['time_fs'] for r in ref]);axes[1].plot(times,displacement(ref),label='PBE AIMD')
        axes[1].plot(times,displacements.mean(0),label='Our GNN mean')
        axes[1].fill_between(times,displacements.min(0),displacements.max(0),alpha=.2,label=f'{len(traces)} completed seed range')
        axes[1].set(xlabel='Time (fs)',ylabel='Displacement from initial frame (Å²)',title='COM-corrected; not a diffusion estimate')
        for ax in axes:ax.legend(fontsize=8)
        fig.suptitle(name+' | 300 K | PBE → r2SCAN transfer');fig.tight_layout();fig.savefig(out/(name+'_metrics.png'),dpi=170);plt.close(fig)
        render(ref,traces[0],out/(name+'.gif'),name)
        summary['cases'][name]={'force_mae_eV_A':float(np.abs(errors).mean()),'force_rmse_eV_A':float(np.sqrt(np.mean(np.square(errors)))),
                               'force_frames':len(ref[::5]),'model_simulations':reports,'failed_runs':failures,'reference':meta}
        write_json(out/'metrics.json',summary)
        print(name,json.dumps(summary['cases'][name]),flush=True)
    cards=''.join(f'<h2>{name}</h2><p>Failed seeds: {len(value["failed_runs"])}</p>'+
                  (f'<img width="900" src="{name}.gif"><img width="900" src="{name}_metrics.png">' if 'model_simulations' in value else '<p>All simulations failed.</p>')
                  for name,value in summary['cases'].items())
    (out/'index.html').write_text('<html><meta charset="utf-8"><body><h1>External AIMD comparison</h1><p>'+summary['limitations']+'</p>'+cards+'</body></html>')

if __name__=='__main__':main()
