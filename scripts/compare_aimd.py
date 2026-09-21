"""External AIMD transfer diagnostic; never used for selecting hyperparameters."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from semi_mlip.data import read_jsonl, write_json
from semi_mlip.simulate import Calculator, md
from semi_mlip.visualize import plotting, draw_structure

def minimum_image(delta, cell, pbc=(True,True,True)):
    """Closest periodic image using reciprocal-plane bounds, including skew cells."""
    cell=np.asarray(cell,float);delta=np.asarray(delta,float);periodic=np.asarray(pbc,bool)
    if cell.shape!=(3,3) or abs(np.linalg.det(cell))<1e-10:
        raise ValueError('Nonsingular 3x3 cell required')
    if not np.isfinite(cell).all() or not np.isfinite(delta).all():
        raise ValueError('Nonfinite trajectory geometry')
    inverse=np.linalg.inv(cell);fractional=delta@inverse
    fractional[...,periodic]-=np.round(fractional[...,periodic])
    reduced=fractional@cell;best=reduced.copy();best2=np.sum(best*best,axis=-1)
    if best2.size==0:return best
    gram=cell@cell.T
    if np.allclose(gram,np.diag(np.diag(gram)),rtol=0,atol=1e-10):return best
    # A winning vector has norm <= the current largest candidate radius.
    # Its fractional component is bounded by radius * reciprocal-plane norm.
    # Periodic components of the reduced vector lie in [-1/2, 1/2].
    radius=float(np.sqrt(best2.max()))
    bounds=np.floor(.5+radius*np.linalg.norm(inverse,axis=0)+1e-12).astype(int)
    ranges=[range(-n,n+1) if flag else (0,) for n,flag in zip(bounds,periodic)]
    for shift in itertools.product(*ranges):
        if shift==(0,0,0):continue
        candidate=reduced+np.asarray(shift)@cell
        squared=np.sum(candidate*candidate,axis=-1);replace=squared<best2
        best=np.where(replace[...,None],candidate,best);best2=np.minimum(best2,squared)
    return best


def displacement(rows):
    # Incremental nearest-image unwrapping assumes sufficiently frequent samples;
    # long displacements between saved frames remain fundamentally ambiguous.
    cells=np.array([r['cell'] for r in rows])
    if not np.allclose(cells,cells[0]):raise ValueError('A fixed cell is required')
    pbc=rows[0].get('pbc',[True]*3)
    if any(r.get('pbc',[True]*3)!=pbc for r in rows):raise ValueError('PBC changed within trajectory')
    x=np.array([r['positions'] for r in rows]);delta=minimum_image(np.diff(x,axis=0),cells[0],pbc)
    unwrapped=np.concatenate([x[:1],x[:1]+np.cumsum(delta,axis=0)])
    d=unwrapped-unwrapped[:1];d-=d.mean(axis=1,keepdims=True)
    # Initial-time displacement, not time-origin-averaged transport MSD.
    return np.mean(np.sum(d*d,axis=2),axis=1)


def pair_histogram(rows,bins):
    total=np.zeros(len(bins)-1)
    for row in rows:
        x=np.array(row['positions']);cell=np.array(row['cell'])
        i,j=np.triu_indices(len(x),1)
        d=minimum_image(x[j]-x[i],cell,row.get('pbc',[True]*3))
        total+=np.histogram(np.linalg.norm(d,axis=-1),bins)[0]*2/len(x)
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
    protocol={'checkpoint':str(Path(a.checkpoint)),
              'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
              'reference_manifest_sha256':hashlib.sha256(Path('reports/aimd_manifest.json').read_bytes()).hexdigest(),
              'reference_sha256':{name:hashlib.sha256(Path(meta['path']).read_bytes()).hexdigest()
                                  for name,meta in manifest['cases'].items()},
              'steps':1000,'dt_fs':.5,'temperature_K':300,'gamma_fs_inverse':.005,
              'seeds':[42,43,44],'device':a.device,'used_for_selection':False}
    lock=out/'protocol.json'
    if lock.exists() and json.loads(lock.read_text())!=protocol:
        raise ValueError('AIMD comparison protocol changed; use a separate study output')
    write_json(lock,protocol)
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
                cached=Path(str(path)+'.summary.json')
                if cached.exists() and path.exists():
                    saved=list(read_jsonl(path))
                    if len(saved)!=1001 or saved[-1]['time_fs']!=500:
                        raise ValueError('Incomplete cached trajectory')
                    reports.append(json.loads(cached.read_text()))
                else:
                    reports.append(md(ref[0],calculate,path,steps=1000,dt=.5,temperature=300,ensemble='nvt',gamma=.005,seed=seed))
                print(f'{name}: completed seed {seed}',flush=True)
            except (ValueError,RuntimeError,FloatingPointError) as exc:
                failures.append({'seed':seed,'error':str(exc)})
                continue
            traces.append(list(read_jsonl(path))[::10])
        if not traces:
            summary['cases'][name]={'failed_runs':failures,'reference':meta,
                'force_mae_eV_A':float(np.abs(errors).mean()),
                'force_rmse_eV_A':float(np.sqrt(np.mean(np.square(errors)))),
                'force_frames':len(ref[::5])}
            write_json(out/'metrics.json',summary)
            continue
        bins=np.linspace(0,5,101);centers=(bins[1:]+bins[:-1])/2
        refhist=pair_histogram(ref[20:],bins)
        hists=np.array([pair_histogram(t[20:],bins) for t in traces])
        displacements=np.array([displacement(t) for t in traces])
        plt=plotting();fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        axes[0].plot(centers,refhist,label='PBE AIMD');axes[0].plot(centers,hists.mean(0),label='Our GNN mean')
        axes[0].fill_between(centers,hists.min(0),hists.max(0),color='#ff7f0e',alpha=.18,label=f'{len(traces)} completed seed range')
        axes[0].set(xlabel='Pair distance (Å)',ylabel='Neighbors / atom / Å',title='Slab pair-distance density (100–500 fs)')
        times=np.array([r['time_fs'] for r in ref]);axes[1].plot(times,displacement(ref),label='PBE AIMD')
        axes[1].plot(times,displacements.mean(0),label='Our GNN mean')
        axes[1].fill_between(times,displacements.min(0),displacements.max(0),color='#ff7f0e',alpha=.18,label=f'{len(traces)} completed seed range')
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
