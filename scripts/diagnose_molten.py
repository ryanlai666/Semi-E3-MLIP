"""Post-hoc frozen Ti cold/melt diagnostic. Not a blind-test score."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from semi_mlip.train import load_potential,load_graphs
from semi_mlip.graph import collate

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4)
    frozen=json.loads(Path('reports/focused/frozen.json').read_text())
    t=next(t for t in frozen['checkpoints'] if t['name']=='confirm_ti_tensor_n900_s42')
    assert sha(t['checkpoint'])==t['sha256']
    model,_=load_potential(t['checkpoint'],'cpu');model.eval()
    result={'checkpoint_sha256':t['sha256'],'selection':'Post-hoc: worst cached molten energy error and first cold test frame. No training or model selection.', 'cases':{}}
    for split in ['test_cold','test_melt']:
        file=Path('data/focused/ti_cold_900')/(split+'.jsonl')
        rows,graphs,excluded=load_graphs(file,5.,1024,64000);assert not excluded
        ix=0
        if split=='test_melt':
            key=hashlib.sha256((t['sha256']+sha(file)+'parity-v1-cuda-stress').encode()).hexdigest()
            with np.load(Path('runs/parity_cache')/(key+'.npz')) as cache:
                bad=str(cache['id'][np.argmax(np.abs(cache['ep']-cache['er']))])
            ix=next(i for i,r in enumerate(rows) if r['id']==bad)
        row,graph=rows[ix],graphs[ix];trace=[];handles=[]
        for k,layer in enumerate(model.interactions):
            def hook(module,args,out,k=k):
                trace.append({'block':k+1,**{name:float(x.detach().abs().max()) for name,x in zip(['scalar_absmax','vector_absmax','tensor_absmax'],out) if x is not None}})
            handles.append(layer.register_forward_hook(hook))
        pred=model(collate([row],[graph],'cpu'))
        for h in handles:h.remove()
        i,j,shifts=graph;distance=np.linalg.norm(np.array(row['positions'])[j]-np.array(row['positions'])[i]+shifts@np.array(row['cell']),axis=1)
        result['cases'][split]={'id':row['id'],'minimum_distance_A':float(distance.min()),'edges':len(i),'mean_neighbors':len(i)/len(row['z']),'trace':trace,'force_absmax_eV_A':float(pred['forces'].detach().abs().max()),'force_mae_eV_A':float(np.abs(pred['forces'].detach().numpy()-np.array(row['forces'])).mean()),'reference_force_absmax_eV_A':float(np.abs(row['forces']).max())}
    trainrows,traingraphs,_=load_graphs('data/focused/ti_cold_900/train.jsonl',5.,1024,64000)
    minima=[]
    for r,(i,j,s) in zip(trainrows,traingraphs):minima.append(float(np.linalg.norm(np.array(r['positions'])[j]-np.array(r['positions'])[i]+s@np.array(r['cell']),axis=1).min()))
    result['training_minimum_distance_quantiles_A']={str(q):float(np.quantile(minima,q)) for q in [0,.01,.5,.99,1]}
    Path('reports/physics_research').mkdir(exist_ok=True)
    Path('reports/physics_research/ti_failure.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
