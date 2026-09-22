"""Controlled priority pilots, isolated from the running width-study checkout."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from dataclasses import asdict
from semi_mlip.model import ModelConfig,Potential
from semi_mlip.train import TrainConfig,train,evaluate_checkpoint

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def prepare(workspace):
    data=read(workspace/'reports/physics_research/priority_data.json');trials=[]
    for study in ['shared_ta_o_expanded','mixed_ti','mixed_ru','mixed_ta']:
        for variant in ['plain','stabilized']:
            trials.append((study,variant))
    trials.append(('shared_ta_o_expanded','stabilized_core'))
    configs=[]
    for study,variant in trials:
        tc=TrainConfig(epochs=60,max_train=0,atom_budget=1024,edge_budget=64000,accumulation=1,seed=42,lr=.001,loss='pseudo_huber',stress_weight=1 if study.startswith('shared') else 0,microbatch_atoms=256,microbatch_edges=16000,monitor_train_every=10,selection_metric='balanced_macro')
        mc=ModelConfig(scalar_channels=128,vector_channels=64,tensor_channels=32,blocks=4,radial_basis=32,attention=True,equivariant_norm=variant!='plain',bounded_gates=variant!='plain',repulsive_core=variant=='stabilized_core')
        configs.append({'name':study+'_'+variant+'_s42','study':study,'variant':variant,'train':asdict(tc),'model':asdict(mc),'parameters':sum(p.numel() for p in Potential(mc).parameters())})
    code=Path(__file__).resolve().parents[1]
    protocol={'phase':'Exploratory single-seed, matched 60-epoch priority pilots; not a converged three-seed confirmation.',
      'dependency':'Wait for the entire current width queue to finish successfully before using the GPU.',
      'studies':data['studies'],'trials':configs,'priority_data_sha256':sha(workspace/'reports/physics_research/priority_data.json'),
      'code_sha256':{p:sha(code/p) for p in ['semi_mlip/model.py','semi_mlip/train.py','semi_mlip/repulsion.py']},
      'selection':'Equal-chemistry mean of 0.5*(energy_MAE/0.01 + force_MAE/0.1); no test selection.',
      'evaluation':'Freeze all checkpoints before reading model errors on tests. TM23 and historical r2SCAN tests are post-hoc; new Ta-O groups are untouched at protocol time.',
      'claims':'Changing data and selection confounds comparisons with old published models; causal architecture comparisons use matched plain/stabilized pairs within this protocol.'}
    p=workspace/'reports/physics_research/retraining_protocol.json'
    if p.exists():assert read(p)==protocol,'Protocol mismatch'
    else:write(p,protocol)
    return protocol

def report(workspace,protocol):
    status=[]
    for trial in protocol['trials']:
        folder=workspace/'runs/physics_research/trials'/trial['name'];entry={'name':trial['name'],'complete':(folder/'complete.json').exists()}
        p=folder/'history.jsonl'
        if p.exists():
            rows=[json.loads(x) for x in p.read_text().splitlines()];entry.update(epochs=rows[-1]['epoch'],best_selection_score=min(r['selection_score'] for r in rows),peak_gpu_MB=max(r['peak_gpu_MB'] for r in rows))
        status.append(entry)
    write(workspace/'reports/physics_research/retraining_progress.json',{'complete':all(t['complete'] for t in status),'trials':status})

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--trial');p.add_argument('--prepare',action='store_true');p.add_argument('--queue',action='store_true');args=p.parse_args();workspace=args.workspace.resolve();protocol=prepare(workspace)
    if args.prepare:report(workspace,protocol);return
    if args.trial:
        trial=next(t for t in protocol['trials'] if t['name']==args.trial);folder=workspace/'runs/physics_research/trials'/trial['name']
        if (folder/'complete.json').exists():return
        parts=protocol['studies'][trial['study']]['partitions']
        for name in ['train','valid']:
            part=parts[name];assert sha(workspace/part['path'])==part['sha256']
        checkpoint=train((workspace/parts['train']['path']).parent,folder,TrainConfig(**trial['train']),ModelConfig(**trial['model']),folder/'last.pt' if (folder/'last.pt').exists() else None)
        assert not any(read(folder/'excluded.json').values()),'Unexpected excluded frames'
        write(folder/'complete.json',{'checkpoint':str(checkpoint),'sha256':sha(checkpoint)});report(workspace,protocol);return
    if not args.queue:p.error('Choose --prepare, --trial or --queue')
    lock=workspace/'runs/physics_research/queue.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    out=workspace/'reports/physics_research'
    try:
        while True:
            active=workspace/'reports/width_scaling/active.json';progress=workspace/'reports/width_scaling/progress.json'
            if (workspace/'reports/width_scaling/failure.json').exists():raise RuntimeError('Width queue failed; physics queue will not silently bypass it')
            if active.exists() and progress.exists() and read(active).get('complete') and read(progress).get('complete'):break
            write(out/'queue_status.json',{'status':'waiting_for_width_queue','updated_unix':time.time()});time.sleep(30)
        for trial in protocol['trials']:
            write(out/'queue_status.json',{'status':'training','trial':trial['name'],'updated_unix':time.time()})
            subprocess.run([sys.executable,'-u','-m','scripts.priority_retrain','--workspace',str(workspace),'--trial',trial['name']],check=True,cwd=Path(__file__).resolve().parents[1])
        frozen={t['name']:read(workspace/'runs/physics_research/trials'/t['name']/'complete.json') for t in protocol['trials']}
        write(out/'retraining_frozen.json',{'protocol_sha256':sha(out/'retraining_protocol.json'),'trials':frozen})
        results={}
        for trial in protocol['trials']:
            checkpoint=frozen[trial['name']]['checkpoint'];assert sha(checkpoint)==frozen[trial['name']]['sha256'];results[trial['name']]={}
            for split,part in protocol['studies'][trial['study']]['partitions'].items():
                assert sha(workspace/part['path'])==part['sha256']
                target=out/'evaluations'/(trial['name']+'_'+split+'.json')
                metrics=evaluate_checkpoint(checkpoint,workspace/part['path'],target,'cuda',1024,64000)
                results[trial['name']][split]=metrics['overall']
            write(out/'retraining_results.json',{'complete':False,'results':results})
        write(out/'retraining_results.json',{'complete':True,'results':results});write(out/'queue_status.json',{'status':'complete'})
    except Exception as exc:
        write(out/'queue_status.json',{'status':'failed','error':str(exc)});raise
    finally:lock.unlink()

if __name__=='__main__':main()
