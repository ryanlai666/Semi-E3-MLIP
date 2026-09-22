"""Frozen-data channel-width study. Resume safely; never select on test results."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from semi_mlip.model import ModelConfig, Potential
from semi_mlip.train import TrainConfig, train

ROOT=Path('reports/width_scaling')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)
def configuration(width,seed):
    original=read('runs/device/expanded_tensor/config.json')
    tc=TrainConfig(**{**original['train'],'seed':seed,'microbatch_atoms':256,'microbatch_edges':16000})
    mc=ModelConfig(**{**original['model'],'scalar_channels':64*width,'vector_channels':32*width,'tensor_channels':16*width})
    return tc,mc

def protocol():
    trials=[]
    for seed in (42,43,44):
        for width in (2,4,1):
            tc,mc=configuration(width,seed)
            trials.append({'name':f'w{width}_s{seed}','width':width,'seed':seed,'train':asdict(tc),'model':asdict(mc),'parameters':sum(p.numel() for p in Potential(mc).parameters())})
    d={'scope':'Channel width only; 1x, 2x, 4x shared architecture; three seeds per width.',
       'data_sha256':{str(p):sha(p) for p in [Path('data/device_expanded/train.jsonl'),Path('data/device_expanded/valid.jsonl')]},
       'code_sha256':{str(p):sha(p) for p in [Path('semi_mlip/train.py'),Path('semi_mlip/model.py')]},
       'selection':'Validation force MAE, unchanged for this capacity ablation; energy and force MAE/RMSE reported together. No automatic promotion.',
       'batching':'Original optimizer batches/schedule retained. Microbatches weighted by structure count; no new data exclusions.',
       'implementation_note':'After startup, added a configuration guard rejecting microbatching with auxiliary loss. All study trials use auxiliary_weight=0; numerical training code unchanged.',
       'test_used':False,'trials':trials}
    p=ROOT/'protocol.json'
    if p.exists():assert read(p)==d,'Protocol changed; use a new study directory.'
    else:write(p,d)
    return d

def report(p):
    status=[]
    for t in p['trials']:
        folder=Path('runs/width_scaling')/t['name'];entry={k:t[k] for k in ('name','width','seed','parameters')}
        history=folder/'history.jsonl'
        if history.exists():
            rows=[json.loads(x) for x in history.read_text().splitlines()]
            entry.update(epochs=rows[-1]['epoch'],seconds=sum(r['epoch_seconds'] for r in rows),peak_gpu_MB=max(r['peak_gpu_MB'] for r in rows),validation=read(folder/'validation_best.json')['overall'])
        entry['complete']=(folder/'complete.json').exists()
        status.append(entry)
    write(ROOT/'progress.json',{'complete':all(t['complete'] for t in status),'test_used':False,'trials':status})

def main():
    a=argparse.ArgumentParser();a.add_argument('--trial');a.add_argument('--probe',action='store_true');a.add_argument('--report',action='store_true');args=a.parse_args()
    p=protocol()
    if args.report:report(p);return
    if args.trial:
        t=next(t for t in p['trials'] if t['name']==args.trial);folder=Path('runs/width_scaling')/t['name']
        if (folder/'complete.json').exists():return
        tc,mc=TrainConfig(**t['train']),ModelConfig(**t['model']);last=folder/'last.pt'
        if args.probe and last.exists():return
        checkpoint=train('data/device_expanded',folder,tc,mc,last if last.exists() else None,stop_after_epochs=1 if args.probe else None)
        excluded=read(folder/'excluded.json');assert not any(excluded.values()),'Unexpected data exclusion'
        if not args.probe:write(folder/'complete.json',{'checkpoint':str(checkpoint),'sha256':sha(checkpoint)})
        report(p);return
    # Separate processes release CUDA allocations between trials. Probe both wide
    # models first; probe checkpoints resume the same 200-epoch LR schedule.
    phases=[(t,True) for t in p['trials'][:2]]+[(t,False) for t in p['trials']]
    for t,probe in phases:
        command=[sys.executable,'-u','-m','scripts.width_scaling','--trial',t['name']]+(['--probe'] if probe else [])
        write(ROOT/'active.json',{'trial':t['name'],'probe':probe,'started_unix':time.time()})
        result=subprocess.run(command)
        if result.returncode:
            write(ROOT/'failure.json',{'trial':t['name'],'probe':probe,'returncode':result.returncode});sys.exit(result.returncode)
    report(p);write(ROOT/'active.json',{'complete':True})

if __name__=='__main__':main()
