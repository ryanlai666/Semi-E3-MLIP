"""Measure end-to-end single-structure inference on this machine, no pretrained comparators."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from semi_mlip.data import read_jsonl,write_json
from semi_mlip.simulate import Calculator


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,default=30);args=parser.parse_args()
    selection=json.loads(Path('reports/aimd_comparison/selection.json').read_text())
    rows=list(read_jsonl('data/device/test.jsonl'))
    examples=[]
    for system in ('Si','Cu','Ti','W','O-Si','Al-O','Hf-O','O-Ti'):
        choices=sorted([r for r in rows if r['chemsys']==system],key=lambda r:r['id'])
        if choices:examples.append(choices[0])
    # Include one actual 216-atom AIMD surface to expose size-dependent cost.
    surface=next(read_jsonl('data/aimd/Si_110_elong0.500.jsonl'))
    examples.append(surface)
    results=[]
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        calc=Calculator(selection['checkpoint'].replace('\\','/'),device)
        for row in examples:
            for _ in range(5):calc(row)
            durations=[]
            for _ in range(args.repeats):
                if device=='cuda':torch.cuda.synchronize()
                start=time.perf_counter();calc(row)
                if device=='cuda':torch.cuda.synchronize()
                durations.append(time.perf_counter()-start)
            results.append({'device':device,'system':row['chemsys'],'id':row['id'],'atoms':len(row['z']),
                'median_ms':float(np.median(durations)*1000),'p10_ms':float(np.percentile(durations,10)*1000),
                'p90_ms':float(np.percentile(durations,90)*1000),'atoms_per_second':len(row['z'])/float(np.median(durations))})
    write_json('reports/benchmark/inference.json',{'checkpoint_sha256':selection['checkpoint_sha256'],
        'torch':torch.__version__,'python':platform.python_version(),'platform':platform.platform(),
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cpu_threads':torch.get_num_threads(),'warmups':5,'repeats':args.repeats,'cases':results,
        'scope':'Batch size one; includes neighbor enumeration, tensor transfer, energy, force and stress gradients. Primitive test cells plus one 216-atom Si surface. No cross-model speed claim.'})
    from semi_mlip.visualize import plotting
    plt=plotting();fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    labels=[f"{r['chemsys']} ({len(r['z'])} atoms)" for r in examples];x=np.arange(len(examples))
    for device,shift,color in [('cpu',-.18,'#38748c'),('cuda',.18,'#d78a36')]:
        subset=[r for r in results if r['device']==device]
        if not subset:continue
        med=np.array([r['median_ms'] for r in subset]);low=np.array([r['p10_ms'] for r in subset]);high=np.array([r['p90_ms'] for r in subset])
        ax.bar(x+shift,med,.35,yerr=np.vstack([med-low,high-med]),color=color,capsize=3,label=device.upper())
    ax.set_xticks(x,labels,rotation=30,ha='right');ax.set(ylabel='End-to-end inference time (ms)',yscale='log',title='Local inference cost | median and 10th-90th percentile interval')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
    for ext in ('png','svg','pdf'):fig.savefig(Path('docs/assets')/f'inference_benchmark.{ext}',dpi=170)
    plt.close(fig)
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
