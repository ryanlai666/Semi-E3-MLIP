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
    results=[]
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        calc=Calculator(selection['checkpoint'],device)
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
        'scope':'Batch size one; includes neighbor enumeration, tensor transfer, energy, force and stress gradients. Small cells only. No cross-model speed claim.'})
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
