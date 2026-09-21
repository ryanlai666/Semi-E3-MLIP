"""Reproduce a fixed published baseline into a new local run, preserving reference reports."""
import argparse
import json
from pathlib import Path
from semi_mlip.data import write_json
from semi_mlip.model import ModelConfig
from semi_mlip.train import TrainConfig,train,evaluate_checkpoint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='runs/reproduction/broad')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    selection=json.loads(Path('reports/aimd_comparison/selection.json').read_text())
    folder=Path(args.run)
    reference=Path(selection['checkpoint'].replace('\\','/')).parent.resolve()
    if folder.resolve()==reference:raise ValueError('Choose a new run directory; the benchmark reference is frozen')
    config=TrainConfig(**selection['config']);model=ModelConfig(**selection['model'])
    checkpoint=train('data/device_expanded',folder,config,model,folder/'last.pt' if args.resume else None)
    records={}
    for split,file in [('train','data/device_expanded/train.jsonl'),('validation','data/device_expanded/valid.jsonl'),
                       ('test','data/device/test.jsonl'),('oxide_test','data/device_expanded/aloe_test.jsonl')]:
        records[split]=evaluate_checkpoint(checkpoint,file,folder/f'{split}_evaluation.json',atom_budget=1024,edge_budget=64000)
    write_json(folder/'reproduction.json',{'reference_checkpoint_sha256':selection['checkpoint_sha256'],
        'checkpoint':str(checkpoint),'selection':'Fixed reference configuration; best epoch chosen on validation only',
        'note':'CUDA scatter operations may change numerical results across runs/hardware. This reproduction does not replace the frozen reference reports.',
        'metrics':{s:r['overall'] for s,r in records.items()}})


if __name__=='__main__':main()
