"""Predeclared temperature-separated TM23 experiments, before model fitting."""
import hashlib
import json
from pathlib import Path
from semi_mlip.data import read_jsonl, write_json


def choose(rows, count):
    return sorted(rows, key=lambda r: hashlib.sha256(r['id'].encode()).digest())[:count]


def main():
    root = Path('data/focused'); root.mkdir(exist_ok=True)
    audit = {}
    for metal in ('Cu', 'Ti'):
        source = Path('data/source_isolated/tm23')
        cold = list(read_jsonl(source/f'{metal}_cold_train.jsonl'))
        warm = list(read_jsonl(source/f'{metal}_warm_train.jsonl'))
        for count in (300, 900):
            folder = root/f'{metal.lower()}_cold_{count}'; folder.mkdir(exist_ok=True)
            partitions = {'train': choose(cold, count), 'valid': choose(warm, 90)}
            for regime in ('cold','warm','melt'):
                # Published test files stay out of model/hyperparameter selection.
                partitions[f'test_{regime}'] = list(read_jsonl(source/f'{metal}_{regime}_test.jsonl'))
            for split, rows in partitions.items():
                with (folder/f'{split}.jsonl').open('w') as out:
                    for row in rows:
                        row = {**row, 'split': split}
                        out.write(json.dumps(row)+'\n')
            assert {r['group'] for r in partitions['train']}.isdisjoint(r['group'] for r in partitions['valid'])
            keys = {split:{r['id'] for r in rows} for split,rows in partitions.items()}
            assert all(not keys[a]&keys[b] for a in keys for b in keys if a<b)
            # Exact geometry audit independent of labels and predictions.
            hashes = {}
            for split,rows in partitions.items():
                hashes[split] = {hashlib.sha256(json.dumps([r['z'],r['cell'],r['positions']]).encode()).hexdigest() for r in rows}
            overlap = {a+'|'+b:len(hashes[a]&hashes[b]) for a in hashes for b in hashes if a<b}
            assert not any(overlap.values())
            audit[str(folder)] = {'counts':{s:len(r) for s,r in partitions.items()},
                'groups':{s:sorted({x['group'] for x in r}) for s,r in partitions.items()},
                'exact_geometry_overlap':overlap,
                'file_sha256':{s:hashlib.sha256((folder/f'{s}.jsonl').read_bytes()).hexdigest() for s in partitions}}
    write_json('reports/focused_data_manifest.json', {'datasets':audit,
        'selection':'ID-hash nested sample; independent of label values and model predictions',
        'label_family':'TM23 VASP PBE nonspin, dense-k labels on Gamma-AIMD geometries',
        'limitations':['Warm validation is a temperature-transfer selection set, not in-domain validation.',
                      'Cold and warm final tests share source trajectories with training/validation respectively.',
                      'Molten test is a held-out temperature/trajectory, not an independent seed replicate.',
                      'Exact geometry checks do not establish near-duplicate independence.']})
    print(json.dumps({k:v['counts'] for k,v in audit.items()},indent=2))


if __name__ == '__main__':
    main()
