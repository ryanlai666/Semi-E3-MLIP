"""Predeclared chemistry scope; retain original parent-disjoint partitions."""
from collections import Counter
from pathlib import Path
import json
from math import gcd
from functools import reduce
from semi_mlip.data import read_jsonl, write_json

ELEMENTS = {13,14,22,27,29,40,44,72,73,74}
OXIDES = {(14,1,2),(13,2,3),(29,2,1),(29,1,1),(72,1,2),
          (40,1,2),(22,1,2),(73,2,5),(74,1,3),(44,1,2),
          (27,1,1),(27,3,4)}

def included(row):
    counts = Counter(row['z'])
    if len(counts) == 1:
        return next(iter(counts)) in ELEMENTS
    if len(counts) != 2 or 8 not in counts:
        return False
    metal = next(z for z in counts if z != 8)
    divisor = reduce(gcd, counts.values())
    return (metal, counts[metal]//divisor, counts[8]//divisor) in OXIDES

def main():
    out = Path('data/device'); out.mkdir(exist_ok=True)
    audit = {'selection': 'composition only; no filtering by prediction errors, forces or energies',
             'split_policy': 'retain original parent-disjoint split', 'splits': {}}
    for split in ('train','valid','test'):
        rows = list(read_jsonl(Path('data/processed') / (split+'.jsonl')))
        selected = [r for r in rows if included(r)]
        with (out/(split+'.jsonl')).open('w') as f:
            for row in selected:
                f.write(json.dumps(row)+'\n')
        audit['splits'][split] = {'frames': len(selected), 'formulas': dict(Counter(r['formula'] for r in selected)),
                                 'parents': len({r['group'] for r in selected})}
    write_json('reports/device_scope.json', audit)
    print(json.dumps(audit, indent=2))

if __name__ == '__main__':
    main()
