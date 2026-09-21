"""Read public OMat ASE-LMDB storage directly, without FAIRChem/ASE/PyG.

Use uncorrected DFT energy. Preserve official partitions and parent identifiers.
These are randomly subsampled snapshots, not continuous trajectories.
"""
import json
import zlib
from pathlib import Path
from collections import Counter
from math import gcd
from functools import reduce
import lmdb
import numpy as np
from semi_mlip.data import SYMBOLS, write_json
from device_subset import included


def decode_array(value):
    if isinstance(value, dict) and '__ndarray__' in value:
        shape, dtype, values = value['__ndarray__']
        return np.asarray(values, dtype=dtype).reshape(shape)
    return np.asarray(value)


def main():
    source = Path('data/raw/omat24_1M')
    target = Path('data/source_isolated/omat24'); target.mkdir(parents=True, exist_ok=True)
    audit = {}; parents = {}; total = 0
    for path in sorted(source.rglob('*.aselmdb')):
        rel = path.relative_to(source); partition = rel.parts[0]; sampling = rel.parts[1]
        scanned = 0; selected = 0; formulas = Counter(); parent_set = set()
        env = lmdb.open(str(path), subdir=False, readonly=True, lock=False)
        with env.begin() as tx, (target/f'{partition}__{sampling}.jsonl').open('w') as out:
            for key, value in tx.cursor():
                if not key.isdigit(): continue
                raw = json.loads(zlib.decompress(value)); scanned += 1
                numbers = decode_array(raw['numbers']).astype(int).tolist()
                if not included({'z': numbers}): continue
                info = raw['data']; parent = info['parent_id']
                positions = decode_array(raw['positions']); forces = decode_array(raw['forces'])
                cell = decode_array(raw['cell']); energy = float(raw['energy'])
                assert positions.shape == forces.shape == (len(numbers), 3)
                assert all(np.isfinite(x).all() for x in (positions, forces, cell, energy))
                assert abs(np.linalg.det(cell)) > 1e-6
                counts = Counter(numbers); divisor = reduce(gcd, counts.values())
                formula = ''.join(SYMBOLS[z]+(str(n//divisor) if n//divisor>1 else '') for z,n in sorted(counts.items(), key=lambda kv: kv[0]==8))
                is_aimd = sampling.startswith('aimd-')
                row = {'id': info['sid'], 'z': numbers, 'positions': positions.tolist(),
                       'cell': cell.tolist(), 'pbc': decode_array(raw['pbc']).tolist(),
                       'energy': energy, 'forces': forces.tolist(), 'stress': None,
                       'source_stress_ase_eV_A3': decode_array(raw['stress']).tolist() if 'stress' in raw else None,
                       'source': 'OMat24-1M-251210', 'label_source': 'VASP PBE(+U), OMat PAW54 settings',
                       'group': parent, 'parent': parent, 'source_split': partition, 'split': 'unassigned',
                       'sampling': sampling, 'is_aimd_sample': is_aimd,
                       'temperature_target_K': (int(sampling.split('-')[-2]) if is_aimd else None),
                       'formula': formula, 'chemsys': '-'.join(sorted(SYMBOLS[z] for z in counts)),
                       'time_fs': None, 'source_metadata': info}
                out.write(json.dumps(row)+'\n'); selected += 1; formulas[formula] += 1; parent_set.add(parent)
        env.close(); total += selected
        audit[str(rel)] = {'scanned': scanned, 'selected': selected, 'formulas': dict(formulas), 'parents': len(parent_set)}
        parents.setdefault(partition, set()).update(parent_set)
        print(str(rel), scanned, selected, flush=True)
    overlap = {f'{a}|{b}': len(parents[a]&parents[b]) for a in parents for b in parents if a < b}
    write_json('reports/omat_device_audit.json', {'files': audit, 'selected_total': total,
        'parent_overlap_across_official_partitions': overlap,
        'training_enabled': False, 'split_policy': 'Repartition by parent before training; official test parents remain excluded from training.',
        'energy': 'raw total energy; MP2020 corrections deliberately unused',
        'trajectory_order_verified': False})


if __name__ == '__main__':
    main()
