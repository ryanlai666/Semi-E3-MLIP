"""Audit all original AIMD arrays; keep original set boundaries and missing metadata."""
from collections import Counter
import hashlib
from pathlib import Path
import numpy as np
from semi_mlip.data import write_json


def main():
    root = Path('data/raw/alsi_interface_2025/Dataset')
    report = {}; total = 0
    for types_path in sorted(root.rglob('type.raw')):
        system = types_path.parent; names = (system/'type_map.raw').read_text().split()
        types = np.loadtxt(types_path, dtype=int, ndmin=1)
        species = [names[i] for i in types]; seen = set(); sets = {}; atoms = len(types)
        force_max = 0.
        for folder in sorted(system.glob('set.*')):
            arrays = {k: np.load(folder/f'{k}.npy', mmap_mode='r', allow_pickle=False)
                      for k in ('coord','force','energy','box','virial')}
            n = len(arrays['energy']); total += n
            assert all(len(a) == n and np.isfinite(a).all() for a in arrays.values())
            positions = arrays['coord'].reshape(n,atoms,3)
            forces = arrays['force'].reshape(n,atoms,3)
            cells = arrays['box'].reshape(n,3,3)
            assert (np.abs(np.linalg.det(cells)) > 1e-6).all()
            force_max = max(force_max, float(np.abs(forces).max()))
            duplicate = 0
            for p,c in zip(positions,cells):
                key = hashlib.sha256(p.tobytes()+c.tobytes()).hexdigest()
                duplicate += key in seen; seen.add(key)
            sets[folder.name] = {'frames': n, 'exact_duplicates_previously_seen': duplicate}
        report[str(system.relative_to(root))] = {'atoms': atoms, 'composition': dict(Counter(species)),
            'frames': sum(s['frames'] for s in sets.values()), 'unique_exact_geometries': len(seen),
            'sets': sets, 'maximum_absolute_force_component': force_max,
            'per_frame_temperature_available': False, 'chronology_verified': False}
    write_json('reports/alsi_interface_audit.json', {'frames':total,'systems':report,
        'source':'https://github.com/krutarth24/Al-Si-DeePMD-NNP',
        'paper':'https://doi.org/10.1063/5.0243641', 'classification':'AIMD-derived Al, Si, Al/Si interface DFT labels',
        'training_enabled':False,
        'split_policy':'set.NNN is a storage shard, not a verified independent trajectory; do not random-split correlated frames.',
        'notes':'Use audited array counts, not paper table totals. Temperature range in paper is not a per-frame temperature annotation.'})
    print(total, {k:v['frames'] for k,v in report.items()}, flush=True)


if __name__ == '__main__':
    main()
