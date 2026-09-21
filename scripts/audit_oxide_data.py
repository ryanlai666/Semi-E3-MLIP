"""Normalize new oxide labels while keeping incompatible DFT sources separate.

No random frame split: the files lack trustworthy trajectory identifiers.
No time axes are invented for shuffled/selected configurations.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from semi_mlip.extxyz import read_extxyz
from semi_mlip.data import ELEMENTS, write_json


def main():
    sources = [
        ('hafnia_pbesol', 'data/raw/hafnia_aimd/HfO2_structure_data.xyz',
         'VASP PBEsol; on-the-fly DFT selection', 'https://arxiv.org/abs/2511.09976'),
        ('zirconia_pbe', 'data/raw/oxide_2025/57225278.xyz',
         'VASP PBE; mixed AIMD, model-sampled and perturbed structures', 'https://doi.org/10.6084/m9.figshare.29923850'),
        ('alumina_pbe', 'data/raw/oxide_2025/57225281.xyz',
         'VASP PBE; mixed AIMD, model-sampled and perturbed structures', 'https://doi.org/10.6084/m9.figshare.29923850'),
    ]
    root = Path('data/source_isolated'); root.mkdir(exist_ok=True)
    report = {}
    for name, filename, method, citation in sources:
        counts = Counter(); sizes = Counter(); seen = set(); duplicates = 0
        path = Path(filename)
        with path.open(encoding='utf-8') as stream, (root / f'{name}.jsonl').open('w') as out:
            for i, (meta, cols) in enumerate(read_extxyz(stream)):
                cell = np.asarray(meta['Lattice'].split(), float).reshape(3, 3)
                assert abs(np.linalg.det(cell)) > 1e-6
                symbols = cols['species'].tolist()
                z = [ELEMENTS[s] for s in symbols]
                energy = float(meta['energy']); assert np.isfinite(energy)
                assert cols['pos'].shape == cols['forces'].shape == (len(z), 3)
                # Byte-identical label duplicate audit only, not symmetry deduplication.
                key = hashlib.sha256(cell.tobytes()+cols['pos'].tobytes()+cols['forces'].tobytes()+str(energy).encode()).hexdigest()
                duplicates += key in seen; seen.add(key)
                counts['-'.join(sorted(set(symbols)))] += 1; sizes[len(z)] += 1
                row = {'id': f'{name}-{i}', 'z': z, 'positions': cols['pos'].tolist(),
                       'cell': cell.tolist(), 'pbc': [True]*3, 'energy': energy,
                       'forces': cols['forces'].tolist(), 'stress': None,
                       'source': name, 'label_source': method, 'source_index': i,
                       'group': name, 'split': 'unassigned', 'trajectory_id': None,
                       'time_fs': None, 'temperature_target_K': None,
                       'chemsys': '-'.join(sorted(set(symbols))),
                       'source_metadata': meta}
                # Preserve virial in source metadata, pending convention verification.
                out.write(json.dumps(row)+'\n')
        report[name] = {'frames': sum(counts.values()), 'chemical_systems': dict(counts),
                        'atom_counts': dict(sizes), 'exact_label_duplicates': duplicates,
                        'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
                        'citation': citation, 'method': method,
                        'trajectory_order_verified': False, 'training_enabled': False}
    write_json('reports/oxide_data_audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
