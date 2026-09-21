"""Import only selected device metals; preserve published temperature/test splits."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from semi_mlip.extxyz import read_extxyz
from semi_mlip.data import ELEMENTS, write_json


def main():
    archive = Path('data/raw/tm23.zip')
    assert hashlib.file_digest(archive.open('rb'), 'md5').hexdigest() == '52b2c471591c8a8a88717f3fbdff6e37'
    root = Path('data/source_isolated/tm23'); root.mkdir(parents=True, exist_ok=True)
    audit = {}
    with zipfile.ZipFile(archive) as z:
        for metal in ('Cu', 'W', 'Ti', 'Ta', 'Co', 'Ru', 'Hf', 'Zr'):
            for regime, ratio in [('cold', .25), ('warm', .75), ('melt', 1.25)]:
                for split in ('train', 'test'):
                    name = f'{metal}_{regime}_{split}'
                    member = f'benchmarking_master_collection/{metal}_{regime}_nequip_{split}.xyz'
                    seen = set(); frames = 0
                    with z.open(member) as binary, (root/f'{name}.jsonl').open('w') as out:
                        for i, (meta, cols) in enumerate(read_extxyz(io.TextIOWrapper(binary))):
                            assert set(cols['species']) == {metal}
                            cell = np.asarray(meta['Lattice'].split(), float).reshape(3, 3)
                            assert abs(np.linalg.det(cell)) > 1e-6
                            energy = float(meta['energy']); assert np.isfinite(energy)
                            key = hashlib.sha256(cell.tobytes()+cols['pos'].tobytes()).hexdigest()
                            assert key not in seen; seen.add(key)
                            row = {'id': f'TM23-{name}-{i}', 'z': [ELEMENTS[metal]]*len(cols['pos']),
                                   'positions': cols['pos'].tolist(), 'forces': cols['forces'].tolist(),
                                   'cell': cell.tolist(), 'pbc': [True]*3, 'energy': energy,
                                   'stress': None, 'source_metadata': meta,
                                   'source': 'TM23-PBE-nonspin', 'formula': metal, 'chemsys': metal,
                                   'group': f'TM23-{metal}-{regime}', 'source_index': i,
                                   'source_split': split, 'split': split,
                                   'temperature_over_melting': ratio, 'reported_sampling_interval_fs': 50,
                                   'time_fs': None, 'label_source': 'PBE dense-k relabel of Gamma-point AIMD'}
                            out.write(json.dumps(row)+'\n'); frames += 1
                    audit[name] = {'frames': frames, 'source_member': member, 'path': str(root/f'{name}.jsonl')}
    write_json('reports/tm23_audit.json', {
        'source': 'https://archive.materialscloud.org/record/2024.48',
        'frames': sum(v['frames'] for v in audit.values()), 'files': audit,
        'notes': ['Do not also ingest 2700cwm files: they repeat these labels.',
                  'Original test files remain test-only; train and test share a trajectory.',
                  'Use whole-temperature holdouts for transfer testing; no independent-seed claim.',
                  'No explicit time in XYZ: ordering/timing not verified for animation.',
                  'Stress preserved as metadata pending unit/sign verification.',
                  'Nonspin labels: particularly important for Co; never pool with spin-polarized labels.']})
    print('Imported', sum(v['frames'] for v in audit.values()), 'frames in', len(audit), 'files')


if __name__ == '__main__':
    main()
