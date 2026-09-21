"""Audit actual files, labels and duplicate leakage before enabling training."""
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import zipfile
import numpy as np
from semi_mlip.extxyz import read_extxyz
from semi_mlip.data import write_json


def main():
    archive = Path('data/raw/unialcu.zip')
    assert hashlib.file_digest(archive.open('rb'), 'md5').hexdigest() == 'e88745ca2cfd25ef0b152f5fa3ed6bd9'
    report = {}; sets = {}
    with zipfile.ZipFile(archive) as z:
        for split in ('train', 'valid'):
            counts = Counter(); unique_counts = Counter(); seen = set(); bad = 0
            with z.open(f'dataset/{split}.xyz') as f:
                for i, (meta, cols) in enumerate(read_extxyz(io.TextIOWrapper(f))):
                    cell = np.asarray(meta['Lattice'].split(), float).reshape(3, 3)
                    assert abs(np.linalg.det(cell)) > 1e-6
                    assert cols['pos'].shape == cols['ref_forces'].shape
                    assert np.isfinite(float(meta['ref_energy']))
                    chem = '-'.join(sorted(set(cols['species'])))
                    assert set(cols['species']) <= {'Al', 'Cu', 'O'}
                    key = hashlib.sha256('|'.join(cols['species']).encode()+cell.tobytes()+cols['pos'].tobytes()).hexdigest()
                    counts[chem] += 1
                    if key not in seen: unique_counts[chem] += 1
                    seen.add(key)
                    if (i+1) % 25000 == 0: print(split, i+1, flush=True)
            sets[split] = seen
            report[split] = {'frames': sum(counts.values()), 'unique_exact_geometries': len(seen),
                             'chemical_systems': dict(counts), 'unique_chemical_systems': dict(unique_counts)}
    report['cross_split_exact_geometry_overlap'] = len(sets['train'] & sets['valid'])
    report['union_unique_exact_geometries'] = len(sets['train'] | sets['valid'])
    report['source'] = 'https://zenodo.org/records/15865800'
    report['classification'] = 'Mixed DFT-labelled training snapshots; individual AIMD identity and chronology absent'
    report['training_enabled'] = False
    report['label_settings'] = 'VASP PBE, 400 eV; source PAW labels and inputs in unialcu_vasp_inputs.zip'
    report['dedup_limit'] = 'Exact geometry only, not rotation/permutation/translation invariant'
    write_json('reports/unialcu_audit.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
