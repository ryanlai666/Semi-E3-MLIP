"""Inventory AIMD source groups from the original n2p2 text; no unit assumptions."""
from collections import Counter
import io
import zipfile
import numpy as np
from semi_mlip.data import write_json


def main():
    report = {}; all_groups = {}
    with zipfile.ZipFile('data/raw/alsi_2025.zip') as z:
        for split in ('train','test'):
            counts = Counter(); groups = Counter(); atom_counts = Counter(); active = False
            with z.open(f'dataset-AlSi/{split}.data') as binary:
                for line in io.TextIOWrapper(binary):
                    fields = line.split()
                    if not fields: continue
                    if fields[0] == 'begin':
                        assert not active
                        active = True; species = []; lattice = []; energy = None; group = None
                    elif fields[0] == 'comment':
                        group = next(x.split('=',1)[1] for x in fields if x.startswith('source_file_name='))
                    elif fields[0] == 'atom':
                        assert len(fields) == 10 and np.isfinite([float(x) for x in fields[1:4]+fields[5:]]).all()
                        species.append(fields[4])
                    elif fields[0] == 'lattice': lattice.append([float(x) for x in fields[1:]])
                    elif fields[0] == 'energy': energy = float(fields[1])
                    elif fields[0] == 'end':
                        assert active and group and energy is not None and np.isfinite(energy)
                        assert np.asarray(lattice).shape == (3,3) and abs(np.linalg.det(lattice))>1e-6
                        counts['-'.join(sorted(set(species)))] += 1
                        groups[group] += 1; atom_counts[len(species)] += 1; active = False
            assert not active
            report[split] = {'frames': sum(counts.values()), 'chemical_systems': dict(counts),
                             'atom_counts': dict(atom_counts), 'source_path_groups': dict(groups)}
            all_groups[split] = set(groups)
    report['source_path_overlap'] = len(all_groups['train']&all_groups['test'])
    report['source'] = 'https://doi.org/10.24435/materialscloud:3h-sc'
    report['training_enabled'] = False
    report['limitations'] = ['Units require verification before conversion.',
                             'Relative source paths can collide between compositions; combine with composition/cell.',
                             'Original train/test share AIMD source families; use grouped temperature holdouts.']
    write_json('reports/alsi_alloy_audit.json', report)
    print({k:v['frames'] for k,v in report.items() if isinstance(v,dict)})


if __name__ == '__main__':
    main()
