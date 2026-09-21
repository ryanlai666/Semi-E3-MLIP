"""Recover complete silica AIMD position/energy references and match sparse forces."""
import io
import json
from pathlib import Path
import re
import zipfile
import numpy as np
from semi_mlip.extxyz import read_extxyz
from semi_mlip.data import ELEMENTS, write_json


def main():
    root = Path('data/aimd_silica_reference'); root.mkdir(exist_ok=True)
    cases = {}; rejected = {}
    with zipfile.ZipFile('data/raw/Si_SiO2.zip') as z:
        for member in z.namelist():
            if '/SiO2/surf/' not in member or not member.endswith('/traj.xyz'): continue
            prefix = member[:-8]
            inp = z.read(prefix+'input.inp').decode()
            output = z.read(prefix+'output').decode()
            assert 'RUN_TYPE MD' in inp and 'ENSEMBLE NVT' in inp
            assert not re.search(r'SCF.*NOT converged', output, re.I)
            energies = np.asarray([float(x) for x in re.findall(r'ENERGY\| Total FORCE_EVAL.*?:\s+([-\d.]+)', output)])*27.211386245988
            frames = list(read_extxyz(io.StringIO(z.read(member).decode())))
            steps = [int(x) for x in re.findall(r'MD\| Step number\s+(\d+)', output)]
            if not (steps == list(range(1,len(frames))) and len(frames) == len(energies) == 101):
                rejected[prefix] = {'reason': 'position/energy/step count mismatch',
                                    'positions': len(frames), 'energies': len(energies), 'steps': len(steps)}
                continue
            dt = float(re.search(r'TIMESTEP\s+(\S+)',inp)[1])
            temp = float(re.search(r'\n\s*TEMPERATURE\s+(\S+)',inp)[1])
            def array(name):
                return np.load(io.BytesIO(z.read(prefix+'dpgen.init/set.000/'+name+'.npy')), allow_pickle=False)
            coords = array('coord'); e = array('energy').reshape(-1); forces = array('force')
            positions = np.asarray([cols['pos'] for meta, cols in frames])
            labels = {}
            for j, point in enumerate(coords):
                differences = np.max(np.abs(positions-point.reshape(-1,3)),axis=(1,2))
                matches = np.flatnonzero(differences < 2e-7)
                assert len(matches) == 1
                # Float64 storage contains values quantized to float32 upstream.
                assert float(e[j]) == float(np.float32(e[j]))
                k = int(matches[0])
                tolerance = max(2e-5, .51*abs(float(np.spacing(np.float32(e[j])))))
                np.testing.assert_allclose(energies[k], e[j], atol=tolerance, rtol=0)
                labels[k] = forces[j].reshape(-1,3).tolist()
            parts = prefix.rstrip('/').split('/'); name = 'SiO2_'+parts[-2]+'_'+parts[-1]
            with (root/f'{name}.jsonl').open('w') as out:
                for k, (meta, cols) in enumerate(frames):
                    row = {'id': f'{name}-{k}', 'z': [ELEMENTS[s] for s in cols['species']],
                           'positions': cols['pos'].tolist(), 'cell': np.asarray(meta['Lattice'].split(),float).reshape(3,3).tolist(),
                           'pbc': [True]*3, 'energy': float(energies[k]), 'forces': labels.get(k), 'stress': None,
                           'group': name, 'split': 'external', 'formula': 'SiO2', 'chemsys': 'O-Si',
                           'time_fs': k*dt, 'temperature_target_K': temp,
                           'label_source': 'CP2K PBE AIMD; energy from output, sparse matched forces from NPY'}
                    out.write(json.dumps(row)+'\n')
            cases[name] = {'frames': len(frames), 'force_labelled_frames': len(labels), 'dt_fs': dt,
                            'duration_fs': steps[-1]*dt, 'temperature_K': temp, 'source_prefix': prefix,
                            'path': str(root/f'{name}.jsonl')}
    write_json('reports/silica_reference_manifest.json', {'cases': cases, 'rejected': rejected,
        'source': 'https://doi.org/10.6084/m9.figshare.29422061.v1', 'training_use': False,
        'limitation': '5 fs integration, 500 fs per trajectory, no velocities, sparse force labels; no transport-coefficient claim'})
    print('cases', len(cases), 'frames', sum(x['frames'] for x in cases.values()),
          'force labels', sum(x['force_labelled_frames'] for x in cases.values()))


if __name__ == '__main__':
    main()
