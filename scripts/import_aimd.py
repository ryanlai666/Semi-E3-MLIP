"""Read public CP2K trajectories with stdlib/NumPy; no external MLIP code."""
import io
import hashlib
import json
from pathlib import Path
import re
import shlex
import zipfile
import argparse
import numpy as np
from semi_mlip.data import write_json

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--all-si-surfaces', action='store_true')
    args=parser.parse_args()
    archive=Path('data/raw/Si_SiO2.zip')
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest!='55a97cab4a3a61c251bf61cefcce0e3990badc67089b1e6a33609560f8660994':
        raise ValueError('Unexpected source archive')
    z=zipfile.ZipFile(archive)
    root='Si_SiO2/s4-dpgen/input/data.init/Si/surf/gendata/'
    cases={}
    selection=([(s,e) for s in ('110','111') for e in ('0.500','1.000','1.500','2.000','4.000','6.000')]
               if args.all_si_surfaces else [('110','0.500'),('111','0.500'),('110','1.500')])
    for surface,elong in selection:
        name=f'Si_{surface}_elong{elong}'
        prefix=root+f'surf-{surface}/elong-{elong}/'
        text=z.read(prefix+'input.inp').decode()
        output=z.read(prefix+'output').decode(errors='replace')
        assert 'RUN_TYPE MD' in text and 'ENSEMBLE NVT' in text
        dt=float(re.search(r'TIMESTEP\s+(\S+)',text)[1])
        temperature=float(re.search(r'\n\s*TEMPERATURE\s+(\S+)',text)[1])
        steps=[int(x) for x in re.findall(r'MD\| Step number\s+(\d+)',output)]
        coords=np.load(io.BytesIO(z.read(prefix+'data/set.000/coord.npy')))
        cells=np.load(io.BytesIO(z.read(prefix+'data/set.000/box.npy'))).reshape(-1,3,3)
        energies=np.load(io.BytesIO(z.read(prefix+'data/set.000/energy.npy')))
        forces=np.load(io.BytesIO(z.read(prefix+'data/set.000/force.npy')))
        types=np.loadtxt(io.BytesIO(z.read(prefix+'data/type.raw')),dtype=int,ndmin=1)
        symbols=z.read(prefix+'data/type_map.raw').decode().split()
        numbers=[{'Si':14,'O':8}[symbols[i]] for i in types]
        assert steps==list(range(1,len(coords)))
        xyz=io.StringIO(z.read(prefix+'traj.xyz').decode())
        rows=[]
        for k in range(len(coords)):
            n=int(xyz.readline()); header=dict(t.split('=',1) for t in shlex.split(xyz.readline()))
            positions=np.array([[float(x) for x in xyz.readline().split()[1:4]] for _ in range(n)])
            np.testing.assert_allclose(positions,coords[k].reshape(n,3),atol=2e-7)
            rows.append({'id':f'{name}-{k}','z':numbers,'positions':positions.tolist(),'cell':cells[k].tolist(),
                'pbc':[True]*3,'energy':float(energies[k]),'forces':forces[k].reshape(n,3).tolist(),
                'stress':None,'chemsys':'Si','formula':'Si','time_fs':k*dt,'label_source':'CP2K PBE AIMD',
                'group':name,'split':'external','temperature_target_K':temperature})
        directory=Path('data/aimd');directory.mkdir(exist_ok=True)
        path=directory/(name+'.jsonl')
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        cases[name]={'path':str(path),'frames':len(rows),'dt_fs':dt,'temperature_K':temperature,
                     'source_prefix':prefix,'duration_fs':steps[-1]*dt,'ensemble':'NVT Nose-Hoover',
                     'circumstance':f'Si ({surface}) surface, source elongation parameter {elong}',
                     'limitation':'500 fs, PBE reference versus r2SCAN model; no reference velocities'}
    manifest='reports/aimd_extended_manifest.json' if args.all_si_surfaces else 'reports/aimd_manifest.json'
    write_json(manifest,{'archive_sha256':digest,'source':'https://doi.org/10.6084/m9.figshare.29422061.v1',
               'cases':cases,'training_use':False})
    print(json.dumps(cases,indent=2))

if __name__=='__main__':
    main()
