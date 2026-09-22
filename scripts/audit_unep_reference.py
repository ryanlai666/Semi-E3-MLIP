"""Audit the published UNEP-v1 DFT tests, without using trained NEP weights."""
import io,json,hashlib,zipfile,urllib.request
from collections import Counter
from pathlib import Path
import numpy as np
from semi_mlip.extxyz import read_extxyz

def main():
    metadata=Path('data/raw/unep_source.json');metadata.parent.mkdir(parents=True,exist_ok=True)
    if not metadata.exists():
        with urllib.request.urlopen('https://zenodo.org/api/records/11533864',timeout=60) as response:
            metadata.write_bytes(response.read())
    source=json.loads(metadata.read_text())
    for name in ('testset.zip','INCAR','readme.txt'):
        entry=next(f for f in source['files'] if f['key']==name)
        target=Path('data/raw/unep')/name;target.parent.mkdir(exist_ok=True)
        if not target.exists():
            with urllib.request.urlopen(entry['links']['self'],timeout=60) as response:
                content=response.read()
            assert 'md5:'+hashlib.md5(content).hexdigest()==entry['checksum']
            target.write_bytes(content)
        assert 'md5:'+hashlib.md5(target.read_bytes()).hexdigest()==entry['checksum']
    archive=Path('data/raw/unep/testset.zip');f=next(f for f in source['files'] if f['key']=='testset.zip')
    assert 'md5:'+hashlib.md5(archive.read_bytes()).hexdigest()==f['checksum']
    supported={'Al','Si','Cu','Ti','Co','W','Ta','Ru','Hf','Zr','O'}
    files={};total=Counter();usable=Counter()
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if not name.endswith('.xyz'):continue
            counts=Counter();n_supported=0;fields=set();atoms=Counter()
            with z.open(name) as binary:
                for meta,cols in read_extxyz(io.TextIOWrapper(binary)):
                    symbols=set(cols['species']);chem='-'.join(sorted(symbols));counts[chem]+=1;total[chem]+=1
                    assert 'energy' in {k.lower() for k in meta} and ('force' in cols or 'forces' in cols)
                    assert np.isfinite(float(next(v for k,v in meta.items() if k.lower()=='energy')))
                    cell=np.fromstring(meta['Lattice'],sep=' ').reshape(3,3)
                    assert np.isfinite(cell).all() and abs(np.linalg.det(cell))>1e-6
                    fields.update(meta);atoms[len(cols['species'])]+=1
                    if symbols<=supported:usable[chem]+=1;n_supported+=1
            files[name]={'frames':sum(counts.values()),'systems':dict(counts),'supported_frames':n_supported,'atom_counts':dict(atoms),'metadata_fields':sorted(fields)}
    report={'source':'https://doi.org/10.5281/zenodo.11533864','paper':'https://www.nature.com/articles/s41467-024-54554-x',
       'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_md5':f['checksum'],
       'calculation_settings':Path('data/raw/unep/INCAR').read_text(),'frames':sum(total.values()),'systems':dict(total),'supported_systems':dict(usable),'files':files,
       'status':'Downloaded and schema-audited; not trained on or evaluated by Semi-E3-MLIP.',
       'limitations':['PBE versus current r2SCAN checkpoint: external transfer only.','Heating/deformation snapshots do not certify a continuous AIMD trajectory.','Before evaluation freeze a supported-chemistry protocol and verify provenance, overlap, units and energy references.','No pretrained NEP weights used.']}
    Path('reports/unep_reference_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Total',report['frames'],'supported',sum(usable.values()),dict(usable))

if __name__=='__main__':main()
