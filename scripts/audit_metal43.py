"""Acquire and inventory Metal-43 priority trajectories; no fitting/evaluation."""
import io,json,hashlib,urllib.request,zipfile
from pathlib import Path
from semi_mlip.extxyz import read_extxyz

def main():
    root=Path('data/raw/metal43');root.mkdir(parents=True,exist_ok=True)
    metadata=Path('data/raw/metal43_metadata.json')
    if not metadata.exists():
        with urllib.request.urlopen('https://archive.materialscloud.org/api/records/desfh-3ma25',timeout=60) as r:metadata.write_bytes(r.read())
    meta=json.loads(metadata.read_text());files={}
    for name in ['Readme.md','Metal-43-Dataset.zip']:
        info=meta['files']['entries'][name];target=root/name
        if not target.exists():
            tmp=target.with_suffix(target.suffix+'.part')
            with urllib.request.urlopen(info['links']['content'],timeout=60) as r,tmp.open('wb') as w:
                while block:=r.read(8*1024*1024):w.write(block)
            with tmp.open('rb') as r:assert 'md5:'+hashlib.file_digest(r,'md5').hexdigest()==info['checksum']
            tmp.replace(target)
        with target.open('rb') as r:assert 'md5:'+hashlib.file_digest(r,'md5').hexdigest()==info['checksum']
        with target.open('rb') as r:files[name]={'md5':info['checksum'],'sha256':hashlib.file_digest(r,'sha256').hexdigest(),'bytes':target.stat().st_size}
    inventory={}
    with zipfile.ZipFile(root/'Metal-43-Dataset.zip') as z:
        for name in sorted(z.namelist()):
            if Path(name).name.split('_')[0] not in ['Ti','Ta','Ru','Al','Hf'] or not name.endswith('.extxyz'):continue
            count=0;fields=set();species=set();atoms=set()
            with z.open(name) as binary:
                for meta,cols in read_extxyz(io.TextIOWrapper(binary)):
                    count+=1;fields.update(meta);species.update(cols['species']);atoms.add(len(cols['species']))
            inventory[name]={'frames':count,'elements':sorted(species),'atom_counts':sorted(atoms),'metadata_fields':sorted(fields)}
    result={'source':'https://doi.org/10.24435/materialscloud:hm-6z','paper':'https://www.nature.com/articles/s41524-026-01977-3','files':files,'priority_inventory':inventory,'status':'Downloaded, checksummed and inventoried only; no model predictions or training use.','limitations':['Audit DFT functional, pseudopotentials, smearing, magnetism and units before comparisons.','Do not assume cells above the melting temperature have actually melted; inspect structural statistics.','Temperature files contain correlated snapshots, not independent replicates.','Do not fit an energy shift on the final test set.']}
    Path('reports/physics_research/metal43_audit.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v['frames'] for k,v in inventory.items()})

if __name__=='__main__':main()
