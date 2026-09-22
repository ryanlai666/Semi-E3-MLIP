"""Download two access samples for AFLOW Al-Si; do not claim a complete hull."""
import urllib.request,json,hashlib
from pathlib import Path

def main():
    base='https://aflowlib.duke.edu/AFLOWDATA/LIB2_RAW/AlSi/'
    report={'source':base,'scope':'Reference access audit only; not a phase diagram or model evaluation.','records':[]}
    for entry in ['281','282']:
        folder=Path('data/raw/aflow_alsi')/entry;folder.mkdir(parents=True,exist_ok=True)
        for name in ['aflowlib.json','CONTCAR.relax']:
            target=folder/name
            if not target.exists():
                with urllib.request.urlopen(base+entry+'/'+name,timeout=30) as r:target.write_bytes(r.read())
        d=json.loads((folder/'aflowlib.json').read_text())
        report['records'].append({'entry':entry,'url':base+entry+'/',**{k:d.get(k) for k in ['auid','compound','dft_type','energy_atom','enthalpy_formation_atom','species_pp','species_pp_version','pressure_residual','delta_electronic_energy_convergence']},'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}})
    report['requirements']=['Inventory all relevant competing structures and matched elemental endpoints.','Verify relaxation/convergence and energy-reference consistency.','PBE is not directly matched to the r2SCAN model.','Keep source data separate from repository code licensing.']
    Path('reports/aflow_alsi_reference_audit.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
