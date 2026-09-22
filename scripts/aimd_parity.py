"""Parity on every frozen external AIMD geometry; no dynamics rerun or fitting."""
import json
from pathlib import Path
import numpy as np
from scripts.parity_gallery import read, sha, predict, render, joined, OUT, CACHE

def main():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    protocol=read('reports/aimd_comparison/protocol.json')
    manifest=read('reports/aimd_manifest.json')
    assert sha('reports/aimd_manifest.json')==protocol['reference_manifest_sha256']
    report={'checkpoint_sha256':protocol['checkpoint_sha256'],'selection':'All stored reference frames and all force components; frozen model on reference geometries.',
        'energy':'Each trajectory and each method separately referenced to its first-frame energy; this removes an arbitrary constant, is not a fitted slope, and does not test absolute energy accuracy.',
        'limitation':'CP2K/PBE reference versus r2SCAN model; cross-method surface transfer diagnostic, not matched-Hamiltonian validation.', 'cases':{}}
    parts=[]
    for name,meta in manifest['cases'].items():
        assert sha(meta['path'])==protocol['reference_sha256'][name]
        d=predict(protocol['checkpoint'],protocol['checkpoint_sha256'],meta['path'])
        raw_mae=float(np.abs(d['ep']-d['er']).mean())
        d={k:v.copy() for k,v in d.items()}
        d['er']-=d['er'][0];d['ep']-=d['ep'][0]
        metrics=render('aimd_'+name,name+' | all reference geometries | PBE to r2SCAN transfer',{'External AIMD snapshots':d},'Relative energy (eV/atom)')
        report['cases'][name]={'frames':len(d['er']),'reference_sha256':sha(meta['path']),'raw_energy_mae_eV_atom_not_comparable_zero':raw_mae,'parity':metrics}
        parts.append(d);print(name,metrics,flush=True)
    report['overall']=render('aimd_overall','External Si AIMD | three trajectories | PBE to r2SCAN transfer',{'All reference snapshots':joined(parts)},'Relative energy (eV/atom)')
    report['complete']=True
    Path('reports/aimd_comparison/parity.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# AIMD reference parity','','[Trajectory animations and structural statistics](results.md#external-aimd-comparison) | [All train/test parity](parity.md)','','The frozen shared model is evaluated on **every stored reference geometry**: 101 frames per trajectory, 303 frames total, and 196,344 signed Cartesian force components. These are external tests, not training data. No predictions are compared between independently evolved trajectories.','','**Energy convention:** each method is separately referenced to its own first-frame energy within each trajectory. The plot tests energy changes; it does not establish absolute energy agreement. The first point is zero by construction. Raw energy errors are retained in the report but have incompatible reference zeros. **CP2K/PBE reference versus r2SCAN model:** both force and relative-energy plots are cross-method transfer diagnostics.','','![Overall AIMD parity](assets/parity/aimd_overall.png)','','[Overall PDF](assets/parity/aimd_overall.pdf)','','## Each reference trajectory','']
    for name in manifest['cases']:
        lines += [f'### {name}', '',f'![{name} parity](assets/parity/aimd_{name}.png)','',f'[PDF](assets/parity/aimd_{name}.pdf)','']
    lines+=['## Interpretation','','The dashed line denotes agreement, not a fitted regression. MAE/RMSE include all plotted points. The earlier trajectory report evaluated every fifth stored frame; this report uses all frames, so the force numbers can differ. Temporal correlation means 303 frames are not 303 independent experiments. These 500 fs surface trajectories cannot establish diffusion or molten-metal suitability.','','The existing [TM23 specialist parity figures](parity.md#independent-specialists) already compare cold, warm and molten AIMD-derived DFT snapshots. They are distinct from these verified continuous Si references and are not pooled with them.','','[Metrics and frozen hashes](../reports/aimd_comparison/parity.json) | [Reference source](https://doi.org/10.6084/m9.figshare.29422061.v1)']
    overall=report['overall']['All reference snapshots']
    lines+=['', f"Overall force MAE is **{overall['fr']['mae']:.4f} eV/A**; relative-energy MAE is **{1000*overall['er']['mae']:.2f} meV/atom**. These discrepancies do not support a quantitative accuracy claim for these surfaces."]
    Path('docs/aimd_parity.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
