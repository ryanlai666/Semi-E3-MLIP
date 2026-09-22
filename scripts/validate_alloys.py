"""Frozen shared-model verification on previously unused r2SCAN binary chemistry."""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from semi_mlip.data import read_jsonl,write_json,convert,fingerprint,ELEMENTS
from semi_mlip.train import evaluate_checkpoint
from parity_gallery import predict,render,compare_metrics,OUT,CACHE

SYSTEMS=('Al-Si','Al-Ti','Cu-Zr','Hf-Zr','Ta-W','Ti-W')
ROOT=Path('reports/alloy_validation');DATA=Path('data/alloy_validation')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lock(path,value):
    if path.exists():assert read(path)==value,'Frozen alloy protocol changed'
    else:write_json(path,value)

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    selected=read('reports/aimd_comparison/selection.json');archive=Path('data/raw/MP_ALOE_data.jsonl.gz')
    protocol={'systems':list(SYSTEMS),'source':'https://doi.org/10.6084/m9.figshare.29452190.v2',
        'paper':'https://www.nature.com/articles/s41524-025-01834-9','archive_md5':'0cac41b76cdc936848d360a14bdc9bf2',
        'checkpoint':selected['checkpoint'],'checkpoint_sha256':selected['checkpoint_sha256'],
        'selection':'All valid non-MP-sourced records in six declared binary chemical systems; composition-only scope, no prediction/energy/force filtering.',
        'role':'Untouched chemistry holdout for the existing shared checkpoint. No fitting, fine-tuning, checkpoint selection or new hyperparameter search.',
        'classification':'Off-equilibrium r2SCAN DFT snapshots, not verified continuous AIMD or equilibrium phase-diagram data.'}
    lock(ROOT/'protocol.json',protocol)
    if (ROOT/'data_manifest.json').exists():
        manifest=read(ROOT/'data_manifest.json')
        for system,p in manifest['systems'].items():assert sha(p['path'])==p['sha256']
        return protocol,manifest
    assert hashlib.file_digest(archive.open('rb'),'md5').hexdigest()==protocol['archive_md5']
    known=[]
    for split in ('train','valid','test','aloe_valid','aloe_test'):known.extend(read_jsonl(f'data/device_expanded/{split}.jsonl'))
    assert not set(SYSTEMS)&{r['chemsys'] for r in known}
    seen={r.get('fingerprint') or fingerprint(r) for r in known};ids={r['id'] for r in known}
    rows={s:[] for s in SYSTEMS};counts=Counter()
    with gzip.open(archive,'rt') as f:
        for line in f:
            raw=json.loads(line);counts['scanned']+=1
            system='-'.join(sorted(raw['elements']))
            if system not in rows:continue
            counts['composition_candidates']+=1
            if raw['provenance'].get('original_mp_id'):counts['mp_sourced_excluded']+=1;continue
            if raw.get('prototype_number') is None:counts['missing_prototype']+=1;continue
            assert str(raw['functional']).lower().replace('-','')=='r2scan',raw['functional']
            raw['matpes_id']=raw['mp_aloe_id'];parent=f"aloe-prototype-{raw['prototype_number']}:{raw['formula_pretty']}"
            raw['provenance']={**raw['provenance'],'original_mp_id':parent}
            try:r=convert(raw,allow_alloys=True)
            except ValueError as exc:counts['invalid_'+str(exc)]+=1;continue
            fp=fingerprint(r)
            if fp in seen or r['id'] in ids:counts['duplicate_excluded']+=1;continue
            seen.add(fp);ids.add(r['id'])
            r.update(source='MP-ALOE-r2SCAN',group=parent,split='alloy_holdout',fingerprint=fp,
                provenance={'prototype_number':raw['prototype_number'],'ionic_step_number':raw['ionic_step_number'],'original_mp_id':None})
            rows[system].append(r)
    manifest={'counts':dict(counts),'systems':{},'training_chemistry_overlap':[],
        'limitations':['Chemistry-disjoint from training, but same MP-ALOE dataset family; not independent DFT methodology.',
            'Prototype/ionic-step snapshots are correlated; configuration counts are not trajectory replicates.',
            'Geometry deduplication does not prove absence of structural similarity.',
            'No verified fully relaxed ground-state set or matched elemental endpoints; cannot establish an equilibrium convex hull.']}
    for system,items in rows.items():
        assert items
        items.sort(key=lambda r:r['id']);p=DATA/(system.replace('-','_')+'.jsonl')
        p.write_text(''.join(json.dumps(r)+'\n' for r in items))
        manifest['systems'][system]={'path':p.as_posix(),'sha256':sha(p),'frames':len(items),'parent_groups':len({r['group'] for r in items}),
            'formulas':dict(Counter(r['formula'] for r in items))}
    write_json(ROOT/'data_manifest.json',manifest)
    print('Alloy import',json.dumps(manifest['counts']),flush=True)
    return protocol,manifest

def main():
    protocol,manifest=prepare();metrics={};predictions={}
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    assert sha(protocol['checkpoint'])==protocol['checkpoint_sha256']
    for system,part in manifest['systems'].items():
        out=ROOT/(system.replace('-','_')+'.json')
        metric=read(out) if out.exists() else evaluate_checkpoint(protocol['checkpoint'],part['path'],out,device='cuda',atom_budget=1024,edge_budget=64000)
        assert not metric['excluded'];metrics[system]=metric['overall']
        d=predict(protocol['checkpoint'],protocol['checkpoint_sha256'],part['path']);compare_metrics(d,metric['overall'])
        predictions[system]=d
        render('alloy_'+system.replace('-','_').lower(),f'Unseen {system} chemistry | shared checkpoint | r2SCAN DFT',{'Alloy holdout':d})
        print(system,json.dumps(metric['overall']),flush=True)
    from parity_gallery import joined
    overall=render('alloy_overall','Shared model | six unseen binary systems | r2SCAN DFT',{'Alloy holdout':joined(list(predictions.values()))})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax,(system,part) in zip(axes.flat,manifest['systems'].items()):
        rows=list(read_jsonl(part['path']));byid={r['id']:r for r in rows};d=predictions[system];second=system.split('-')[1]
        fraction=[byid[i]['z'].count(ELEMENTS[second])/len(byid[i]['z']) for i in d['id']]
        error=(d['ep']-d['er'])*1000
        ax.scatter(fraction,error,s=8,alpha=.4);ax.axhline(0,color='black',lw=.8);ax.axhspan(-10,10,color='#328c70',alpha=.15)
        ax.set(title=system,xlabel=f'Atomic fraction {second}',ylabel='Energy error (meV/atom)',xlim=(0,1));ax.grid(alpha=.15)
    fig.suptitle('Composition-resolved energy errors | not an equilibrium phase diagram')
    for ext in ('png','pdf'):fig.savefig(OUT/f'alloy_composition_errors.{ext}',dpi=160)
    plt.close(fig)
    write_json(ROOT/'summary.json',{'complete':True,'metrics':metrics,'checkpoint_sha256':protocol['checkpoint_sha256'],
        'not_phase_diagram':True,'not_continuous_aimd':True,'overall_parity':overall,'alloy_holdout_frames':sum(v['frames'] for v in metrics.values())})
    lines=['# Alloy verification and phase-stability limits','','[Material suitability](material_studies.md) | [Train/test parity gallery](parity.md)','','## What was tested','',
        'The existing shared checkpoint was evaluated without retraining on six binary chemistries absent from its elemental/oxide training and validation sets. All eligible, deduplicated configurations in the declared systems were used. These are off-equilibrium r2SCAN DFT snapshots from [MP-ALOE v2](https://doi.org/10.6084/m9.figshare.29452190.v2), not verified continuous alloy AIMD. The [source paper](https://www.nature.com/articles/s41524-025-01834-9) describes the dataset generation.', '',
        'This is a chemistry-transfer test within the same dataset family, not an independent DFT-method benchmark. Model and chemistry scope were frozen before predictions. MP-derived records and exact duplicate geometries were excluded. No alloy labels entered training or selection.', '',
        '| Binary system | Frames / parent groups | Energy MAE (meV/atom) | Force MAE / RMSE (eV/A) | Both project targets met? | Parity |',
        '| --- | ---: | ---: | ---: | --- | --- |']
    for system,m in metrics.items():
        part=manifest['systems'][system];passed=m['energy_mae_eV_atom']<=.01 and m['force_mae_eV_A']<=.1
        name='alloy_'+system.replace('-','_').lower()
        lines.append(f'| {system} | {part["frames"]} / {part["parent_groups"]} | {m["energy_mae_eV_atom"]*1000:.2f} | {m["force_mae_eV_A"]:.3f} / {m["force_rmse_eV_A"]:.3f} | {"Yes, on these samples only" if passed else "No"} | [PNG](assets/parity/{name}.png) / [PDF](assets/parity/{name}.pdf) |')
    passed=sum(m['energy_mae_eV_atom']<=.01 and m['force_mae_eV_A']<=.1 for m in metrics.values())
    zero_comparison='; '.join(f'{system}: model {m["force_mae_eV_A"]:.3f} versus zero-force {m["zero_force_baseline_mae_eV_A"]:.3f} eV/A' for system,m in metrics.items())
    lines+=['', '**Trivial-force baseline:** '+zero_comparison+'. Cu-Zr is worse than predicting zero forces; improvements over that baseline are limited for several other systems. This prevents interpreting a plausible-looking parity cloud as sufficient evidence of accuracy.']
    lines+=['',f'**{passed} of six systems meet both working targets** (10 meV/atom and 0.1 eV/A). Passing a snapshot test would still not validate long MD or phase stability. Large energy errors directly weaken a phase-energy claim; force accuracy alone cannot establish correct phase ordering.', '',
        '![Alloy parity](assets/parity/alloy_overall.png)','','## Can this validate a phase diagram?','',
        '**No equilibrium phase diagram is established by these records.** A zero-temperature convex-hull comparison needs consistently calculated, sufficiently relaxed competing structures over composition plus matched elemental ground-state references. This off-equilibrium subset does not certify that set. Taking its lowest sampled energies would produce a sample-dependent lower envelope, not a verified ground-state phase diagram.', '',
        'For the distinction between a DFT formation-energy hull and a finite-temperature phase diagram, see the [Materials Project phase-diagram methodology](https://docs.materialsproject.org/methodology/materials-methodology/thermodynamic-stability/phase-diagrams-pds).', '',
        'A finite-temperature phase diagram additionally needs free-energy differences, configurational/vibrational contributions, and sampling of relevant phases. Neither a static parity plot nor short energy-conserving MD supplies that evidence. The diagnostic below shows composition-dependent prediction bias; it is explicitly not a phase diagram.', '',
        '![Energy errors across composition](assets/parity/alloy_composition_errors.png)', '', 'Energy error is prediction minus DFT. The green band marks +/-10 meV/atom, the project screening target; it is not a phase boundary.','','## Other alloy/AIMD references','',
        '- [Al-Si nucleation dataset](https://doi.org/10.24435/materialscloud:3h-sc): genuine AIMD-derived elemental and alloy configurations already audited locally, but LDA labels and unresolved text-unit confirmation prevent treating it as a directly matched r2SCAN validation set. Source-path overlap also requires grouped holdouts.',
        '- The Al/Si interface collection is retired as an alloy benchmark candidate. [Replacement references](alloy_references.md): downloaded UNEP alloy DFT tests and verified access to AFLOW relaxed Al-Si structures, with provenance and limitations.', '',
        'Next phase-stability work should use a dedicated matched-fidelity set of relaxed elemental and competing alloy structures, then compare formation energies, relative phase ordering and hull membership without fitting to those tests. First address any large errors found here.', '',
        '[Frozen protocol, source counts, hashes and full metrics](../reports/alloy_validation).']
    Path('docs/alloy_validation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
