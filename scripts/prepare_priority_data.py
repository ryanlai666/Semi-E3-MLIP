"""Prepare new priority development data; never alter frozen study partitions."""
import gzip,hashlib,json
from collections import Counter
from pathlib import Path
from semi_mlip.data import read_jsonl,convert,fingerprint,assign_splits

ROOT=Path('data/priority_v1')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_parts(name,parts):
    folder=ROOT/name;folder.mkdir(parents=True,exist_ok=True);result={}
    for split,rows in parts.items():
        text=''.join(json.dumps({**r,'split':split})+'\n' for r in rows);p=folder/(split+'.jsonl')
        if p.exists():assert p.read_text()==text,'Frozen priority dataset changed'
        else:p.write_text(text)
        result[split]={'path':str(p),'frames':len(rows),'groups':len({r.get('group',r.get('parent')) for r in rows}),'sha256':sha(p),'systems':dict(Counter(r['chemsys'] for r in rows))}
    ids=[{r['id'] for r in rows} for rows in parts.values()]
    for i,a in enumerate(ids):
        for b in ids[i+1:]:assert not a&b
    return result

def main():
    report={'selection':'No prediction-error filtering. New manifests; original partitions remain unchanged.','studies':{}}
    for metal in ['Ti','Ru','Ta']:
        source=Path('data/source_isolated/tm23');cold=list(read_jsonl(source/f'{metal}_cold_train.jsonl'))
        old=Path('data/focused/ti_cold_900' if metal=='Ti' else f'data/material_studies/tm23_{metal.lower()}')
        warm=list(read_jsonl(source/f'{metal}_warm_train.jsonl'));melt=list(read_jsonl(source/f'{metal}_melt_train.jsonl'))
        warmvalid=list(read_jsonl(old/'valid.jsonl'));validids={r['id'] for r in warmvalid}
        # Stable ID selection; temporal ordering is not certified by source.
        meltvalid=sorted(melt,key=lambda r:hashlib.sha256(r['id'].encode()).digest())[:90]
        validids|={r['id'] for r in meltvalid}
        parts={'train':cold+[r for r in warm+melt if r['id'] not in validids],'valid':warmvalid+meltvalid}
        for regime in ['cold','warm','melt']:parts['test_'+regime]=list(read_jsonl(source/f'{metal}_{regime}_test.jsonl'))
        report['studies']['mixed_'+metal.lower()]={'label_family':'PBE nonspin TM23','partitions':write_parts('mixed_'+metal.lower(),parts),'limitation':'Original tests already inspected and share source trajectories. Adding molten training changes this to within-temperature validation, not blind molten extrapolation. ID holdouts are not independent trajectories.'}
    existing={s:list(read_jsonl(Path('data/device_expanded')/(s+'.jsonl'))) for s in ['train','valid','test','aloe_valid','aloe_test']}
    protectedparents={r['parent'] for rows in existing.values() for r in rows if r['chemsys']=='O-Ta'}
    seen={r.get('fingerprint') or fingerprint(r) for rows in existing.values() for r in rows if r['chemsys']=='O-Ta'}
    additions=[];counts=Counter();source=Path('data/raw/MP_ALOE_data.jsonl.gz')
    with source.open('rb') as f:assert hashlib.file_digest(f,'md5').hexdigest()=='0cac41b76cdc936848d360a14bdc9bf2'
    with gzip.open(source,'rt') as f:
        for line in f:
            raw=json.loads(line)
            if set(raw['elements'])!={'Ta','O'}:continue
            counts['raw_ta_o']+=1
            if raw['provenance'].get('original_mp_id') or raw.get('prototype_number') is None:continue
            parent=f"aloe-prototype-{raw['prototype_number']}:{raw['formula_pretty']}"
            if parent in protectedparents:counts['known_parent_excluded']+=1;continue
            raw['matpes_id']=raw['mp_aloe_id'];raw['provenance']={**raw['provenance'],'original_mp_id':parent}
            try:r=convert(raw)
            except ValueError:counts['conversion_excluded']+=1;continue
            key=fingerprint(r)
            if key in seen:counts['duplicate_excluded']+=1;continue
            seen.add(key);r.update(source='MP-ALOE-r2SCAN',fingerprint=key);additions.append(r)
    assign_splits(additions)
    parts={'train':existing['train']+[r for r in additions if r['split']=='train'],
           'valid':existing['valid']+[r for r in additions if r['split']=='valid'],
           'new_ta_o_test':[r for r in additions if r['split']=='test'],
           'historical_test':existing['test'],'historical_aloe_test':existing['aloe_test']}
    groups={s:{r['group'] for r in rows} for s,rows in parts.items()}
    for i,(a,aa) in enumerate(groups.items()):
        for b in list(groups)[i+1:]:assert not aa&groups[b],(a,b)
    report['studies']['shared_ta_o_expanded']={'label_family':'r2SCAN','partitions':write_parts('shared_ta_o_expanded',parts),'counts':dict(counts),'added':dict(Counter(r['split'] for r in additions)),'added_formulas':dict(Counter(r['formula'] for r in additions)),'limitation':'New Ta-O parent groups exclude known Ta-O parents. Composition/prototype grouping does not establish complete structural independence. Historical tests remain post-hoc diagnostics.'}
    out=Path('reports/physics_research');out.mkdir(exist_ok=True)
    target=out/'priority_data.json'
    if target.exists():assert json.loads(target.read_text())==report
    else:target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{s:p['frames'] for s,p in v['partitions'].items()} for k,v in report['studies'].items()},indent=2))

if __name__=='__main__':main()
