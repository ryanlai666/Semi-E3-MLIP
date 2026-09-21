"""Add compatible off-equilibrium data, retaining the original validation set."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
from semi_mlip.data import convert, fingerprint, assign_splits, read_jsonl, write_json, ELEMENTS
from device_subset import included

def main():
    archive=Path('data/raw/MP_ALOE_data.jsonl.gz')
    meta=json.loads(Path('reports/mpaloe_metadata.json').read_text())
    source=next(f for f in meta['files'] if f['name']==archive.name)
    assert archive.stat().st_size==source['size']
    assert hashlib.md5(archive.read_bytes()).hexdigest()==source['computed_md5']
    existing={s:list(read_jsonl(Path('data/device')/(s+'.jsonl'))) for s in ('train','valid','test')}
    seen={r['fingerprint'] for rows in existing.values() for r in rows}
    candidates=[];counts=collections.Counter()
    with gzip.open(archive,'rt') as f:
        for line in f:
            raw=json.loads(line);counts['scanned']+=1
            if not set(raw['elements'])<=ELEMENTS.keys():continue
            # Exclude MP-derived entries, as recommended by the dataset authors.
            if raw['provenance'].get('original_mp_id'):
                counts['mp_sourced_excluded']+=1;continue
            if raw.get('prototype_number') is None:
                counts['missing_prototype']+=1;continue
            parent=f"aloe-prototype-{raw['prototype_number']}:{raw['formula_pretty']}"
            raw['matpes_id']=raw['mp_aloe_id']
            raw['provenance']={**raw['provenance'],'original_mp_id':parent}
            try:r=convert(raw)
            except ValueError:continue
            if not included(r):continue
            r['source']='MP-ALOE-r2SCAN';r['provenance']={'prototype_number':raw['prototype_number'],
                'ionic_step_number':raw['ionic_step_number'],'original_mp_id':None}
            r['fingerprint']=fingerprint(r)
            if r['fingerprint'] in seen:
                counts['duplicate_excluded']+=1;continue
            seen.add(r['fingerprint']);candidates.append(r)
    assign_splits(candidates)
    out=Path('data/device_expanded');out.mkdir(exist_ok=True)
    additions={s:[r for r in candidates if r['split']==s] for s in existing}
    # Model selection remains directly comparable on the same 145 MatPES frames.
    # ALOE development/test partitions are stored separately and never trained on.
    outputs={'train':existing['train']+additions['train'],'valid':existing['valid'],
             'test':existing['test'],'aloe_valid':additions['valid'],'aloe_test':additions['test']}
    for split,rows in outputs.items():
        (out/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report={'source':'https://doi.org/10.6084/m9.figshare.29452190','license':meta['license'],
            'md5':source['computed_md5'],'counts':dict(counts),'new_frames':len(candidates),
            'splits':{s:dict(collections.Counter(r['formula'] for r in rows)) for s,rows in outputs.items()},
            'sizes':{s:len(rows) for s,rows in outputs.items()},
            'grouping':'All frames sharing composition and prototype number stay together; MP-sourced ALOE entries excluded',
            'limitation':'Exact same-basis geometry deduplication; structural similarity across independently generated families may remain'}
    write_json('reports/mpaloe_merge.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
