"""Audit unused MP-ALOE chemistry without changing any training or test partition."""
from collections import Counter
import gzip
import json
from semi_mlip.data import ELEMENTS,write_json

SCOPE={'Al','Si','Ti','Co','Cu','Zr','Ru','Hf','Ta','W','SiO2','Al2O3','Cu2O','CuO','HfO2',
       'ZrO2','TiO2','Ta2O5','WO3','RuO2','CoO','Co3O4'}


def main():
    counts=Counter();eligible=Counter();outside=Counter();parents={}
    with gzip.open('data/raw/MP_ALOE_data.jsonl.gz','rt') as f:
        for line in f:
            raw=json.loads(line);counts['all_frames']+=1
            elements=set(raw['elements'])
            if not elements<=ELEMENTS.keys():continue
            counts['supported_elements']+=1
            if raw['provenance'].get('original_mp_id'):
                counts['mp_sourced_excluded']+=1;continue
            if raw.get('prototype_number') is None:
                counts['missing_prototype_excluded']+=1;continue
            formula=raw['formula_pretty'];system='-'.join(sorted(elements))
            eligible[system]+=1
            parents.setdefault(system,set()).add((formula,raw['prototype_number']))
            if formula in SCOPE:counts['current_formula_scope_before_conversion']+=1
            else:outside[formula]+=1
    write_json('reports/mpaloe_expansion_audit.json',{'source':'https://doi.org/10.6084/m9.figshare.29452190.v2',
        'counts':dict(counts),'eligible_by_system':dict(eligible),'prototype_formula_groups':{s:len(g) for s,g in parents.items()},
        'outside_current_formula_scope':dict(outside.most_common()),'partitions_modified':False,
        'note':'Raw metadata eligibility, not validated extra training frames. New stoichiometries/alloys require conversion, geometry deduplication and a new group-disjoint study. Existing ALOE validation/test groups remain protected.'})
    print(json.dumps({'counts':dict(counts),'outside_top20':outside.most_common(20)},indent=2),flush=True)


if __name__=='__main__':main()
