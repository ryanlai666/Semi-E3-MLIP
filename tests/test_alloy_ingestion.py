import copy
import pytest
from semi_mlip.data import convert


def alloy_record():
    return {'elements':['Al','Si'],'functional':'r2scan','matpes_id':'alloy-fixture','formula_pretty':'AlSi',
        'provenance':{'original_mp_id':'fixture-parent'},'energy':-8.,'forces':[[.1,0,0],[-.1,0,0]],
        'structure':{'lattice':{'matrix':[[5,0,0],[0,5,0],[0,0,5]]},'sites':[
            {'species':[{'element':'Al','occu':1}],'xyz':[0,0,0]},
            {'species':[{'element':'Si','occu':1}],'xyz':[2,2,2]}]}}


def test_alloy_scope_requires_explicit_opt_in():
    raw=alloy_record()
    with pytest.raises(ValueError,match='oxygen_free_alloy'):convert(raw)
    result=convert(raw,allow_alloys=True)
    assert result['z']==[13,14] and result['chemsys']=='Al-Si'
    assert result['energy']==raw['energy'] and result['forces']==raw['forces']
    assert result['cell']==raw['structure']['lattice']['matrix']


@pytest.mark.parametrize('bad,reason', [('functional','wrong_functional'),('occupancy','partial_occupancy'),('forces','nonfinite')])
def test_alloy_opt_in_preserves_label_and_structure_checks(bad,reason):
    raw=copy.deepcopy(alloy_record())
    if bad=='functional':raw['functional']='pbe'
    elif bad=='occupancy':raw['structure']['sites'][0]['species'][0]['occu']=.5
    else:raw['forces'][0][0]=float('nan')
    with pytest.raises(ValueError,match=reason):convert(raw,allow_alloys=True)
