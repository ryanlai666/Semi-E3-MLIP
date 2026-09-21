"""Freeze the validation-selected research checkpoint before test evaluation."""
import json
from pathlib import Path
from collect_search import main as collect
from device_subset import included
from semi_mlip.data import read_jsonl,write_json
from semi_mlip.train import evaluate_checkpoint

def main():
    collect();trials=[]
    for name in ('device_search','tensor_search','expanded_search'):
        p=Path('reports')/(name+'.json')
        if p.exists():trials+=json.loads(p.read_text())['trials']
    eligible=[t for t in trials if t['model']['attention'] and t['status']=='completed_or_early_stopped'
              and t['best_epoch']>=10
              and t['validation']['overall']['force_mae_eV_A'] < .99*t['validation']['overall']['zero_force_baseline_mae_eV_A']
              and t['validation']['overall']['energy_mae_eV_atom'] < .8*t['validation']['overall']['offset_baseline_mae_eV_atom']]
    if not eligible:raise RuntimeError('No trained attention candidate beats the trivial baselines')
    best=min(t['validation']['overall']['force_mae_eV_A'] for t in eligible)
    chosen=min([t for t in eligible if t['validation']['overall']['force_mae_eV_A']<=1.02*best],
               key=lambda t:t['validation']['overall']['energy_mae_eV_atom'])
    report={**chosen,'selection':'Completed attention candidates beating trivial baselines; within 2% of best validation force MAE, choose lower energy MAE',
            'test_used_for_selection':False,'status':'research_prototype',
            'targets_met_on_validation':chosen['validation']['overall']['force_mae_eV_A']<=.1 and chosen['validation']['overall']['energy_mae_eV_atom']<=.01}
    write_json('reports/frozen_model.json',report)  # Freeze BEFORE opening test labels.
    report['device_test']=evaluate_checkpoint(chosen['checkpoint'],'data/device/test.jsonl','reports/device_test.json',atom_budget=1024,edge_budget=64000)
    if chosen['name'].startswith('expanded'):
        report['aloe_test']=evaluate_checkpoint(chosen['checkpoint'],'data/device_expanded/aloe_test.jsonl','reports/aloe_test.json',atom_budget=1024,edge_budget=64000)
    unusual=[r for r in read_jsonl('data/processed/test.jsonl') if not included(r)]
    path=Path('data/device/unusual_test.jsonl');path.write_text(''.join(json.dumps(r)+'\n' for r in unusual))
    report['unusual_test']=evaluate_checkpoint(chosen['checkpoint'],path,'reports/unusual_test.json',atom_budget=1024,edge_budget=64000)
    write_json('reports/frozen_model.json',report)
    print('FROZEN',chosen['checkpoint'],flush=True)

if __name__=='__main__':main()
