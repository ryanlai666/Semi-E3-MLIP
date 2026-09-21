"""Collect measured runs, preserving interrupted-run status and best epoch."""
import json
from pathlib import Path
from semi_mlip.data import write_json

def main():
    groups={'device_search':[],'tensor_search':[],'expanded_search':[]}
    for folder in sorted(Path('runs/device').iterdir()):
        if not (folder/'validation_best.json').exists():continue
        info=json.loads((folder/'config.json').read_text())
        history=[json.loads(l) for l in (folder/'history.jsonl').read_text().splitlines()]
        best=min(history,key=lambda x:x['valid']['force_mae_eV_A'])
        finished=(folder/'timing.json').exists()
        entry={'name':folder.name,'checkpoint':str(folder/'best.pt'),'model':info['model'],'config':info['train'],
               'validation':json.loads((folder/'validation_best.json').read_text()),'best_epoch':best['epoch'],
               'epochs_recorded':history[-1]['epoch'],'status':'completed_or_early_stopped' if finished else 'in_progress_or_interrupted',
               'parameters':info['parameters'],'train_frames':info['train_frames'],'valid_frames':info['valid_frames']}
        key='expanded_search' if folder.name.startswith('expanded') else 'tensor_search' if folder.name.startswith('tensor') else 'device_search'
        groups[key].append(entry)
    for name,trials in groups.items():
        if trials:write_json('reports/'+name+'.json',{'trials':trials,'test_used':False,
                         'comparison_limit':'Some pilots were interrupted to prioritize data expansion; not equal-duration architecture benchmarks'})

if __name__=='__main__':main()
