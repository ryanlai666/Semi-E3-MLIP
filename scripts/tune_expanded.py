"""Data-expansion ablation against the unchanged MatPES validation partition."""
import json
from pathlib import Path
from semi_mlip.model import ModelConfig
from semi_mlip.train import TrainConfig,train
from semi_mlip.data import write_json

results=[]
for name,tensor,seed in [('expanded_attention',0,42),('expanded_tensor',16,42),('expanded_tensor_seed43',16,43)]:
    folder=Path('runs/device')/name
    cfg=TrainConfig(epochs=200,max_train=0,atom_budget=1024,edge_budget=64000,accumulation=1,
                    lr=1e-3,loss='pseudo_huber',patience=50,minimum_epochs=100,monitor_train_every=10,seed=seed)
    mc=ModelConfig(attention=True,tensor_channels=tensor)
    last=folder/'last.pt'
    checkpoint=train('data/device_expanded',folder,cfg,mc,last if last.exists() else None)
    actual=json.loads((folder/'config.json').read_text())
    results.append({'name':name,'checkpoint':str(checkpoint),'model':actual['model'],'config':actual['train'],
                    'validation':json.loads((folder/'validation_best.json').read_text())})
    write_json('reports/expanded_search.json',{'trials':results,'test_used':False,'validation_unchanged':True})
