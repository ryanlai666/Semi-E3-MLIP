"""Angular-backbone comparison on the fixed device validation partition."""
from dataclasses import asdict
import json
from pathlib import Path
from semi_mlip.model import ModelConfig
from semi_mlip.train import TrainConfig, train
from semi_mlip.data import write_json

results=[]
for name,width,vectors,tensors,seed in [('tensor16',64,32,16,42),('tensor16_seed43',64,32,16,43),
                                      ('tensor16_wide',96,48,16,42)]:
    cfg=TrainConfig(epochs=250,max_train=0,atom_budget=512,edge_budget=32000,accumulation=1,
                    lr=1e-3,loss='pseudo_huber',patience=60,minimum_epochs=120,
                    monitor_train_every=10,seed=seed)
    mc=ModelConfig(scalar_channels=width,vector_channels=vectors,tensor_channels=tensors,attention=True)
    folder=Path('runs/device')/name
    last=folder/'last.pt'
    checkpoint=train('data/device',folder,cfg,mc,last if last.exists() else None)
    results.append({'name':name,'checkpoint':str(checkpoint),'model':asdict(mc),'config':asdict(cfg),
                    'validation':json.loads((folder/'validation_best.json').read_text())})
    write_json('reports/tensor_search.json',{'trials':results,'test_used':False})
