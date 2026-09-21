"""Bounded validation-only search; never evaluates test labels."""
from dataclasses import asdict
import json
from pathlib import Path
from semi_mlip.model import ModelConfig
from semi_mlip.train import TrainConfig, train
from semi_mlip.data import write_json

def main():
    results=[]
    variants=[('attention_lr1e3',1e-3,True,'swiglu',False),
              ('gated_lr1e3',1e-3,False,'swiglu',False),
              ('attention_lr3e4',3e-4,True,'swiglu',False),
              ('attention_geglu',1e-3,True,'geglu',False),
              ('attention_chem',1e-3,True,'swiglu',True)]
    for name,lr,attention,activation,chem in variants:
        cfg=TrainConfig(epochs=250,max_train=0,atom_budget=512,edge_budget=32000,
                        accumulation=1,lr=lr,loss='pseudo_huber',patience=60,
                        minimum_epochs=120,monitor_train_every=10)
        mc=ModelConfig(attention=attention,activation=activation,chemical_descriptors=chem)
        folder=Path('runs/device')/name
        last=folder/'last.pt'
        checkpoint=train('data/device',folder,cfg,mc,last if last.exists() else None)
        metrics=json.loads((folder/'validation_best.json').read_text())
        results.append({'name':name,'checkpoint':str(checkpoint),'model':asdict(mc),
                        'config':asdict(cfg),'validation':metrics})
        write_json('reports/device_search.json',{'trials':results,'test_used':False,
                   'targets':{'energy_mae_eV_atom':0.01,'force_mae_eV_A':0.1}})

if __name__=='__main__':
    main()
