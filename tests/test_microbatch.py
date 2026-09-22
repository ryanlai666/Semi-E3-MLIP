import copy
import numpy as np
import torch
from semi_mlip.model import ModelConfig, Potential
from semi_mlip.graph import collate, neighbor_list
from semi_mlip.train import loss_terms, split_optimizer_batches
from test_data_training import tiny_rows


def test_microbatch_force_stress_gradients_match_full_batch():
    rows=tiny_rows()[:3]
    rows[1]['stress']=(np.eye(3)*.02).tolist()
    rows[2]['z'].append(14);rows[2]['positions'].append([0.,2.,0.]);rows[2]['forces'].append([.1,.2,0.])
    graphs=[neighbor_list(r['positions'],r['cell'],r['pbc'],3.) for r in rows]
    cfg=ModelConfig(8,4,2,6,3.,attention=True,tensor_channels=4)
    torch.manual_seed(42)
    full=Potential(cfg).double();micro=copy.deepcopy(full)
    scales={'energy':1.,'forces':1.,'stress':1.}
    def backward(model,indices,weight):
        batch=collate([rows[i] for i in indices],[graphs[i] for i in indices],'cpu')
        batch={k:v.double() if isinstance(v,torch.Tensor) and v.is_floating_point() else v for k,v in batch.items()}
        pred=model(batch,create_graph=True,compute_stress=True)
        e,f,s=loss_terms(pred,batch,scales,'pseudo_huber',1.)
        ((e+10*f+s)*weight).backward()
    backward(full,[0,1,2],1.)
    batches,groups=split_optimizer_batches(rows,graphs,[[0,1,2]],2,100)
    assert groups==[[0,1,2]] and batches==[[0],[1],[2]]
    for indices in batches:backward(micro,indices,len(indices)/3)
    for a,b in zip(full.parameters(),micro.parameters()):
        if a.grad is None:assert b.grad is None
        else:torch.testing.assert_close(a.grad,b.grad,rtol=1e-8,atol=1e-10)




def test_microbatch_rejects_atom_normalized_auxiliary_loss():
    import pytest
    from semi_mlip.train import TrainConfig
    with pytest.raises(ValueError, match="energy/force/stress"):
        TrainConfig(microbatch_atoms=256, auxiliary_weight=1.)
