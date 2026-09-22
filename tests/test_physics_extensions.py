import copy
import numpy as np
import pytest
import torch
from semi_mlip.model import ModelConfig,Potential
from semi_mlip.graph import collate,neighbor_list
from semi_mlip.repulsion import switched_zbl

def graph(row):return collate([row],[neighbor_list(row['positions'],row['cell'],row['pbc'],3.)],dtype=torch.float64)
def row():return {'z':[22,8],'positions':[[.1,.2,.3],[1.2,.4,.5]],'cell':[[4.,0,0],[.5,4.1,0],[.1,.3,4.3]],'pbc':[True]*3}
def model():return Potential(ModelConfig(8,4,2,6,3.,attention=True,tensor_channels=4,equivariant_norm=True,bounded_gates=True,repulsive_core=True)).double()

def test_core_monotone_and_smooth_switch():
    r=torch.linspace(.2,1.6,300,dtype=torch.float64,requires_grad=True)
    e=switched_zbl(r,torch.full_like(r,22),torch.full_like(r,8))
    d=torch.autograd.grad(e.sum(),r,create_graph=True)[0]
    assert bool((e>=0).all()) and bool((d<=1e-10).all())
    for x in [.8,1.5]:
        values=[]
        for dx in [-1e-7,1e-7]:
            a=torch.tensor([x+dx],dtype=torch.float64,requires_grad=True)
            u=switched_zbl(a,torch.tensor([22]),torch.tensor([8]))
            du=torch.autograd.grad(u.sum(),a,create_graph=True)[0]
            ddu=torch.autograd.grad(du.sum(),a)[0];values.append([u.item(),du.item(),ddu.item()])
        np.testing.assert_allclose(values[0],values[1],rtol=1e-4,atol=1e-3)

def test_stabilized_core_equivariance_and_finite_difference():
    torch.manual_seed(3);m=model();r=row();g=graph(r);out=m(g)
    q,_=np.linalg.qr(np.random.default_rng(2).normal(size=(3,3)));q[:,0]*=-1
    transformed=copy.deepcopy(r);transformed['positions']=(np.array(r['positions'])@q+2).tolist();transformed['cell']=(np.array(r['cell'])@q).tolist()
    rotated=m(graph(transformed))
    np.testing.assert_allclose(rotated['energy'].detach(),out['energy'].detach(),atol=1e-9)
    np.testing.assert_allclose(rotated['forces'].detach(),out['forces'].detach().numpy()@q,atol=1e-8)
    np.testing.assert_allclose(rotated['stress'].detach()[0],q.T@out['stress'].detach()[0].numpy()@q,atol=1e-8)
    h=1e-5;energies=[]
    for sign in [-1,1]:
        moved=copy.deepcopy(r);moved['positions'][1][0]+=sign*h;energies.append(m(graph(moved))['energy'].item())
    assert -(energies[1]-energies[0])/(2*h)==pytest.approx(out['forces'][1,0].item(),rel=1e-6)
    # Check stress derivative under xx strain with fixed fractional coordinates.
    energies=[]
    for sign in [-1,1]:
        deformation=np.eye(3);deformation[0,0]+=sign*h
        moved=copy.deepcopy(r);moved['positions']=(np.array(r['positions'])@deformation).tolist();moved['cell']=(np.array(r['cell'])@deformation).tolist();energies.append(m(graph(moved))['energy'].item())
    assert (energies[1]-energies[0])/(2*h*np.linalg.det(r['cell']))==pytest.approx(out['stress'][0,0,0].item(),rel=1e-6)

def test_directed_core_pair_count_and_periodic_replication():
    m=model()
    with torch.no_grad():
        for p in m.parameters():p.zero_()
    r=row();out=m(graph(r))
    distance=torch.tensor([np.linalg.norm(np.array(r['positions'])[1]-r['positions'][0])],dtype=torch.float64)
    expected=switched_zbl(distance,torch.tensor([22]),torch.tensor([8]))
    torch.testing.assert_close(out['energy'],expected)
    rr=copy.deepcopy(r);rr['z']+=r['z'];rr['positions']+=(np.array(r['positions'])+r['cell'][0]).tolist();rr['cell'][0]=(2*np.array(r['cell'][0])).tolist()
    replicated=m(graph(rr));torch.testing.assert_close(replicated['energy'],2*out['energy'])
    torch.testing.assert_close(replicated['forces'][:2],out['forces'])
    torch.testing.assert_close(replicated['stress'],out['stress'])

def test_balanced_selection_macro_weights_chemistries():
    from semi_mlip.train import TrainConfig, validation_score
    cfg=TrainConfig(selection_metric='balanced_macro')
    metrics={'overall':{'force_mae_eV_A':99},'Ti':{'energy_mae_eV_atom':.01,'force_mae_eV_A':.1,'frames':1000},'O-Ta':{'energy_mae_eV_atom':.03,'force_mae_eV_A':.2,'frames':1},'formula:Ta2O5':{'force_mae_eV_A':99}}
    assert validation_score(metrics,cfg)==pytest.approx(1.75)
    assert validation_score(metrics,TrainConfig())==99
