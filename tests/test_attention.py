import copy
import numpy as np
import pytest
import torch
from semi_mlip.graph import neighbor_list, collate
from semi_mlip.model import ModelConfig, Potential


def make_model(tensor_channels=0):
    torch.manual_seed(17)
    return Potential(ModelConfig(16,8,2,8,3., attention=True, attention_heads=4,tensor_channels=tensor_channels)).double()


def graph(row):
    return collate([row],[neighbor_list(row["positions"],row["cell"],row["pbc"],3.)],dtype=torch.float64)


@pytest.mark.parametrize("tensor_channels",[0,4])
def test_attention_equivariance_derivatives_and_batch_independence(tensor_channels):
    model = make_model(tensor_channels)
    row = {"z":[14,8,8],"positions":[[0.,0.,0.],[1.3,.2,0.],[.4,1.5,.3]],
           "cell":(np.eye(3)*5).tolist(),"pbc":[True]*3}
    g = graph(row)
    result = model(g,create_graph=True)
    q,_ = np.linalg.qr(np.random.default_rng(9).normal(size=(3,3)))
    q[:,0] *= -np.sign(np.linalg.det(q))  # Explicit improper orthogonal transform.
    transformed = copy.deepcopy(row)
    transformed["positions"] = (np.array(row["positions"]) @ q).tolist()
    transformed["cell"] = (np.array(row["cell"]) @ q).tolist()
    rotated = model(graph(transformed))
    np.testing.assert_allclose(rotated["energy"].detach(),result["energy"].detach(),atol=1e-10)
    np.testing.assert_allclose(rotated["forces"].detach(),result["forces"].detach().numpy() @ q,atol=1e-9)
    eps = 1e-5
    p1,p2 = g["positions"].clone(),g["positions"].clone()
    p1[1,0] += eps
    p2[1,0] -= eps
    numeric = -(model.energy(g,p1)-model.energy(g,p2))/(2*eps)
    torch.testing.assert_close(numeric[0],result["forces"][1,0],atol=1e-8,rtol=1e-5)
    (result["forces"].square().mean()+result["stress"].square().mean()).backward()
    assert model.interactions[0].query.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.interactions[0].query.weight.grad).all()
    gr = neighbor_list(row["positions"],row["cell"],row["pbc"],3.)
    doubled = model(collate([row,row],[gr,gr],dtype=torch.float64))
    torch.testing.assert_close(doubled["energy"],result["energy"].repeat(2),atol=1e-10,rtol=1e-9)


@pytest.mark.parametrize("other_neighbor",[False,True])
def test_attention_cutoff_crossing(other_neighbor):
    model=make_model()
    def calculate(distance):
        row={"z":[14,8],"positions":[[0.,0.,0.],[distance,0.,0.]],
             "cell":(np.eye(3)*20).tolist(),"pbc":[False]*3}
        if other_neighbor:
            row["z"].append(8)
            row["positions"].append([-1.,0.,0.])
        return model(graph(row))
    left=calculate(3.-1e-5)
    right=calculate(3.+1e-5)
    torch.testing.assert_close(left["energy"],right["energy"],atol=1e-10,rtol=0)
    torch.testing.assert_close(left["forces"],right["forces"],atol=1e-8,rtol=0)
