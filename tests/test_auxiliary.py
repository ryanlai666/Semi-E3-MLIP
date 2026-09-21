import copy
import numpy as np
import torch
from semi_mlip.graph import collate, neighbor_list
from semi_mlip.model import ModelConfig, Potential
from semi_mlip.data import training_statistics
from semi_mlip.train import auxiliary_loss, loss_terms, penalty


def record():
    return {"z": [14,8,8], "positions": [[0,0,0],[1.5,.2,0],[.2,1.5,0]],
            "cell": (np.eye(3)*8).tolist(), "pbc": [True]*3, "energy": -15.,
            "forces": [[0,0,0]]*3, "stress": None,
            "auxiliary": {"bader_population": [2.,7.,7.], "bader_abs_magmom": [0.,0.,0.]}}


def test_auxiliary_labels_are_not_inputs_and_gradients_work():
    torch.manual_seed(42)
    row = record()
    gr = neighbor_list(row["positions"], row["cell"], row["pbc"], 3.)
    batch = collate([row], [gr], dtype=torch.float64)
    model = Potential(ModelConfig(12,6,2,8,3., chemical_descriptors=True, auxiliary_heads=True)).double()
    out = model(batch, create_graph=True)
    changed = copy.deepcopy(batch)
    changed["auxiliary"] += 500
    other = model(changed, create_graph=True)
    torch.testing.assert_close(out["energy"], other["energy"], rtol=0, atol=0)
    torch.testing.assert_close(out["forces"], other["forces"], rtol=0, atol=0)
    loss = auxiliary_loss(out, batch, training_statistics([row])) + out["forces"].square().mean()
    loss.backward()
    assert model.auxiliary_readout.last.weight.grad.abs().sum() > 0
    assert model.embedding.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_missing_stress_and_auxiliary_are_masked():
    row = record()
    row["auxiliary"] = {}
    gr = neighbor_list(row["positions"], row["cell"], row["pbc"], 3.)
    batch = collate([row], [gr])
    model = Potential(ModelConfig(8,4,1,6,3., auxiliary_heads=True))
    out = model(batch, create_graph=True)
    stats = training_statistics([row])
    assert auxiliary_loss(out, batch, stats).item() == 0
    _, _, stress = loss_terms(out, batch, stats["scales"])
    assert stress.item() == 0


def test_robust_loss_has_finite_second_derivatives():
    x = torch.tensor([0.,1.,100.], dtype=torch.float64, requires_grad=True)
    y = penalty(x, "pseudo_huber")
    first, = torch.autograd.grad(y.sum(), x, create_graph=True)
    second, = torch.autograd.grad(first.sum(), x)
    assert torch.isfinite(second).all()
    assert first[-1] < 2.01
    assert second[0] == 2.
