import copy
import itertools
import numpy as np
import pytest
import torch
from semi_mlip.graph import neighbor_list, collate
from semi_mlip.model import ModelConfig, Potential, envelope


@pytest.fixture
def structure():
    return {"z": [14, 8, 8], "positions": [[0.2, 0.3, 0.4], [1.5, 0.8, 0.9], [0.8, 1.8, 1.4]],
            "cell": [[4., 0, 0], [0.7, 4.2, 0], [0.3, 0.5, 4.4]], "pbc": [True]*3}


def graph(record, cutoff=3.):
    return collate([record], [neighbor_list(record["positions"], record["cell"], record["pbc"], cutoff)], dtype=torch.float64)


def model(activation="swiglu"):
    torch.manual_seed(1)
    return Potential(ModelConfig(12, 6, 2, 8, 3., activation)).double()


@pytest.mark.parametrize("reflection", [False, True])
def test_e3_and_permutation(structure, reflection):
    m = model()
    reference = m(graph(structure))
    q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    if reflection:
        q[:, 0] *= -1
    transformed = copy.deepcopy(structure)
    order = [2, 0, 1]
    transformed["positions"] = (np.array(structure["positions"])[order] @ q + [8., -3., 2.]).tolist()
    transformed["cell"] = (np.array(structure["cell"]) @ q).tolist()
    transformed["z"] = [structure["z"][i] for i in order]
    out = m(graph(transformed))
    np.testing.assert_allclose(out["energy"].detach(), reference["energy"].detach(), atol=1e-10)
    np.testing.assert_allclose(out["forces"].detach(), reference["forces"].detach().numpy()[order] @ q, atol=1e-9)
    np.testing.assert_allclose(out["stress"].detach()[0], q.T @ reference["stress"].detach()[0].numpy() @ q, atol=1e-9)


def test_periodic_wrap(structure):
    m = model()
    reference = m(graph(structure))
    moved = copy.deepcopy(structure)
    moved["positions"] = (np.array(moved["positions"]) + np.array([[4,-3,2],[0,5,-2],[-1,0,0]]) @ np.array(moved["cell"])).tolist()
    out = m(graph(moved))
    torch.testing.assert_close(out["energy"], reference["energy"], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(out["forces"], reference["forces"], atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize('cutoff', [.6, 1.8, 3.2])
def test_triclinic_images_against_brute_force(cutoff):
    cell = np.array([[1.8, 0, 0], [1.1, 2., 0], [0.7, 0.6, 2.1]])
    p = np.array([[0.1, 0.2, 0.3], [1., 0.5, 0.8]])
    i, j, shifts = neighbor_list(p, cell, [True]*3, cutoff)
    actual = {(int(a), int(b), *s) for a, b, s in zip(i, j, shifts)}
    expected = set()
    for a, b, s in itertools.product(range(2), range(2), itertools.product(range(-5,6), repeat=3)):
        distance = np.linalg.norm(p[b] - p[a] + np.array(s) @ cell)
        if 1e-8 < distance < cutoff:
            expected.add((a, b, *s))
    assert actual == expected
    if cutoff == 3.2:
        assert any(a == b for a, b, *_ in actual)


@pytest.mark.parametrize("activation", ["swiglu", "geglu", "silu"])
def test_finite_difference_and_force_loss_backward(structure, activation):
    m, g = model(activation), graph(structure)
    result = m(g, create_graph=True)
    eps = 1e-5
    plus, minus = g["positions"].clone(), g["positions"].clone()
    plus[1, 2] += eps
    minus[1, 2] -= eps
    numeric_force = -(m.energy(g, plus) - m.energy(g, minus)) / (2*eps)
    torch.testing.assert_close(result["forces"][1, 2], numeric_force[0], atol=1e-8, rtol=1e-5)
    for a, b in ((0, 0), (0, 1), (1, 2)):
        strain = torch.zeros((3,3), dtype=torch.float64)
        strain[a,b] += eps/2
        strain[b,a] += eps/2
        ip, im = torch.eye(3) + strain, torch.eye(3) - strain
        ep = m.energy(g, g["positions"] @ ip, g["cell"] @ ip)
        em = m.energy(g, g["positions"] @ im, g["cell"] @ im)
        numeric_stress = (ep-em)/(2*eps*torch.linalg.det(g["cell"]).abs())
        torch.testing.assert_close(result["stress"][0,a,b], numeric_stress[0], atol=1e-9, rtol=1e-4)
    (result["forces"].square().sum() + result["stress"].square().sum()).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_extensivity_and_batching(structure):
    m = model()
    base = m(graph(structure))
    repeated = copy.deepcopy(structure)
    repeated["z"] *= 2
    cell = np.array(structure["cell"])
    repeated["positions"] = np.concatenate([structure["positions"], np.array(structure["positions"])+cell[0]]).tolist()
    cell[0] *= 2
    repeated["cell"] = cell.tolist()
    out = m(graph(repeated))
    torch.testing.assert_close(out["energy"], 2*base["energy"], atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(out["stress"], base["stress"], atol=1e-10, rtol=1e-9)
    gr = neighbor_list(structure["positions"], structure["cell"], structure["pbc"], 3.)
    batched = m(collate([structure, structure], [gr, gr], dtype=torch.float64))
    torch.testing.assert_close(batched["energy"], base["energy"].repeat(2))


def test_cutoff_and_isolated():
    near = torch.linspace(.99, 1.00001, 1000, dtype=torch.float32)
    assert (envelope(near) >= 0).all()
    x = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    y = envelope(x)
    dy, = torch.autograd.grad(y, x, create_graph=True)
    ddy, = torch.autograd.grad(dy, x)
    assert y == dy == ddy == 0
    record = {"z": [14], "positions": [[0.,0.,0.]], "cell": np.eye(3).tolist(), "pbc": [False]*3}
    out = model()(graph(record), create_graph=True)
    assert torch.count_nonzero(out["forces"]) == 0
    assert torch.count_nonzero(out["stress"]) == 0
