import json
import numpy as np
from semi_mlip.simulate import relaxation, md


def harmonic(record):
    x = np.array(record["positions"])
    return {"energy": float(0.5 * (x*x).sum()), "forces": -x, "stress": np.zeros((3,3))}


def structure():
    return {"z": [14,14], "positions": [[-0.5,0,0],[0.5,0,0]], "cell": (np.eye(3)*20).tolist(), "pbc": [False]*3}


def test_fire():
    result = relaxation(structure(), harmonic, steps=500, fmax=1e-4)
    assert result["converged"]
    assert result["structure"]["energy"] < 1e-6


def test_verlet_timestep_convergence(tmp_path):
    coarse = md(structure(), harmonic, tmp_path / "coarse.jsonl", steps=100, dt=2., temperature=0.)
    fine = md(structure(), harmonic, tmp_path / "fine.jsonl", steps=200, dt=1., temperature=0.)
    assert fine["energy_range_eV_atom"] < coarse["energy_range_eV_atom"] * 0.3


def test_langevin_reproducible(tmp_path):
    a = md(structure(), harmonic, tmp_path / "a.jsonl", steps=10, ensemble="nvt")
    b = md(structure(), harmonic, tmp_path / "b.jsonl", steps=10, ensemble="nvt")
    assert a == b
    assert np.isfinite(a["final_temperature_K"])
