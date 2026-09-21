"""Small-system FIRE, NVE velocity Verlet, and BAOAB Langevin MD."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from .data import SYMBOLS, write_json
from .graph import collate, neighbor_list
from .train import load_potential, resolve_device

# (eV/Angstrom)/amu -> Angstrom/fs^2, and Boltzmann constant in eV/K.
ACCEL = 0.009648533215665327
KB = 8.617333262145e-5
MASSES = {8: 15.999, 13: 26.9815385, 14: 28.085, 22: 47.867, 27: 58.933194,
          29: 63.546, 40: 91.224, 44: 101.07, 72: 178.49, 73: 180.94788, 74: 183.84}


class Calculator:
    def __init__(self, checkpoint, device="auto"):
        self.device = resolve_device(device)
        self.model, self.metadata = load_potential(checkpoint, self.device)

    def __call__(self, record):
        if not set(record["z"]) <= set(self.metadata["supported_elements"]):
            raise ValueError("Structure includes elements absent from training")
        chemistry = "-".join(sorted({SYMBOLS[z] for z in record["z"]}))
        if chemistry not in self.metadata["training_chemsys"]:
            raise ValueError(f"Chemical system {chemistry} absent from training")
        graph = neighbor_list(record["positions"], record["cell"], record.get("pbc", [True]*3), self.model.config.cutoff)
        batch = collate([record], [graph], self.device)
        result = self.model(batch)
        return {"energy": float(result["energy"].detach()[0]),
                "forces": result["forces"].detach().cpu().numpy(),
                "stress": result["stress"].detach().cpu().numpy()[0]}


def relaxation(record, calculate, steps=500, fmax=0.03):
    """Fixed-cell FIRE; output reports convergence instead of assuming it."""
    record = copy.deepcopy(record)
    x = np.asarray(record["positions"], float)
    v = np.zeros_like(x)
    dt, alpha, positive = 0.05, 0.1, 0
    history = []
    for step in range(steps + 1):
        record["positions"] = x.tolist()
        result = calculate(record)
        force = result["forces"]
        maximum = float(np.linalg.norm(force, axis=1).max())
        history.append({"step": step, "energy": result["energy"], "max_force": maximum})
        if maximum <= fmax or step == steps:
            break
        # FIRE uses fictitious equal masses and algorithmic time, not MD time.
        v += dt * force
        power = float((v * force).sum())
        if power > 0:
            positive += 1
            if positive > 5:
                dt = min(dt * 1.1, 0.5)
                alpha *= 0.99
        else:
            positive, dt, alpha = 0, dt * 0.5, 0.1
            v[:] = 0
        v = (1 - alpha) * v + alpha * force * np.linalg.norm(v) / max(np.linalg.norm(force), 1e-15)
        displacement = dt * v
        displacement *= min(1., 0.1 / max(np.linalg.norm(displacement, axis=1).max(), 1e-15))
        x += displacement
    record.update(energy=result["energy"], forces=force.tolist(), stress=result["stress"].tolist())
    return {"structure": record, "converged": maximum <= fmax, "history": history, "fixed_cell": True}


def kinetic_energy(velocity, masses):
    return float(0.5 * (masses[:, None] * velocity**2).sum() / ACCEL)


def initial_velocities(masses, temperature, rng):
    velocity = rng.normal(size=(len(masses), 3)) * np.sqrt(KB * temperature * ACCEL / masses[:, None])
    if len(masses) > 1:
        velocity -= (velocity * masses[:, None]).sum(axis=0) / masses.sum()
    return velocity


def md(record, calculate, output, steps=1000, dt=0.5, temperature=300., ensemble="nve", gamma=0.01, seed=42):
    if steps < 1 or dt <= 0 or temperature < 0 or gamma < 0 or ensemble not in ("nve", "nvt"):
        raise ValueError("Invalid MD settings")
    record = copy.deepcopy(record)
    masses = np.array([MASSES[z] for z in record["z"]])
    rng = np.random.default_rng(seed)
    x = np.array(record["positions"], float)
    v = initial_velocities(masses, temperature, rng)
    dof = 3 * len(masses) - (3 if len(masses) > 1 else 0)
    result = calculate(record)
    trace = []
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for step in range(steps + 1):
            kinetic = kinetic_energy(v, masses)
            frame = {"step": step, "time_fs": step * dt, "z": record["z"], "positions": x.tolist(),
                     "cell": record["cell"], "pbc": record.get("pbc", [True]*3),
                     "energy_eV": result["energy"], "kinetic_eV": kinetic,
                     "temperature_K": 2 * kinetic / (dof * KB), "velocities_A_fs": v.tolist(),
                     "forces_eV_A": result["forces"].tolist(), "label_source": "model_prediction"}
            f.write(json.dumps(frame, allow_nan=False) + "\n")
            trace.append(result["energy"] + kinetic)
            if step == steps:
                break
            v += 0.5 * dt * ACCEL * result["forces"] / masses[:, None]
            if ensemble == "nve":
                x += dt * v
            else:
                x += 0.5 * dt * v
                decay = np.exp(-gamma * dt)
                noise = initial_velocities(masses, temperature, rng)
                v = decay * v + np.sqrt(1 - decay**2) * noise
                x += 0.5 * dt * v
            record["positions"] = x.tolist()
            result = calculate(record)
            v += 0.5 * dt * ACCEL * result["forces"] / masses[:, None]
            if not np.isfinite(x).all() or not np.isfinite(v).all():
                raise FloatingPointError(f"Unstable MD at step {step}")
    drift = (trace[-1] - trace[0]) / len(masses)
    report = {"ensemble": ensemble, "steps": steps, "dt_fs": dt, "seed": seed,
              "temperature_target_K": temperature, "final_temperature_K": frame["temperature_K"],
              "energy_change_eV_atom": drift,
              "energy_range_eV_atom": float(np.ptp(trace) / len(masses)),
              "energy_slope_eV_atom_ps": float(np.polyfit(np.arange(len(trace))*dt / 1000, trace, 1)[0] / len(masses)),
              "drift_is_conservation_check": ensemble == "nve"}
    write_json(str(output) + ".summary.json", report)
    return report
