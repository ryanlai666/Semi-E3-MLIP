"""Reproducible small-data experiment suite; test labels are read only at end.

Runs serially on one GPU. Re-running resumes identical completed/interrupted
trials. Initial sweeps are deliberately short, so their ranking is exploratory.
"""
from dataclasses import asdict
import json
from pathlib import Path
from semi_mlip.model import ModelConfig
from semi_mlip.train import TrainConfig, ablate, train, evaluate_checkpoint
from semi_mlip.data import write_json


def trial(name, cfg, mc):
    directory = Path("runs/research") / name
    last = directory / "last.pt"
    checkpoint = train("data/processed", directory, cfg, mc, last if last.exists() else None)
    report = json.loads((directory / "validation_best.json").read_text())
    metadata = json.loads((directory / "config.json").read_text())
    return {"name": name, "checkpoint": str(checkpoint), "model": asdict(mc), "train": asdict(cfg),
            "parameters": metadata["parameters"], "validation": report["overall"]}


def main():
    base = TrainConfig(epochs=20, atom_budget=1024, edge_budget=64000, accumulation=1)
    activations = ablate("data/processed", "runs/optimized_activation", base)
    activation = activations["selected"]
    trials = []
    trials.append(trial("gated_mse", base, ModelConfig(activation=activation)))
    robust = TrainConfig(**{**asdict(base), "loss": "pseudo_huber"})
    trials.append(trial("gated_robust", robust, ModelConfig(activation=activation)))
    trials.append(trial("attention_robust", robust, ModelConfig(activation=activation, attention=True)))
    trials.append(trial("attention_chem", robust, ModelConfig(activation=activation, attention=True, chemical_descriptors=True)))
    auxiliary = TrainConfig(**{**asdict(robust), "auxiliary_weight": .05})
    trials.append(trial("attention_aux", auxiliary, ModelConfig(activation=activation, attention=True, auxiliary_heads=True)))
    candidates = [t for t in trials if t["model"]["attention"]]
    selected = min(candidates, key=lambda t: t["validation"]["force_mae_eV_A"])
    write_json("reports/experiments.json", {"activation_comparison": activations, "trials": trials,
               "selected_attention_variant": selected["name"], "selection_uses_test": False,
               "limitation": "20-epoch pilot rankings, feature/attention comparisons use one seed; not a converged architecture benchmark"})
    cfg = TrainConfig(**{**selected["train"], "epochs": 200, "max_train": 0})
    final = trial("attention_final", cfg, ModelConfig(**selected["model"]))
    write_json("reports/pilot_model.json", final)
    print("PILOT_MODEL " + final["checkpoint"], flush=True)


if __name__ == "__main__":
    main()
