"""Small real-data learning sanity check. Not a held-out accuracy benchmark."""
import json
from pathlib import Path
import torch
from semi_mlip.data import read_jsonl, training_statistics, write_json
from semi_mlip.graph import collate, neighbor_list
from semi_mlip.model import ModelConfig, Potential
from semi_mlip.train import loss_terms, seed_all


def main():
    seed_all(42)
    torch.set_num_threads(4)
    rows = [r for r in read_jsonl("data/processed/train.jsonl") if r["chemsys"] == "O-Si" and len(r["z"]) <= 9][:4]
    stats = training_statistics(rows)
    graphs = [neighbor_list(r["positions"], r["cell"], r["pbc"], 5.) for r in rows]
    batch = collate(rows, graphs)
    model = Potential(ModelConfig(24,12,2,16,5.), stats["offsets"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1e-5)
    trace = []
    for step in range(301):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch, create_graph=True)
        e, f, s = loss_terms(prediction, batch, stats["scales"])
        loss = e + 10*f + s
        trace.append(float(loss.detach()))
        if step < 300:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
    result = {"label": "training-set overfit diagnostic only", "frames": [r["id"] for r in rows],
              "initial_loss": trace[0], "final_loss": trace[-1], "reduction_fraction": 1-trace[-1]/trace[0],
              "steps": 300, "passed": trace[-1] < .3*trace[0], "history": trace}
    write_json("reports/overfit.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "history"}))


if __name__ == "__main__":
    main()
