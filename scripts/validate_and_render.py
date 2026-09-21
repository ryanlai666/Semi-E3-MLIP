"""Frozen-model held-out case animations and short simulation diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
from semi_mlip.data import read_jsonl, write_json
from semi_mlip.simulate import Calculator, md, relaxation
from semi_mlip.visualize import compare_cases, choose_cases, learning_curves, md_comparison


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/research/attention_final/best.pt")
    parser.add_argument("--data",default="data/processed/test.jsonl")
    parser.add_argument("--output",default="reports/visualizations")
    args=parser.parse_args()
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    calculate=Calculator(args.checkpoint,"cpu")
    cases=choose_cases(list(read_jsonl(args.data)))
    results={}
    for name,rows in cases.items():
        row=rows[0]
        folder=Path("runs/simulations") / name
        folder.mkdir(parents=True,exist_ok=True)
        try:
            relaxed=relaxation(row,calculate,steps=300,fmax=.03)
            write_json(folder/"relaxation.json",relaxed)
            # Dynamics begin from the DFT-labelled input, not the model-relaxed
            # minimum, so failed relaxation does not hide unstable trajectories.
            coarse=md(row,calculate,folder/"nve_05.jsonl",steps=200,dt=.5,temperature=300.)
            fine=md(row,calculate,folder/"nve_025.jsonl",steps=400,dt=.25,temperature=300.)
            thermostat=md(row,calculate,folder/"nvt.jsonl",steps=200,dt=.5,temperature=300.,ensemble="nvt")
            results[name]={"initial_id":row["id"],"relaxation_converged":relaxed["converged"],
                           "relaxation_final_force":relaxed["history"][-1]["max_force"],
                           "nve_05":coarse,"nve_025":fine,"nvt":thermostat,
                           "smaller_dt_reduces_energy_range":fine["energy_range_eV_atom"] < coarse["energy_range_eV_atom"],
                           "validation_limit":"100 fs numerical diagnostic, not long-time DFT agreement"}
            md_comparison(folder/"nve_05.jsonl",folder/"nve_025.jsonl",output/f"{name}_md.gif")
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            results[name]={"initial_id":row["id"],"failed":str(exc)}
        write_json("reports/simulation_checks.json",results)
        print(name, json.dumps(results[name]),flush=True)
    compare_cases(args.checkpoint,args.data,args.output,"cpu")
    learning_curves(Path(args.checkpoint).parent,"reports/learning_curves.png")
    gallery=output/"index.html"
    page=gallery.read_text(encoding="utf-8")
    cards=['<h1>Model-only MD: timestep comparison</h1><p>Both panels use the trained GNN. '
           'These are 100 fs numerical checks, not DFT/AIMD comparisons. The initial state is the held-out reference geometry.</p>']
    for name,report in results.items():
        if "failed" in report:
            cards.append(f'<section><h2>{name}</h2><p>Simulation failed; see simulation_checks.json.</p></section>')
        else:
            cards.append(f'<section><h2>{name.title()} MD</h2><img src="{name}_md.gif" alt="Two model MD timesteps">'
                         f'<p>NVE energy range: {report["nve_05"]["energy_range_eV_atom"]:.3g} versus '
                         f'{report["nve_025"]["energy_range_eV_atom"]:.3g} eV/atom.</p></section>')
    gallery.write_text(page.replace('</body>', ''.join(cards)+'</body>'),encoding="utf-8")


if __name__ == "__main__":
    main()
