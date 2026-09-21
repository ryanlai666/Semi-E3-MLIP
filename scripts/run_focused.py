"""Resumable controlled experiments; never evaluates protected tests during tuning."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from semi_mlip.model import ModelConfig
from semi_mlip.train import TrainConfig, train
from semi_mlip.data import write_json


def model_config(architecture):
    return ModelConfig(scalar_channels=48, vector_channels=24, blocks=3, radial_basis=24,
                       attention=architecture!='gated', tensor_channels=8 if architecture=='tensor' else 0)


def run(metal, architecture, count=300, seed=42, screen=True):
    name = f'screen_{metal}_{architecture}' if screen else f'confirm_{metal}_{architecture}_n{count}_s{seed}'
    folder = Path('runs/focused')/name
    cfg = TrainConfig(epochs=60, max_train=0, atom_budget=1024, edge_budget=64000,
                      accumulation=1, lr=.001, seed=seed, loss='pseudo_huber', stress_weight=0,
                      monitor_train_every=10)
    checkpoint = train(f'data/focused/{metal}_cold_{count}', folder, cfg,
                       model_config(architecture), folder/'last.pt' if (folder/'last.pt').exists() else None)
    config = json.loads((folder/'config.json').read_text())
    metric = json.loads((folder/'validation_best.json').read_text())['overall']
    return {'name':name, 'metal':metal, 'architecture':architecture, 'train_frames':count, 'seed':seed,
            'checkpoint':str(checkpoint), 'validation':metric, 'parameters':config['parameters'],
            'train_config':asdict(cfg), 'model_config':config['model']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['screen','confirm'], default='screen')
    args = parser.parse_args()
    output = Path('reports/focused'); output.mkdir(exist_ok=True)
    if (output/'frozen.json').exists():
        raise ValueError('Final tests have been frozen; do not retune this study against them')
    if args.stage == 'screen':
        trials = []
        for architecture in ('gated','attention','tensor'):
            for metal in ('cu','ti'):
                trials.append(run(metal, architecture))
                write_json(output/'screen.json', {'complete':False,'trials':trials,'test_used':False})
        choices = {}
        for metal in ('cu','ti'):
            candidates = [t for t in trials if t['metal']==metal]
            best = min(t['validation']['force_mae_eV_A'] for t in candidates)
            chosen = min((t for t in candidates if t['validation']['force_mae_eV_A']<=1.02*best),
                         key=lambda t:t['parameters'])
            choices[metal] = chosen['architecture']
        write_json(output/'screen.json', {'complete':True,'trials':trials,'test_used':False,
                                          'selected':choices,'rule':'Lowest warm force MAE; within 2%, fewer parameters'})
    else:
        screen = json.loads((output/'screen.json').read_text()); assert screen['complete']
        trials = []
        for count in (300,900):
            for seed in (42,43,44):
                for metal in ('cu','ti'):
                    architecture = screen['selected'][metal]
                    if count==300 and seed==42:
                        trials.append(next(t for t in screen['trials'] if t['metal']==metal and t['architecture']==architecture))
                    else:
                        trials.append(run(metal,architecture,count,seed,screen=False))
                    write_json(output/'confirm.json', {'complete':False,'trials':trials,'test_used':False})
        write_json(output/'confirm.json', {'complete':True,'trials':trials,'test_used':False})


if __name__ == '__main__':
    main()
