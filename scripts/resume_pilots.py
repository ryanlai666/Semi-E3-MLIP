"""Resume only existing interrupted device pilots with their saved configurations."""
import json
from pathlib import Path
import torch
from semi_mlip.data import write_json
from semi_mlip.model import ModelConfig
from semi_mlip.train import TrainConfig, train


def pilot_finished(folder, payload, config):
    # Legacy pilot scripts never use stop_after_epochs: timing.json marks completion.
    return (folder/'timing.json').exists() or payload['epoch']+1>=config.epochs or payload.get('early_stopped',False)


def main():
    from collect_search import main as collect
    output = Path('reports/benchmark/pilot_completion.json')
    completed = []
    for folder in sorted(Path('runs/device').iterdir()):
        last = folder / 'last.pt'
        if not last.exists():
            continue
        payload = torch.load(last, map_location='cpu', weights_only=False)
        config = TrainConfig(**payload['train_config'])
        # Legacy completed runs predate the checkpoint early_stopped field.
        # These pilot scripts write timing.json only after finishing, never as a stop-after probe.
        finished = pilot_finished(folder,payload,config)
        start = payload['epoch'] + 1
        if not finished:
            data = 'data/device_expanded' if folder.name.startswith('expanded') else 'data/device'
            print(f'Resuming {folder.name} after epoch {start}', flush=True)
            train(data, folder, config, ModelConfig(**payload['model_config']), last)
        completed.append({'name':folder.name, 'resumed_from_epoch':start if not finished else None,
                          'checkpoint':str(folder/'best.pt')})
        write_json(output, {'complete':False, 'trials':completed, 'test_used':False})
        collect()
    write_json(output, {'complete':True, 'trials':completed, 'test_used':False,
                       'scope':'Existing pilots only; unstarted search variants are future work.'})


if __name__ == '__main__':
    main()
