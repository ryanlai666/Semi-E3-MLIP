import copy
import json
import numpy as np
import pytest
import torch
from semi_mlip.data import assign_splits, fingerprint, stress_from_matpes, training_statistics
from semi_mlip.model import ModelConfig
from semi_mlip.train import TrainConfig, learning_rate, train


def test_stress_conversion():
    stress = np.array(stress_from_matpes([1,2,3,4,5,6]))
    np.testing.assert_allclose(stress, -0.1 / 160.21766208 * np.array([[1,6,5],[6,2,4],[5,4,3]]))
    assert stress_from_matpes(None) is None


def test_parent_and_duplicate_grouping():
    rows = [{"parent": "A", "fingerprint": "one"}, {"parent": "A", "fingerprint": "two"},
            {"parent": "B", "fingerprint": "two"}, {"parent": "B", "fingerprint": "three"}]
    assign_splits(rows)
    assert len({r["split"] for r in rows}) == 1
    assert len({r["group"] for r in rows}) == 1


def test_fingerprint_invariance():
    row = {"z": [14,8], "positions": [[0.1,0.2,0.3],[1.,1.2,1.3]], "cell": (np.eye(3)*4).tolist(), "pbc": [True]*3}
    moved = copy.deepcopy(row)
    moved["positions"] = (np.array(row["positions"])[::-1] + [1.3,2.1,0.8]).tolist()
    moved["z"] = row["z"][::-1]
    assert fingerprint(row) == fingerprint(moved)


def test_warmup_cosine():
    cfg = TrainConfig()
    rates = [learning_rate(i, 100, cfg) for i in range(100)]
    assert rates[0] == cfg.min_lr
    assert rates[5] == cfg.lr
    assert rates[-1] == cfg.min_lr
    assert all(a <= b for a,b in zip(rates[:5],rates[1:6]))
    assert all(a >= b for a,b in zip(rates[5:],rates[6:]))


def tiny_rows():
    rows = []
    for i, distance in enumerate([1.2,1.4,1.6,1.8]):
        force = 2 * (distance - 1.5)
        rows.append({"id": str(i), "parent": str(i), "chemsys": "Si", "z": [14,14],
                     "positions": [[0.,0.,0.],[distance,0.,0.]], "cell": (np.eye(3)*10).tolist(),
                     "pbc": [False]*3, "energy": -4 + (distance-1.5)**2,
                     "forces": [[force,0,0],[-force,0,0]], "stress": None})
    return rows


@pytest.mark.parametrize("microbatch_atoms", [0, 2])
def test_exact_cpu_resume(tmp_path, microbatch_atoms):
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "valid"):
        (data / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in tiny_rows()))
    cfg = TrainConfig(epochs=3, max_train=0, atom_budget=4, edge_budget=100, accumulation=2, device="cpu", microbatch_atoms=microbatch_atoms)
    mc = ModelConfig(8,4,1,6,3.,"swiglu")
    train(data, tmp_path / "full", cfg, mc)
    train(data, tmp_path / "resume", cfg, mc, stop_after_epochs=1)
    train(data, tmp_path / "resume", cfg, mc, resume=tmp_path / "resume/last.pt")
    a = torch.load(tmp_path / "full/last.pt", weights_only=False)
    b = torch.load(tmp_path / "resume/last.pt", weights_only=False)
    assert a["update"] == b["update"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)


def test_training_only_offsets():
    stats = training_statistics(tiny_rows())
    expected = np.mean([r["energy"]/2 for r in tiny_rows()])
    assert stats["offsets"]["14"] == pytest.approx(expected)


def test_packed_cache_preserves_training(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module('semi_mlip.train')
    data = tmp_path/'data'; data.mkdir()
    for split in ('train','valid'):
        (data/f'{split}.jsonl').write_text('\n'.join(json.dumps(r) for r in tiny_rows()))
    cfg=TrainConfig(epochs=2, max_train=0, atom_budget=4, edge_budget=100, accumulation=1, device='cpu')
    mc=ModelConfig(8,4,1,6,3.,'swiglu')
    train(data,tmp_path/'cached',cfg,mc)
    def uncached(self,index):
        indices=self.batches[index]
        return module.collate([self.records[i] for i in indices],[self.graphs[i] for i in indices],self.device)
    monkeypatch.setattr(module.BatchCache,'get',uncached)
    train(data,tmp_path/'uncached',cfg,mc)
    a=torch.load(tmp_path/'cached/last.pt',weights_only=False)
    b=torch.load(tmp_path/'uncached/last.pt',weights_only=False)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)


def test_early_stopping_survives_resume(tmp_path):
    data=tmp_path/'data';data.mkdir()
    for split in ('train','valid'):
        (data/f'{split}.jsonl').write_text('\n'.join(json.dumps(r) for r in tiny_rows()))
    cfg=TrainConfig(epochs=10,device='cpu',lr=0,min_lr=0,patience=1,minimum_epochs=2,
                    monitor_train_every=1,atom_budget=8,edge_budget=100)
    mc=ModelConfig(8,4,1,6,3.,'swiglu')
    folder=tmp_path/'run'
    train(data,folder,cfg,mc)
    history=(folder/'history.jsonl').read_text()
    assert len(history.splitlines())==2
    train(data,folder,cfg,mc,resume=folder/'last.pt')
    assert (folder/'history.jsonl').read_text()==history
