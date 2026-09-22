"""Plain PyTorch training, resumable warmup/cosine AdamW, and evaluation."""
import collections
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from .data import read_jsonl, training_statistics, write_json
from .graph import collate, neighbor_list
from .model import ModelConfig, Potential


@dataclass
class TrainConfig:
    epochs: int = 50
    max_train: int = 10000
    max_valid: int = 0
    atom_budget: int = 256
    edge_budget: int = 16000
    accumulation: int = 4
    seed: int = 42
    lr: float = 3e-4
    min_lr: float = 3e-6
    warmup_fraction: float = 0.05
    weight_decay: float = 1e-5
    energy_weight: float = 1.0
    force_weight: float = 10.0
    stress_weight: float = 1.0
    device: str = "auto"
    loss: str = "mse"
    huber_delta: float = 1.0
    auxiliary_weight: float = 0.0
    patience: int = 0
    minimum_epochs: int = 50
    monitor_train_every: int = 0
    microbatch_atoms: int = 0
    microbatch_edges: int = 0
    selection_metric: str = "force_mae"
    selection_energy_scale: float = 0.01
    selection_force_scale: float = 0.1

    def __post_init__(self):
        if self.selection_metric not in ("force_mae", "balanced_macro"):
            raise ValueError("Unknown validation selection metric")
        if self.selection_energy_scale <= 0 or self.selection_force_scale <= 0:
            raise ValueError("Validation score scales must be positive")
        if (self.microbatch_atoms or self.microbatch_edges) and self.auxiliary_weight:
            raise ValueError("Microbatching currently supports energy/force/stress losses only")


def resolve_device(device):
    return "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def sample_records(records, limit):
    # Identical selection for all initialization seeds/activation ablations.
    if not limit or len(records) <= limit:
        return records
    groups = collections.defaultdict(list)
    for r in records:
        groups[r["chemsys"]].append(r)
    queues = [iter(sorted(v, key=lambda r: hashlib.sha256(r["id"].encode()).digest()))
              for _, v in sorted(groups.items())]
    result = []
    while queues and len(result) < limit:
        alive = []
        for q in queues:
            r = next(q, None)
            if r is not None:
                result.append(r)
                alive.append(q)
                if len(result) == limit:
                    break
        queues = alive
    return result


def load_graphs(path, cutoff, atom_budget, edge_budget, limit=0):
    records = sample_records(list(read_jsonl(path)), limit)
    cache_key = hashlib.sha256(json.dumps({"graph_version": 1, "cutoff": cutoff,
        "geometry": [(r["id"], r["positions"], r["cell"], r["pbc"]) for r in records]},
        separators=(",", ":")).encode()).hexdigest()
    cache_dir = Path(path).parent / "graph_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.npz"
    cached = None
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            cached = {key: archive[key] for key in archive.files}
    retained, graphs, excluded = [], [], []
    computed = []
    for index, r in enumerate(records):
        if len(r["z"]) > atom_budget:
            excluded.append({"id": r["id"], "reason": "atom_budget", "atoms": len(r["z"])})
            computed.append(None)
            continue
        try:
            if cached is not None and cached["valid"][index]:
                a, b = cached["ptr"][index:index+2]
                graph = cached["i"][a:b], cached["j"][a:b], cached["shifts"][a:b]
            else:
                graph = neighbor_list(r["positions"], r["cell"], r["pbc"], cutoff)
        except ValueError as exc:
            excluded.append({"id": r["id"], "reason": str(exc)})
            computed.append(None)
            continue
        computed.append(graph)
        if len(graph[0]) > edge_budget:
            excluded.append({"id": r["id"], "reason": "edge_budget", "edges": len(graph[0])})
            continue
        retained.append(r)
        graphs.append(graph)
    if cached is None and any(g is not None for g in computed):
        valid_graphs = [g for g in computed if g is not None]
        np.savez(cache_path, valid=np.array([g is not None for g in computed]),
                 ptr=np.concatenate([[0], np.cumsum([len(g[0]) if g is not None else 0 for g in computed])]),
                 i=np.concatenate([g[0] for g in valid_graphs]), j=np.concatenate([g[1] for g in valid_graphs]),
                 shifts=np.concatenate([g[2] for g in valid_graphs]))
    if not retained:
        raise ValueError(f"No usable records in {path}")
    return retained, graphs, excluded


def make_batches(records, graphs, atom_budget, edge_budget):
    batches, batch, atoms, edges = [], [], 0, 0
    # Fixed packing makes optimizer-update count independent of shuffle/resume.
    for idx, (record, graph) in enumerate(zip(records, graphs)):
        na, ne = len(record["z"]), len(graph[0])
        if batch and (atoms + na > atom_budget or edges + ne > edge_budget):
            batches.append(batch)
            batch, atoms, edges = [], 0, 0
        batch.append(idx)
        atoms, edges = atoms + na, edges + ne
    if batch:
        batches.append(batch)
    return batches


def split_optimizer_batches(records, graphs, batches, atom_budget=0, edge_budget=0):
    """Partition within fixed optimizer batches; retain even oversized single records.

    Budgets limit packing, not dataset admission. The caller weights each microbatch
    by its structure count over the original optimizer-update structure count.
    """
    microbatches, groups = [], []
    for indices in batches:
        local = make_batches([records[i] for i in indices], [graphs[i] for i in indices],
            atom_budget or float("inf"), edge_budget or float("inf"))
        groups.append(list(range(len(microbatches), len(microbatches) + len(local))))
        microbatches.extend([[indices[j] for j in group] for group in local])
    return microbatches, groups


class BatchCache:
    """Bounded tensor-packing cache for immutable, fixed-geometry training batches.

    Forward detaches coordinates before differentiation, so model activations and
    autograd graphs are never cached. Values and optimization order are unchanged.
    """
    def __init__(self, records, graphs, batches, device, max_bytes=128*1024**2):
        self.records, self.graphs, self.batches, self.device = records, graphs, batches, device
        self.max_bytes, self.bytes = max_bytes, 0
        self.cache = collections.OrderedDict()

    def get(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index][0]
        indices = self.batches[index]
        batch = collate([self.records[i] for i in indices], [self.graphs[i] for i in indices], self.device)
        size = sum(t.numel()*t.element_size() for t in batch.values() if isinstance(t,torch.Tensor))
        if size <= self.max_bytes:
            while self.bytes+size > self.max_bytes:
                _, (_, removed) = self.cache.popitem(last=False)
                self.bytes -= removed
            self.cache[index] = (batch,size); self.bytes += size
        return batch


def learning_rate(update, total, config):
    warmup = max(1, math.ceil(total * config.warmup_fraction))
    if update <= warmup:
        return config.min_lr + (config.lr - config.min_lr) * update / warmup
    progress = min(1., (update - warmup) / max(1, total - 1 - warmup))
    return config.min_lr + (config.lr - config.min_lr) * (1 + math.cos(math.pi * progress)) / 2


def optimizer_for(model, config):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 or "norm" in name else decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": config.weight_decay},
                             {"params": no_decay, "weight_decay": 0.}],
                            lr=config.min_lr, betas=(0.9, 0.999), eps=1e-8)


def penalty(error, kind="mse", delta=1.):
    if kind == "mse":
        return error.square()
    if kind == "pseudo_huber" and delta > 0:
        # Smooth robust loss, equivalent to error^2 near zero. Stable at zero.
        x = error / delta
        return 2 * error.square() / (torch.sqrt(1 + x.square()) + 1)
    raise ValueError("Unknown loss or invalid delta")


def loss_terms(prediction, batch, scales, kind="mse", delta=1.):
    e = penalty((prediction["energy"] - batch["energy"]) / batch["n_atoms"] / scales["energy"], kind, delta).mean()
    force_atom = penalty((prediction["forces"] - batch["forces"]) / scales["forces"], kind, delta).mean(dim=-1)
    force_structure = torch.zeros_like(batch["n_atoms"]).index_add(0, batch["batch"], force_atom) / batch["n_atoms"]
    f = force_structure.mean()
    mask = batch["stress_mask"]
    s = prediction["energy"].sum() * 0
    if prediction["stress"] is not None and mask.any():
        # Full batch denominator: missing labels have zero contribution.
        stress_per_structure = penalty((prediction["stress"][mask] - batch["stress"][mask]) / scales["stress"], kind, delta).mean(dim=(1, 2))
        s = stress_per_structure.sum() / len(mask)
    return e, f, s


def auxiliary_loss(prediction, batch, statistics):
    if prediction.get("auxiliary") is None:
        return prediction["energy"].sum() * 0
    total = prediction["energy"].sum() * 0
    for column, key in enumerate(("bader_population", "bader_abs_magmom")):
        stats = statistics["auxiliary"][key]
        mask = batch["auxiliary_mask"][:, column].clone()
        means = torch.zeros(119, device=batch["positions"].device)
        known = torch.zeros(119, dtype=torch.bool, device=means.device)
        for z, mean in stats["element_means"].items():
            means[int(z)], known[int(z)] = mean, True
        mask &= known[batch["z"]]
        if mask.any():
            target = (batch["auxiliary"][:, column] - means[batch["z"]]) / stats["scale"]
            total = total + penalty(prediction["auxiliary"][mask, column] - target[mask], "pseudo_huber").mean()
    return total


def evaluate(model, records, graphs, batches, device, scales, cache=None):
    model.eval()
    metrics = {}
    for batch_index, indices in enumerate(batches):
        rows = [records[i] for i in indices]
        batch = cache.get(batch_index) if cache is not None else collate(rows, [graphs[i] for i in indices], device)
        prediction = model(batch)
        ep = prediction["energy"].detach().cpu().numpy()
        fp = prediction["forces"].detach().cpu().numpy()
        sp = prediction["stress"].detach().cpu().numpy()
        offset = 0
        for row, energy, stress in zip(rows, ep, sp):
            n = len(row["z"])
            fref = np.array(row["forces"])
            ferr = fp[offset:offset+n] - fref
            eerr = float((energy - row["energy"]) / n)
            baseline = float(sum(float(model.offsets[z]) for z in row["z"]))
            high = np.linalg.norm(fref, axis=-1) > 1.0
            for key in ("overall", row["chemsys"], "formula:" + row.get("formula", row["chemsys"])):
                m = metrics.setdefault(key, collections.Counter())
                m["frames"] += 1
                m["atoms"] += n
                m["energy_abs"] += abs(eerr)
                m["energy_sq"] += eerr**2
                m["force_abs"] += float(np.abs(ferr).sum())
                m["force_sq"] += float((ferr**2).sum())
                m["offset_baseline_abs"] += abs((baseline - row["energy"]) / n)
                m["zero_force_abs"] += float(np.abs(fref).sum())
                m["high_force_abs"] += float(np.abs(ferr[high]).sum())
                m["high_force_components"] += int(high.sum()) * 3
                if row["stress"] is not None:
                    err = stress - np.array(row["stress"])
                    m["stress_abs"] += float(np.abs(err).sum())
                    m["stress_components"] += 9
            offset += n
    result = {}
    for key, m in metrics.items():
        nf, nc = m["frames"], m["atoms"] * 3
        result[key] = {"frames": nf, "atoms": m["atoms"], "energy_mae_eV_atom": m["energy_abs"] / nf,
                       "energy_rmse_eV_atom": math.sqrt(m["energy_sq"] / nf),
                       "force_mae_eV_A": m["force_abs"] / nc, "force_rmse_eV_A": math.sqrt(m["force_sq"] / nc),
                       "stress_mae_eV_A3": m["stress_abs"] / m["stress_components"] if m["stress_components"] else None,
                       "offset_baseline_mae_eV_atom": m["offset_baseline_abs"] / nf,
                       "zero_force_baseline_mae_eV_A": m["zero_force_abs"] / nc,
                       "high_force_mae_eV_A": m["high_force_abs"] / m["high_force_components"] if m["high_force_components"] else None}
    return result


def validation_score(validation, config):
    if config.selection_metric == "force_mae":
        return validation["overall"]["force_mae_eV_A"]
    systems=[v for k,v in validation.items() if k != "overall" and not k.startswith("formula:")]
    if not systems:raise ValueError("Balanced selection requires per-system metrics")
    return sum(.5 * (v["energy_mae_eV_atom"] / config.selection_energy_scale
                      + v["force_mae_eV_A"] / config.selection_force_scale) for v in systems) / len(systems)


def dataset_signature(records, graphs):
    h = hashlib.sha256()
    for row, graph in zip(records, graphs):
        h.update(json.dumps(row, sort_keys=True).encode())
        for a in graph:
            h.update(a.tobytes())
    return h.hexdigest()


def save_checkpoint(path, payload):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_potential(checkpoint, device="cpu"):
    torch.set_num_threads(4)
    # Only load locally produced checkpoints: torch pickle is not an interchange format.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Potential(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, payload


def train(data_dir="data/processed", run_dir="runs/pilot", config=None, model_config=None,
          resume=None, stop_after_epochs=None):
    config, model_config = config or TrainConfig(), model_config or ModelConfig()
    if config.epochs < 1 or config.accumulation < 1 or config.atom_budget < 1 or config.edge_budget < 1:
        raise ValueError("Epochs, accumulation, and batch budgets must be positive")
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    if (run_dir / "last.pt").exists() and not resume:
        raise ValueError("Run exists: choose a new directory or resume explicitly")
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_all(config.seed)
    torch.set_num_threads(4)
    device = resolve_device(config.device)
    print(f"Preparing graphs on CPU; training device={device}", flush=True)
    train_rows, train_graphs, train_excluded = load_graphs(data_dir / "train.jsonl", model_config.cutoff,
        config.atom_budget, config.edge_budget, config.max_train)
    valid_rows, valid_graphs, valid_excluded = load_graphs(data_dir / "valid.jsonl", model_config.cutoff,
        config.atom_budget, config.edge_budget, config.max_valid)
    supported = {z for r in train_rows for z in r["z"]}
    if any(not set(r["z"]) <= supported for r in valid_rows):
        raise ValueError("Validation contains elements absent from training")
    tb = make_batches(train_rows, train_graphs, config.atom_budget, config.edge_budget)
    vb = make_batches(valid_rows, valid_graphs, config.atom_budget, config.edge_budget)
    if config.microbatch_atoms < 0 or config.microbatch_edges < 0:
        raise ValueError("Microbatch budgets must be nonnegative")
    microbatches, microgroups = split_optimizer_batches(train_rows, train_graphs, tb,
        config.microbatch_atoms, config.microbatch_edges)
    train_cache = BatchCache(train_rows,train_graphs,tb,device)
    micro_cache = BatchCache(train_rows,train_graphs,microbatches,device)
    valid_cache = BatchCache(valid_rows,valid_graphs,vb,device)
    statistics = training_statistics(train_rows)
    if config.auxiliary_weight > 0 and not model_config.auxiliary_heads:
        raise ValueError("Auxiliary loss requires auxiliary heads")
    if model_config.auxiliary_heads and not any(s["labelled_atoms"] for s in statistics["auxiliary"].values()):
        raise ValueError("Auxiliary heads requested but no training labels exist")
    model_config = ModelConfig(**{**asdict(model_config), "average_neighbors":
        max(1., sum(len(g[0]) for g in train_graphs) / sum(len(r["z"]) for r in train_rows))})
    model = Potential(model_config, statistics["offsets"]).to(device)
    optimizer = optimizer_for(model, config)
    total = math.ceil(len(tb) / config.accumulation) * config.epochs
    signature = dataset_signature(train_rows + valid_rows, train_graphs + valid_graphs)
    start_epoch, update, best = 0, 0, float("inf")
    best_epoch = -1
    already_stopped = False
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload["signature"] != signature or asdict(TrainConfig(**payload["train_config"])) != asdict(config) or asdict(ModelConfig(**payload["model_config"])) != asdict(model_config):
            raise ValueError("Resume data/config mismatch; start a new run for changed settings")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch, update, best = payload["epoch"] + 1, payload["update"], payload["best"]
        best_epoch = payload.get("best_epoch", start_epoch - 1)
        already_stopped = payload.get("early_stopped", False)
        torch.set_rng_state(payload["rng_torch"])
        random.setstate(payload["rng_python"])
        np.random.set_state(payload["rng_numpy"])
        if device.startswith("cuda") and payload["rng_cuda"]:
            torch.cuda.set_rng_state_all(payload["rng_cuda"])
    write_json(run_dir / "config.json", {"train": asdict(config), "model": asdict(model_config), "device": device,
        "parameters": sum(p.numel() for p in model.parameters()), "statistics": statistics,
        "train_frames": len(train_rows), "valid_frames": len(valid_rows), "total_updates": total,
        "signature": signature, "torch_version": torch.__version__,
        "cuda_determinism": "CUDA index_add may be nondeterministic; CPU resume tested exactly"})
    write_json(run_dir / "excluded.json", {"train": train_excluded, "valid": valid_excluded})
    write_json(run_dir / "selection.json", {"train": [r["id"] for r in train_rows], "valid": [r["id"] for r in valid_rows]})
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    final_epoch = min(config.epochs, start_epoch + stop_after_epochs) if stop_after_epochs else config.epochs
    if already_stopped:
        final_epoch = start_epoch
    for epoch in range(start_epoch, final_epoch):
        epoch_start = time.monotonic()
        model.train()
        order = np.random.default_rng(config.seed + epoch).permutation(len(tb))
        total_loss, structures = 0., 0
        for group_start in range(0, len(order), config.accumulation):
            chunk = order[group_start:group_start + config.accumulation]
            denominator = sum(len(tb[b]) for b in chunk)
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(update, total, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            for b in chunk:
                for mb in microgroups[int(b)]:
                    indices = microbatches[mb]
                    batch = micro_cache.get(mb)
                    prediction = model(batch, create_graph=True, compute_stress=config.stress_weight != 0)
                    e, f, s = loss_terms(prediction, batch, statistics["scales"], config.loss, config.huber_delta)
                    loss = config.energy_weight * e + config.force_weight * f + config.stress_weight * s
                    if config.auxiliary_weight:
                        loss = loss + config.auxiliary_weight * auxiliary_loss(prediction, batch, statistics)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at epoch={epoch} update={update}")
                    (loss * len(indices) / denominator).backward()
                    total_loss += float(loss.detach()) * len(indices)
                    structures += len(indices)
                    del prediction, loss, e, f, s
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            update += 1
        validation = evaluate(model, valid_rows, valid_graphs, vb, device, statistics["scales"], valid_cache)
        score = validation_score(validation, config)
        improved = score < best
        best = min(score, best)
        if improved:
            best_epoch = epoch
        early_stopped = bool(config.patience and epoch + 1 >= config.minimum_epochs and epoch - best_epoch >= config.patience)
        payload = {"model": model.state_dict(), "model_config": asdict(model_config), "train_config": asdict(config),
                   "statistics": statistics, "optimizer": optimizer.state_dict(), "epoch": epoch,
                   "update": update, "total_updates": total, "best": best, "best_epoch": best_epoch, "early_stopped": early_stopped, "signature": signature,
                   "rng_torch": torch.get_rng_state(), "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                   "rng_cuda": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
                   "validation": validation, "supported_elements": sorted(supported),
                   "training_chemsys": sorted({r["chemsys"] for r in train_rows})}
        save_checkpoint(run_dir / "last.pt", payload)
        if improved:
            save_checkpoint(run_dir / "best.pt", payload)
            write_json(run_dir / "validation_best.json", validation)
        history = {"epoch": epoch + 1, "updates": update, "lr": lr, "train_loss": total_loss / structures,
                   "valid": validation["overall"], "selection_score": score, "validation_by_system": {k:v for k,v in validation.items() if k != "overall" and not k.startswith("formula:")}, "epoch_seconds": time.monotonic() - epoch_start,
                   "peak_gpu_MB": torch.cuda.max_memory_allocated() / 1e6 if device.startswith("cuda") else 0}
        if config.monitor_train_every and ((epoch + 1) % config.monitor_train_every == 0 or epoch == 0):
            history["train_metrics"] = evaluate(model, train_rows, train_graphs, tb, device, statistics["scales"], train_cache)["overall"]
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(history) + "\n")
        print(json.dumps(history), flush=True)
        if early_stopped:
            print(f"Early stopping at epoch {epoch + 1}; best epoch {best_epoch + 1}", flush=True)
            break
    write_json(run_dir / "timing.json", {"session_seconds": time.monotonic() - start,
                                        "epochs_completed": epoch + 1 if start_epoch < final_epoch else start_epoch, "planned_epochs": config.epochs})
    return run_dir / "best.pt"


def evaluate_checkpoint(checkpoint, data_file, output, device="auto", atom_budget=256, edge_budget=16000):
    device = resolve_device(device)
    model, payload = load_potential(checkpoint, device)
    records, graphs, excluded = load_graphs(data_file, model.config.cutoff, atom_budget, edge_budget)
    if any(not set(r["z"]) <= set(payload["supported_elements"]) for r in records):
        raise ValueError("Evaluation contains unsupported elements")
    batches = make_batches(records, graphs, atom_budget, edge_budget)
    result = evaluate(model, records, graphs, batches, device, payload["statistics"]["scales"])
    result["excluded"] = excluded
    write_json(output, result)
    return result


def ablate(data_dir, output, config):
    summaries = []
    for activation in ("swiglu", "geglu", "silu"):
        trials = []
        for seed in (42, 43, 44):
            cfg = TrainConfig(**{**asdict(config), "seed": seed})
            directory = Path(output) / f"{activation}-{seed}"
            resume = directory / "last.pt"
            train(data_dir, directory, cfg, ModelConfig(activation=activation), resume if resume.exists() else None)
            validation = json.loads((directory / "validation_best.json").read_text())
            history = list(read_jsonl(directory / "history.jsonl"))
            trials.append({"seed": seed, "force_mae": validation["overall"]["force_mae_eV_A"],
                           "seconds": sum(row["epoch_seconds"] for row in history)})
        summaries.append({"activation": activation, "trials": trials,
                          "mean_force_mae": float(np.mean([x["force_mae"] for x in trials])),
                          "mean_seconds": float(np.mean([x["seconds"] for x in trials]))})
    best = min(s["mean_force_mae"] for s in summaries)
    chosen = min((s for s in summaries if s["mean_force_mae"] <= best * 1.02), key=lambda s: s["mean_seconds"])
    report = {"selected": chosen["activation"], "rule": "Lowest mean validation force MAE; within 2%, faster wins",
              "results": summaries, "test_used_for_selection": False}
    write_json(Path(output) / "comparison.json", report)
    return report
