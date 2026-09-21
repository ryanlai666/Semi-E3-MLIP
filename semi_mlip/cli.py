import argparse
import json
from pathlib import Path
from .data import prepare, read_jsonl, write_json
from .download import download


def add_training_arguments(p):
    p.add_argument("--data", default="data/processed")
    p.add_argument("--run", default="runs/pilot")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--max-train", type=int, default=10000)
    p.add_argument("--max-valid", type=int, default=0)
    p.add_argument("--atom-budget", type=int, default=256)
    p.add_argument("--edge-budget", type=int, default=16000)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--loss", choices=["mse", "pseudo_huber"], default="mse")
    p.add_argument("--auxiliary-weight", type=float, default=0.)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--minimum-epochs", type=int, default=50)
    p.add_argument("--monitor-train-every", type=int, default=0)


def main():
    parser = argparse.ArgumentParser(description="From-scratch PyTorch semiconductor MLIP")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("download")
    p.add_argument("--output", default="data/raw")
    p = commands.add_parser("prepare", aliases=["audit"])
    p.add_argument("--source", default="data/raw/MatPES-R2SCAN-2025.2.jsonl")
    p.add_argument("--output", default="data/processed")
    p.add_argument("--seed", type=int, default=42)
    p = commands.add_parser("train")
    add_training_arguments(p)
    p.add_argument("--activation", choices=["swiglu", "geglu", "silu"], default="swiglu")
    p.add_argument("--resume")
    p.add_argument("--stop-after-epochs", type=int)
    p.add_argument("--chemical-descriptors", action="store_true")
    p.add_argument("--attention", action="store_true")
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--scalar-channels", type=int, default=64)
    p.add_argument("--vector-channels", type=int, default=32)
    p.add_argument("--tensor-channels", type=int, default=0)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--cutoff", type=float, default=5.)
    p = commands.add_parser("ablate")
    add_training_arguments(p)
    p = commands.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/processed/test.jsonl")
    p.add_argument("--output", default="runs/evaluation.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--atom-budget", type=int, default=256)
    p.add_argument("--edge-budget", type=int, default=16000)
    p = commands.add_parser("visualize")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/processed/test.jsonl")
    p.add_argument("--output", default="reports/visualizations")
    p.add_argument("--device", default="cpu")
    for command in ("predict", "relax", "md"):
        p = commands.add_parser(command)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--structure", required=True, help="Normalized single-structure JSON")
        p.add_argument("--output", required=True)
        p.add_argument("--device", default="auto")
        if command == "relax":
            p.add_argument("--steps", type=int, default=500)
            p.add_argument("--fmax", type=float, default=0.03)
        if command == "md":
            p.add_argument("--steps", type=int, default=1000)
            p.add_argument("--dt", type=float, default=0.5)
            p.add_argument("--temperature", type=float, default=300)
            p.add_argument("--ensemble", choices=["nve", "nvt"], default="nve")
            p.add_argument("--gamma", type=float, default=0.01, help="Friction in inverse fs")
            p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "download":
        download(args.output)
    elif args.command in ("prepare", "audit"):
        prepare(args.source, args.output, args.seed)
    elif args.command in ("train", "ablate"):
        from .train import TrainConfig, train, ablate
        from .model import ModelConfig
        cfg = TrainConfig(**{k: getattr(args, k) for k in ("epochs", "max_train", "max_valid", "atom_budget",
                             "edge_budget", "accumulation", "seed", "device", "loss", "auxiliary_weight",
                             "lr", "weight_decay", "patience", "minimum_epochs", "monitor_train_every")})
        if args.command == "train":
            train(args.data, args.run, cfg, ModelConfig(activation=args.activation,
                  chemical_descriptors=args.chemical_descriptors, auxiliary_heads=args.auxiliary_weight > 0,
                  attention=args.attention, attention_heads=args.attention_heads,
                  scalar_channels=args.scalar_channels, vector_channels=args.vector_channels,
                  tensor_channels=args.tensor_channels, blocks=args.blocks, cutoff=args.cutoff),
                  args.resume, args.stop_after_epochs)
        else:
            print(json.dumps(ablate(args.data, args.run, cfg), indent=2))
    elif args.command == "evaluate":
        from .train import evaluate_checkpoint
        report = evaluate_checkpoint(args.checkpoint, args.data, args.output, args.device, args.atom_budget, args.edge_budget)
        print(json.dumps(report["overall"], indent=2))
    elif args.command == "visualize":
        from .visualize import compare_cases
        compare_cases(args.checkpoint, args.data, args.output, args.device)
    else:
        from .simulate import Calculator, relaxation, md
        record = json.loads(Path(args.structure).read_text(encoding="utf-8"))
        calculate = Calculator(args.checkpoint, args.device)
        if args.command == "predict":
            result = calculate(record)
            write_json(args.output, {k: v.tolist() if hasattr(v, "tolist") else v for k, v in result.items()})
        elif args.command == "relax":
            write_json(args.output, relaxation(record, calculate, args.steps, args.fmax))
        else:
            print(json.dumps(md(record, calculate, args.output, args.steps, args.dt, args.temperature, args.ensemble, args.gamma, args.seed), indent=2))


if __name__ == "__main__":
    main()
