"""Evaluate a PPO policy on fixed test seeds and report metrics as JSON."""

import argparse
import json
from pathlib import Path

from robot_rl.training.config import add_environment_arguments, experiment_from_args
from robot_rl.training.evaluation import evaluate_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_environment_arguments(parser)
    parser.add_argument("model", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", help="Override the fixed evaluation seeds.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON report.")
    args = parser.parse_args()
    if args.seeds and any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain nonnegative integers.")
    default = Path(__file__).resolve().parents[1] / "configs/experiments/formation_ppo.yaml"
    experiment = experiment_from_args(parser, args, default, model=args.model)
    seeds = args.seeds or experiment.evaluation.seeds
    report = evaluate_policy(args.model, experiment.environment, seeds, args.device)
    report["checkpoint"] = str(args.model.resolve())
    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
