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
    parser.add_argument("--episodes", type=int, help="Evaluate this many consecutive seeds from the first seed.")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel worker processes for evaluation.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON report.")
    args = parser.parse_args()
    if args.seeds and any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain nonnegative integers.")
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes must be a positive integer.")
    if args.num_envs < 1:
        parser.error("--num-envs must be a positive integer.")
    default = Path(__file__).resolve().parents[1] / "configs/experiments/formation_ppo.yaml"
    experiment = experiment_from_args(parser, args, default, model=args.model)
    seeds = args.seeds or experiment.evaluation.seeds
    if args.episodes is not None and not args.seeds:
        seeds = range(seeds[0], seeds[0] + args.episodes)
    report = evaluate_policy(args.model, experiment.environment, seeds, args.device, num_envs=args.num_envs)
    report["checkpoint"] = str(args.model.resolve())
    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
