"""Train or resume a three-leader PPO policy."""

import argparse
from pathlib import Path

from robot_rl.training.config import add_environment_arguments, experiment_from_args
from robot_rl.training.runner import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_environment_arguments(parser)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from", type=Path, help="Continue from a model or periodic checkpoint.")
    parser.add_argument("--new-run", action="store_true", help="Use a separate log when loading existing weights.")
    parser.add_argument("--smoke", action="store_true", help="Run 128 CPU training steps without saving artifacts.")
    args = parser.parse_args()
    default = Path(__file__).resolve().parents[1] / "configs/experiments/formation_ppo.yaml"
    experiment = experiment_from_args(parser, args, default, model=args.resume_from)
    try:
        run_training(experiment, resume_from=args.resume_from, new_run=args.new_run, smoke=args.smoke)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
