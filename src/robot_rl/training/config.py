"""Load an experiment and apply command-line overrides."""

from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path

import yaml

from robot_rl.controllers.formation import FOLLOWER_CONTROL_MODES, NEIGHBOR_VELOCITY_SOURCES
from robot_rl.maps.layouts import OBSTACLE_COURSES
from robot_rl.tasks.formation_navigation.config import EnvironmentConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 0
    device: str = "cuda"
    num_envs: int = 4
    timesteps: int | None = None
    open_timesteps: int = 1_000_000
    obstacle_timesteps: int = 5_000_000
    checkpoint_steps: int = 1_000_000
    output_dir: str = "outputs"


@dataclass(frozen=True, slots=True)
class PPOConfig:
    rollout_steps: int = 4096
    batch_size: int = 256
    n_epochs: int = 5
    clip_range: float = 0.1
    target_kl: float = 0.03
    learning_rate: float = 3e-5
    open_learning_rate: float = 1e-4
    ent_coef: float = 0.001
    open_ent_coef: float = 5e-4


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    seeds: tuple[int, ...] = (1000, 1001, 1002, 1003, 1004)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    environment: EnvironmentConfig = EnvironmentConfig()
    training: TrainingConfig = TrainingConfig()
    ppo: PPOConfig = PPOConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    def validate(self):
        self.environment.validate()
        training = self.training
        if type(training.seed) is not int or training.seed < 0:
            raise ValueError("training.seed must be a nonnegative integer.")
        if not isinstance(training.device, str) or not training.device:
            raise ValueError("training.device must be a nonempty string.")
        if not isinstance(training.output_dir, str) or not training.output_dir:
            raise ValueError("training.output_dir must be a nonempty path.")
        for key in ("num_envs", "open_timesteps", "obstacle_timesteps", "checkpoint_steps", "timesteps"):
            value = getattr(training, key)
            if key == "timesteps" and value is None:
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"training.{key} must be a positive integer.")
        for key, minimum in (("rollout_steps", 1), ("batch_size", 2), ("n_epochs", 1)):
            value = getattr(self.ppo, key)
            if type(value) is not int or value < minimum:
                raise ValueError(f"ppo.{key} must be an integer >= {minimum}.")
        for key in ("clip_range", "target_kl", "learning_rate", "open_learning_rate", "ent_coef", "open_ent_coef"):
            value = getattr(self.ppo, key)
            if type(value) not in (int, float) or not isfinite(value):
                raise ValueError(f"ppo.{key} must be finite.")
            if value < 0 or (value == 0 and key not in ("ent_coef", "open_ent_coef")):
                raise ValueError(f"Invalid ppo.{key} value.")
        if not self.evaluation.seeds or any(type(seed) is not int or seed < 0 for seed in self.evaluation.seeds):
            raise ValueError("evaluation.seeds must contain nonnegative integers.")


def load_experiment(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("An experiment YAML must be a mapping.")
    sections = {
        "environment": EnvironmentConfig,
        "training": TrainingConfig,
        "ppo": PPOConfig,
        "evaluation": EvaluationConfig,
    }
    unknown = set(raw) - sections.keys()
    if unknown:
        raise ValueError(f"Unknown experiment sections: {', '.join(map(str, unknown))}.")
    options = {}
    for name, config_class in sections.items():
        values = raw.get(name, {})
        if not isinstance(values, dict):
            raise ValueError(f"{name} must be a mapping.")
        if name == "evaluation" and "seeds" in values:
            if not isinstance(values["seeds"], list):
                raise ValueError("evaluation.seeds must be a YAML list.")
            values = {**values, "seeds": tuple(values["seeds"])}
        try:
            options[name] = config_class(**values)
        except TypeError as exc:
            raise ValueError(f"Invalid {name} settings: {exc}") from exc
    return ExperimentConfig(**options)


def model_run_dir(checkpoint):
    directory = Path(checkpoint).resolve().parent
    return directory.parent if directory.name == "checkpoints" else directory


def add_environment_arguments(parser):
    parser.add_argument("--config", type=Path, help="Experiment YAML path.")
    parser.add_argument("--stage", type=int, choices=(1, 2), help="1: open course. 2: obstacle course.")
    parser.add_argument("--map", choices=OBSTACLE_COURSES, help="Obstacle course for stage 2.")
    parser.add_argument("--no-walls", dest="walls", action="store_false", default=None)
    parser.add_argument("--follower-control", choices=FOLLOWER_CONTROL_MODES)
    parser.add_argument("--neighbor-velocity", choices=NEIGHBOR_VELOCITY_SOURCES)


def experiment_from_args(parser, args, default_config, model=None):
    try:
        config_path = args.config
        if config_path is None and model is not None:
            saved_config = model_run_dir(model) / "experiment.yaml"
            if saved_config.is_file():
                config_path = saved_config
        experiment = load_experiment(config_path or default_config)
        environment = {
            name: getattr(args, name, None)
            for name in ("stage", "map", "walls", "follower_control", "neighbor_velocity")
        }
        training = {
            name: getattr(args, name, None)
            for name in ("seed", "device", "num_envs", "timesteps", "output_dir")
        }
        result = replace(
            experiment,
            environment=replace(experiment.environment, **{k: v for k, v in environment.items() if v is not None}),
            training=replace(experiment.training, **{k: v for k, v in training.items() if v is not None}),
        )
        result.validate()
        return result
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
