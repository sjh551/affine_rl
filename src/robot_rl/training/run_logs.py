"""Locate an existing run and continue its TensorBoard timeline."""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import mkdtemp
from zipfile import ZipFile

from stable_baselines3.common.logger import HumanOutputFormat, Logger, TensorBoardOutputFormat
from stable_baselines3.common.utils import get_latest_run_id
from torch.utils.tensorboard import SummaryWriter
import yaml

from robot_rl.maps.layouts import OBSTACLE_COURSES
from robot_rl.tasks.formation_navigation.config import DEFAULT_TASK
from robot_rl.training.config import model_run_dir


_SETTINGS = "run_settings.json"
_FINAL_MODEL = "ppo_affine.zip"


def select_run(output_root, prefix, settings, checkpoint, new_run):
    if checkpoint is not None and not new_run:
        candidate = model_run_dir(checkpoint)
        if (candidate / "tensorboard").is_dir():
            previous_settings = candidate / _SETTINGS
            if previous_settings.is_file():
                same_course = json.loads(previous_settings.read_text(encoding="utf-8")) == settings
            else:
                # Runs predating settings files have a stage, and newer names
                # also carry the map and controller.
                stage_prefix = f"stage{settings['stage']}_"
                known_maps = ("open", *OBSTACLE_COURSES)
                has_map = candidate.name.startswith(tuple(f"{stage_prefix}{name}_" for name in known_maps))
                same_course = candidate.name.startswith(prefix if has_map else stage_prefix)
            if same_course:
                return candidate, True

    output_root.mkdir(parents=True, exist_ok=True)
    return Path(mkdtemp(prefix=prefix, dir=output_root)), False


def record_settings(run_dir, settings):
    destination = run_dir / _SETTINGS
    if not destination.exists():
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)


def record_experiment(run_dir, experiment, resume_from, starting_step):
    snapshot = asdict(experiment)
    task = asdict(DEFAULT_TASK)
    task["max_steps"] = experiment.environment.max_steps
    segment = {
        "starting_step": starting_step,
        "resume_from": str(resume_from.resolve()) if resume_from else None,
        "experiment": snapshot,
        "task_defaults": task,
    }
    config_path = run_dir / "experiment.yaml"
    if not config_path.is_file():
        temporary = config_path.with_suffix(".tmp")
        temporary.write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
        temporary.replace(config_path)
    with (run_dir / "training_sessions.jsonl").open("a", encoding="utf-8") as sessions:
        sessions.write(json.dumps(segment) + "\n")


def check_checkpoint(run_dir, current_step):
    final_model = run_dir / _FINAL_MODEL
    if final_model.is_file():
        with ZipFile(final_model) as archive:
            final_step = json.loads(archive.read("data"))["num_timesteps"]
        if current_step < final_step:
            raise ValueError(f"The current run already has a model at step {final_step}. "
                             "Resume from ppo_affine.zip or use --new-run to branch from this checkpoint.")


class _TensorBoardWriter(TensorBoardOutputFormat):
    def __init__(self, log_dir, purge_step):
        self.writer = SummaryWriter(log_dir=str(log_dir), purge_step=purge_step)
        self._is_closed = False


def make_logger(run_dir, *, resume_step=None):
    root = run_dir / "tensorboard"
    run_id = get_latest_run_id(str(root), "PPO")
    # Older resumed runs could use PPO_0 because SB3 starts from run ID zero.
    if run_id == 0 and not (root / "PPO_0").is_dir():
        run_id = 1
    log_dir = root / f"PPO_{run_id}"
    # The checkpoint's step is valid. Discard displayed samples from later steps
    # when recovering from an interruption that wrote logs ahead of its checkpoint.
    purge_step = resume_step + 1 if resume_step is not None else None
    return Logger(str(log_dir), [HumanOutputFormat(sys.stdout), _TensorBoardWriter(log_dir, purge_step)])


def save_model(model, run_dir):
    destination = run_dir / _FINAL_MODEL
    temporary = destination.with_name(".ppo_affine.tmp.zip")
    try:
        model.save(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
