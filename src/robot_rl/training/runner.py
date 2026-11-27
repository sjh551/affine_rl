"""Start or resume PPO using the configured navigation task."""

from contextlib import ExitStack
from functools import partial
from math import ceil
from pathlib import Path

from filelock import FileLock, Timeout
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from robot_rl.tasks.formation_navigation.env import FormationPyBulletEnv

from .policy import load_policy, policy_kwargs
from .run_logs import check_checkpoint, make_logger, record_experiment, record_settings, save_model, select_run


class FormationMetrics(BaseCallback):
    def _on_step(self):
        for info in self.locals["infos"]:
            for key in ("goal_distance", "formation_error", "shape_error"):
                self.logger.record_mean(f"formation/{key}", info[key])
            for key, value in info["reward_terms"].items():
                self.logger.record_mean(f"reward/{key}", value)
        return True


def _make_training_env(environment):
    return Monitor(FormationPyBulletEnv(**environment.env_kwargs()))


def _ppo_options(experiment, smoke):
    training, ppo = experiment.training, experiment.ppo
    obstacle_course = experiment.environment.stage == 2
    num_envs = 1 if smoke else training.num_envs
    stage_steps = training.obstacle_timesteps if obstacle_course else training.open_timesteps
    timesteps = 128 if smoke else (training.timesteps or stage_steps)
    batch_size = 64 if smoke else ppo.batch_size
    rollout_size = min(ppo.rollout_steps, timesteps)
    n_steps = 64 if smoke else batch_size * ceil(rollout_size / (num_envs * batch_size))
    options = dict(
        device="cpu" if smoke else training.device, seed=training.seed, verbose=1,
        n_steps=n_steps, batch_size=batch_size, n_epochs=ppo.n_epochs, gamma=ppo.gamma, clip_range=ppo.clip_range,
        target_kl=ppo.target_kl, learning_rate=ppo.learning_rate if obstacle_course else ppo.open_learning_rate,
        ent_coef=ppo.ent_coef if obstacle_course else ppo.open_ent_coef, tensorboard_log=None,
    )
    return num_envs, timesteps, options


def _start_run(resources, experiment, resume_from, new_run):
    environment = experiment.environment
    mode = environment.follower_control
    if mode == "distributed":
        mode += f"_{environment.neighbor_velocity}"
    prefix = f"stage{environment.stage}_{environment.map_name}_{mode}_"
    settings = environment.run_settings()
    run_dir, continuing = select_run(Path(experiment.training.output_dir), prefix, settings, resume_from, new_run)
    try:
        resources.enter_context(FileLock(run_dir / ".training.lock", timeout=0))
    except Timeout as exc:
        raise ValueError(f"Another training process is using {run_dir}.") from exc
    return run_dir, continuing, settings


def run_training(experiment, *, resume_from=None, new_run=False, smoke=False):
    experiment.validate()
    resume_from = Path(resume_from) if resume_from is not None else None
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    environment = experiment.environment
    num_envs, timesteps, options = _ppo_options(experiment, smoke)
    factories = [partial(_make_training_env, environment)] * num_envs
    env = DummyVecEnv(factories) if num_envs == 1 else SubprocVecEnv(factories, start_method="spawn")

    with ExitStack() as resources:
        resources.callback(env.close)
        print(f"Map: {environment.map_name}")
        print(f"Follower control: {environment.follower_control}")
        if environment.follower_control == "distributed":
            print(f"Neighbor velocity: {environment.neighbor_velocity}")
        callbacks = [FormationMetrics()]
        run_dir = None
        if not smoke:
            run_dir, continuing, settings = _start_run(resources, experiment, resume_from, new_run)
            frequency = max(experiment.training.checkpoint_steps // num_envs, 1)
            callbacks.append(CheckpointCallback(frequency, str(run_dir / "checkpoints")))

        if resume_from:
            model = load_policy(resume_from, env=env, **options)
        else:
            model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs(), **options)
        if run_dir is not None:
            if continuing:
                check_checkpoint(run_dir, model.num_timesteps)
            record_settings(run_dir, settings)
            record_experiment(run_dir, experiment, resume_from, model.num_timesteps)
            logger = make_logger(run_dir, resume_step=model.num_timesteps if continuing else None)
            resources.callback(logger.close)
            model.set_logger(logger)
            action = "Continuing log" if continuing else "Logging"
            print(f"{action} at step {model.num_timesteps}: {logger.get_dir()}")

        model.learn(total_timesteps=timesteps, callback=callbacks, reset_num_timesteps=resume_from is None)
        if run_dir is not None:
            model.logger.dump(step=model.num_timesteps)
            destination = save_model(model, run_dir)
            print(f"Saved model: {destination}")
            return destination
    return None
