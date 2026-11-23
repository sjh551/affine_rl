"""Train a PPO policy for the three affine leaders."""

import argparse
from functools import partial
from math import ceil
from pathlib import Path
from tempfile import mkdtemp

import torch as th
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from controllers.distributed import FOLLOWER_CONTROL_MODES
from POMDP_env import FormationPyBulletEnv
from training_common import FormationFeaturesExtractor


class FormationMetrics(BaseCallback):
    def _on_step(self):
        for info in self.locals["infos"]:
            for key in ("goal_distance", "formation_error", "shape_error"):
                self.logger.record_mean(f"formation/{key}", info[key])
        return True


def make_env(enable_obstacles, enable_walls, follower_control_mode="centralized"):
    return Monitor(FormationPyBulletEnv(enable_obstacles=enable_obstacles, enable_walls=enable_walls,
                                       follower_control_mode=follower_control_mode))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), default=2,
                        help="1: navigation without obstacles. 2: navigation with obstacles (default).")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-from", type=Path, help="Path to a checkpoint of the three-leader policy.")
    parser.add_argument("--no-walls", action="store_true")
    parser.add_argument("--follower-control", choices=FOLLOWER_CONTROL_MODES, default="centralized",
                        help="Follower velocity controller (default: centralized).")
    parser.add_argument("--smoke", action="store_true", help="Run 128 CPU training steps without saving artifacts.")
    args = parser.parse_args()
    if args.num_envs < 1 or (args.timesteps is not None and args.timesteps < 1) or args.seed < 0:
        parser.error("Use positive --num-envs and --timesteps, and a nonnegative --seed.")

    num_envs = 1 if args.smoke else args.num_envs
    device = "cpu" if args.smoke else args.device
    th.set_num_threads(min(th.get_num_threads(), 4))
    obstacles = args.stage == 2
    timesteps = 128 if args.smoke else (args.timesteps or (5_000_000 if obstacles else 1_000_000))
    batch_size = 64 if args.smoke else 256
    rollout_size = min(4096, timesteps)
    n_steps = 64 if args.smoke else batch_size * ceil(rollout_size / (num_envs * batch_size))
    factories = [partial(make_env, obstacles, not args.no_walls, args.follower_control)] * num_envs
    env = DummyVecEnv(factories) if num_envs == 1 else SubprocVecEnv(factories, start_method="spawn")

    try:
        print(f"Follower control: {args.follower_control}")
        callbacks = [FormationMetrics()]
        run_dir = None
        if not args.smoke:
            output_root = Path(__file__).resolve().parent / "outputs"
            output_root.mkdir(exist_ok=True)
            run_dir = Path(mkdtemp(prefix=f"stage{args.stage}_{args.follower_control}_", dir=output_root))
            callbacks.append(CheckpointCallback(max(1_000_000 // num_envs, 1), str(run_dir / "checkpoints")))

        kwargs = dict(env=env, device=device, seed=args.seed, verbose=1, n_steps=n_steps,
                      batch_size=batch_size, n_epochs=5, clip_range=0.1, target_kl=0.03,
                      learning_rate=3e-5 if obstacles else 1e-4, ent_coef=0.001 if obstacles else 5e-4,
                      tensorboard_log=str(run_dir / "tensorboard") if run_dir else None)
        if args.resume_from:
            model = PPO.load(args.resume_from, **kwargs)
        else:
            model = PPO("MultiInputPolicy", policy_kwargs={
                "features_extractor_class": FormationFeaturesExtractor,
                "activation_fn": nn.ReLU,
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            }, **kwargs)
        model.learn(total_timesteps=timesteps, callback=callbacks, reset_num_timesteps=args.resume_from is None)
        if run_dir:
            model.save(run_dir / "ppo_affine.zip")
            print(f"Saved model: {run_dir / 'ppo_affine.zip'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
