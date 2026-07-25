import argparse
from pathlib import Path

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from MDP_env import FormationPyBulletEnv
from project_paths import (
    CHECKPOINTS_DIR,
    MODEL_FINAL_NAME,
    MODEL_STAGE1_NAME,
    TB_STAGE1_DIR,
    TB_STAGE2_DIR,
    ensure_output_dirs,
    model_load_path,
    model_save_base,
)
from train import CustomCombinedExtractor, TensorboardRewardCallback


MDP_STAGE1_MODEL_NAME = f"mdp_{MODEL_STAGE1_NAME}"
MDP_FINAL_MODEL_NAME = f"mdp_{MODEL_FINAL_NAME}"
MDP_STAGE1_TB_DIR = TB_STAGE1_DIR / "mdp_full_state"
MDP_STAGE2_TB_DIR = TB_STAGE2_DIR / "mdp_full_state"
MDP_CHECKPOINT_DIR = CHECKPOINTS_DIR / "mdp_full_state"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO with full follower-state observations."
    )
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Model name or path. Stage 2 defaults to the MDP Stage 1 model.",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Do not initialize Stage 2 from the MDP Stage 1 model.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short CPU-friendly training smoke test.",
    )
    return parser.parse_args()


def make_env(rank, enable_obstacles, seed):
    def _init():
        np.random.seed(seed + rank)
        env = FormationPyBulletEnv(
            render_mode=False,
            enable_obstacles=enable_obstacles,
        )
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def create_vectorized_env(num_envs, enable_obstacles, seed):
    factories = [
        make_env(rank, enable_obstacles, seed) for rank in range(num_envs)
    ]
    if num_envs > 1:
        return SubprocVecEnv(factories)
    return DummyVecEnv(factories)


def policy_kwargs():
    return {
        "features_extractor_class": CustomCombinedExtractor,
        "features_extractor_kwargs": {
            "pretrained_path": None,
            "freeze_encoder": False,
        },
        "activation_fn": nn.ReLU,
        "net_arch": {"pi": [128, 128], "vf": [128, 128]},
    }


def existing_model_path(name_or_path):
    resolved = Path(model_load_path(name_or_path))
    if not resolved.exists():
        raise FileNotFoundError(f"Model not found: {resolved}")
    return str(resolved)


def main():
    args = parse_args()
    ensure_output_dirs()
    for output_dir in (
        MDP_STAGE1_TB_DIR,
        MDP_STAGE2_TB_DIR,
        MDP_CHECKPOINT_DIR,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.num_envs = 1
        args.device = "cpu"

    stage_timesteps = 3_000_000 if args.stage == 1 else 5_000_000
    total_timesteps = args.timesteps or stage_timesteps
    if args.smoke:
        total_timesteps = 128
        n_steps = 64
        batch_size = 64
    else:
        n_steps = max(4096 // args.num_envs, 256)
        batch_size = 256

    env = create_vectorized_env(
        args.num_envs,
        enable_obstacles=args.stage == 2,
        seed=args.seed,
    )

    kinematics_shape = env.observation_space["kinematics"].shape
    if kinematics_shape != (36,):
        env.close()
        raise RuntimeError(
            f"Expected full-state kinematics shape (36,), got {kinematics_shape}"
        )

    tensorboard_dir = (
        MDP_STAGE1_TB_DIR if args.stage == 1 else MDP_STAGE2_TB_DIR
    )
    callbacks = [TensorboardRewardCallback()]

    resume_from = args.resume_from
    if args.stage == 2 and not args.from_scratch and resume_from is None:
        resume_from = MDP_STAGE1_MODEL_NAME

    if resume_from is not None:
        model = PPO.load(
            existing_model_path(resume_from),
            env=env,
            batch_size=batch_size,
            n_steps=n_steps,
            learning_rate=3e-4 if args.stage == 1 else 3e-5,
            n_epochs=10 if args.stage == 1 else 5,
            clip_range=0.2 if args.stage == 1 else 0.1,
            ent_coef=0.0 if args.stage == 1 else 0.001,
            tensorboard_log=str(tensorboard_dir),
            device=args.device,
        )
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            policy_kwargs=policy_kwargs(),
            learning_rate=3e-4,
            batch_size=batch_size,
            n_steps=n_steps,
            tensorboard_log=str(tensorboard_dir),
            device=args.device,
            seed=args.seed,
        )

    if args.stage == 2 and not args.smoke:
        save_freq = max(1_000_000 // args.num_envs, 1)
        callbacks.append(
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(MDP_CHECKPOINT_DIR),
                name_prefix=f"mdp_stage2_from_{resume_from or 'scratch'}",
            )
        )

    run_name = f"mdp_stage{args.stage}_full_state"
    if args.smoke:
        run_name += "_smoke"
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=run_name,
    )

    if not args.smoke:
        output_name = (
            MDP_STAGE1_MODEL_NAME if args.stage == 1 else MDP_FINAL_MODEL_NAME
        )
        model.save(model_save_base(output_name))
        print(f"Saved MDP model: {model_save_base(output_name)}.zip")

    env.close()


if __name__ == "__main__":
    main()
