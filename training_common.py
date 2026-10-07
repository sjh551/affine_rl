import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from project_paths import ensure_output_dirs, model_load_path, model_save_base


class LaplacianAttentionEncoder(nn.Module):
    """Encode padded matrix-valued Laplacian rows with masked self-attention."""

    def __init__(self, observation_shape, features_dim=32, num_heads=4):
        super().__init__()
        max_agents, rows_per_agent, matrix_width = observation_shape
        if features_dim % num_heads != 0:
            raise ValueError("features_dim must be divisible by num_heads.")

        self.max_agents = max_agents
        self.input_projection = nn.Linear(
            rows_per_agent * matrix_width,
            features_dim,
        )
        self.pool_token = nn.Parameter(th.zeros(1, 1, features_dim))
        self.position_embedding = nn.Parameter(
            th.zeros(1, max_agents + 1, features_dim)
        )
        self.input_norm = nn.LayerNorm(features_dim)
        self.attention = nn.MultiheadAttention(
            features_dim,
            num_heads,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(features_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(features_dim, 2 * features_dim),
            nn.ReLU(),
            nn.Linear(2 * features_dim, features_dim),
        )
        nn.init.normal_(self.pool_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, laplacian, agent_mask):
        batch_size = laplacian.shape[0]
        node_tokens = self.input_projection(
            laplacian.reshape(batch_size, self.max_agents, -1)
        )
        pool_token = self.pool_token.expand(batch_size, -1, -1)
        tokens = th.cat((pool_token, node_tokens), dim=1)
        tokens = tokens + self.position_embedding

        inactive_agents = agent_mask <= 0.5
        pool_mask = th.zeros(
            (batch_size, 1),
            dtype=th.bool,
            device=agent_mask.device,
        )
        key_padding_mask = th.cat((pool_mask, inactive_agents), dim=1)

        normalized = self.input_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.feed_forward(self.output_norm(tokens))
        return self.output_norm(tokens[:, 0])


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, pretrained_path=None, freeze_encoder=False):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0
        self.lidar_modules = {}
        self.laplacian_encoder = None

        has_laplacian = "laplacian" in observation_space.spaces
        has_agent_mask = "agent_mask" in observation_space.spaces
        if has_laplacian != has_agent_mask:
            raise ValueError(
                "Observation space must provide both laplacian and agent_mask."
            )
        if has_laplacian:
            self.laplacian_encoder = LaplacianAttentionEncoder(
                observation_space["laplacian"].shape
            )
            total_concat_size += 32

        for key, subspace in observation_space.spaces.items():
            if key in ("laplacian", "agent_mask"):
                continue
            if key == "kinematics":
                extractors[key] = nn.Identity()
                total_concat_size += subspace.shape[0]
            elif "lidar" in key:
                cnn = nn.Sequential(
                    nn.Conv1d(subspace.shape[0], 16, kernel_size=5, stride=2),
                    nn.ReLU(),
                    nn.Conv1d(16, 32, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv1d(32, 32, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                with th.no_grad():
                    dummy_input = th.as_tensor(subspace.sample()[None]).float()
                    n_flatten = cnn(dummy_input).shape[1]

                lidar_module = nn.Sequential(
                    cnn,
                    nn.Linear(n_flatten, 64),
                    nn.ReLU(),
                )
                extractors[key] = lidar_module
                self.lidar_modules[key] = lidar_module
                total_concat_size += 64

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained_encoder(pretrained_path)
        if freeze_encoder:
            self.freeze_encoder()

    def load_pretrained_encoder(self, pretrained_path):
        checkpoint = th.load(pretrained_path, map_location="cpu")
        saved_tensors = list(checkpoint.values())

        for lidar_name, lidar_module in self.lidar_modules.items():
            current_params = list(lidar_module.parameters())
            if len(saved_tensors) != len(current_params):
                raise ValueError(
                    f"Pretrained encoder parameter count does not match {lidar_name}."
                )
            with th.no_grad():
                for current_param, saved_param in zip(current_params, saved_tensors):
                    current_param.copy_(saved_param.view_as(current_param))

    def freeze_encoder(self):
        for lidar_module in self.lidar_modules.values():
            for parameter in lidar_module.parameters():
                parameter.requires_grad = False

    def forward(self, observations):
        encoded = [
            extractor(observations[key])
            for key, extractor in self.extractors.items()
        ]
        if self.laplacian_encoder is not None:
            encoded.append(
                self.laplacian_encoder(
                    observations["laplacian"],
                    observations["agent_mask"],
                )
            )
        return th.cat(encoded, dim=1)


class TensorboardRewardCallback(BaseCallback):
    def _on_step(self):
        reward_keys = (
            "ep_rew_progress",
            "ep_rew_shape",
            "ep_rew_collision",
            "ep_rew_smooth",
            "ep_rew_time",
        )
        outcome_keys = {
            "crash_team_rate": "outcome_crash_team",
            "crash_obstacle_rate": "outcome_crash_obstacle",
            "wall_out_rate": "outcome_wall_out",
            "timeout_rate": "outcome_timeout",
        }

        for info in self.locals.get("infos", []):
            if "ep_rew_progress" not in info:
                continue

            for key in reward_keys:
                self.logger.record_mean(f"custom_rewards/{key}", info[key])
            self.logger.record_mean(
                "custom_rewards/ep_rew_total",
                sum(info[key] for key in reward_keys),
            )
            self.logger.record_mean(
                "custom_metrics/success_rate",
                float(info.get("reason") == "Success"),
            )
            for metric_name, info_key in outcome_keys.items():
                self.logger.record_mean(
                    f"custom_outcomes/{metric_name}",
                    float(info.get(info_key, 0.0)),
                )
            if "ep_formation_error_avg" in info:
                self.logger.record_mean(
                    "custom_metrics/formation_error_avg",
                    info["ep_formation_error_avg"],
                )
            if "ep_formation_error_max" in info:
                self.logger.record_mean(
                    "custom_metrics/formation_error_max",
                    info["ep_formation_error_max"],
                )
        return True


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


def ppo_training_kwargs(stage):
    if stage == 1:
        return {
            "learning_rate": 1e-4,
            "n_epochs": 5,
            "clip_range": 0.1,
            "ent_coef": 5e-4,
            "target_kl": 0.03,
        }
    return {
        "learning_rate": 3e-5,
        "n_epochs": 5,
        "clip_range": 0.1,
        "ent_coef": 0.001,
        "target_kl": 0.03,
    }


def build_stage_configs(
    prefix,
    stage1_model,
    stage2_model,
    stage3_model,
    stage1_tensorboard_dir,
    stage2_tensorboard_dir,
    stage3_tensorboard_dir,
):
    return {
        1: {
            "enable_obstacles": False,
            "enable_followers": False,
            "timesteps": 1_000_000,
            "tensorboard_dir": stage1_tensorboard_dir,
            "model_name": stage1_model,
            "default_resume": None,
            "run_name": f"{prefix}_stage1_leaders_navigation",
        },
        2: {
            "enable_obstacles": True,
            "enable_followers": False,
            "timesteps": 3_000_000,
            "tensorboard_dir": stage2_tensorboard_dir,
            "model_name": stage2_model,
            "default_resume": stage1_model,
            "run_name": f"{prefix}_stage2_leaders_obstacles",
        },
        3: {
            "enable_obstacles": True,
            "enable_followers": True,
            "timesteps": 5_000_000,
            "tensorboard_dir": stage3_tensorboard_dir,
            "model_name": stage3_model,
            "default_resume": stage2_model,
            "run_name": f"{prefix}_stage3_full_formation",
        },
    }


def parse_training_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Base environment seed. Defaults to 0 for a new stage and must be "
            "specified explicitly when continuing the same stage."
        ),
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Model name or path; defaults to the previous stage for stages 2/3.",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Disable automatic previous-stage initialization.",
    )
    parser.add_argument(
        "--continue-training",
        action="store_true",
        help=(
            "Continue the current stage model and TensorBoard curve without "
            "resetting timesteps. May be combined with --resume-from for a "
            "same-stage checkpoint."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a 128-step CPU smoke test without saving a model.",
    )
    return parser.parse_args()


def make_env_factory(env_class, rank, config, seed):
    def _init():
        env_seed = seed + rank
        np.random.seed(env_seed)
        env = env_class(
            render_mode=False,
            enable_obstacles=config["enable_obstacles"],
            enable_followers=config["enable_followers"],
        )
        env.reset(seed=env_seed)
        return Monitor(env)

    return _init


def create_vectorized_env(env_class, num_envs, config, seed):
    factories = [
        make_env_factory(env_class, rank, config, seed)
        for rank in range(num_envs)
    ]
    if num_envs > 1:
        return SubprocVecEnv(factories)
    return DummyVecEnv(factories)


def existing_model_path(name_or_path):
    resolved = Path(model_load_path(name_or_path))
    if not resolved.exists():
        raise FileNotFoundError(f"Model not found: {resolved}")
    return str(resolved)


def is_current_stage_model(name_or_path, current_model_name):
    """Recognize the saved model of the stage being trained."""
    if name_or_path is None:
        return False
    candidate = Path(model_load_path(name_or_path)).stem
    current = Path(current_model_name).stem
    return candidate == current


def rollout_parameters(num_envs, smoke):
    if smoke:
        return 64, 64

    batch_size = 256
    n_steps = max(
        256,
        math.ceil(4096 / (num_envs * batch_size)) * batch_size,
    )
    return n_steps, batch_size


def run_training(
    env_class,
    stage_configs,
    expected_kinematics_dim,
    checkpoint_dir,
    description,
):
    args = parse_training_args(description)
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1.")
    if args.from_scratch and (args.resume_from or args.continue_training):
        raise ValueError(
            "--from-scratch cannot be combined with --resume-from or "
            "--continue-training."
        )
    if args.smoke and args.continue_training:
        raise ValueError("Do not append a smoke test to a real training run.")

    config = stage_configs[args.stage]
    resume_from = args.resume_from
    if args.continue_training and resume_from is None:
        resume_from = config["model_name"]
    elif not args.from_scratch and resume_from is None:
        resume_from = config["default_resume"]

    continue_same_stage = args.continue_training or is_current_stage_model(
        resume_from,
        config["model_name"],
    )
    if args.seed is None:
        if continue_same_stage:
            raise ValueError(
                "Same-stage continuation requires an explicit new --seed to "
                "avoid replaying the previous environment sequence."
            )
        args.seed = 0

    ensure_output_dirs()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for stage_config in stage_configs.values():
        stage_config["tensorboard_dir"].mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.num_envs = 1
        args.device = "cpu"

    total_timesteps = args.timesteps or config["timesteps"]
    if args.smoke:
        total_timesteps = 128
    n_steps, batch_size = rollout_parameters(args.num_envs, args.smoke)

    env = create_vectorized_env(env_class, args.num_envs, config, args.seed)
    try:
        shape = env.observation_space["kinematics"].shape
        if shape != (expected_kinematics_dim,):
            raise RuntimeError(
                f"Expected kinematics shape ({expected_kinematics_dim},), got {shape}."
            )

        print(
            f"Stage {args.stage}: obstacles={config['enable_obstacles']}, "
            f"followers={config['enable_followers']}, resume={resume_from}, "
            f"continue={continue_same_stage}, env_seeds="
            f"{args.seed}..{args.seed + args.num_envs - 1}"
        )
        training_kwargs = ppo_training_kwargs(args.stage)
        common_kwargs = {
            "env": env,
            "batch_size": batch_size,
            "n_steps": n_steps,
            "tensorboard_log": str(config["tensorboard_dir"]),
            "device": args.device,
            "seed": args.seed,
            **training_kwargs,
        }

        if resume_from is not None:
            model = PPO.load(existing_model_path(resume_from), **common_kwargs)
        else:
            model = PPO(
                "MultiInputPolicy",
                policy_kwargs=policy_kwargs(),
                verbose=1,
                **common_kwargs,
            )

        callbacks = [TensorboardRewardCallback()]
        if args.stage in (2, 3) and not args.smoke:
            callbacks.append(
                CheckpointCallback(
                    save_freq=max(1_000_000 // args.num_envs, 1),
                    save_path=str(checkpoint_dir),
                    name_prefix=(
                        f"stage{args.stage}_from_{resume_from or 'scratch'}"
                    ),
                )
            )

        run_name = config["run_name"] + ("_smoke" if args.smoke else "")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=run_name,
            reset_num_timesteps=not continue_same_stage,
        )

        if not args.smoke:
            output_base = model_save_base(config["model_name"])
            model.save(output_base)
            print(f"Saved model: {output_base}.zip")
    finally:
        env.close()
