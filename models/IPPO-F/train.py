from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch as th

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from IPPO_env import FormationIPPOEnv
from ippo_shared import SharedIPPOActorCritic, actions_to_dict, build_multi_agent_obs_batch
from project_paths import CHECKPOINTS_DIR, MODELS_DIR, ensure_output_dirs


DEFAULT_MODEL_NAME = "ippo_shared_policy.pt"


def get_agent_names(env: FormationIPPOEnv) -> List[str]:
    return list(getattr(env, "agent_names", list(env.observation_space.spaces.keys())))


def get_agent_space(env: FormationIPPOEnv):
    agent_names = get_agent_names(env)
    return env.observation_space.spaces[agent_names[0]]


def infer_dims(agent_space) -> Tuple[int, int, int]:
    kinematics_dim = int(agent_space.spaces["kinematics"].shape[0])
    lidar_shape = agent_space.spaces["lidar"].shape
    return kinematics_dim, int(lidar_shape[0]), int(lidar_shape[1])


def clone_observations(observations: Dict[str, Dict[str, np.ndarray]]):
    return {
        name: {
            "kinematics": np.asarray(agent_obs["kinematics"], dtype=np.float32).copy(),
            "lidar": np.asarray(agent_obs["lidar"], dtype=np.float32).copy(),
        }
        for name, agent_obs in observations.items()
    }


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_value: float, gamma: float, gae_lambda: float):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    values_ext = np.append(values, last_value).astype(np.float32)

    for step in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * values_ext[step + 1] * next_non_terminal - values_ext[step]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[step] = gae

    returns = advantages + values
    return advantages, returns


def collect_rollout(env: FormationIPPOEnv, policy: SharedIPPOActorCritic, agent_names: List[str], horizon: int, device: th.device):
    observations, _ = env.reset()
    rollout = []
    episode_returns = []

    for _ in range(horizon):
        obs_batch = build_multi_agent_obs_batch(observations, agent_names, device)
        with th.no_grad():
            actions, log_probs, values = policy.act(obs_batch, deterministic=False)

        action_dict = actions_to_dict(actions, agent_names)
        next_observations, rewards, terminated, truncated, infos = env.step(action_dict)

        step_record = {
            "obs": clone_observations(observations),
            "actions": {
                name: np.asarray(action_dict[name], dtype=np.float32).copy() for name in agent_names
            },
            "log_probs": {
                name: float(log_probs[i].item()) for i, name in enumerate(agent_names)
            },
            "values": {
                name: float(values[i].item()) for i, name in enumerate(agent_names)
            },
            "rewards": {
                name: float(rewards[name]) for name in agent_names
            },
            "dones": {
                name: bool(terminated[name] or truncated[name]) for name in agent_names
            },
        }
        rollout.append(step_record)

        episode_info = infos.get("__all__", {}) if isinstance(infos, dict) else {}
        if episode_info.get("reason"):
            episode_returns.append(float(episode_info.get("team_reward", 0.0)))

        observations = next_observations
        if any(step_record["dones"].values()):
            observations, _ = env.reset()

    return rollout, observations, episode_returns


def build_training_batch(policy: SharedIPPOActorCritic, rollout, next_observations, agent_names: List[str], gamma: float, gae_lambda: float, device: th.device):
    next_obs_batch = build_multi_agent_obs_batch(next_observations, agent_names, device)
    with th.no_grad():
        last_values = policy.value(next_obs_batch).detach().cpu().numpy().astype(np.float32)

    obs_kinematics = []
    obs_lidar = []
    actions = []
    old_log_probs = []
    returns = []
    advantages = []

    for agent_index, agent_name in enumerate(agent_names):
        rewards = np.array([step["rewards"][agent_name] for step in rollout], dtype=np.float32)
        values = np.array([step["values"][agent_name] for step in rollout], dtype=np.float32)
        dones = np.array([step["dones"][agent_name] for step in rollout], dtype=np.float32)
        adv, ret = compute_gae(rewards, values, dones, float(last_values[agent_index]), gamma, gae_lambda)

        for step_index, step in enumerate(rollout):
            agent_obs = step["obs"][agent_name]
            obs_kinematics.append(agent_obs["kinematics"])
            obs_lidar.append(agent_obs["lidar"])
            actions.append(step["actions"][agent_name])
            old_log_probs.append(step["log_probs"][agent_name])
            returns.append(ret[step_index])
            advantages.append(adv[step_index])

    obs_batch = {
        "kinematics": th.as_tensor(np.stack(obs_kinematics, axis=0), dtype=th.float32, device=device),
        "lidar": th.as_tensor(np.stack(obs_lidar, axis=0), dtype=th.float32, device=device),
    }
    action_batch = th.as_tensor(np.stack(actions, axis=0), dtype=th.float32, device=device)
    old_log_prob_batch = th.as_tensor(np.asarray(old_log_probs, dtype=np.float32), dtype=th.float32, device=device)
    return_batch = th.as_tensor(np.asarray(returns, dtype=np.float32), dtype=th.float32, device=device)
    advantage_batch = th.as_tensor(np.asarray(advantages, dtype=np.float32), dtype=th.float32, device=device)
    advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std(unbiased=False) + 1e-8)

    return {
        "obs": obs_batch,
        "actions": action_batch,
        "old_log_probs": old_log_prob_batch,
        "returns": return_batch,
        "advantages": advantage_batch,
    }


def update_policy(
    policy: SharedIPPOActorCritic,
    optimizer: th.optim.Optimizer,
    batch,
    batch_size: int,
    epochs: int,
    clip_range: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
):
    total_samples = batch["actions"].shape[0]
    indices = np.arange(total_samples)
    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "updates": 0,
    }

    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, total_samples, batch_size):
            mb_idx = indices[start : start + batch_size]
            obs_mb = {
                "kinematics": batch["obs"]["kinematics"][mb_idx],
                "lidar": batch["obs"]["lidar"][mb_idx],
            }
            action_mb = batch["actions"][mb_idx]
            old_log_prob_mb = batch["old_log_probs"][mb_idx]
            return_mb = batch["returns"][mb_idx]
            advantage_mb = batch["advantages"][mb_idx]

            new_log_prob, entropy, values = policy.evaluate_actions(obs_mb, action_mb)
            ratio = th.exp(new_log_prob - old_log_prob_mb)
            unclipped = ratio * advantage_mb
            clipped = th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage_mb
            policy_loss = -th.min(unclipped, clipped).mean()
            value_loss = 0.5 * (return_mb - values).pow(2).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            approx_kl = (old_log_prob_mb - new_log_prob).mean().abs().item()
            clip_fraction = (th.abs(ratio - 1.0) > clip_range).float().mean().item()

            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["entropy"] += float(entropy_loss.item())
            stats["approx_kl"] += approx_kl
            stats["clip_fraction"] += clip_fraction
            stats["updates"] += 1

    if stats["updates"] > 0:
        for key in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"]:
            stats[key] /= stats["updates"]

    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Train a shared-parameter IPPO policy for all robots.")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--rollout-horizon", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--enable-obstacles", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    return parser.parse_args()


def resolve_device(requested: str):
    if requested == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    return th.device(requested)


def main():
    args = parse_args()
    ensure_output_dirs()
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    env = FormationIPPOEnv(render_mode=False, enable_obstacles=args.enable_obstacles)
    agent_names = get_agent_names(env)
    agent_space = get_agent_space(env)
    kinematics_dim, lidar_channels, lidar_length = infer_dims(agent_space)

    policy = SharedIPPOActorCritic(
        kinematics_dim=kinematics_dim,
        lidar_channels=lidar_channels,
        lidar_length=lidar_length,
    ).to(device)
    optimizer = th.optim.Adam(policy.parameters(), lr=args.learning_rate)

    target_transitions = max(int(args.total_timesteps), 1)
    transitions_per_rollout = args.rollout_horizon * len(agent_names)
    total_updates = max(math.ceil(target_transitions / transitions_per_rollout), 1)
    global_transitions = 0
    start_time = time.time()

    final_model_path = MODELS_DIR / DEFAULT_MODEL_NAME

    try:
        for update_index in range(1, total_updates + 1):
            rollout, next_observations, episode_returns = collect_rollout(
                env=env,
                policy=policy,
                agent_names=agent_names,
                horizon=args.rollout_horizon,
                device=device,
            )
            batch = build_training_batch(
                policy=policy,
                rollout=rollout,
                next_observations=next_observations,
                agent_names=agent_names,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                device=device,
            )
            stats = update_policy(
                policy=policy,
                optimizer=optimizer,
                batch=batch,
                batch_size=args.batch_size,
                epochs=args.epochs,
                clip_range=args.clip_range,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
            )

            global_transitions += transitions_per_rollout
            elapsed = time.time() - start_time
            sps = global_transitions / max(elapsed, 1e-8)
            avg_episode_return = float(np.mean(episode_returns)) if episode_returns else float("nan")

            print(
                f"Update {update_index:04d}/{total_updates:04d} | "
                f"transitions={global_transitions}/{target_transitions} | "
                f"ep_return={avg_episode_return:.3f} | "
                f"policy_loss={stats['policy_loss']:.4f} | value_loss={stats['value_loss']:.4f} | "
                f"entropy={stats['entropy']:.4f} | kl={stats['approx_kl']:.4f} | "
                f"clip_frac={stats['clip_fraction']:.4f} | sps={sps:.1f}"
            )

            if update_index % max(args.checkpoint_interval, 1) == 0:
                checkpoint_path = CHECKPOINTS_DIR / f"ippo_shared_update_{update_index:04d}.pt"
                policy.save(checkpoint_path, metadata={"update": update_index, "timesteps": global_transitions})

        policy.save(final_model_path, metadata={"timesteps": global_transitions, "agent_names": agent_names})
        print(f"Saved final policy to {final_model_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
