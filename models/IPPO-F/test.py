from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import torch as th

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from IPPO_env import FormationIPPOEnv
from ippo_shared import SharedIPPOActorCritic, actions_to_dict, build_multi_agent_obs_batch
from project_paths import MODELS_DIR, ensure_output_dirs
from draw.trajectory_formation_plot import plot_trajectory_and_formation


plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_MODEL_PATH = MODELS_DIR / "ippo_shared_policy.pt"


def get_agent_names(env: FormationIPPOEnv):
    return list(getattr(env, "agent_names", list(env.observation_space.spaces.keys())))


def get_agent_space(env: FormationIPPOEnv):
    agent_names = get_agent_names(env)
    return env.observation_space.spaces[agent_names[0]]


def infer_dims(agent_space):
    kinematics_dim = int(agent_space.spaces["kinematics"].shape[0])
    lidar_shape = agent_space.spaces["lidar"].shape
    return kinematics_dim, int(lidar_shape[0]), int(lidar_shape[1])


def load_policy(env: FormationIPPOEnv, device: th.device):
    agent_space = get_agent_space(env)
    kinematics_dim, lidar_channels, lidar_length = infer_dims(agent_space)

    if DEFAULT_MODEL_PATH.exists():
        policy, payload = SharedIPPOActorCritic.load(DEFAULT_MODEL_PATH, map_location=device)
        policy.to(device)
        print(f"Loaded IPPO policy from {DEFAULT_MODEL_PATH}")
        metadata = payload.get("metadata", {})
        if metadata:
            print(f"Metadata: {metadata}")
        return policy

    print("No saved IPPO policy found, using a randomly initialized policy.")
    return SharedIPPOActorCritic(
        kinematics_dim=kinematics_dim,
        lidar_channels=lidar_channels,
        lidar_length=lidar_length,
    ).to(device)


def main():
    ensure_output_dirs()
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    env = FormationIPPOEnv(render_mode=True, enable_obstacles=True)
    agent_names = get_agent_names(env)
    policy = load_policy(env, device)
    policy.eval()

    observations, _ = env.reset()
    done = False

    num_robots = int(getattr(env, "num_robots", len(getattr(env, "robot_ids", [])) or 0))
    if num_robots <= 0:
        num_robots = len(agent_names)

    all_trajectories = {i: ([], []) for i in range(num_robots)}

    print("Running simulation...")

    while not done:
        obs_batch = build_multi_agent_obs_batch(observations, agent_names, device)
        with th.no_grad():
            actions, _, _ = policy.act(obs_batch, deterministic=True)

        action_dict = actions_to_dict(actions, agent_names)
        next_observations, rewards, terminated, truncated, infos = env.step(action_dict)

        for i, robot_id in enumerate(env.robot_ids):
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            all_trajectories[i][0].append(pos[0])
            all_trajectories[i][1].append(pos[1])

        episode_info = infos.get("__all__", {}) if isinstance(infos, dict) else {}
        done = any(bool(terminated[name] or truncated[name]) for name in agent_names)
        observations = next_observations

        if done:
            reason = episode_info.get("reason", "unknown")
            print(f"Episode ended: {reason}")
            print(
                f"Progress: {env.ep_rew_progress:.2f}, Shape: {env.ep_rew_shape:.2f}, "
                f"Collision: {env.ep_rew_collision:.2f}, Smooth: {env.ep_rew_smooth:.2f}, "
                f"Time: {env.ep_rew_time:.2f}, Total: {env.ep_rew_progress + env.ep_rew_shape + env.ep_rew_collision + env.ep_rew_smooth + env.ep_rew_time:.2f}"
            )
            break

    obstacle_footprints = []
    for obs_id in getattr(env, "obstacles", []):
        try:
            aabb_min, aabb_max = p.getAABB(obs_id)
            obstacle_footprints.append((aabb_min, aabb_max))
        except Exception:
            pass

    wall_footprints = []
    for wall_id in getattr(env, "wall_ids", []):
        try:
            aabb_min, aabb_max = p.getAABB(wall_id)
            wall_footprints.append((aabb_min, aabb_max))
        except Exception:
            pass

    env.close()

    print("Drawing trajectories...")
    plot_trajectory_and_formation(
        env=env,
        all_trajectories=all_trajectories,
        obstacle_footprints=obstacle_footprints,
        wall_footprints=wall_footprints,
        output_path=None,
        show=True,
    )


if __name__ == "__main__":
    main()
