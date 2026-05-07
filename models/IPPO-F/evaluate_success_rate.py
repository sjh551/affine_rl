from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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
from project_paths import MODELS_DIR, ensure_output_dirs


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


def load_policy(env: FormationIPPOEnv, device: th.device, model_path: Path):
    agent_space = get_agent_space(env)
    kinematics_dim, lidar_channels, lidar_length = infer_dims(agent_space)

    if model_path.exists():
        policy, _ = SharedIPPOActorCritic.load(model_path, map_location=device)
        policy.to(device)
        return policy

    return SharedIPPOActorCritic(
        kinematics_dim=kinematics_dim,
        lidar_channels=lidar_channels,
        lidar_length=lidar_length,
    ).to(device)


def evaluate_model(model_path: Path, num_episodes: int = 100):
    print(f"Loading policy: {model_path}")
    print(f"Evaluating {num_episodes} episodes without rendering...")

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    env = FormationIPPOEnv(render_mode=False, enable_obstacles=True)
    policy = load_policy(env, device, model_path)
    policy.eval()
    agent_names = get_agent_names(env)

    stats = {"Success": 0, "Crash_Obstacle": 0, "Crash_Team": 0, "Wall_Out": 0, "Timeout": 0, "Unknown": 0}
    step_times = []
    success_steps = []
    formation_error_avgs = []
    formation_error_maxs = []

    start_time = time.time()

    for episode in range(num_episodes):
        observations, _ = env.reset()
        done = False
        steps = 0

        while not done:
            t0 = time.perf_counter()
            obs_batch = build_multi_agent_obs_batch(observations, agent_names, device)
            with th.no_grad():
                actions, _, _ = policy.act(obs_batch, deterministic=True)
            action_dict = actions_to_dict(actions, agent_names)
            observations, rewards, terminated, truncated, infos = env.step(action_dict)
            t1 = time.perf_counter()
            step_times.append(t1 - t0)

            steps += 1
            done = any(bool(terminated[name] or truncated[name]) for name in agent_names)

            if done:
                reason = infos.get("__all__", {}).get("reason", "Unknown") if isinstance(infos, dict) else "Unknown"
                episode_info = infos.get("__all__", {}) if isinstance(infos, dict) else {}
                if "ep_formation_error_avg" in episode_info:
                    formation_error_avgs.append(episode_info["ep_formation_error_avg"])
                if "ep_formation_error_max" in episode_info:
                    formation_error_maxs.append(episode_info["ep_formation_error_max"])

                if "Success" in reason:
                    stats["Success"] += 1
                    success_steps.append(steps)
                elif "Crash_Obstacle" in reason:
                    stats["Crash_Obstacle"] += 1
                elif "Crash_Team" in reason:
                    stats["Crash_Team"] += 1
                elif "Wall_Out" in reason:
                    stats["Wall_Out"] += 1
                elif "Timeout" in reason:
                    stats["Timeout"] += 1
                else:
                    stats["Unknown"] += 1

        if (episode + 1) % 10 == 0:
            print(f"Completed {episode + 1}/{num_episodes} episodes...")

    env.close()

    total_time = time.time() - start_time
    success_rate = (stats["Success"] / num_episodes) * 100.0
    avg_step_time = float(np.mean(step_times)) if step_times else 0.0
    max_step_time = float(np.max(step_times)) if step_times else 0.0
    control_freq = 1.0 / avg_step_time if avg_step_time > 0 else 0.0

    print("\n" + "=" * 45)
    print(" Evaluation Report ")
    print("=" * 45)
    print(f"Total episodes: {num_episodes} | Total time: {total_time:.2f} s")
    print("-" * 45)
    print(f"Success: {stats['Success']}")
    print(f"Obstacle crashes: {stats['Crash_Obstacle']}")
    print(f"Team crashes: {stats['Crash_Team']}")
    print(f"Wall hits: {stats['Wall_Out']}")
    print(f"Timeouts: {stats['Timeout']}")
    print("-" * 45)
    print(f"Success rate: {success_rate:.1f}%")
    if success_steps:
        print(f"Average success steps: {np.mean(success_steps):.1f}")
    if formation_error_avgs:
        print(f"Average formation error: {np.mean(formation_error_avgs):.4f}")
        print(f"Max formation error: {np.mean(formation_error_maxs):.4f}")
    print("-" * 45)
    print(f"Average step time: {avg_step_time * 1000:.2f} ms")
    print(f"Max step time: {max_step_time * 1000:.2f} ms")
    print(f"Control frequency: {control_freq:.1f} Hz")
    print("=" * 45)


if __name__ == "__main__":
    ensure_output_dirs()
    parser = argparse.ArgumentParser(description="Evaluate shared IPPO policy success rate.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    evaluate_model(Path(args.model_path), args.episodes)
