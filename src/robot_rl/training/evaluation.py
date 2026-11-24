"""Deterministic rollouts and summary statistics for held-out seeds."""

from collections import Counter
from statistics import fmean

import torch

from robot_rl.tasks.formation_navigation.env import FormationPyBulletEnv

from .policy import load_policy


def _run_episode(env, model, seed):
    obs, _ = env.reset(seed=seed)
    episode_return = tracking_sum = 0.0
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        tracking_sum += info["formation_error"]
        steps += 1
        if terminated or truncated:
            return {
                "seed": seed,
                "steps": steps,
                "return": episode_return,
                "success": info["is_success"],
                "reason": info["reason"],
                "mean_tracking_error_m": tracking_sum / steps,
                "final_goal_distance_m": info["goal_distance"],
            }


def evaluate_policy(checkpoint, environment, seeds, device="cpu"):
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = load_policy(checkpoint, device=device)
    env = FormationPyBulletEnv(**environment.env_kwargs())
    try:
        episodes = [_run_episode(env, model, seed) for seed in seeds]
    finally:
        env.close()
    reasons = Counter(episode["reason"] for episode in episodes)
    collisions = sum(reasons[key] for key in ("Crash_Obstacle", "Crash_Team", "Wall_Out"))
    return {
        "task": environment.run_settings(),
        "summary": {
            "episodes": len(episodes),
            "success_rate": fmean(episode["success"] for episode in episodes),
            "collision_rate": collisions / len(episodes),
            "timeout_rate": reasons["Timeout"] / len(episodes),
            "mean_return": fmean(episode["return"] for episode in episodes),
            "mean_steps": fmean(episode["steps"] for episode in episodes),
            "mean_tracking_error_m": fmean(episode["mean_tracking_error_m"] for episode in episodes),
            "mean_final_goal_distance_m": fmean(episode["final_goal_distance_m"] for episode in episodes),
            "termination_counts": dict(reasons),
        },
        "episodes": episodes,
    }
