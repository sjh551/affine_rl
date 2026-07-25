import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from MDP_env import FormationPyBulletEnv
from draw.trajectory_formation_plot import plot_trajectory_and_formation
from project_paths import model_load_path
from train_mdp import MDP_FINAL_MODEL_NAME, MDP_STAGE1_MODEL_NAME
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = [
    'Times New Roman',
    'Noto Sans CJK SC',
    'DejaVu Sans',
]
plt.rcParams['axes.unicode_minus'] = False

EXPECTED_KINEMATICS_DIM = 36


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO policy in the full-state MDP environment."
    )
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--model",
        default=None,
        help="Model name or path. Defaults to the MDP model for the selected stage.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional evaluation step limit; the environment limit is used by default.",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample policy actions instead of using deterministic predictions.",
    )
    parser.add_argument(
        "--random-policy",
        action="store_true",
        help="Skip model loading and sample random actions (for smoke testing).",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional trajectory figure output path.",
    )
    return parser.parse_args()


def resolve_model_path(model_name):
    path = Path(model_load_path(model_name))
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Train it with train_mdp.py or pass --model."
        )
    return str(path)


def record_positions(env, trajectories):
    positions = []
    for index, robot_id in enumerate(env.robot_ids):
        position, _ = p.getBasePositionAndOrientation(robot_id)
        trajectories[index][0].append(float(position[0]))
        trajectories[index][1].append(float(position[1]))
        positions.append(position[:2])
    return np.asarray(positions, dtype=np.float64)


def capture_footprints(body_ids):
    footprints = []
    for body_id in body_ids:
        try:
            footprints.append(p.getAABB(body_id))
        except p.error:
            continue
    return footprints


def cross_track_error(point, start, target):
    route = target - start
    route_length = np.linalg.norm(route)
    if route_length < 1e-8:
        return 0.0
    offset = point - start
    return float(abs(route[0] * offset[1] - route[1] * offset[0]) / route_length)


def run_episode(env, model, args):
    np.random.seed(args.seed)
    obs, _ = env.reset(seed=args.seed)
    kinematics = np.asarray(obs["kinematics"])
    if kinematics.shape != (EXPECTED_KINEMATICS_DIM,):
        raise RuntimeError(
            "Expected the full-state MDP observation to have shape "
            f"({EXPECTED_KINEMATICS_DIM},), got {kinematics.shape}."
        )

    trajectories = {i: ([], []) for i in range(len(env.robot_ids))}
    initial_positions = record_positions(env, trajectories)
    start = initial_positions.mean(axis=0)
    target = np.asarray(env.target_pos[:2], dtype=np.float64)
    cross_track_errors = [cross_track_error(start, start, target)]
    action_changes = []
    previous_action = None
    episode_return = 0.0
    steps = 0
    done = False
    info = {}

    while not done:
        if args.random_policy:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(
                obs,
                deterministic=not args.stochastic,
            )

        action = np.asarray(action, dtype=np.float32)
        if previous_action is not None:
            action_changes.append(float(np.linalg.norm(action - previous_action)))
        previous_action = action.copy()

        obs, reward, terminated, truncated, info = env.step(action)
        positions = record_positions(env, trajectories)
        center = positions.mean(axis=0)
        cross_track_errors.append(cross_track_error(center, start, target))
        episode_return += float(reward)
        steps += 1
        done = bool(terminated or truncated)

        if args.max_steps is not None and steps >= args.max_steps and not done:
            done = True
            info = dict(info)
            info["reason"] = "EvaluationStepLimit"

    reason = info.get("reason", "Unknown")
    metrics = {
        "observation": "full_state_mdp",
        "stage": args.stage,
        "seed": args.seed,
        "policy": "random" if args.random_policy else "ppo",
        "model": None if args.random_policy else args.model,
        "steps": steps,
        "return": episode_return,
        "success": reason == "Success",
        "reason": reason,
        "final_goal_distance": float(np.linalg.norm(center - target)),
        "cross_track_error_mean": float(np.mean(cross_track_errors)),
        "cross_track_error_max": float(np.max(cross_track_errors)),
        "action_change_mean": (
            float(np.mean(action_changes)) if action_changes else 0.0
        ),
        "formation_error_mean": float(
            env.ep_formation_error_sum / max(steps, 1)
        ),
        "formation_error_max": float(env.ep_formation_error_max),
        "reward_progress": float(env.ep_rew_progress),
        "reward_shape": float(env.ep_rew_shape),
        "reward_collision": float(env.ep_rew_collision),
        "reward_smooth": float(env.ep_rew_smooth),
        "reward_time": float(env.ep_rew_time),
    }
    return metrics, trajectories


def main():
    args = parse_args()
    default_model = MDP_STAGE1_MODEL_NAME if args.stage == 1 else MDP_FINAL_MODEL_NAME
    args.model = args.model or default_model

    model = None
    env = None
    obstacle_footprints = []
    wall_footprints = []
    try:
        if not args.random_policy:
            model = PPO.load(resolve_model_path(args.model))

        env = FormationPyBulletEnv(
            render_mode=args.render,
            enable_obstacles=args.stage == 2,
        )
        metrics, trajectories = run_episode(env, model, args)
        obstacle_footprints = capture_footprints(getattr(env, "obstacles", []))
        wall_footprints = capture_footprints(getattr(env, "wall_ids", []))
    finally:
        if env is not None:
            env.close()

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if not args.no_plot:
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        plot_trajectory_and_formation(
            env=env,
            all_trajectories=trajectories,
            obstacle_footprints=obstacle_footprints,
            wall_footprints=wall_footprints,
            output_path=str(args.output) if args.output is not None else None,
            show=args.output is None,
        )


if __name__ == "__main__":
    main()
