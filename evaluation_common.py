import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from draw.trajectory_formation_plot import plot_trajectory_and_formation
from project_paths import model_load_path


plt.rcParams["font.family"] = [
    "Times New Roman",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

REASON_CATEGORIES = (
    "Success",
    "Crash_Obstacle",
    "Crash_Team",
    "Wall_Out",
    "Timeout",
    "EvaluationStepLimit",
    "Unknown",
)


def parse_evaluation_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--model",
        default=None,
        help="Model name or path; defaults to the selected stage model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Single seed, or the first seed used with --num-seeds.",
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate explicit seeds, for example: --seeds 0 1 2 3 4.",
    )
    seed_group.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Evaluate consecutive seeds starting at --seed.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--no-walls",
        action="store_true",
        help="Do not create boundary walls or apply Wall_Out termination.",
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional JSON path for per-seed metrics and aggregate statistics.",
    )
    return parser.parse_args()


def resolve_evaluation_seeds(args):
    if args.seeds is not None:
        return args.seeds
    if args.num_seeds is not None:
        if args.num_seeds < 1:
            raise ValueError("--num-seeds must be at least 1.")
        return list(range(args.seed, args.seed + args.num_seeds))
    return [args.seed]


def resolve_model_path(model_name):
    path = Path(model_load_path(model_name))
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Train it first or pass --model."
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
    cross_product = route[0] * offset[1] - route[1] * offset[0]
    return float(abs(cross_product) / route_length)


def run_episode(
    env,
    model,
    args,
    seed,
    expected_kinematics_dim,
    observation_name,
):
    np.random.seed(seed)
    env.action_space.seed(seed)
    if model is not None:
        model.set_random_seed(seed)
    obs, _ = env.reset(seed=seed)
    kinematics = np.asarray(obs["kinematics"])
    expected_shape = (expected_kinematics_dim,)
    if kinematics.shape != expected_shape:
        raise RuntimeError(
            f"Expected kinematics shape {expected_shape}, got {kinematics.shape}."
        )

    trajectories = {index: ([], []) for index in range(len(env.robot_ids))}
    initial_positions = record_positions(env, trajectories)
    center = initial_positions.mean(axis=0)
    start = center.copy()
    target = np.asarray(env.target_pos[:2], dtype=np.float64)
    cross_track_errors = [0.0]
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
        "observation": observation_name,
        "stage": args.stage,
        "obstacles_enabled": bool(env.enable_obstacles),
        "walls_enabled": bool(env.enable_walls),
        "followers_enabled": bool(env.enable_followers),
        "seed": seed,
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


def normalize_reason(reason):
    for category in REASON_CATEGORIES[:-1]:
        if reason.startswith(category):
            return category
    return "Unknown"


def summarize_episodes(episodes):
    numeric_metrics = (
        "steps",
        "return",
        "final_goal_distance",
        "cross_track_error_mean",
        "cross_track_error_max",
        "action_change_mean",
        "formation_error_mean",
        "formation_error_max",
        "reward_progress",
        "reward_shape",
        "reward_collision",
        "reward_smooth",
        "reward_time",
    )
    observed_reason_counts = Counter(
        normalize_reason(episode["reason"]) for episode in episodes
    )
    episode_count = len(episodes)
    reason_counts = {
        reason: int(observed_reason_counts.get(reason, 0))
        for reason in REASON_CATEGORIES
    }
    statistics = {}
    for metric in numeric_metrics:
        values = np.asarray([episode[metric] for episode in episodes], dtype=float)
        statistics[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return {
        "num_episodes": episode_count,
        "seeds": [episode["seed"] for episode in episodes],
        "success_count": reason_counts["Success"],
        "success_rate": float(reason_counts["Success"] / episode_count),
        "reason_counts": reason_counts,
        "reason_rates": {
            reason: float(count / episode_count)
            for reason, count in reason_counts.items()
        },
        "metrics": statistics,
    }


def seed_output_path(output_path, seed, multiple_seeds):
    if output_path is None or not multiple_seeds:
        return output_path
    suffix = output_path.suffix or ".png"
    return output_path.with_name(f"{output_path.stem}_seed{seed}{suffix}")


def run_evaluation(
    env_class,
    stage_configs,
    expected_kinematics_dim,
    observation_name,
    description,
):
    args = parse_evaluation_args(description)
    seeds = resolve_evaluation_seeds(args)
    multiple_seeds = len(seeds) > 1
    config = stage_configs[args.stage]
    args.model = args.model or config["model_name"]

    env = None
    episodes = []
    try:
        model = None
        if not args.random_policy:
            model = PPO.load(resolve_model_path(args.model))

        env = env_class(
            render_mode=args.render,
            enable_obstacles=config["enable_obstacles"],
            enable_followers=config["enable_followers"],
            enable_walls=not args.no_walls,
        )
        if multiple_seeds and not args.no_plot and args.output is None:
            print(
                "Interactive plots are disabled for multi-seed evaluation. "
                "Pass --output to save one image per seed.",
                file=sys.stderr,
            )

        for index, seed in enumerate(seeds, start=1):
            metrics, trajectories = run_episode(
                env,
                model,
                args,
                seed,
                expected_kinematics_dim,
                observation_name,
            )
            episodes.append(metrics)

            should_plot = not args.no_plot and (
                not multiple_seeds or args.output is not None
            )
            if should_plot:
                output_path = seed_output_path(args.output, seed, multiple_seeds)
                if output_path is not None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                plot_trajectory_and_formation(
                    env=env,
                    all_trajectories=trajectories,
                    obstacle_footprints=capture_footprints(
                        getattr(env, "obstacles", [])
                    ),
                    wall_footprints=capture_footprints(
                        getattr(env, "wall_ids", [])
                    ),
                    output_path=(
                        str(output_path) if output_path is not None else None
                    ),
                    show=output_path is None,
                    title=f"Trajectory (seed={seed})",
                )

            if multiple_seeds:
                print(
                    f"Completed {index}/{len(seeds)}: seed={seed}, "
                    f"reason={metrics['reason']}, return={metrics['return']:.3f}",
                    file=sys.stderr,
                )
    finally:
        if env is not None:
            env.close()

    report = (
        episodes[0]
        if not multiple_seeds
        else {
            "summary": summarize_episodes(episodes),
            "episodes": episodes,
        }
    )
    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    print(report_json)

    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(report_json + "\n", encoding="utf-8")
