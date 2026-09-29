"""Headless PyBullet comparison of two saved formation configurations.

Optimization is not performed here.  This keeps physical evaluation separate
from both optimizers while applying exactly the same environment settings to
the spectral-radius and spectral-norm results.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from POMDP_env import FormationPyBulletEnv
from utils.formation_utils import (
    get_five_agent_topology,
    get_nominal_configuration,
    get_square_topology,
    get_topology,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_RADIUS_CONFIG = RESULTS_DIR / "optimized_configuration.npy"
DEFAULT_NORM_CONFIG = (
    RESULTS_DIR / "spectral_norm" / "optimized_configuration.npy"
)
DEFAULT_REPORT = RESULTS_DIR / "pybullet_comparison.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two saved formations in the same PyBullet setup"
    )
    parser.add_argument("--radius-config", type=Path, default=DEFAULT_RADIUS_CONFIG)
    parser.add_argument("--norm-config", type=Path, default=DEFAULT_NORM_CONFIG)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--control-mode",
        choices=("centralized", "distributed", "distributed_iterative"),
        default="distributed",
    )
    parser.add_argument("--inner-iterations", type=int, default=1)
    parser.add_argument("--formation-gain", type=float, default=1.0)
    parser.add_argument("--physics-timestep", type=float, default=1.0 / 240.0)
    parser.add_argument("--physics-substeps", type=int, default=24)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def topology_for(configuration: np.ndarray):
    if len(configuration) == len(get_nominal_configuration()):
        return get_topology()
    if len(configuration) == 4:
        return get_square_topology()
    if len(configuration) == 5:
        return get_five_agent_topology()
    raise ValueError(
        "Cannot infer topology for a configuration with "
        f"{len(configuration)} agents."
    )


def planar_constraint_matrix(env: FormationPyBulletEnv) -> np.ndarray:
    indices = np.array([
        agent * env.agent_state_dim + axis
        for agent in range(env.num_robots)
        for axis in (0, 1)
    ])
    return env.M_3d[np.ix_(indices, indices)]


def planar_state(env: FormationPyBulletEnv, states: Dict[int, dict]) -> np.ndarray:
    return np.concatenate([states[i]["pos"][:2] for i in range(env.num_robots)])


def minimum_robot_distance(states: Dict[int, dict]) -> float:
    return float(min(
        np.linalg.norm(states[i]["pos"][:2] - states[j]["pos"][:2])
        for i, j in combinations(range(len(states)), 2)
    ))


def evaluate_configuration(
    configuration: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, object]:
    # POMDP_env currently samples target/start positions through NumPy's
    # legacy global RNG rather than Gym's ``self.np_random``.  Reset it before
    # every candidate so both formations receive exactly the same scenario.
    np.random.seed(args.seed)
    env = FormationPyBulletEnv(
        render_mode=False,
        enable_obstacles=False,
        draw_line=False,
        follower_control_mode=args.control_mode,
        follower_inner_iterations=args.inner_iterations,
        follower_formation_gain=args.formation_gain,
        physics_timestep=args.physics_timestep,
        physics_substeps_per_control=args.physics_substeps,
        nominal_configuration=configuration,
        formation_edges=topology_for(configuration),
        print_follower_spectrum=False,
        max_steps=None,
    )
    try:
        env.reset(seed=args.seed)
        matrix = planar_constraint_matrix(env)
        residual_history: List[float] = []
        max_command = 0.0
        max_actual_speed = 0.0
        min_distance = float("inf")
        terminated = False
        truncated = False

        for _ in range(args.steps):
            _, _, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            command = np.asarray(
                info["follower_formation_velocity_commands"],
                dtype=float,
            )
            if command.size:
                max_command = max(
                    max_command,
                    float(np.max(np.linalg.norm(command[:, :2], axis=1))),
                )

            states = env._get_all_robot_states()
            max_actual_speed = max(
                max_actual_speed,
                max(
                    float(np.linalg.norm(states[i]["vel"][:2]))
                    for i in range(env.n_leaders, env.num_robots)
                ),
            )
            min_distance = min(min_distance, minimum_robot_distance(states))
            residual_history.append(float(np.linalg.norm(
                matrix @ planar_state(env, states)
            )))
            if terminated or truncated:
                break

        return {
            "requested_steps": int(args.steps),
            "executed_steps": int(len(residual_history)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "max_follower_command": max_command,
            "max_actual_follower_speed": max_actual_speed,
            "initial_planar_constraint_residual": residual_history[0],
            "final_planar_constraint_residual": residual_history[-1],
            "max_planar_constraint_residual": max(residual_history),
            "minimum_robot_distance": min_distance,
        }
    finally:
        env.close()


def print_results(results: Dict[str, Dict[str, object]]) -> None:
    print("=" * 78)
    print("Headless PyBullet comparison (zero leader command)")
    for name, metrics in results.items():
        print("-" * 78)
        print(name)
        for key, value in metrics.items():
            print(f"  {key:<36}: {value}")
    print("=" * 78)


def main() -> None:
    args = parse_args()
    paths = {
        "spectral_radius": resolve_path(args.radius_config),
        "spectral_norm": resolve_path(args.norm_config),
    }
    configurations = {
        name: np.asarray(np.load(path), dtype=float)
        for name, path in paths.items()
    }
    results = {
        name: evaluate_configuration(configuration, args)
        for name, configuration in configurations.items()
    }
    print_results(results)

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "settings": {
                    "steps": args.steps,
                    "seed": args.seed,
                    "control_mode": args.control_mode,
                    "inner_iterations": args.inner_iterations,
                    "formation_gain": args.formation_gain,
                    "physics_timestep": args.physics_timestep,
                    "physics_substeps": args.physics_substeps,
                },
                "configuration_files": {
                    name: str(path) for name, path in paths.items()
                },
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved PyBullet comparison: {output_path}")


if __name__ == "__main__":
    main()
