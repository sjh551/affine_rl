"""Compare spectral-radius and spectral-norm optimized configurations.

This script is deliberately matrix-only.  PyBullet validation remains a
separate step through ``evaluate_formation.py --config ...``.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formation_optimization.optimizer import (
    build_follower_planar_iteration_matrix,
)
from utils.formation_graph import FormationGraph
from utils.formation_utils import get_nominal_configuration


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_RADIUS_CONFIG = RESULTS_DIR / "optimized_configuration.npy"
DEFAULT_NORM_CONFIG = (
    RESULTS_DIR / "spectral_norm" / "optimized_configuration.npy"
)
DEFAULT_REPORT = RESULTS_DIR / "optimization_comparison.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare radius-constrained and norm-constrained formations"
    )
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--radius-config", type=Path, default=DEFAULT_RADIUS_CONFIG)
    parser.add_argument("--norm-config", type=Path, default=DEFAULT_NORM_CONFIG)
    parser.add_argument("--n-leaders", type=int, default=2)
    parser.add_argument("--formation-gain", type=float, default=1.0)
    parser.add_argument("--control-interval", type=float, default=0.1)
    parser.add_argument("--power-horizon", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def minimum_pairwise_distance(configuration: np.ndarray) -> float:
    return float(min(
        np.linalg.norm(configuration[i, :2] - configuration[j, :2])
        for i, j in combinations(range(len(configuration)), 2)
    ))


def calculate_metrics(
    configuration: np.ndarray,
    nominal: np.ndarray,
    n_leaders: int,
    formation_gain: float,
    control_interval: float,
    power_horizon: int,
    triplets,
) -> Dict[str, float]:
    degree, _, iteration, sigma_min = build_follower_planar_iteration_matrix(
        configuration,
        n_leaders=n_leaders,
        theta=0.0,
        triplets=triplets,
    )
    eigenvalues = np.linalg.eigvals(iteration)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    spectral_norm = float(np.linalg.norm(iteration, ord=2))

    power = np.eye(iteration.shape[0])
    maximum_power_norm = 1.0
    maximum_power_step = 0
    for step in range(1, power_horizon + 1):
        power = power @ iteration
        power_norm = float(np.linalg.norm(power, ord=2))
        if power_norm > maximum_power_norm:
            maximum_power_norm = power_norm
            maximum_power_step = step

    identity = np.eye(iteration.shape[0])
    correction = identity - iteration
    closed_loop = np.block([
        [
            identity - control_interval * formation_gain * correction,
            control_interval * iteration,
        ],
        [
            -formation_gain * correction,
            iteration,
        ],
    ])

    follower_delta = (
        configuration[n_leaders:, :2] - nominal[n_leaders:, :2]
    )
    return {
        "position_deviation_J": float(np.sum(follower_delta ** 2)),
        "spectral_radius_T": spectral_radius,
        "spectral_norm_T": spectral_norm,
        "sigma_min_D": float(sigma_min),
        "condition_number_D": float(np.linalg.cond(degree, p=2)),
        "inverse_norm_D": float(1.0 / sigma_min),
        "nonnormality_T": float(np.linalg.norm(
            iteration.T @ iteration - iteration @ iteration.T,
            ord=2,
        )),
        "max_T_power_norm": maximum_power_norm,
        "max_T_power_step": int(maximum_power_step),
        "closed_loop_spectral_radius": float(np.max(np.abs(
            np.linalg.eigvals(closed_loop)
        ))),
        "raw_closed_loop_norm": float(np.linalg.norm(closed_loop, ord=2)),
        "minimum_pairwise_distance": minimum_pairwise_distance(configuration),
    }


def print_table(results: Dict[str, Dict[str, float]]) -> None:
    columns = list(results)
    metrics = [
        ("J position", "position_deviation_J"),
        ("rho(T)", "spectral_radius_T"),
        ("||T||_2", "spectral_norm_T"),
        ("sigma_min(D)", "sigma_min_D"),
        ("cond_2(D)", "condition_number_D"),
        ("||D^-1||_2", "inverse_norm_D"),
        ("nonnormality(T)", "nonnormality_T"),
        ("max ||T^k||_2", "max_T_power_norm"),
        ("argmax k", "max_T_power_step"),
        ("rho(F)", "closed_loop_spectral_radius"),
        ("raw ||F||_2", "raw_closed_loop_norm"),
        ("min pair distance", "minimum_pairwise_distance"),
    ]
    width = 22
    print("=" * (width * (len(columns) + 1)))
    print(f"{'Metric':<{width}}" + "".join(
        f"{name:>{width}}" for name in columns
    ))
    print("-" * (width * (len(columns) + 1)))
    for label, key in metrics:
        values = []
        for name in columns:
            value = results[name][key]
            if isinstance(value, int):
                values.append(f"{value:>{width}d}")
            else:
                values.append(f"{value:>{width}.8g}")
        print(f"{label:<{width}}" + "".join(values))
    print("=" * (width * (len(columns) + 1)))
    print(
        "Note: raw ||F||_2 mixes position and velocity units; use it only as "
        "a same-scaling comparison.  ||T||_2 and ||T^k||_2 map velocity to "
        "velocity and are the primary transient-amplification metrics."
    )


def main() -> None:
    args = parse_args()
    nominal = (
        np.asarray(get_nominal_configuration(), dtype=float)
        if args.base_config is None
        else np.asarray(np.load(resolve_path(args.base_config)), dtype=float)
    )
    configurations = {
        "baseline": nominal,
        "radius": np.asarray(
            np.load(resolve_path(args.radius_config)),
            dtype=float,
        ),
        "norm": np.asarray(
            np.load(resolve_path(args.norm_config)),
            dtype=float,
        ),
    }
    for name, configuration in configurations.items():
        if configuration.shape != nominal.shape:
            raise ValueError(
                f"{name} configuration shape {configuration.shape} does not "
                f"match baseline shape {nominal.shape}"
            )

    triplets = tuple(FormationGraph(nominal).triplets)
    results = {
        name: calculate_metrics(
            configuration,
            nominal=nominal,
            n_leaders=args.n_leaders,
            formation_gain=args.formation_gain,
            control_interval=args.control_interval,
            power_horizon=args.power_horizon,
            triplets=triplets,
        )
        for name, configuration in configurations.items()
    }
    print_table(results)

    report_path = resolve_path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_leaders": args.n_leaders,
        "formation_gain": args.formation_gain,
        "control_interval": args.control_interval,
        "power_horizon": args.power_horizon,
        "configuration_files": {
            "baseline": "built-in nominal configuration"
            if args.base_config is None
            else str(resolve_path(args.base_config)),
            "radius": str(resolve_path(args.radius_config)),
            "norm": str(resolve_path(args.norm_config)),
        },
        "metrics": results,
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved comparison report: {report_path}")


if __name__ == "__main__":
    main()
