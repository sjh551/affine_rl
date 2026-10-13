"""Minimize follower spectral radius on strictly ordered leader-follower paths.

This experiment complements ``verify_triplet_betweenness.py``.  The latter
samples feasible configurations; this script actually optimizes the adjacent
path gaps for each requested agent count and compares the optimum with equal
spacing.  Leaders stay fixed and every normalized adjacent gap is constrained
to be at least ``gap_margin``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formation_optimization.optimizer import (
    build_follower_planar_iteration_matrix,
)
from formation_optimization.verify_triplet_betweenness import (
    _configuration_from_path_fractions,
    _path_base_configuration,
    _path_triplets,
    analyze_configuration,
)


def _configuration_from_gaps(
    n_agents: int,
    gaps: np.ndarray,
    theta: float,
) -> np.ndarray:
    n_gaps = n_agents - 1
    gaps = np.asarray(gaps, dtype=float)
    if gaps.shape == (n_gaps,):
        gaps = np.tile(gaps, (2, 1))
    elif gaps.size == 2 * n_gaps:
        gaps = gaps.reshape(2, n_gaps)
    else:
        raise ValueError(
            f"gaps must contain {n_gaps} or {2 * n_gaps} values"
        )
    fractions = np.cumsum(gaps, axis=1)[:, :-1]
    return _configuration_from_path_fractions(
        _path_base_configuration(n_agents),
        fractions,
        theta=theta,
    )


def _spectral_metrics(
    n_agents: int,
    gaps: np.ndarray,
    theta: float,
) -> tuple[float, float]:
    configuration = _configuration_from_gaps(n_agents, gaps, theta)
    _, _, iteration, _ = build_follower_planar_iteration_matrix(
        configuration,
        n_leaders=2,
        theta=theta,
        triplets=_path_triplets(n_agents),
    )
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(iteration))))
    spectral_norm = float(np.linalg.norm(iteration, 2))
    return spectral_radius, spectral_norm


def _minimum_pairwise_distance(configuration: np.ndarray) -> float:
    positions = np.asarray(configuration, dtype=float)[:, :2]
    distances = [
        np.linalg.norm(positions[i] - positions[j])
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
    ]
    return float(min(distances))


def _pairwise_squared_distance_residuals(
    configuration: np.ndarray,
    minimum_distance: float,
) -> np.ndarray:
    """Return ``||p_i-p_j||^2-d_min^2`` for every unordered pair."""

    positions = np.asarray(configuration, dtype=float)[:, :2]
    return np.asarray(
        [
            np.dot(positions[i] - positions[j], positions[i] - positions[j])
            - minimum_distance**2
            for i in range(len(positions))
            for j in range(i + 1, len(positions))
        ],
        dtype=float,
    )


def _maximize_minimum_distance(
    n_agents: int,
    gap_margin: float,
    starts: int,
    seed: int,
    theta: float,
    maxiter: int,
    tolerance: float,
) -> tuple[float, np.ndarray]:
    """Numerically maximize the all-pairs minimum distance on the monotone path."""

    n_gaps = n_agents - 1
    rng = np.random.default_rng(seed)
    residual = 1.0 - n_gaps * gap_margin
    gap_starts = [
        np.full((2, n_gaps), 1.0 / n_gaps, dtype=float).reshape(-1)
    ]
    gap_starts.extend(
        np.vstack(
            [
                gap_margin + residual * rng.dirichlet(np.ones(n_gaps)),
                gap_margin + residual * rng.dirichlet(np.ones(n_gaps)),
            ]
        ).reshape(-1)
        for _ in range(max(0, starts - 1))
    )

    bounds = [
        *[(gap_margin, 1.0) for _ in range(2 * n_gaps)],
        (0.0, None),
    ]

    def equality(variable: np.ndarray) -> np.ndarray:
        gaps = variable[:-1].reshape(2, n_gaps)
        return np.sum(gaps, axis=1) - 1.0

    def distance_residuals(variable: np.ndarray) -> np.ndarray:
        gaps = variable[:-1]
        distance = float(variable[-1])
        return _pairwise_squared_distance_residuals(
            _configuration_from_gaps(n_agents, gaps, theta),
            distance,
        )

    candidates = []
    for start_index, gaps in enumerate(gap_starts):
        configuration = _configuration_from_gaps(n_agents, gaps, theta)
        initial_distance = _minimum_pairwise_distance(configuration)
        variable = np.concatenate((gaps, [0.95 * initial_distance]))
        result = minimize(
            lambda values: -float(values[-1]),
            x0=variable,
            method="SLSQP",
            bounds=bounds,
            constraints=(
                {"type": "eq", "fun": equality},
                {"type": "ineq", "fun": distance_residuals},
            ),
            options={
                "maxiter": int(maxiter),
                "ftol": float(tolerance),
                "disp": False,
            },
        )
        optimized_gaps = np.asarray(result.x[:-1], dtype=float)
        configuration = _configuration_from_gaps(
            n_agents,
            optimized_gaps,
            theta,
        )
        actual_distance = _minimum_pairwise_distance(configuration)
        gap_matrix = optimized_gaps.reshape(2, n_gaps)
        feasible = bool(
            np.max(np.abs(np.sum(gap_matrix, axis=1) - 1.0)) <= 1.0e-7
            and np.min(gap_matrix) >= gap_margin - 1.0e-8
        )
        if feasible:
            candidates.append((actual_distance, start_index, optimized_gaps))

    if not candidates:
        raise RuntimeError(
            f"minimum-distance feasibility search failed for n={n_agents}"
        )
    maximum_distance, _, best_gaps = max(candidates, key=lambda item: item[0])
    return float(maximum_distance), best_gaps


def optimize_agent_count(
    n_agents: int,
    gap_margin: float,
    minimum_distance: float,
    starts: int,
    seed: int,
    theta: float,
    maxiter: int,
    tolerance: float,
) -> Dict[str, object]:
    n_gaps = n_agents - 1
    if minimum_distance < 0.0:
        raise ValueError("minimum_distance must be nonnegative")
    leader_positions = _path_base_configuration(n_agents)[:2, :2]
    leader_delta = leader_positions[1] - leader_positions[0]
    leader_distance = float(np.linalg.norm(leader_delta))
    monotone_path_length_upper_bound = float(np.sum(np.abs(leader_delta)))
    distance_necessary_upper_bound = monotone_path_length_upper_bound / n_gaps
    if gap_margin <= 0.0 or n_gaps * gap_margin >= 1.0:
        raise ValueError(
            f"gap_margin={gap_margin} is infeasible for n={n_agents}; "
            f"require 0 < gap_margin < {1.0 / n_gaps:.10f}"
        )
    if n_gaps * minimum_distance >= monotone_path_length_upper_bound - 1.0e-12:
        raise ValueError(
            f"infeasible for n={n_agents}: d_min={minimum_distance}; every "
            f"componentwise-monotone path has length below "
            f"{monotone_path_length_upper_bound:.10f}, while "
            f"(n-1)*d_min={n_gaps * minimum_distance:.10f}"
        )
    if starts <= 0:
        raise ValueError("starts must be positive")

    maximum_distance, maximum_distance_gaps = _maximize_minimum_distance(
        n_agents=n_agents,
        gap_margin=gap_margin,
        starts=max(10, min(starts, 30)),
        seed=seed + 500000,
        theta=theta,
        maxiter=maxiter,
        tolerance=tolerance,
    )
    if minimum_distance > maximum_distance + 1.0e-7:
        raise ValueError(
            f"numerically infeasible for n={n_agents}: d_min={minimum_distance}, "
            f"largest distance found under strict componentwise ordering is "
            f"{maximum_distance:.10f}"
        )

    rng = np.random.default_rng(seed)
    residual = 1.0 - n_gaps * gap_margin
    initial_points = [
        maximum_distance_gaps.copy(),
        np.full((2, n_gaps), 1.0 / n_gaps, dtype=float).reshape(-1),
    ]
    initial_points.extend(
        np.vstack(
            [
                gap_margin + residual * rng.dirichlet(np.ones(n_gaps)),
                gap_margin + residual * rng.dirichlet(np.ones(n_gaps)),
            ]
        ).reshape(-1)
        for _ in range(max(0, starts - len(initial_points)))
    )

    bounds = [(gap_margin, 1.0) for _ in range(2 * n_gaps)]
    equality_constraint = {
        "type": "eq",
        "fun": lambda gaps: np.sum(
            np.asarray(gaps).reshape(2, n_gaps),
            axis=1,
        ) - 1.0,
    }

    def objective(gaps: np.ndarray) -> float:
        try:
            return _spectral_metrics(n_agents, gaps, theta)[0]
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 1.0e6

    distance_constraint = {
        "type": "ineq",
        "fun": lambda gaps: _pairwise_squared_distance_residuals(
            _configuration_from_gaps(n_agents, gaps, theta),
            minimum_distance,
        ),
    }

    candidates = []
    for start_index, initial in enumerate(initial_points):
        result = minimize(
            objective,
            x0=initial,
            method="SLSQP",
            bounds=bounds,
            constraints=(equality_constraint, distance_constraint),
            options={
                "maxiter": int(maxiter),
                "ftol": float(tolerance),
                "disp": False,
            },
        )
        gaps = np.asarray(result.x, dtype=float)
        gaps_2d = gaps.reshape(2, n_gaps)
        feasible = bool(
            np.max(np.abs(np.sum(gaps_2d, axis=1) - 1.0)) <= 1.0e-7
            and np.min(gaps_2d) >= gap_margin - 1.0e-8
            and _minimum_pairwise_distance(
                _configuration_from_gaps(n_agents, gaps, theta)
            ) >= minimum_distance - 1.0e-7
        )
        if feasible:
            candidates.append((float(objective(gaps)), start_index, result, gaps))

    if not candidates:
        raise RuntimeError(
            f"No feasible SLSQP result for n={n_agents}, margin={gap_margin}"
        )

    optimized_rho, best_start, best_result, optimized_gaps = min(
        candidates,
        key=lambda item: item[0],
    )
    optimized_configuration = _configuration_from_gaps(
        n_agents,
        optimized_gaps,
        theta,
    )
    optimized_diagnostics = analyze_configuration(
        f"optimized_{n_agents}_agents_margin_{gap_margin}",
        optimized_configuration,
        n_leaders=2,
        theta=theta,
        tolerance=1.0e-10,
        triplets=_path_triplets(n_agents),
    )
    optimized_norm = optimized_diagnostics.full_spectral_norm

    equal_gaps = np.full((2, n_gaps), 1.0 / n_gaps, dtype=float)
    equal_rho, equal_norm = _spectral_metrics(n_agents, equal_gaps, theta)
    equal_formula = float(np.cos(np.pi / (n_agents - 1)))

    return {
        "n_agents": int(n_agents),
        "n_followers": int(n_agents - 2),
        "gap_margin": float(gap_margin),
        "minimum_distance": float(minimum_distance),
        "leader_distance": leader_distance,
        "monotone_path_length_upper_bound": monotone_path_length_upper_bound,
        "distance_necessary_upper_bound": distance_necessary_upper_bound,
        "maximum_distance_found": maximum_distance,
        "starts": int(starts),
        "seed": int(seed),
        "equal_spacing": {
            "gaps": equal_gaps.tolist(),
            "spectral_radius": equal_rho,
            "spectral_norm": equal_norm,
            "analytic_spectral_radius": equal_formula,
            "formula_error": abs(equal_rho - equal_formula),
        },
        "optimized": {
            "gaps": optimized_gaps.tolist(),
            "spectral_radius": optimized_rho,
            "spectral_norm": optimized_norm,
            "improvement_from_equal": equal_rho - optimized_rho,
            "minimum_pairwise_distance": _minimum_pairwise_distance(
                optimized_configuration
            ),
            "strict_betweenness": optimized_diagnostics.strict_betweenness,
            "every_follower_reaches_leader": all(
                direction.every_follower_reaches_leader
                for direction in optimized_diagnostics.directions
            ),
            "configuration": optimized_configuration.tolist(),
        },
        "solver": {
            "best_start": int(best_start),
            "feasible_results": int(len(candidates)),
            "success": bool(best_result.success),
            "message": str(best_result.message),
            "iterations": int(best_result.nit),
            "function_evaluations": int(best_result.nfev),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize rho(T_ff) under strict path-betweenness gaps",
    )
    parser.add_argument(
        "--agent-counts",
        type=int,
        nargs="+",
        default=[4, 5, 6, 7, 8],
    )
    parser.add_argument(
        "--gap-margins",
        type=float,
        nargs="+",
        default=[0.02, 0.10],
    )
    parser.add_argument(
        "--min-distances",
        type=float,
        nargs="+",
        default=[0.0],
        help="Euclidean all-pairs distance limits to test",
    )
    parser.add_argument("--starts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "formation_optimization/results/path_betweenness_optimization.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    agent_counts = list(dict.fromkeys(int(value) for value in args.agent_counts))
    gap_margins = list(dict.fromkeys(float(value) for value in args.gap_margins))
    minimum_distances = list(
        dict.fromkeys(float(value) for value in args.min_distances)
    )

    runs: List[Dict[str, object]] = []
    infeasible_runs: List[Dict[str, object]] = []
    for distance_index, minimum_distance in enumerate(minimum_distances):
        for gap_index, gap_margin in enumerate(gap_margins):
            print("=" * 80)
            print(
                f"gap_margin={gap_margin:.6f}, "
                f"minimum_distance={minimum_distance:.6f}"
            )
            for n_agents in agent_counts:
                try:
                    run = optimize_agent_count(
                        n_agents=n_agents,
                        gap_margin=gap_margin,
                        minimum_distance=minimum_distance,
                        starts=args.starts,
                        seed=(
                            args.seed
                            + 100000 * distance_index
                            + 1000 * gap_index
                            + n_agents
                        ),
                        theta=args.theta,
                        maxiter=args.maxiter,
                        tolerance=args.tolerance,
                    )
                except (ValueError, RuntimeError) as exc:
                    infeasible = {
                        "n_agents": int(n_agents),
                        "gap_margin": float(gap_margin),
                        "minimum_distance": float(minimum_distance),
                        "reason": str(exc),
                    }
                    infeasible_runs.append(infeasible)
                    print(f"n={n_agents}: infeasible - {exc}")
                    continue
                runs.append(run)
                equal_rho = run["equal_spacing"]["spectral_radius"]
                optimized = run["optimized"]
                print(
                    f"n={n_agents}: equal rho={equal_rho:.10f}, "
                    f"optimized rho={optimized['spectral_radius']:.10f}, "
                    f"improvement={optimized['improvement_from_equal']:.10f}, "
                    f"min_distance={optimized['minimum_pairwise_distance']:.6f}"
                )

    report = {
        "experiment": "path spectral-radius minimization under strict betweenness",
        "topology_family": "two-leader path 0-2-3-...-(n-1)-1",
        "agent_counts": agent_counts,
        "gap_margins": gap_margins,
        "minimum_distances": minimum_distances,
        "starts_per_case": int(args.starts),
        "theta": float(args.theta),
        "runs": runs,
        "infeasible_runs": infeasible_runs,
        "all_strict": all(run["optimized"]["strict_betweenness"] for run in runs),
        "all_rho_less_than_one": all(
            run["optimized"]["spectral_radius"] < 1.0 for run in runs
        ),
    }

    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 80)
    print(f"all strict: {report['all_strict']}")
    print(f"all rho < 1: {report['all_rho_less_than_one']}")
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()
