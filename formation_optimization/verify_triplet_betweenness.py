"""Verify the local-betweenness condition for follower velocity iteration.

For a control triplet ``(i, j, k)`` and one coordinate in the fixed theta
frame, the zero-formation-error velocity relation is

    v_k = ((z_j-z_k)/(z_j-z_i)) v_i
        + ((z_k-z_i)/(z_j-z_i)) v_j.

If ``z_k`` lies *strictly* between ``z_i`` and ``z_j``, the two coefficients
are positive and sum to one.  With one triplet per follower, this script checks
that the follower-only iteration matrix is substochastic, every follower can
reach a leader through positive dependency weights, and ``rho(T_ff) < 1``.

Scope: this certifies convergence of the simultaneous/Jacobi follower velocity
iteration ``v_f^(r+1) = T_ff v_f^(r) + b``.  It does not by itself certify the
complete sampled-data position/velocity loop, actuator dynamics, APF, contact,
or PyBullet simulation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formation_optimization.optimizer import (
    build_follower_planar_iteration_matrix,
)
from utils.formation_graph import FormationGraph
from utils.formation_utils import get_square_configuration


@dataclass
class DirectionDiagnostics:
    coordinate: str
    strict_betweenness: bool
    inclusive_betweenness: bool
    nonnegative_iteration_matrix: bool
    substochastic_iteration_matrix: bool
    every_follower_reaches_leader: bool
    spectral_radius: float
    spectral_norm: float
    maximum_row_sum: float
    minimum_row_sum: float
    deficient_rows: List[int]
    follower_reaches_leader: List[bool]


@dataclass
class ConfigurationDiagnostics:
    name: str
    configuration: List[List[float]]
    triplets: List[List[int]]
    one_triplet_per_follower: bool
    degree_min_singular_value: float
    full_spectral_radius: float
    full_spectral_norm: float
    directions: List[DirectionDiagnostics]

    @property
    def strict_betweenness(self) -> bool:
        return all(item.strict_betweenness for item in self.directions)

    @property
    def inclusive_betweenness(self) -> bool:
        return all(item.inclusive_betweenness for item in self.directions)

    @property
    def velocity_iteration_stable(self) -> bool:
        return self.full_spectral_radius < 1.0


def _theta_rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def _positions_in_theta_frame(
    configuration: np.ndarray,
    theta: float,
) -> np.ndarray:
    # Row-vector form of z = R(theta)^T p.
    return np.asarray(configuration, dtype=float)[:, :2] @ _theta_rotation(theta)


def _iteration_in_theta_frame(iteration: np.ndarray, theta: float) -> np.ndarray:
    n_followers = iteration.shape[0] // 2
    rotation = _theta_rotation(theta)
    transform = np.kron(np.eye(n_followers), rotation)
    return transform.T @ iteration @ transform


def _triplet_counts(
    triplets: Sequence[Tuple[int, int, int]],
    n_leaders: int,
    n_agents: int,
) -> Dict[int, int]:
    counts = {k: 0 for k in range(n_leaders, n_agents)}
    for _, _, k in triplets:
        if k in counts:
            counts[k] += 1
    return counts


def _local_weights(
    positions_theta: np.ndarray,
    triplets: Sequence[Tuple[int, int, int]],
    coordinate: int,
    tolerance: float,
) -> Tuple[Dict[int, List[Tuple[int, float]]], bool, bool]:
    dependencies: Dict[int, List[Tuple[int, float]]] = {}
    strict_all = True
    inclusive_all = True

    for i, j, k in triplets:
        zi = float(positions_theta[i, coordinate])
        zj = float(positions_theta[j, coordinate])
        zk = float(positions_theta[k, coordinate])
        denominator = zj - zi
        if abs(denominator) <= tolerance:
            dependencies[k] = []
            strict_all = False
            inclusive_all = False
            continue

        weight_i = (zj - zk) / denominator
        weight_j = (zk - zi) / denominator
        dependencies.setdefault(k, []).extend(((i, weight_i), (j, weight_j)))

        inclusive = (
            weight_i >= -tolerance
            and weight_j >= -tolerance
            and weight_i <= 1.0 + tolerance
            and weight_j <= 1.0 + tolerance
        )
        strict = weight_i > tolerance and weight_j > tolerance
        inclusive_all = inclusive_all and inclusive
        strict_all = strict_all and strict

    return dependencies, strict_all, inclusive_all


def _followers_reach_leader(
    dependencies: Dict[int, List[Tuple[int, float]]],
    n_leaders: int,
    n_agents: int,
    tolerance: float,
) -> List[bool]:
    results: List[bool] = []
    for start in range(n_leaders, n_agents):
        stack = [start]
        visited = set()
        reached = False
        while stack:
            node = stack.pop()
            if node < n_leaders:
                reached = True
                break
            if node in visited:
                continue
            visited.add(node)
            for neighbour, weight in dependencies.get(node, []):
                if weight > tolerance:
                    stack.append(neighbour)
        results.append(reached)
    return results


def analyze_configuration(
    name: str,
    configuration: np.ndarray,
    n_leaders: int = 2,
    theta: float = 0.0,
    tolerance: float = 1.0e-10,
    triplets: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> ConfigurationDiagnostics:
    configuration = np.asarray(configuration, dtype=float)
    if triplets is None:
        triplets = tuple(FormationGraph(configuration).triplets)
    else:
        triplets = tuple((int(i), int(j), int(k)) for i, j, k in triplets)
    degree, _, iteration, sigma_min = build_follower_planar_iteration_matrix(
        configuration,
        n_leaders=n_leaders,
        theta=theta,
        triplets=triplets,
    )
    del degree

    positions_theta = _positions_in_theta_frame(configuration, theta)
    iteration_theta = _iteration_in_theta_frame(iteration, theta)
    n_followers = len(configuration) - n_leaders
    counts = _triplet_counts(triplets, n_leaders, len(configuration))
    one_triplet = all(value == 1 for value in counts.values())

    direction_diagnostics: List[DirectionDiagnostics] = []
    for coordinate, coordinate_name in enumerate(("x_theta", "y_theta")):
        indices = np.arange(coordinate, 2 * n_followers, 2)
        directional_iteration = iteration_theta[np.ix_(indices, indices)]
        eigenvalues = np.linalg.eigvals(directional_iteration)
        row_sums = np.sum(directional_iteration, axis=1)
        dependencies, strict_all, inclusive_all = _local_weights(
            positions_theta,
            triplets,
            coordinate,
            tolerance,
        )
        reaches = _followers_reach_leader(
            dependencies,
            n_leaders,
            len(configuration),
            tolerance,
        )

        direction_diagnostics.append(
            DirectionDiagnostics(
                coordinate=coordinate_name,
                strict_betweenness=strict_all,
                inclusive_betweenness=inclusive_all,
                nonnegative_iteration_matrix=bool(
                    np.all(directional_iteration >= -tolerance)
                ),
                substochastic_iteration_matrix=bool(
                    np.all(directional_iteration >= -tolerance)
                    and np.all(row_sums <= 1.0 + tolerance)
                ),
                every_follower_reaches_leader=all(reaches),
                spectral_radius=float(np.max(np.abs(eigenvalues))),
                spectral_norm=float(np.linalg.norm(directional_iteration, 2)),
                maximum_row_sum=float(np.max(row_sums)),
                minimum_row_sum=float(np.min(row_sums)),
                deficient_rows=[
                    int(row)
                    for row, row_sum in enumerate(row_sums)
                    if row_sum < 1.0 - tolerance
                ],
                follower_reaches_leader=reaches,
            )
        )

    full_eigenvalues = np.linalg.eigvals(iteration)
    return ConfigurationDiagnostics(
        name=name,
        configuration=configuration.tolist(),
        triplets=[list(item) for item in triplets],
        one_triplet_per_follower=one_triplet,
        degree_min_singular_value=float(sigma_min),
        full_spectral_radius=float(np.max(np.abs(full_eigenvalues))),
        full_spectral_norm=float(np.linalg.norm(iteration, 2)),
        directions=direction_diagnostics,
    )


def _path_triplets(n_agents: int) -> Tuple[Tuple[int, int, int], ...]:
    """Return triplets for path ``0-2-3-...-(n-1)-1``."""

    if n_agents < 3:
        raise ValueError("a two-leader path needs at least three agents")
    path = [0, *range(2, n_agents), 1]
    triplets = []
    for path_index in range(1, len(path) - 1):
        previous_node = path[path_index - 1]
        follower = path[path_index]
        next_node = path[path_index + 1]
        i, j = sorted((previous_node, next_node))
        triplets.append((i, j, follower))
    return tuple(triplets)


def _path_base_configuration(n_agents: int) -> np.ndarray:
    """Create a planar path test configuration with fixed square leaders."""

    square = get_square_configuration()
    configuration = np.zeros((n_agents, 3), dtype=float)
    configuration[:2] = square[:2]
    if n_agents >= 4:
        configuration[2, 2] = square[2, 2]
        configuration[3, 2] = square[3, 2]
    return configuration


def _configuration_from_path_fractions(
    base_configuration: np.ndarray,
    fractions_by_coordinate: np.ndarray,
    theta: float = 0.0,
) -> np.ndarray:
    """Place followers monotonically in each fixed-theta coordinate."""

    configuration = np.asarray(base_configuration, dtype=float).copy()
    n_followers = len(configuration) - 2
    fractions_by_coordinate = np.asarray(fractions_by_coordinate, dtype=float)
    if fractions_by_coordinate.shape != (2, n_followers):
        raise ValueError(
            "fractions_by_coordinate must have shape "
            f"(2, {n_followers}), got {fractions_by_coordinate.shape}"
        )
    rotation = _theta_rotation(theta)
    positions_theta = configuration[:, :2] @ rotation
    leader_0 = positions_theta[0].copy()
    leader_1 = positions_theta[1].copy()
    for coordinate in range(2):
        delta = leader_1[coordinate] - leader_0[coordinate]
        for follower_offset, fraction in enumerate(
            fractions_by_coordinate[coordinate]
        ):
            follower = 2 + follower_offset
            positions_theta[follower, coordinate] = (
                leader_0[coordinate] + fraction * delta
            )
    configuration[:, :2] = positions_theta @ rotation.T
    return configuration


def _sample_strict_fractions(
    rng: np.random.Generator,
    fraction_margin: float,
    n_followers: int,
) -> np.ndarray:
    """Sample ordered path fractions with every adjacent gap bounded below."""

    n_gaps = n_followers + 1
    if not 0.0 < fraction_margin < 1.0 / n_gaps:
        raise ValueError(
            f"fraction_margin must be in (0, 1/{n_gaps}) for "
            f"{n_followers} followers"
        )
    residual = 1.0 - n_gaps * fraction_margin
    result = np.empty((2, n_followers), dtype=float)
    for coordinate in range(2):
        gaps = fraction_margin + residual * rng.dirichlet(np.ones(n_gaps))
        result[coordinate] = np.cumsum(gaps[:-1])
    return result


def _run_monte_carlo(
    n_agents: int,
    samples: int,
    seed: int,
    fraction_margin: float,
    theta: float,
    tolerance: float,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    base = _path_base_configuration(n_agents)
    triplets = _path_triplets(n_agents)
    n_followers = n_agents - 2
    radii = np.empty(samples, dtype=float)
    norms = np.empty(samples, dtype=float)
    implication_violations = 0
    strict_samples = 0
    worst_configuration: Optional[np.ndarray] = None

    for sample in range(samples):
        fractions = _sample_strict_fractions(
            rng,
            fraction_margin,
            n_followers=n_followers,
        )
        configuration = _configuration_from_path_fractions(
            base,
            fractions,
            theta=theta,
        )
        diagnostics = analyze_configuration(
            f"sample_{sample}",
            configuration,
            n_leaders=2,
            theta=theta,
            tolerance=tolerance,
            triplets=triplets,
        )
        radii[sample] = diagnostics.full_spectral_radius
        norms[sample] = diagnostics.full_spectral_norm
        strict_samples += int(diagnostics.strict_betweenness)
        if diagnostics.strict_betweenness and diagnostics.full_spectral_radius >= 1.0 - tolerance:
            implication_violations += 1
        if worst_configuration is None or radii[sample] >= np.max(radii[:sample]):
            worst_configuration = configuration.copy()

    assert worst_configuration is not None
    return {
        "n_agents": int(n_agents),
        "n_followers": int(n_followers),
        "triplets": [list(item) for item in triplets],
        "samples": int(samples),
        "seed": int(seed),
        "fraction_margin": float(fraction_margin),
        "strict_betweenness_samples": int(strict_samples),
        "rho_less_than_one_samples": int(np.sum(radii < 1.0 - tolerance)),
        "implication_violations": int(implication_violations),
        "spectral_radius_min": float(np.min(radii)),
        "spectral_radius_mean": float(np.mean(radii)),
        "spectral_radius_max": float(np.max(radii)),
        "spectral_norm_max": float(np.max(norms)),
        "worst_configuration": worst_configuration.tolist(),
    }


def _homogeneous_iteration_trace(
    configuration: np.ndarray,
    triplets: Sequence[Tuple[int, int, int]],
    theta: float,
    iterations: int,
    seed: int,
) -> Dict[str, object]:
    _, _, iteration, _ = build_follower_planar_iteration_matrix(
        configuration,
        n_leaders=2,
        theta=theta,
        triplets=triplets,
    )
    rng = np.random.default_rng(seed)
    velocity = rng.normal(size=iteration.shape[0])
    initial_norm = float(np.linalg.norm(velocity))
    norms = [initial_norm]
    for _ in range(iterations):
        velocity = iteration @ velocity
        norms.append(float(np.linalg.norm(velocity)))
    return {
        "iterations": int(iterations),
        "initial_norm": initial_norm,
        "final_norm": norms[-1],
        "norm_ratio": norms[-1] / initial_norm,
        "norms": norms,
    }


def _print_case(diagnostics: ConfigurationDiagnostics) -> None:
    print("-" * 72)
    print(diagnostics.name)
    print(f"  strict betweenness   : {diagnostics.strict_betweenness}")
    print(f"  inclusive between    : {diagnostics.inclusive_betweenness}")
    print(f"  one triplet/follower : {diagnostics.one_triplet_per_follower}")
    print(f"  rho(T_ff)            : {diagnostics.full_spectral_radius:.10f}")
    print(f"  ||T_ff||_2           : {diagnostics.full_spectral_norm:.10f}")
    for item in diagnostics.directions:
        print(
            f"  {item.coordinate}: rho={item.spectral_radius:.10f}, "
            f"substochastic={item.substochastic_iteration_matrix}, "
            f"leader_reachable={item.every_follower_reaches_leader}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify strict triplet betweenness => rho(T_ff) < 1",
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--agent-counts",
        type=int,
        nargs="+",
        default=[4, 5, 6, 7, 8],
        help="path sizes to test; default: 4 5 6 7 8",
    )
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--fraction-margin", type=float, default=0.02)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("formation_optimization/results/betweenness_verification.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    agent_counts = list(dict.fromkeys(int(value) for value in args.agent_counts))
    if not agent_counts or any(value < 3 for value in agent_counts):
        raise ValueError("agent-counts must contain integers greater than or equal to 3")

    square = get_square_configuration()
    four_agent_triplets = _path_triplets(4)
    endpoint_configuration = _configuration_from_path_fractions(
        square,
        np.array([[0.5, 0.5]] * 2),
        theta=args.theta,
    )

    original_case = analyze_configuration(
        "original_square_not_between",
        square,
        theta=args.theta,
        tolerance=args.tolerance,
        triplets=four_agent_triplets,
    )
    strict_cases: List[ConfigurationDiagnostics] = []
    strict_configurations: Dict[int, np.ndarray] = {}
    for n_agents in agent_counts:
        n_followers = n_agents - 2
        evenly_spaced = np.arange(1, n_followers + 1, dtype=float) / (
            n_followers + 1
        )
        configuration = _configuration_from_path_fractions(
            _path_base_configuration(n_agents),
            np.tile(evenly_spaced, (2, 1)),
            theta=args.theta,
        )
        strict_configurations[n_agents] = configuration
        strict_cases.append(
            analyze_configuration(
                f"strict_between_{n_agents}_agents",
                configuration,
                theta=args.theta,
                tolerance=args.tolerance,
                triplets=_path_triplets(n_agents),
            )
        )

    endpoint_case = analyze_configuration(
        "inclusive_endpoint_counterexample_4_agents",
        endpoint_configuration,
        theta=args.theta,
        tolerance=args.tolerance,
        triplets=four_agent_triplets,
    )
    cases = [
        original_case,
        *strict_cases,
        endpoint_case,
    ]
    monte_carlo_runs = [
        _run_monte_carlo(
            n_agents=n_agents,
            samples=args.samples,
            seed=args.seed + n_agents,
            fraction_margin=args.fraction_margin,
            theta=args.theta,
            tolerance=args.tolerance,
        )
        for n_agents in agent_counts
    ]

    trace_agent_count = agent_counts[0]
    iteration_trace = _homogeneous_iteration_trace(
        strict_configurations[trace_agent_count],
        triplets=_path_triplets(trace_agent_count),
        theta=args.theta,
        iterations=args.iterations,
        seed=args.seed,
    )
    iteration_trace["n_agents"] = trace_agent_count

    checks = {
        "all_strict_examples_are_strict": all(
            item.strict_betweenness for item in strict_cases
        ),
        "all_strict_examples_use_one_triplet_per_follower": all(
            item.one_triplet_per_follower for item in strict_cases
        ),
        "all_strict_examples_reach_leaders": all(
            direction.every_follower_reaches_leader
            for item in strict_cases
            for direction in item.directions
        ),
        "all_strict_examples_have_rho_less_than_one": all(
            item.full_spectral_radius < 1.0 for item in strict_cases
        ),
        "endpoint_example_is_inclusive": endpoint_case.inclusive_betweenness,
        "endpoint_example_is_not_strict": not endpoint_case.strict_betweenness,
        "endpoint_example_rho_equals_one": bool(
            np.isclose(endpoint_case.full_spectral_radius, 1.0, atol=1.0e-10)
        ),
        "all_monte_carlo_samples_are_strict": all(
            item["strict_betweenness_samples"] == item["samples"]
            for item in monte_carlo_runs
        ),
        "monte_carlo_found_no_counterexample": all(
            item["implication_violations"] == 0
            for item in monte_carlo_runs
        ),
    }

    report = {
        "claim_scope": (
            "Strict componentwise betweenness in the fixed theta frame, with "
            "one nondegenerate control triplet per follower, certifies the "
            "follower velocity iteration rho(T_ff)<1. It does not certify the "
            "complete sampled-data/PyBullet closed loop."
        ),
        "topology_family": "two-leader path 0-2-3-...-(n-1)-1",
        "agent_counts": agent_counts,
        "samples_per_agent_count": args.samples,
        "cases": [
            {
                **asdict(item),
                "strict_betweenness": item.strict_betweenness,
                "inclusive_betweenness": item.inclusive_betweenness,
                "velocity_iteration_stable": item.velocity_iteration_stable,
            }
            for item in cases
        ],
        "monte_carlo_by_agent_count": monte_carlo_runs,
        "homogeneous_velocity_iteration": iteration_trace,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }

    for item in cases:
        _print_case(item)
    print("-" * 72)
    print("Monte Carlo strict-betweenness falsification search")
    for monte_carlo in monte_carlo_runs:
        print(
            f"  n={monte_carlo['n_agents']}: samples={monte_carlo['samples']}, "
            f"violations={monte_carlo['implication_violations']}, "
            f"rho min/mean/max={monte_carlo['spectral_radius_min']:.10f} / "
            f"{monte_carlo['spectral_radius_mean']:.10f} / "
            f"{monte_carlo['spectral_radius_max']:.10f}"
        )
    print(f"  all checks passed     : {report['all_checks_passed']}")
    print()
    print("Scope warning: rho(T_ff)<1 only certifies the follower velocity iteration;")
    print("it is not a certificate for the full sampled-data/PyBullet closed loop.")

    output = args.output
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report: {output}")

    if not report["all_checks_passed"]:
        raise RuntimeError("At least one verification check failed; inspect the report")


if __name__ == "__main__":
    main()
