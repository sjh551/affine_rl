"""Optimize the original four-agent formation without fixing coordinate order.

The two leaders and every yaw component are fixed.  Only the planar positions
of followers 2 and 3 are decision variables.  Unlike ``optimizer.py``, this
experiment does not preserve the nominal signs of coordinate differences.  It
uses the original non-degeneracy constraint directly,

    abs(p_u^q - p_v^q) >= coordinate_margin,  q in {x_theta, y_theta},

for the reference pair of every control triplet and for every inferred DEP
edge.  The feasible set is therefore a union of disconnected sign/order
branches, so differential evolution is used as the global-search heuristic.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formation_optimization.optimizer import (  # noqa: E402
    build_follower_planar_iteration_matrix,
)
from utils.formation_graph import FormationGraph  # noqa: E402
from utils.formation_utils import get_square_configuration  # noqa: E402


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "results" / "four_agent_absolute_order"
)


@dataclass(frozen=True)
class AbsoluteOrderOptimizationResult:
    configuration: np.ndarray
    initial_spectral_radius: float
    optimized_spectral_radius: float
    objective_value: float
    optimization_objective: str
    minimized_value: float
    min_degree_singular_value: float
    minimum_absolute_coordinate_separation: float
    minimum_pairwise_distance: float
    changed_coordinate_signs: int
    success: bool
    solver_success: bool
    message: str
    iterations: int
    function_evaluations: int


def _unique_pairs(
    triplets: Sequence[Tuple[int, int, int]],
) -> Tuple[Tuple[int, int], ...]:
    """Return triplet reference pairs plus all inferred DEP/sensing edges."""

    pairs = set()
    for raw_i, raw_j, raw_k in triplets:
        i, j, k = int(raw_i), int(raw_j), int(raw_k)
        for u, v in ((i, j), (i, k), (j, k)):
            pairs.add((min(u, v), max(u, v)))
    return tuple(sorted(pairs))


def _positions_in_theta_frame(
    configuration: np.ndarray,
    theta: float,
) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    return np.asarray(configuration, dtype=float)[:, :2] @ rotation


def _coordinate_differences(
    configuration: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    theta: float,
) -> np.ndarray:
    positions = _positions_in_theta_frame(configuration, theta)
    return np.asarray(
        [positions[u] - positions[v] for u, v in pairs],
        dtype=float,
    )


def _pairwise_squared_distances(configuration: np.ndarray) -> np.ndarray:
    positions = np.asarray(configuration, dtype=float)[:, :2]
    return np.asarray(
        [
            np.dot(positions[i] - positions[j], positions[i] - positions[j])
            for i, j in combinations(range(len(positions)), 2)
        ],
        dtype=float,
    )


def _spectral_metrics(
    configuration: np.ndarray,
    n_leaders: int,
    theta: float,
    triplets: Sequence[Tuple[int, int, int]],
) -> Tuple[float, float, np.ndarray]:
    degree, _, iteration, sigma_min = build_follower_planar_iteration_matrix(
        configuration,
        n_leaders=n_leaders,
        theta=theta,
        triplets=triplets,
    )
    del degree
    eigenvalues = np.linalg.eigvals(iteration)
    radius = float(np.max(np.abs(eigenvalues)))
    return radius, float(sigma_min), eigenvalues


def _axis_bounds(
    name: str,
    lower: Optional[float],
    upper: Optional[float],
) -> Optional[Tuple[float, float]]:
    if (lower is None) != (upper is None):
        raise ValueError(f"{name}-min and {name}-max must be specified together")
    if lower is None:
        return None
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"{name} bounds must be finite and satisfy min < max")
    return float(lower), float(upper)


def optimize_four_agent_absolute_order(
    nominal_configuration: np.ndarray,
    *,
    n_leaders: int = 2,
    theta: float = 0.0,
    spectral_radius_limit: float = 0.95,
    search_radius: float = 10.0,
    coordinate_margin: float = 0.8,
    minimum_pairwise_distance: float = 1.0,
    min_degree_singular_value: float = 1.0e-3,
    x_bounds: Optional[Tuple[float, float]] = None,
    y_bounds: Optional[Tuple[float, float]] = None,
    maxiter: int = 300,
    popsize: int = 15,
    seed: int = 0,
    polish: bool = True,
    objective_mode: str = "deviation",
) -> Tuple[np.ndarray, AbsoluteOrderOptimizationResult, dict]:
    """Solve the four-agent, free-order spectral-radius constrained problem."""

    from scipy.optimize import NonlinearConstraint, differential_evolution

    nominal = np.asarray(nominal_configuration, dtype=float)
    if nominal.shape != (4, 3):
        raise ValueError("This experiment requires a (4, 3) configuration")
    if n_leaders != 2:
        raise ValueError("This experiment is defined for exactly two leaders")
    if not 0.0 < spectral_radius_limit < 1.0:
        raise ValueError("spectral_radius_limit must lie strictly between 0 and 1")
    if search_radius <= 0.0 or not np.isfinite(search_radius):
        raise ValueError("search_radius must be finite and positive")
    if coordinate_margin <= 0.0:
        raise ValueError("coordinate_margin must be positive")
    if minimum_pairwise_distance < 0.0:
        raise ValueError("minimum_pairwise_distance must be non-negative")
    if min_degree_singular_value <= 0.0:
        raise ValueError("min_degree_singular_value must be positive")
    if objective_mode not in {"deviation", "spectral-radius"}:
        raise ValueError(
            "objective_mode must be either 'deviation' or 'spectral-radius'"
        )

    triplets = tuple(FormationGraph(nominal).triplets)
    constrained_pairs = _unique_pairs(triplets)
    follower_slice = slice(n_leaders, len(nominal))
    initial_positions = nominal[follower_slice, :2].reshape(-1).copy()
    nominal_differences = _coordinate_differences(
        nominal,
        constrained_pairs,
        theta,
    )
    nominal_signs = np.sign(nominal_differences)

    lower = initial_positions - float(search_radius)
    upper = initial_positions + float(search_radius)
    for flat_index in range(len(initial_positions)):
        bounds_for_axis = x_bounds if flat_index % 2 == 0 else y_bounds
        if bounds_for_axis is not None:
            lower[flat_index], upper[flat_index] = bounds_for_axis
    if np.any(lower >= upper):
        raise ValueError("Follower search bounds have an empty interval")
    bounds = list(zip(lower, upper))

    def make_configuration(position_vector: np.ndarray) -> np.ndarray:
        candidate = nominal.copy()
        candidate[follower_slice, :2] = np.asarray(
            position_vector,
            dtype=float,
        ).reshape(-1, 2)
        return candidate

    def safe_spectral_metrics(
        position_vector: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        try:
            return _spectral_metrics(
                make_configuration(position_vector),
                n_leaders,
                theta,
                triplets,
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return float("inf"), 0.0, np.full(4, np.nan)

    def geometry_values(position_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        candidate = make_configuration(position_vector)
        absolute_separations = np.abs(
            _coordinate_differences(candidate, constrained_pairs, theta)
        ).reshape(-1)
        return absolute_separations, _pairwise_squared_distances(candidate)

    def position_deviation(position_vector: np.ndarray) -> float:
        delta = np.asarray(position_vector, dtype=float) - initial_positions
        return float(delta @ delta)

    def objective(position_vector: np.ndarray) -> float:
        if objective_mode == "deviation":
            return position_deviation(position_vector)
        radius, _, _ = safe_spectral_metrics(position_vector)
        return radius if np.isfinite(radius) else 1.0e6

    def feasibility_values(position_vector: np.ndarray) -> np.ndarray:
        radius, sigma_min, _ = safe_spectral_metrics(position_vector)
        absolute_separations, squared_distances = geometry_values(position_vector)
        scalar_constraints = [sigma_min - min_degree_singular_value]
        if objective_mode == "deviation":
            scalar_constraints.insert(0, spectral_radius_limit - radius)
        return np.concatenate(
            (
                np.asarray(scalar_constraints, dtype=float),
                absolute_separations - coordinate_margin,
                squared_distances - minimum_pairwise_distance**2,
            )
        )

    nonlinear_constraint = NonlinearConstraint(
        feasibility_values,
        lb=0.0,
        ub=np.inf,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="delta_grad == 0.0.*",
            category=UserWarning,
        )
        scipy_result = differential_evolution(
            objective,
            bounds=bounds,
            constraints=(nonlinear_constraint,),
            maxiter=int(maxiter),
            popsize=int(popsize),
            seed=int(seed),
            polish=bool(polish),
            updating="immediate",
            workers=1,
            tol=1.0e-8,
            atol=1.0e-10,
        )

    optimized = make_configuration(scipy_result.x)
    initial_radius, initial_sigma, initial_eigenvalues = safe_spectral_metrics(
        initial_positions
    )
    optimized_radius, optimized_sigma, optimized_eigenvalues = safe_spectral_metrics(
        scipy_result.x
    )
    absolute_separations, squared_distances = geometry_values(scipy_result.x)
    optimized_differences = _coordinate_differences(
        optimized,
        constrained_pairs,
        theta,
    )
    optimized_signs = np.sign(optimized_differences)
    changed_signs = int(np.count_nonzero(optimized_signs != nominal_signs))
    tolerance = 1.0e-7
    feasible = bool(
        np.all(np.isfinite(optimized))
        and (
            objective_mode == "spectral-radius"
            or optimized_radius <= spectral_radius_limit + tolerance
        )
        and optimized_sigma >= min_degree_singular_value - tolerance
        and np.all(absolute_separations >= coordinate_margin - tolerance)
        and np.all(
            squared_distances
            >= minimum_pairwise_distance**2 - tolerance
        )
    )
    result = AbsoluteOrderOptimizationResult(
        configuration=optimized.copy(),
        initial_spectral_radius=initial_radius,
        optimized_spectral_radius=optimized_radius,
        objective_value=position_deviation(scipy_result.x),
        optimization_objective=objective_mode,
        minimized_value=objective(scipy_result.x),
        min_degree_singular_value=optimized_sigma,
        minimum_absolute_coordinate_separation=float(
            np.min(absolute_separations)
        ),
        minimum_pairwise_distance=float(np.sqrt(np.min(squared_distances))),
        changed_coordinate_signs=changed_signs,
        success=feasible,
        solver_success=bool(scipy_result.success),
        message=str(scipy_result.message),
        iterations=int(getattr(scipy_result, "nit", -1)),
        function_evaluations=int(getattr(scipy_result, "nfev", -1)),
    )
    diagnostics = {
        "triplets": [list(item) for item in triplets],
        "constrained_pairs": [list(item) for item in constrained_pairs],
        "initial_eigenvalues": [
            [float(value.real), float(value.imag)]
            for value in initial_eigenvalues
        ],
        "optimized_eigenvalues": [
            [float(value.real), float(value.imag)]
            for value in optimized_eigenvalues
        ],
        "nominal_coordinate_signs": nominal_signs.astype(int).tolist(),
        "optimized_coordinate_signs": optimized_signs.astype(int).tolist(),
        "x_order": np.argsort(
            _positions_in_theta_frame(optimized, theta)[:, 0]
        ).tolist(),
        "y_order": np.argsort(
            _positions_in_theta_frame(optimized, theta)[:, 1]
        ).tolist(),
        "initial_sigma_min": float(initial_sigma),
    }
    return optimized, result, diagnostics


def _normalized_violation(
    result: AbsoluteOrderOptimizationResult,
    rho_limit: float,
    coordinate_margin: float,
    minimum_distance: float,
    min_sigma: float,
    enforce_rho_limit: bool,
) -> float:
    distance_violation = 0.0
    if minimum_distance > 0.0:
        distance_violation = max(
            0.0,
            minimum_distance - result.minimum_pairwise_distance,
        ) / minimum_distance
    return float(
        (
            max(0.0, result.optimized_spectral_radius - rho_limit)
            / max(rho_limit, np.finfo(float).eps)
            if enforce_rho_limit
            else 0.0
        )
        + max(
            0.0,
            coordinate_margin
            - result.minimum_absolute_coordinate_separation,
        )
        / max(coordinate_margin, np.finfo(float).eps)
        + distance_violation
        + max(0.0, min_sigma - result.min_degree_singular_value)
        / max(min_sigma, np.finfo(float).eps)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Four-agent spectral-radius constrained optimization with "
            "absolute-value coordinate separation and free order"
        )
    )
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--rho-limit", type=float, default=0.95)
    parser.add_argument(
        "--objective",
        choices=("deviation", "spectral-radius"),
        default="deviation",
        help=(
            "deviation: minimize formation change subject to rho limit; "
            "spectral-radius: minimize rho directly while retaining geometry "
            "and nonsingularity constraints"
        ),
    )
    parser.add_argument("--search-radius", type=float, default=10.0)
    parser.add_argument("--coordinate-margin", type=float, default=0.8)
    parser.add_argument("--min-distance", type=float, default=1.0)
    parser.add_argument("--min-degree-singular-value", type=float, default=1.0e-3)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--popsize", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seeds", type=int, nargs="+", default=None)
    seed_group.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--no-polish", action="store_true")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x_bounds = _axis_bounds("x", args.x_min, args.x_max)
    y_bounds = _axis_bounds("y", args.y_min, args.y_max)
    if args.num_seeds is not None and args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    if args.seeds is not None:
        seeds = list(dict.fromkeys(args.seeds))
    elif args.num_seeds is not None:
        seeds = list(range(args.seed, args.seed + args.num_seeds))
    else:
        seeds = [args.seed]
    if any(seed < 0 for seed in seeds):
        raise ValueError("random seeds must be non-negative integers")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    seed_dir = output_dir / "seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    nominal = np.asarray(get_square_configuration(), dtype=float)

    runs = []
    print(f"Running {len(seeds)} free-order seed(s): {seeds}")
    for run_index, seed in enumerate(seeds, start=1):
        print(f"\n[{run_index}/{len(seeds)}] seed={seed}")
        optimized, result, diagnostics = optimize_four_agent_absolute_order(
            nominal,
            theta=args.theta,
            spectral_radius_limit=args.rho_limit,
            search_radius=args.search_radius,
            coordinate_margin=args.coordinate_margin,
            minimum_pairwise_distance=args.min_distance,
            min_degree_singular_value=args.min_degree_singular_value,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            maxiter=args.maxiter,
            popsize=args.popsize,
            seed=seed,
            polish=not args.no_polish,
            objective_mode=args.objective,
        )
        run_summary = asdict(result)
        run_summary["configuration"] = result.configuration.tolist()
        run_summary.update({"seed": seed, **diagnostics})
        config_path = seed_dir / f"seed_{seed}_configuration.npy"
        summary_path = seed_dir / f"seed_{seed}_summary.json"
        np.save(config_path, optimized)
        run_summary["configuration_file"] = str(config_path)
        summary_path.write_text(
            json.dumps(run_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        runs.append((seed, optimized, result, run_summary))
        print(
            f"rho={result.optimized_spectral_radius:.10f}, "
            f"J={result.objective_value:.10f}, "
            f"min_abs_coord={result.minimum_absolute_coordinate_separation:.6f}, "
            f"min_distance={result.minimum_pairwise_distance:.6f}, "
            f"sign_changes={result.changed_coordinate_signs}, "
            f"feasible={result.success}"
        )

    feasible_runs = [run for run in runs if run[2].success]
    if feasible_runs:
        if args.objective == "spectral-radius":
            best_radius = min(
                run[2].optimized_spectral_radius for run in feasible_runs
            )
            # The minimum-rho problem can have a flat family of solutions.
            # Treat differences below numerical tolerance as ties and then
            # prefer the configuration closest to the nominal formation.
            radius_ties = [
                run
                for run in feasible_runs
                if run[2].optimized_spectral_radius <= best_radius + 1.0e-8
            ]
            best = min(radius_ties, key=lambda run: run[2].objective_value)
            selection_reason = (
                "minimum position deviation within 1e-8 of the best "
                "feasible spectral radius"
            )
        else:
            best = min(
                feasible_runs,
                key=lambda run: (
                    run[2].objective_value,
                    run[2].optimized_spectral_radius,
                ),
            )
            selection_reason = "minimum position deviation among feasible runs"
    else:
        best = min(
            runs,
            key=lambda run: (
                _normalized_violation(
                    run[2],
                    args.rho_limit,
                    args.coordinate_margin,
                    args.min_distance,
                    args.min_degree_singular_value,
                    args.objective == "deviation",
                ),
                run[2].objective_value,
            ),
        )
        selection_reason = "minimum normalized violation; no feasible run found"

    best_seed, best_configuration, best_result, best_summary = best
    best_config_path = output_dir / "optimized_configuration.npy"
    report_path = output_dir / "optimization_summary.json"
    np.save(best_config_path, best_configuration)
    report = {
        **best_summary,
        "experiment": "four-agent free-order absolute-separation optimization",
        "best_seed": best_seed,
        "selection_reason": selection_reason,
        "number_of_runs": len(runs),
        "number_of_feasible_runs": len(feasible_runs),
        "requested_seeds": seeds,
        "rho_limit": args.rho_limit,
        "optimization_objective": args.objective,
        "rho_limit_enforced": args.objective == "deviation",
        "theta": args.theta,
        "search_radius": args.search_radius,
        "coordinate_margin": args.coordinate_margin,
        "minimum_pairwise_distance_limit": args.min_distance,
        "min_degree_singular_value_limit": args.min_degree_singular_value,
        "x_bounds": x_bounds,
        "y_bounds": y_bounds,
        "configuration_file": str(best_config_path),
        "runs": [run[3] for run in runs],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n" + "=" * 72)
    print(f"Best seed              : {best_seed}")
    print(f"Feasible runs          : {len(feasible_runs)}/{len(runs)}")
    print(f"Spectral radius        : {best_result.optimized_spectral_radius:.10f}")
    print(f"Position deviation J   : {best_result.objective_value:.10f}")
    print(f"Optimization objective : {args.objective}")
    print(f"Changed coord signs    : {best_result.changed_coordinate_signs}")
    print(f"Selection              : {selection_reason}")
    print("Optimized configuration:")
    print(np.array2string(best_configuration, precision=8, suppress_small=True))
    print(f"Saved configuration    : {best_config_path}")
    print(f"Saved summary          : {report_path}")
    print("=" * 72)
    if not feasible_runs and not args.non_strict:
        raise RuntimeError(
            "No seed found a feasible free-order configuration. Per-seed "
            f"results were saved under {seed_dir}."
        )


if __name__ == "__main__":
    main()
