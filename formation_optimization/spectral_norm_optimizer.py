"""Formation optimization using the planar spectral norm of ``T``.

This module intentionally lives beside, rather than replaces, the existing
spectral-radius optimizer.  It solves the directly comparable problem

    min ||p_f - p_f_nom||_2^2
    s.t. ||D_ff(p)^-1 A_ff(p)||_2 <= gamma,
         sigma_min(D_ff(p)) >= delta.

Only follower x/y coordinates are decision variables.  Leaders and all yaw
components remain fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence, Tuple, Union
import warnings

import numpy as np

from formation_optimization.optimizer import (
    _build_non_degeneracy_pairs,
    _minimum_pairwise_distance,
    _pair_coordinate_differences,
    build_follower_planar_iteration_matrix,
)
from utils.formation_graph import FormationGraph


@dataclass(frozen=True)
class SpectralNormOptimizationResult:
    """Diagnostics returned by the spectral-norm optimizer."""

    configuration: np.ndarray
    initial_spectral_norm: float
    optimized_spectral_norm: float
    initial_spectral_radius: float
    optimized_spectral_radius: float
    objective_value: float
    min_degree_singular_value: float
    degree_condition_number: float
    min_signed_coordinate_separation: float
    minimum_pairwise_distance: float
    success: bool
    message: str
    iterations: int
    function_evaluations: int


@dataclass(frozen=True)
class MinimumSpectralNormResult:
    """Result of directly minimizing ``||T||_2`` over a search box."""

    configuration: np.ndarray
    initial_spectral_norm: float
    optimized_spectral_norm: float
    initial_spectral_radius: float
    optimized_spectral_radius: float
    position_deviation: float
    min_degree_singular_value: float
    degree_condition_number: float
    min_signed_coordinate_separation: float
    minimum_pairwise_distance: float
    feasible: bool
    solver_success: bool
    message: str
    iterations: int
    function_evaluations: int
    active_bound_coordinates: int


def follower_planar_matrix_metrics(
    configuration: np.ndarray,
    n_leaders: int,
    theta: float = 0.0,
    triplets: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> Tuple[float, float, float, float]:
    """Return ``||T||_2``, ``rho(T)``, ``sigma_min(D)`` and ``cond_2(D)``."""

    degree, _, iteration, min_singular_value = (
        build_follower_planar_iteration_matrix(
            configuration,
            n_leaders=n_leaders,
            theta=theta,
            triplets=triplets,
        )
    )
    spectral_norm = float(np.linalg.norm(iteration, ord=2))
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(iteration))))
    degree_condition_number = float(np.linalg.cond(degree, p=2))
    return (
        spectral_norm,
        spectral_radius,
        min_singular_value,
        degree_condition_number,
    )


def optimize_nominal_configuration_by_spectral_norm(
    nominal_configuration: np.ndarray,
    n_leaders: int,
    theta: float = 0.0,
    spectral_norm_limit: float = 1.05,
    search_radius: Union[float, Sequence[float]] = 3.0,
    min_degree_singular_value: float = 1.0e-3,
    coordinate_margin: float = 1.0e-3,
    minimum_pairwise_distance: float = 0.8,
    x_bounds: Optional[Tuple[float, float]] = None,
    y_bounds: Optional[Tuple[float, float]] = None,
    constraint_edges: Optional[Sequence[Tuple[int, int]]] = None,
    maxiter: int = 300,
    popsize: int = 15,
    seed: Optional[int] = 0,
    polish: bool = True,
    strict: bool = True,
    verbose: bool = True,
    return_result: bool = False,
) -> Union[
    np.ndarray,
    Tuple[np.ndarray, SpectralNormOptimizationResult],
]:
    """Find the closest follower configuration satisfying ``||T||_2`` limit."""

    try:
        from scipy.optimize import NonlinearConstraint, differential_evolution
    except ImportError as exc:
        raise ImportError(
            "Formation optimization requires SciPy. Install project "
            "dependencies with `pip install -r requirements.txt`."
        ) from exc

    g_nom = np.asarray(nominal_configuration, dtype=float)
    if g_nom.ndim != 2 or g_nom.shape[1] < 2:
        raise ValueError("nominal_configuration must have shape (n_agents, >=2)")
    if not np.isfinite(spectral_norm_limit) or spectral_norm_limit <= 0.0:
        raise ValueError("spectral_norm_limit must be finite and positive")
    if min_degree_singular_value <= 0.0:
        raise ValueError("min_degree_singular_value must be positive")
    if coordinate_margin <= 0.0:
        raise ValueError("coordinate_margin must be positive")
    if minimum_pairwise_distance < 0.0:
        raise ValueError("minimum_pairwise_distance must be non-negative")
    if not 0 < n_leaders < len(g_nom):
        raise ValueError("n_leaders must be between 1 and n_agents - 1")

    def validate_axis_bounds(
        name: str,
        axis_bounds: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        if axis_bounds is None:
            return None
        if len(axis_bounds) != 2:
            raise ValueError(f"{name} must contain exactly (minimum, maximum)")
        lower, upper = map(float, axis_bounds)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"{name} must contain finite values with min < max")
        return lower, upper

    x_bounds = validate_axis_bounds("x_bounds", x_bounds)
    y_bounds = validate_axis_bounds("y_bounds", y_bounds)

    optimized = g_nom.copy()
    triplets = tuple(FormationGraph(g_nom).triplets)
    non_degeneracy_pairs = _build_non_degeneracy_pairs(
        triplets,
        n_agents=len(g_nom),
        constraint_edges=constraint_edges,
    )
    nominal_differences = _pair_coordinate_differences(
        g_nom,
        non_degeneracy_pairs,
        theta=theta,
    )
    if np.any(np.isclose(nominal_differences, 0.0, atol=1.0e-12)):
        zero_locations = np.argwhere(
            np.isclose(nominal_differences, 0.0, atol=1.0e-12)
        )
        details = [
            (
                non_degeneracy_pairs[int(pair_index)],
                "x" if int(axis_index) == 0 else "y",
            )
            for pair_index, axis_index in zero_locations
        ]
        raise ValueError(
            "The sign-preserving formulation is undefined because the "
            f"nominal separation is zero for: {details}"
        )
    nominal_coordinate_signs = np.sign(nominal_differences)

    follower_slice = slice(n_leaders, len(g_nom))
    initial_positions = g_nom[follower_slice, :2].reshape(-1).copy()

    radius_array = np.asarray(search_radius, dtype=float)
    if radius_array.ndim == 0:
        radius_array = np.full_like(initial_positions, float(radius_array))
    else:
        radius_array = np.broadcast_to(
            radius_array,
            initial_positions.shape,
        ).copy()
    if np.any(~np.isfinite(radius_array)) or np.any(radius_array <= 0.0):
        raise ValueError("search_radius must contain finite positive values")

    lower_bounds = initial_positions - radius_array
    upper_bounds = initial_positions + radius_array
    for flat_index in range(len(initial_positions)):
        axis_bounds = x_bounds if flat_index % 2 == 0 else y_bounds
        if axis_bounds is not None:
            lower_bounds[flat_index] = axis_bounds[0]
            upper_bounds[flat_index] = axis_bounds[1]
    bounds = list(zip(lower_bounds, upper_bounds))

    def make_configuration(position_vector: np.ndarray) -> np.ndarray:
        candidate = g_nom.copy()
        candidate[follower_slice, :2] = np.asarray(position_vector).reshape(-1, 2)
        return candidate

    def evaluate(position_vector: np.ndarray) -> Tuple[float, float, float, float]:
        try:
            return follower_planar_matrix_metrics(
                make_configuration(position_vector),
                n_leaders=n_leaders,
                theta=theta,
                triplets=triplets,
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return float("inf"), float("inf"), 0.0, float("inf")

    def geometry_values(
        position_vector: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        candidate = make_configuration(position_vector)
        coordinate_differences = _pair_coordinate_differences(
            candidate,
            non_degeneracy_pairs,
            theta=theta,
        )
        signed_coordinate_separations = (
            nominal_coordinate_signs * coordinate_differences
        ).reshape(-1)
        positions = candidate[:, :2]
        squared_distances = np.asarray(
            [
                np.sum((positions[i] - positions[j]) ** 2)
                for i, j in combinations(range(len(candidate)), 2)
            ],
            dtype=float,
        )
        return signed_coordinate_separations, squared_distances

    (
        initial_norm,
        initial_radius,
        initial_sigma_min,
        initial_condition_number,
    ) = evaluate(initial_positions)
    initial_coordinate_separations, initial_squared_distances = geometry_values(
        initial_positions
    )
    initial_within_bounds = bool(
        np.all(initial_positions >= lower_bounds)
        and np.all(initial_positions <= upper_bounds)
    )

    if (
        initial_norm <= spectral_norm_limit
        and initial_sigma_min >= min_degree_singular_value
        and np.all(initial_coordinate_separations >= coordinate_margin)
        and np.all(
            initial_squared_distances >= minimum_pairwise_distance ** 2
        )
        and initial_within_bounds
    ):
        result = SpectralNormOptimizationResult(
            configuration=optimized.copy(),
            initial_spectral_norm=initial_norm,
            optimized_spectral_norm=initial_norm,
            initial_spectral_radius=initial_radius,
            optimized_spectral_radius=initial_radius,
            objective_value=0.0,
            min_degree_singular_value=initial_sigma_min,
            degree_condition_number=initial_condition_number,
            min_signed_coordinate_separation=float(
                np.min(initial_coordinate_separations)
            ),
            minimum_pairwise_distance=_minimum_pairwise_distance(g_nom),
            success=True,
            message="The nominal configuration already satisfies the constraints.",
            iterations=0,
            function_evaluations=1,
        )
        if verbose:
            _print_spectral_norm_result(result, spectral_norm_limit)
        if return_result:
            return optimized, result
        return optimized

    def objective(position_vector: np.ndarray) -> float:
        delta = np.asarray(position_vector) - initial_positions
        return float(delta @ delta)

    def feasibility_values(position_vector: np.ndarray) -> np.ndarray:
        spectral_norm, _, sigma_min, _ = evaluate(position_vector)
        signed_separations, squared_distances = geometry_values(position_vector)
        return np.concatenate(
            (
                np.array(
                    [
                        spectral_norm_limit - spectral_norm,
                        sigma_min - min_degree_singular_value,
                    ],
                    dtype=float,
                ),
                signed_separations - coordinate_margin,
                squared_distances - minimum_pairwise_distance ** 2,
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
            seed=seed,
            x0=np.clip(initial_positions, lower_bounds, upper_bounds),
            polish=bool(polish),
            updating="immediate",
            workers=1,
            tol=1.0e-8,
            atol=1.0e-10,
        )

    optimized[follower_slice, :2] = scipy_result.x.reshape(-1, 2)
    (
        optimized_norm,
        optimized_radius,
        optimized_sigma_min,
        optimized_condition_number,
    ) = evaluate(scipy_result.x)
    optimized_coordinate_separations, optimized_squared_distances = (
        geometry_values(scipy_result.x)
    )
    optimized_min_coordinate_separation = float(
        np.min(optimized_coordinate_separations)
    )
    optimized_min_pairwise_distance = float(
        np.sqrt(np.min(optimized_squared_distances))
    )

    constraint_tolerance = 1.0e-7
    feasible = bool(
        optimized_norm <= spectral_norm_limit + constraint_tolerance
        and optimized_sigma_min
        >= min_degree_singular_value - constraint_tolerance
        and np.all(
            optimized_coordinate_separations
            >= coordinate_margin - constraint_tolerance
        )
        and np.all(
            optimized_squared_distances
            >= minimum_pairwise_distance ** 2 - constraint_tolerance
        )
    )
    success = bool(feasible and np.all(np.isfinite(optimized)))
    result = SpectralNormOptimizationResult(
        configuration=optimized.copy(),
        initial_spectral_norm=initial_norm,
        optimized_spectral_norm=optimized_norm,
        initial_spectral_radius=initial_radius,
        optimized_spectral_radius=optimized_radius,
        objective_value=objective(scipy_result.x),
        min_degree_singular_value=optimized_sigma_min,
        degree_condition_number=optimized_condition_number,
        min_signed_coordinate_separation=optimized_min_coordinate_separation,
        minimum_pairwise_distance=optimized_min_pairwise_distance,
        success=success,
        message=str(scipy_result.message),
        iterations=int(getattr(scipy_result, "nit", -1)),
        function_evaluations=int(getattr(scipy_result, "nfev", -1)),
    )

    if verbose:
        _print_spectral_norm_result(result, spectral_norm_limit)

    if strict and not success:
        raise RuntimeError(
            "Spectral-norm formation optimization did not find a feasible "
            f"configuration. Best ||T||_2={optimized_norm:.8f}, required "
            f"||T||_2<={spectral_norm_limit:.8f}; minimum signed coordinate "
            f"separation={optimized_min_coordinate_separation:.8f}, required "
            f">={coordinate_margin:.8f}; minimum pairwise distance="
            f"{optimized_min_pairwise_distance:.8f}, required "
            f">={minimum_pairwise_distance:.8f}. Increase the limit, "
            "search_radius/maxiter, or use --non-strict to inspect the best "
            "infeasible result."
        )

    if return_result:
        return optimized, result
    return optimized


def minimize_follower_planar_spectral_norm(
    nominal_configuration: np.ndarray,
    n_leaders: int,
    theta: float = 0.0,
    search_radius: Union[float, Sequence[float]] = 3.0,
    min_degree_singular_value: float = 1.0e-3,
    min_pairwise_distance: float = 0.8,
    coordinate_margin: float = 1.0e-3,
    x_bounds: Optional[Tuple[float, float]] = None,
    y_bounds: Optional[Tuple[float, float]] = None,
    constraint_edges: Optional[Sequence[Tuple[int, int]]] = None,
    maxiter: int = 300,
    popsize: int = 15,
    seed: Optional[int] = 0,
    polish: bool = True,
    verbose: bool = True,
    return_result: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, MinimumSpectralNormResult]]:
    """Directly minimize ``||D_ff^-1 A_ff||_2`` within a position box.

    Unlike :func:`optimize_nominal_configuration_by_spectral_norm`, this
    routine does not minimize formation deviation subject to a prescribed
    norm limit.  Its objective is the norm itself, so it is suitable for
    testing whether a larger search region can push the best attainable norm
    below one.  The position deviation is reported only as a diagnostic.
    """

    try:
        from scipy.optimize import NonlinearConstraint, differential_evolution
    except ImportError as exc:
        raise ImportError(
            "Formation optimization requires SciPy. Install project "
            "dependencies with `pip install -r requirements.txt`."
        ) from exc

    g_nom = np.asarray(nominal_configuration, dtype=float)
    if g_nom.ndim != 2 or g_nom.shape[1] < 2:
        raise ValueError("nominal_configuration must have shape (n_agents, >=2)")
    if not 0 < n_leaders < len(g_nom):
        raise ValueError("n_leaders must be between 1 and n_agents - 1")
    if min_degree_singular_value <= 0.0:
        raise ValueError("min_degree_singular_value must be positive")
    if min_pairwise_distance < 0.0 or not np.isfinite(min_pairwise_distance):
        raise ValueError("min_pairwise_distance must be finite and non-negative")
    if coordinate_margin <= 0.0:
        raise ValueError("coordinate_margin must be positive")

    def validate_axis_bounds(
        name: str,
        axis_bounds: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        if axis_bounds is None:
            return None
        if len(axis_bounds) != 2:
            raise ValueError(f"{name} must contain exactly (minimum, maximum)")
        lower, upper = map(float, axis_bounds)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"{name} must contain finite values with min < max")
        return lower, upper

    x_bounds = validate_axis_bounds("x_bounds", x_bounds)
    y_bounds = validate_axis_bounds("y_bounds", y_bounds)

    triplets = tuple(FormationGraph(g_nom).triplets)
    non_degeneracy_pairs = _build_non_degeneracy_pairs(
        triplets,
        n_agents=len(g_nom),
        constraint_edges=constraint_edges,
    )
    nominal_differences = _pair_coordinate_differences(
        g_nom,
        non_degeneracy_pairs,
        theta=theta,
    )
    if np.any(np.isclose(nominal_differences, 0.0, atol=1.0e-12)):
        raise ValueError(
            "The sign-preserving formulation is undefined because at least "
            "one nominal constrained coordinate separation is zero"
        )
    nominal_coordinate_signs = np.sign(nominal_differences)

    follower_slice = slice(n_leaders, len(g_nom))
    initial_positions = g_nom[follower_slice, :2].reshape(-1).copy()
    radius_array = np.asarray(search_radius, dtype=float)
    if radius_array.ndim == 0:
        radius_array = np.full_like(initial_positions, float(radius_array))
    else:
        radius_array = np.broadcast_to(
            radius_array,
            initial_positions.shape,
        ).copy()
    if np.any(~np.isfinite(radius_array)) or np.any(radius_array <= 0.0):
        raise ValueError("search_radius must contain finite positive values")

    lower_bounds = initial_positions - radius_array
    upper_bounds = initial_positions + radius_array
    for flat_index in range(len(initial_positions)):
        axis_bounds = x_bounds if flat_index % 2 == 0 else y_bounds
        if axis_bounds is not None:
            lower_bounds[flat_index] = axis_bounds[0]
            upper_bounds[flat_index] = axis_bounds[1]
    bounds = list(zip(lower_bounds, upper_bounds))

    def make_configuration(position_vector: np.ndarray) -> np.ndarray:
        candidate = g_nom.copy()
        candidate[follower_slice, :2] = np.asarray(position_vector).reshape(-1, 2)
        return candidate

    def evaluate(position_vector: np.ndarray) -> Tuple[float, float, float, float]:
        try:
            return follower_planar_matrix_metrics(
                make_configuration(position_vector),
                n_leaders=n_leaders,
                theta=theta,
                triplets=triplets,
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return float("inf"), float("inf"), 0.0, float("inf")

    def candidate_minimum_distance(position_vector: np.ndarray) -> float:
        positions = make_configuration(position_vector)[:, :2]
        difference = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(difference, axis=2)
        upper_triangle = np.triu_indices(len(positions), k=1)
        return float(np.min(distances[upper_triangle]))

    def geometry_values(
        position_vector: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        candidate = make_configuration(position_vector)
        coordinate_differences = _pair_coordinate_differences(
            candidate,
            non_degeneracy_pairs,
            theta=theta,
        )
        signed_coordinate_separations = (
            nominal_coordinate_signs * coordinate_differences
        ).reshape(-1)
        positions = candidate[:, :2]
        squared_distances = np.asarray(
            [
                np.sum((positions[i] - positions[j]) ** 2)
                for i, j in combinations(range(len(candidate)), 2)
            ],
            dtype=float,
        )
        return signed_coordinate_separations, squared_distances

    (
        initial_norm,
        initial_radius,
        _,
        _,
    ) = evaluate(initial_positions)

    def objective(position_vector: np.ndarray) -> float:
        spectral_norm, _, _, _ = evaluate(position_vector)
        return spectral_norm

    def physical_constraints(position_vector: np.ndarray) -> np.ndarray:
        _, _, sigma_min, _ = evaluate(position_vector)
        signed_separations, squared_distances = geometry_values(position_vector)
        return np.concatenate(
            (
                np.array([sigma_min - min_degree_singular_value]),
                signed_separations - coordinate_margin,
                squared_distances - min_pairwise_distance ** 2,
            )
        )

    nonlinear_constraint = NonlinearConstraint(
        physical_constraints,
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
            seed=seed,
            x0=np.clip(initial_positions, lower_bounds, upper_bounds),
            polish=bool(polish),
            updating="immediate",
            workers=1,
            tol=1.0e-8,
            atol=1.0e-10,
        )

    optimized = make_configuration(scipy_result.x)
    (
        optimized_norm,
        optimized_radius,
        optimized_sigma_min,
        optimized_condition_number,
    ) = evaluate(scipy_result.x)
    delta = np.asarray(scipy_result.x) - initial_positions
    position_deviation = float(delta @ delta)
    optimized_minimum_distance = candidate_minimum_distance(scipy_result.x)
    optimized_coordinate_separations, _ = geometry_values(scipy_result.x)
    optimized_min_coordinate_separation = float(
        np.min(optimized_coordinate_separations)
    )
    bound_tolerance = 1.0e-5 * np.maximum(1.0, radius_array)
    active_bounds = int(np.count_nonzero(
        (np.abs(scipy_result.x - lower_bounds) <= bound_tolerance)
        | (np.abs(scipy_result.x - upper_bounds) <= bound_tolerance)
    ))
    feasible = bool(
        np.isfinite(optimized_norm)
        and optimized_sigma_min >= min_degree_singular_value - 1.0e-7
        and optimized_minimum_distance >= min_pairwise_distance - 1.0e-7
        and optimized_min_coordinate_separation
        >= coordinate_margin - 1.0e-7
    )
    result = MinimumSpectralNormResult(
        configuration=optimized.copy(),
        initial_spectral_norm=initial_norm,
        optimized_spectral_norm=optimized_norm,
        initial_spectral_radius=initial_radius,
        optimized_spectral_radius=optimized_radius,
        position_deviation=position_deviation,
        min_degree_singular_value=optimized_sigma_min,
        degree_condition_number=optimized_condition_number,
        min_signed_coordinate_separation=optimized_min_coordinate_separation,
        minimum_pairwise_distance=optimized_minimum_distance,
        feasible=feasible,
        solver_success=bool(scipy_result.success),
        message=str(scipy_result.message),
        iterations=int(getattr(scipy_result, "nit", -1)),
        function_evaluations=int(getattr(scipy_result, "nfev", -1)),
        active_bound_coordinates=active_bounds,
    )

    if verbose:
        _print_minimum_spectral_norm_result(result, search_radius)
    if return_result:
        return optimized, result
    return optimized


def _print_spectral_norm_result(
    result: SpectralNormOptimizationResult,
    spectral_norm_limit: float,
) -> None:
    print("=" * 72)
    print("Nominal formation optimization (spectral norm constraint)")
    print(
        "Spectral norm       : "
        f"{result.initial_spectral_norm:.8f} -> "
        f"{result.optimized_spectral_norm:.8f} "
        f"(limit {spectral_norm_limit:.8f})"
    )
    print(
        "Spectral radius     : "
        f"{result.initial_spectral_radius:.8f} -> "
        f"{result.optimized_spectral_radius:.8f}"
    )
    print(f"Position deviation  : J = {result.objective_value:.8f}")
    print(
        "sigma_min(D_ff)    : "
        f"{result.min_degree_singular_value:.8e}"
    )
    print(
        "cond_2(D_ff)       : "
        f"{result.degree_condition_number:.8e}"
    )
    print(
        "Min signed coord   : "
        f"{result.min_signed_coordinate_separation:.8f}"
    )
    print(
        "Min pair distance  : "
        f"{result.minimum_pairwise_distance:.8f}"
    )
    print(
        "Solver             : "
        f"success={result.success}, iterations={result.iterations}, "
        f"evaluations={result.function_evaluations}"
    )
    print("Optimized formation :")
    print(np.array2string(result.configuration, precision=8, suppress_small=True))
    print("=" * 72)


def _print_minimum_spectral_norm_result(
    result: MinimumSpectralNormResult,
    search_radius: Union[float, Sequence[float]],
) -> None:
    print("=" * 72)
    print("Direct minimization of follower planar spectral norm")
    print(f"Search radius       : {search_radius}")
    print(
        "Spectral norm       : "
        f"{result.initial_spectral_norm:.8f} -> "
        f"{result.optimized_spectral_norm:.8f}"
    )
    print(
        "Spectral radius     : "
        f"{result.initial_spectral_radius:.8f} -> "
        f"{result.optimized_spectral_radius:.8f}"
    )
    print(f"Position deviation  : J = {result.position_deviation:.8f}")
    print(f"sigma_min(D_ff)    : {result.min_degree_singular_value:.8e}")
    print(f"cond_2(D_ff)       : {result.degree_condition_number:.8e}")
    print(
        "Min signed coord   : "
        f"{result.min_signed_coordinate_separation:.8f}"
    )
    print(f"Minimum distance   : {result.minimum_pairwise_distance:.8f}")
    print(f"Active bounds       : {result.active_bound_coordinates}")
    print(
        "Solver             : "
        f"success={result.solver_success}, feasible={result.feasible}, "
        f"iterations={result.iterations}, "
        f"evaluations={result.function_evaluations}"
    )
    print("=" * 72)
