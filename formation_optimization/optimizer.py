"""Nominal-formation optimization for the distributed follower velocity loop.

The optimizer changes only follower planar positions.  Leaders and all yaw
components remain fixed.  The matrices are built with the same triplets and
``compute_W_matrix`` implementation used by :mod:`POMDP_env`, so the optimized
spectral radius refers to the controller that is actually executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence, Tuple, Union
import warnings

import numpy as np

from utils.formation_graph import FormationGraph
from utils.formation_utils import compute_W_matrix


@dataclass(frozen=True)
class FormationOptimizationResult:
    """Diagnostics returned with an optimized nominal configuration."""

    configuration: np.ndarray
    initial_spectral_radius: float
    optimized_spectral_radius: float
    objective_value: float
    min_degree_singular_value: float
    min_signed_coordinate_separation: float
    minimum_pairwise_distance: float
    success: bool
    message: str
    iterations: int
    function_evaluations: int


def _unique_undirected_pairs(
    pairs: Sequence[Tuple[int, int]],
    n_agents: int,
) -> Tuple[Tuple[int, int], ...]:
    """Validate and de-duplicate unordered agent pairs."""

    unique_pairs = set()
    for raw_u, raw_v in pairs:
        u, v = int(raw_u), int(raw_v)
        if not 0 <= u < n_agents or not 0 <= v < n_agents:
            raise ValueError(
                f"constraint pair ({u}, {v}) contains an invalid agent index"
            )
        if u == v:
            raise ValueError(f"constraint pair ({u}, {v}) must use two agents")
        unique_pairs.add((min(u, v), max(u, v)))
    return tuple(sorted(unique_pairs))


def _build_non_degeneracy_pairs(
    triplets: Sequence[Tuple[int, int, int]],
    n_agents: int,
    constraint_edges: Optional[Sequence[Tuple[int, int]]],
) -> Tuple[Tuple[int, int], ...]:
    """Build the pairs appearing in the two signed separation constraints.

    For every local constraint ``(i, j, k)``, the pair ``(i, j)`` enforces
    invertibility of the local diagonal block.  The DEP/sensing-edge pairs are
    either supplied explicitly or inferred as ``(i, k)`` and ``(j, k)`` from
    the same triplets.
    """

    triplet_pairs = [(i, j) for i, j, _ in triplets]
    if constraint_edges is None:
        edge_pairs = [
            pair
            for i, j, k in triplets
            for pair in ((i, k), (j, k))
        ]
    else:
        edge_pairs = list(constraint_edges)
    return _unique_undirected_pairs(
        [*triplet_pairs, *edge_pairs],
        n_agents=n_agents,
    )


def _positions_in_theta_frame(
    configuration: np.ndarray,
    theta: float,
) -> np.ndarray:
    """Return planar positions expressed in the fixed theta frame."""

    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    return np.asarray(configuration, dtype=float)[:, :2] @ rotation


def _pair_coordinate_differences(
    configuration: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    theta: float,
) -> np.ndarray:
    """Return one ``[dx_theta, dy_theta]`` row per agent pair."""

    positions_theta = _positions_in_theta_frame(configuration, theta)
    return np.asarray(
        [positions_theta[u] - positions_theta[v] for u, v in pairs],
        dtype=float,
    )


def _minimum_pairwise_distance(configuration: np.ndarray) -> float:
    """Return the minimum Euclidean distance between any two agents."""

    positions = np.asarray(configuration, dtype=float)[:, :2]
    distances = [
        np.linalg.norm(positions[i] - positions[j])
        for i, j in combinations(range(len(positions)), 2)
    ]
    return float(min(distances)) if distances else float("inf")


def build_follower_planar_iteration_matrix(
    configuration: np.ndarray,
    n_leaders: int,
    theta: float = 0.0,
    triplets: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build ``D_ff``, ``A_ff`` and ``T_ff = D_ff^-1 A_ff`` for x/y states.

    This is deliberately independent of the number of followers.  Each local
    triplet contributes exactly the same blocks as ``POMDP_env``.
    """

    g_nom = np.asarray(configuration, dtype=float)
    if g_nom.ndim != 2 or g_nom.shape[1] < 2:
        raise ValueError("configuration must have shape (n_agents, state_dim>=2)")
    if not 0 < n_leaders < len(g_nom):
        raise ValueError("n_leaders must be between 1 and n_agents - 1")

    if triplets is None:
        triplets = FormationGraph(g_nom).triplets

    planar_dim = 2
    n_followers = len(g_nom) - n_leaders
    matrix_size = n_followers * planar_dim
    degree = np.zeros((matrix_size, matrix_size), dtype=float)
    adjacency = np.zeros_like(degree)
    followers_with_constraints = set()

    for i, j, k in triplets:
        if k < n_leaders:
            continue

        W_jk, W_ki = compute_W_matrix(
            g_nom[i],
            g_nom[j],
            g_nom[k],
            theta=theta,
        )
        W_jk_xy = np.asarray(W_jk[:planar_dim, :planar_dim], dtype=float)
        W_ki_xy = np.asarray(W_ki[:planar_dim, :planar_dim], dtype=float)

        follower_row = k - n_leaders
        row = slice(planar_dim * follower_row, planar_dim * (follower_row + 1))
        degree[row, row] += W_jk_xy + W_ki_xy
        followers_with_constraints.add(k)

        # The block mapping is the same as Eq. (18) and POMDP_env:
        # W_jk multiplies neighbour i, while W_ki multiplies neighbour j.
        if i >= n_leaders:
            follower_col = i - n_leaders
            col = slice(
                planar_dim * follower_col,
                planar_dim * (follower_col + 1),
            )
            adjacency[row, col] += W_jk_xy

        if j >= n_leaders:
            follower_col = j - n_leaders
            col = slice(
                planar_dim * follower_col,
                planar_dim * (follower_col + 1),
            )
            adjacency[row, col] += W_ki_xy

    expected_followers = set(range(n_leaders, len(g_nom)))
    missing = expected_followers - followers_with_constraints
    if missing:
        raise ValueError(
            "Followers without local triplet constraints: "
            f"{sorted(missing)}"
        )

    singular_values = np.linalg.svd(degree, compute_uv=False)
    min_singular_value = float(np.min(singular_values))
    if min_singular_value <= np.finfo(float).eps:
        raise np.linalg.LinAlgError("D_ff is singular")

    iteration = np.linalg.solve(degree, adjacency)
    return degree, adjacency, iteration, min_singular_value


def follower_planar_spectral_radius(
    configuration: np.ndarray,
    n_leaders: int,
    theta: float = 0.0,
    triplets: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> Tuple[float, float]:
    """Return ``rho(D_ff^-1 A_ff)`` and ``sigma_min(D_ff)``."""

    degree, _, iteration, min_singular_value = (
        build_follower_planar_iteration_matrix(
            configuration,
            n_leaders=n_leaders,
            theta=theta,
            triplets=triplets,
        )
    )
    del degree
    eigenvalues = np.linalg.eigvals(iteration)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    return spectral_radius, min_singular_value


def optimize_nominal_configuration(
    nominal_configuration: np.ndarray,
    n_leaders: int,
    theta: float = 0.0,
    spectral_radius_limit: float = 0.95,
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
) -> Union[np.ndarray, Tuple[np.ndarray, FormationOptimizationResult]]:
    """Find the closest follower configuration satisfying the spectral limit.

    The solved problem is

    ``min sum_i ||p_i - p_i_nom||^2``

    subject to the spectral-radius bound, signed coordinate-separation
    constraints in the fixed theta frame, a minimum all-agent distance, and
    follower workspace bounds.  Leaders and yaw values remain exactly fixed.
    A lower bound on ``sigma_min(D_ff)`` additionally protects the numerical
    solve.  Differential evolution is used because the resulting constraints
    are generally non-convex and the spectral radius can be non-smooth.
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
    if not 0.0 < spectral_radius_limit < 1.0:
        raise ValueError("spectral_radius_limit must lie strictly between 0 and 1")
    if min_degree_singular_value <= 0.0:
        raise ValueError("min_degree_singular_value must be positive")
    if coordinate_margin <= 0.0:
        raise ValueError("coordinate_margin must be positive")
    if minimum_pairwise_distance < 0.0:
        raise ValueError("minimum_pairwise_distance must be non-negative")

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
        radius_array = np.broadcast_to(radius_array, initial_positions.shape).copy()
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

    def evaluate(position_vector: np.ndarray) -> Tuple[float, float]:
        try:
            return follower_planar_spectral_radius(
                make_configuration(position_vector),
                n_leaders=n_leaders,
                theta=theta,
                triplets=triplets,
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return float("inf"), 0.0

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

    initial_radius, initial_sigma_min = evaluate(initial_positions)
    initial_coordinate_separations, initial_squared_distances = geometry_values(
        initial_positions
    )
    initial_within_bounds = bool(
        np.all(initial_positions >= lower_bounds)
        and np.all(initial_positions <= upper_bounds)
    )

    # If the nominal formation is already feasible, the zero-deviation point is
    # the global optimum and no numerical search is necessary.
    if (
        initial_radius <= spectral_radius_limit
        and initial_sigma_min >= min_degree_singular_value
        and np.all(initial_coordinate_separations >= coordinate_margin)
        and np.all(
            initial_squared_distances >= minimum_pairwise_distance ** 2
        )
        and initial_within_bounds
    ):
        result = FormationOptimizationResult(
            configuration=optimized.copy(),
            initial_spectral_radius=initial_radius,
            optimized_spectral_radius=initial_radius,
            objective_value=0.0,
            min_degree_singular_value=initial_sigma_min,
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
            _print_optimization_result(result, spectral_radius_limit)
        if return_result:
            return optimized, result
        return optimized

    def objective(position_vector: np.ndarray) -> float:
        delta = np.asarray(position_vector) - initial_positions
        return float(delta @ delta)

    def feasibility_values(position_vector: np.ndarray) -> np.ndarray:
        radius, sigma_min = evaluate(position_vector)
        signed_separations, squared_distances = geometry_values(position_vector)
        return np.concatenate(
            (
                np.array(
                    [
                        spectral_radius_limit - radius,
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
        # SciPy's constrained polishing step may emit this harmless warning
        # because our objective is an exact quadratic function.
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
            polish=bool(polish),
            updating="immediate",
            workers=1,
            tol=1.0e-8,
            atol=1.0e-10,
        )

    optimized[follower_slice, :2] = scipy_result.x.reshape(-1, 2)
    optimized_radius, optimized_sigma_min = evaluate(scipy_result.x)
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
        optimized_radius <= spectral_radius_limit + constraint_tolerance
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
    message = str(scipy_result.message)

    result = FormationOptimizationResult(
        configuration=optimized.copy(),
        initial_spectral_radius=initial_radius,
        optimized_spectral_radius=optimized_radius,
        objective_value=objective(scipy_result.x),
        min_degree_singular_value=optimized_sigma_min,
        min_signed_coordinate_separation=optimized_min_coordinate_separation,
        minimum_pairwise_distance=optimized_min_pairwise_distance,
        success=success,
        message=message,
        iterations=int(getattr(scipy_result, "nit", -1)),
        function_evaluations=int(getattr(scipy_result, "nfev", -1)),
    )

    if verbose:
        _print_optimization_result(result, spectral_radius_limit)

    if strict and not success:
        raise RuntimeError(
            "Formation optimization did not find a feasible configuration. "
            f"Best rho={optimized_radius:.8f}, "
            f"required rho<={spectral_radius_limit:.8f}; "
            "minimum signed coordinate separation="
            f"{optimized_min_coordinate_separation:.8f}, "
            f"required>={coordinate_margin:.8f}; "
            f"minimum pairwise distance={optimized_min_pairwise_distance:.8f}, "
            f"required>={minimum_pairwise_distance:.8f}. "
            "Increase search_radius/maxiter or relax the spectral limit."
        )

    if return_result:
        return optimized, result
    return optimized


def _print_optimization_result(
    result: FormationOptimizationResult,
    spectral_radius_limit: float,
) -> None:
    """Print a compact, reproducible optimization summary."""

    print("=" * 72)
    print("Nominal formation optimization")
    print(
        "Spectral radius     : "
        f"{result.initial_spectral_radius:.8f} -> "
        f"{result.optimized_spectral_radius:.8f} "
        f"(limit {spectral_radius_limit:.8f})"
    )
    print(f"Position deviation  : J = {result.objective_value:.8f}")
    print(
        "sigma_min(D_ff)    : "
        f"{result.min_degree_singular_value:.8e}"
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
