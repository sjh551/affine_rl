"""Validate the one-sample-delay stability bound by changing only the step size.

The experiment uses the formation currently supplied to ``POMDP_env.py`` and
k_f=1.  Both cases use
the same T matrix, controller, initial state, and number of physics substeps.
Only ``physics_timestep`` changes, hence the outer control period
``tau = physics_timestep * physics_substeps_per_control`` changes.

The primary result is the exact discrete closed-loop recurrence derived for the
paper controller.  A small deterministic perturbation is propagated so that a
stable and an unstable case can be compared without APF, wheel saturation, or
other PyBullet effects obscuring the sampling-period condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p

from POMDP_env import FormationPyBulletEnv


FOLLOWER_GAIN = 1.0
PHYSICS_SUBSTEPS_PER_CONTROL = 24
STABLE_CONTROL_PERIOD = 0.10
UNSTABLE_CONTROL_PERIOD = 0.60
SIMULATION_STEPS = 300
INITIAL_PERTURBATION_NORM = 1.0e-6
PYBULLET_STEPS = 150
PYBULLET_POSITION_PERTURBATION = 1.0e-3


@dataclass
class CaseResult:
    tau: float
    physics_timestep: float
    closed_loop_spectral_radius: float
    initial_norm: float
    final_norm: float
    maximum_norm: float
    growth_ratio: float


@dataclass
class PyBulletCaseResult:
    tau: float
    initial_residual: float
    final_residual: float
    maximum_residual: float
    residual_growth_ratio: float
    maximum_follower_speed: float
    executed_steps: int


def mode_gain_limit(eigenvalue: complex) -> float:
    """Return the strict Schur-Cohn upper bound on h=tau*k_f."""
    modulus_squared = float(abs(eigenvalue) ** 2)
    if modulus_squared >= 1.0:
        return 0.0

    distance_squared = float(abs(1.0 - eigenvalue) ** 2)
    denominator = distance_squared**2 + 4.0 * float(eigenvalue.imag**2)
    return (
        2.0
        * (1.0 - modulus_squared)
        * distance_squared
        / denominator
    )


def build_closed_loop_matrix(
    iteration_matrix: np.ndarray,
    tau: float,
    follower_gain: float,
) -> np.ndarray:
    """Build the delayed position-velocity closed-loop matrix."""
    identity = np.eye(iteration_matrix.shape[0])
    correction = identity - iteration_matrix
    return np.block([
        [
            identity - tau * follower_gain * correction,
            tau * iteration_matrix,
        ],
        [
            -follower_gain * correction,
            iteration_matrix,
        ],
    ])


def simulate_case(
    iteration_matrix: np.ndarray,
    tau: float,
    initial_state: np.ndarray,
) -> CaseResult:
    closed_loop = build_closed_loop_matrix(
        iteration_matrix,
        tau=tau,
        follower_gain=FOLLOWER_GAIN,
    )
    closed_loop_radius = float(
        np.max(np.abs(np.linalg.eigvals(closed_loop)))
    )

    state = initial_state.copy()
    norms = [float(np.linalg.norm(state))]
    for _ in range(SIMULATION_STEPS):
        state = closed_loop @ state
        norms.append(float(np.linalg.norm(state)))

    return CaseResult(
        tau=tau,
        physics_timestep=tau / PHYSICS_SUBSTEPS_PER_CONTROL,
        closed_loop_spectral_radius=closed_loop_radius,
        initial_norm=norms[0],
        final_norm=norms[-1],
        maximum_norm=max(norms),
        growth_ratio=norms[-1] / norms[0],
    )


def formation_residual_norm(env: FormationPyBulletEnv) -> float:
    states = env._get_all_robot_states()
    position_indices = np.array([
        3 * agent_index + coordinate
        for agent_index in range(env.num_robots)
        for coordinate in (0, 1)
    ])
    position_matrix = env.M_3d[np.ix_(position_indices, position_indices)]
    joint_state = np.concatenate([
        np.asarray(state["pos"][:2], dtype=float)
        for state in states.values()
    ])
    return float(np.linalg.norm(position_matrix @ joint_state))


def run_pybullet_case(tau: float) -> PyBulletCaseResult:
    """Run the unchanged wheel-driven PyBullet environment for one tau."""
    env = FormationPyBulletEnv(
        render_mode=False,
        enable_obstacles=False,
        follower_control_mode="distributed",
        follower_formation_gain=FOLLOWER_GAIN,
        print_follower_spectrum=False,
        physics_substeps_per_control=PHYSICS_SUBSTEPS_PER_CONTROL,
        physics_timestep=tau / PHYSICS_SUBSTEPS_PER_CONTROL,
    )
    try:
        env.reset(seed=20260919)

        # Disable APF so that only the formation controller and vehicle model
        # remain.  Apply the same small position perturbation in both cases.
        env._get_obstacle_repulsion = (
            lambda _position, safe_margin=1.0, num_rays=90:
            (safe_margin, np.zeros(2, dtype=float))
        )
        follower_id = env.robot_ids[env.n_leaders]
        position, orientation = p.getBasePositionAndOrientation(follower_id)
        perturbed_position = list(position)
        perturbed_position[0] += PYBULLET_POSITION_PERTURBATION
        p.resetBasePositionAndOrientation(
            follower_id,
            perturbed_position,
            orientation,
        )

        residuals = [formation_residual_norm(env)]
        maximum_follower_speed = 0.0
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        executed_steps = 0

        for _ in range(PYBULLET_STEPS):
            env.step(action)
            executed_steps += 1
            residual = formation_residual_norm(env)
            residuals.append(residual)

            states = env._get_all_robot_states()
            follower_speed = max(
                np.linalg.norm(states[index]["vel"][:2])
                for index in range(env.n_leaders, env.num_robots)
            )
            maximum_follower_speed = max(
                maximum_follower_speed,
                float(follower_speed),
            )
            if not np.isfinite(residual) or residual > 1.0e3:
                break

        initial_residual = residuals[0]
        return PyBulletCaseResult(
            tau=tau,
            initial_residual=initial_residual,
            final_residual=residuals[-1],
            maximum_residual=max(residuals),
            residual_growth_ratio=residuals[-1] / initial_residual,
            maximum_follower_speed=maximum_follower_speed,
            executed_steps=executed_steps,
        )
    finally:
        env.close()


def main() -> None:
    # Use the environment to construct the current formation and its actual
    # follower iteration matrix.  No physics episode is run.
    env = FormationPyBulletEnv(
        render_mode=False,
        enable_obstacles=False,
        follower_control_mode="distributed",
        follower_formation_gain=FOLLOWER_GAIN,
        print_follower_spectrum=False,
    )
    try:
        iteration_matrix = env.follower_iteration_matrix.copy()
    finally:
        env.close()

    eigenvalues = np.linalg.eigvals(iteration_matrix)
    gain_limits = np.array([
        mode_gain_limit(complex(value)) for value in eigenvalues
    ])
    limiting_index = int(np.argmin(gain_limits))
    limiting_eigenvalue = complex(eigenvalues[limiting_index])
    h_limit = float(gain_limits[limiting_index])
    tau_limit = h_limit / FOLLOWER_GAIN

    # A real deterministic perturbation excites the conjugate modes without
    # relying on a complex-valued physical state.
    rng = np.random.default_rng(20260919)
    initial_state = rng.normal(size=2 * iteration_matrix.shape[0])
    initial_state *= INITIAL_PERTURBATION_NORM / np.linalg.norm(initial_state)

    cases = [
        simulate_case(iteration_matrix, STABLE_CONTROL_PERIOD, initial_state),
        simulate_case(iteration_matrix, UNSTABLE_CONTROL_PERIOD, initial_state),
    ]

    print("=" * 76)
    print("One-sample-delay timestep stability experiment")
    print(f"Follower gain k_f       : {FOLLOWER_GAIN:.6f}")
    print(f"rho(T)                 : {max(abs(eigenvalues)):.8f}")
    print(
        "Limiting eigenvalue     : "
        f"{limiting_eigenvalue.real:+.8f}"
        f"{limiting_eigenvalue.imag:+.8f}j"
    )
    print(f"Schur-Cohn h limit     : {h_limit:.8f}")
    print(f"Control-period limit   : tau < {tau_limit:.8f} s")
    print(f"Physics substeps       : {PHYSICS_SUBSTEPS_PER_CONTROL}")
    print("Only physics_timestep differs between the two cases.")
    print("-" * 76)
    for result in cases:
        prediction = "stable" if result.tau < tau_limit else "unstable"
        measured = (
            "stable"
            if result.closed_loop_spectral_radius < 1.0
            else "unstable"
        )
        print(
            f"tau={result.tau:.2f} s, "
            f"physics_timestep={result.physics_timestep:.8f} s, "
            f"rho(F)={result.closed_loop_spectral_radius:.8f} | "
            f"predicted={prediction}, measured={measured}"
        )
        print(
            f"  ||state_0||={result.initial_norm:.6e}, "
            f"||state_N||={result.final_norm:.6e}, "
            f"max={result.maximum_norm:.6e}, "
            f"final/initial={result.growth_ratio:.6e}"
        )

    print("-" * 76)
    print("Current wheel-driven PyBullet environment (APF disabled)")
    pybullet_cases = [
        run_pybullet_case(STABLE_CONTROL_PERIOD),
        run_pybullet_case(UNSTABLE_CONTROL_PERIOD),
    ]
    for result in pybullet_cases:
        print(
            f"tau={result.tau:.2f} s | steps={result.executed_steps}, "
            f"residual: {result.initial_residual:.6e} -> "
            f"{result.final_residual:.6e}, "
            f"max={result.maximum_residual:.6e}, "
            f"final/initial={result.residual_growth_ratio:.6e}, "
            f"max follower speed={result.maximum_follower_speed:.6e} m/s"
        )
    print("=" * 76)


if __name__ == "__main__":
    main()
