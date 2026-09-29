"""Synchronous inner iterations for the implicit follower velocity law."""

from __future__ import annotations

import numpy as np


def synchronous_follower_velocity_iterations(
    frozen_agent_states,
    frozen_agent_velocities,
    *,
    n_leaders,
    follower_local_constraints,
    follower_Wkk_inv,
    formation_gain,
    iterations,
):
    """Approximate the same-time follower velocities with Jacobi iterations.

    Positions, leader velocities, and all external quantities are frozen by the
    caller for the complete inner loop.  Follower estimates are initialized
    from the measured velocities at the start of the outer control period.
    Every round reads a copy of the previous round and publishes all new
    follower estimates simultaneously.

    Returns
    -------
    follower_velocity_estimates : ndarray, shape (n_followers, state_dim)
        Estimates after the requested number of synchronous rounds.
    delta_norms : ndarray, shape (iterations, n_followers)
        ``||v_k^(r) - v_k^(r-1)||_2`` for every round and follower.
    planar_delta_norms : ndarray, shape (iterations, n_followers)
        The same diagnostic restricted to the first two translational axes.
    """
    states = np.asarray(frozen_agent_states, dtype=float)
    velocities = np.asarray(frozen_agent_velocities, dtype=float)

    if states.ndim != 2 or velocities.shape != states.shape:
        raise ValueError(
            "frozen_agent_states and frozen_agent_velocities must have the "
            "same two-dimensional shape"
        )
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 1:
        raise ValueError("iterations must be a positive integer")

    iterations = int(iterations)
    n_agents, state_dim = states.shape
    follower_indices = list(range(int(n_leaders), n_agents))
    n_followers = len(follower_indices)

    estimates = velocities[n_leaders:].copy()
    delta_norms = np.empty((iterations, n_followers), dtype=float)
    planar_delta_norms = np.empty((iterations, n_followers), dtype=float)

    for round_index in range(iterations):
        previous_estimates = estimates.copy()
        next_estimates = np.empty_like(previous_estimates)

        for follower_row, k in enumerate(follower_indices):
            local_input = np.zeros(state_dim, dtype=float)
            g_k = states[k]

            for i, j, W_jk, W_ki in follower_local_constraints[k]:
                velocity_i = (
                    velocities[i]
                    if i < n_leaders
                    else previous_estimates[i - n_leaders]
                )
                velocity_j = (
                    velocities[j]
                    if j < n_leaders
                    else previous_estimates[j - n_leaders]
                )

                local_input += (
                    W_jk @ (formation_gain * (states[i] - g_k) + velocity_i)
                    + W_ki @ (formation_gain * (states[j] - g_k) + velocity_j)
                )

            next_estimates[follower_row] = (
                follower_Wkk_inv[k] @ local_input
            )

        estimate_changes = next_estimates - previous_estimates
        delta_norms[round_index] = np.linalg.norm(estimate_changes, axis=1)
        planar_delta_norms[round_index] = np.linalg.norm(
            estimate_changes[:, :min(2, state_dim)],
            axis=1,
        )
        estimates = next_estimates

    return estimates, delta_norms, planar_delta_norms
