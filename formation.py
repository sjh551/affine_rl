"""The nominal planar formation and 12-edge stress in Zhao2018Affine."""

import numpy as np


class AffineFormation:
    n_leaders = 3
    n_agents = 7
    scale = 1.5

    def __init__(self):
        nominal = np.array([
            [2, 0], [1, 1], [1, -1], [0, 1], [0, -1], [-1, 1], [-1, -1],
        ], dtype=float)
        self.nominal = self.scale * (nominal - nominal.mean(axis=0))
        # MATLAB incidence order, with zero-based node IDs and unit-norm stress.
        self.edges = (
            (0, 1), (0, 2), (2, 4), (4, 6), (5, 6), (3, 5),
            (1, 3), (1, 6), (2, 5), (0, 3), (3, 4), (0, 4),
        )
        weights = np.array([4, 4, 8, 4, 2, 4, 8, -2, -2, -2, 1, -2], dtype=float)
        weights /= np.linalg.norm(weights)
        incidence = np.zeros((self.n_agents, len(self.edges)))
        for column, (i, j) in enumerate(self.edges):
            incidence[i, column] = 1
            incidence[j, column] = -1
        self.omega = (incidence * weights) @ incidence.T
        follower_rows = self.omega[self.n_leaders:]
        self.follower_map = -np.linalg.solve(
            follower_rows[:, self.n_leaders:], follower_rows[:, :self.n_leaders],
        )

    def desired_positions(self, leaders):
        return np.vstack((leaders, self.follower_map @ leaders))

    def follower_velocities(self, positions, leader_velocities):
        """Centralized solution of the first-order affine maneuver law (gain 1)."""
        error = positions[self.n_leaders:] - self.follower_map @ positions[:self.n_leaders]
        return self.follower_map @ leader_velocities - error
