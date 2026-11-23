"""One synchronous exchange of measured neighbor states per control tick."""

import numpy as np


FOLLOWER_CONTROL_MODES = ("centralized", "distributed")


class DistributedFollowerController:
    """Cache each follower's incident weights and apply the local stress law."""

    def __init__(self, formation):
        self._follower_edges = []
        for agent in range(formation.n_leaders, formation.n_agents):
            row = formation.omega[agent]
            neighbors = np.flatnonzero(row)
            neighbors = neighbors[neighbors != agent]
            # For i != j, the stress matrix entry is -w_ij.
            weights = -row[neighbors] / row[agent]
            self._follower_edges.append((agent, neighbors, weights))

    def follower_velocities(self, positions, measured_velocities):
        """Use the same pre-command snapshot for every follower and its neighbors."""
        return np.stack([
            weights @ (measured_velocities[neighbors] + (positions[neighbors] - positions[agent]))
            for agent, neighbors, weights in self._follower_edges
        ])
