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


FOLLOWER_CONTROL_MODES = ("centralized", "distributed")
NEIGHBOR_VELOCITY_SOURCES = ("measured", "command")


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

    def follower_velocities(self, positions, neighbor_velocities):
        """Use the same pre-command snapshot for every follower and its neighbors."""
        return np.stack([
            weights @ (neighbor_velocities[neighbors] + (positions[neighbors] - positions[agent]))
            for agent, neighbors, weights in self._follower_edges
        ])


def limit_speed(velocities, limit):
    norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    return velocities * (limit / np.maximum(norms, limit))


class FollowerController:
    """Select one follow law and keep any velocity messages between ticks."""

    def __init__(self, formation, mode, velocity_source, speed_limit):
        self.formation = formation
        self.speed_limit = speed_limit
        self.distributed = DistributedFollowerController(formation) if mode == "distributed" else None
        self.previous_commands = np.zeros((formation.n_agents, 2)) if velocity_source == "command" else None

    def reset(self):
        if self.previous_commands is not None:
            self.previous_commands.fill(0)

    def velocities(self, positions, measured_velocities, leader_velocities):
        if self.distributed is None:
            raw = self.formation.follower_velocities(positions, leader_velocities)
        else:
            messages = measured_velocities if self.previous_commands is None else self.previous_commands
            raw = self.distributed.follower_velocities(positions, messages)
        commands = limit_speed(raw, self.speed_limit)
        if self.previous_commands is not None:
            count = self.formation.n_leaders
            self.previous_commands[:count] = leader_velocities
            self.previous_commands[count:] = commands
        return commands
