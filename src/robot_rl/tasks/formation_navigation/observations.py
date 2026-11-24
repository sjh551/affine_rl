"""Leader kinematics and LiDAR history in the fixed episode frame."""

from collections import deque

from gymnasium import spaces
import numpy as np


class ObservationEncoder:
    _lidar_range = 7.0
    _lidar_bins = 60
    _history_indices = (0, 4, 9, 14, 19)

    def __init__(self, formation, world, arena, leader_speed):
        self.formation = formation
        self.world = world
        self.arena = arena
        self.leader_speed = leader_speed
        count = formation.n_leaders
        self.space = spaces.Dict({
            "kinematics": spaces.Box(-np.inf, np.inf, (2 + 9 * count,), dtype=np.float32),
            "lidar": spaces.Box(0.0, 1.0, (count, len(self._history_indices), self._lidar_bins), dtype=np.float32),
        })
        self._history = deque(maxlen=self._history_indices[-1] + 1)
        angles = np.linspace(-np.pi, np.pi, 3 * self._lidar_bins, endpoint=False)
        self._ray_vectors = np.column_stack((np.cos(angles), np.sin(angles))) * self._lidar_range

    def reset(self):
        self._history.clear()

    def encode(self, states, *, goal, frame, heading, previous_action):
        positions, velocities, yaws, yaw_rates = states
        count = self.formation.n_leaders
        leaders = positions[:count]
        center = self.formation.desired_positions(leaders).mean(axis=0)
        relative_goal = (goal - center) @ frame.T / self.arena
        offsets = (leaders - center) @ frame.T / self.formation.scale
        leader_velocities = velocities[:count] @ frame.T / self.leader_speed
        yaw_error = yaws[:count] - heading
        attitudes = np.column_stack((np.sin(yaw_error), np.cos(yaw_error), yaw_rates[:count]))
        kinematics = np.concatenate((relative_goal, offsets.ravel(), leader_velocities.ravel(),
                                     attitudes.ravel(), previous_action)).astype(np.float32)
        scan = self._scan_lidar(leaders, frame)
        if not self._history:
            self._history.extend([scan] * self._history.maxlen)
        else:
            self._history.append(scan)
        lidar = np.stack([self._history[i] for i in self._history_indices], axis=1)
        return {"kinematics": kinematics, "lidar": lidar}

    def _scan_lidar(self, leaders, frame):
        origins = np.repeat(leaders, len(self._ray_vectors), axis=0)
        rays = self._ray_vectors @ frame
        ends = origins + np.tile(rays, (len(leaders), 1))
        height = np.full(len(origins), 0.32)
        fractions = self.world.ray_fractions(np.column_stack((origins, height)),
                                             np.column_stack((ends, height)))
        return fractions.reshape(len(leaders), self._lidar_bins, 3).min(axis=2)
