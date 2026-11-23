"""Gymnasium navigation task for three affine leaders and four Ranger followers."""

from collections import deque

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

from formation import AffineFormation
from ranger import Ranger


class FormationPyBulletEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    _physics_dt = 1 / 240
    _substeps = 24
    _arena = (20.0, 10.0)
    _leader_speed = 1.0
    _follower_speed = 1.3
    _lidar_range = 7.0
    _lidar_bins = 60
    _lidar_history_indices = (0, 4, 9, 14, 19)

    def __init__(self, render_mode=None, enable_obstacles=True, max_steps=1500, enable_walls=True):
        super().__init__()
        if isinstance(render_mode, bool):
            render_mode = "human" if render_mode else None
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'.")
        if not isinstance(max_steps, (int, np.integer)) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer.")
        self.render_mode = render_mode
        self.enable_obstacles = enable_obstacles
        self.enable_walls = enable_walls
        self.max_steps = max_steps
        self.formation = AffineFormation()
        self.n_leaders = self.formation.n_leaders
        self.num_robots = self.formation.n_agents

        self.action_space = spaces.Box(-1.0, 1.0, (2 * self.n_leaders,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "kinematics": spaces.Box(-np.inf, np.inf, (2 + 9 * self.n_leaders,), dtype=np.float32),
            "lidar": spaces.Box(0.0, 1.0, (self.n_leaders, len(self._lidar_history_indices),
                                          self._lidar_bins), dtype=np.float32),
        })
        self._lidar_history = deque(maxlen=self._lidar_history_indices[-1] + 1)
        angles = np.linspace(-np.pi, np.pi, 3 * self._lidar_bins, endpoint=False)
        self._ray_vectors = np.column_stack((np.cos(angles), np.sin(angles))) * self._lidar_range
        self._pair_indices = np.triu_indices(self.num_robots, k=1)
        self._client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        if self._client < 0:
            raise RuntimeError("Unable to connect to PyBullet.")
        self._needs_reset = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        client = self._client
        if client is None:
            raise RuntimeError("The PyBullet environment is closed.")
        self._needs_reset = True
        p.resetSimulation(physicsClientId=client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        p.setTimeStep(self._physics_dt, physicsClientId=client)
        p.setPhysicsEngineParameter(numSolverIterations=100, deterministicOverlappingPairs=1,
                                    physicsClientId=client)
        p.loadURDF("plane.urdf", physicsClientId=client)

        start = np.array([self.np_random.uniform(-14.0, -13.8), self.np_random.uniform(-2.0, 2.0)])
        self._goal = np.array([self.np_random.uniform(15.5, 16.0), self.np_random.uniform(-4.0, 4.0)])
        direction = self._goal - start
        self._heading = np.arctan2(direction[1], direction[0])
        forward = direction / np.linalg.norm(direction)
        self._frame = np.array([forward, [-forward[1], forward[0]]])
        initial = self.formation.nominal @ self._frame + start

        self._wall_ids, self._obstacle_ids = set(), set()
        self._build_scene()
        self._robots = [Ranger(client, xy, self._heading, i < self.n_leaders)
                        for i, xy in enumerate(initial)]
        self._robot_ids = {robot.body for robot in self._robots}
        for _ in range(round(0.5 / self._physics_dt)):
            p.stepSimulation(physicsClientId=client)

        self._steps = 0
        self._previous_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._lidar_history.clear()
        positions, velocities, yaws, yaw_rates = self._robot_states()
        self._goal_distance = float(np.linalg.norm(self._goal - positions.mean(axis=0)))
        self._debug_lines = [-1] * len(self.formation.edges)
        self._draw_formation(positions)
        obs = self._observation(positions, velocities, yaws, yaw_rates)
        self._needs_reset = False
        return obs, {}

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset() before stepping the environment.")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError("Expected six finite action components in leader order.")
        action = np.clip(action, -1, 1)
        action_change = float(np.mean(np.square(action - self._previous_action)))
        self._previous_action = action
        self._steps += 1

        positions, _, yaws, _ = self._robot_states()
        leader_velocities = self._limit_speed(action.reshape(self.n_leaders, 2) @ self._frame,
                                              self._leader_speed)
        follower_velocities = self.formation.follower_velocities(positions, leader_velocities)
        commands = np.vstack((leader_velocities, self._limit_speed(follower_velocities, self._follower_speed)))
        for robot, velocity, yaw in zip(self._robots, commands, yaws):
            robot.command(velocity, yaw, self._heading)

        collision = None
        for _ in range(self._substeps):
            p.stepSimulation(physicsClientId=self._client)
            collision = self._contact_reason()
            if collision:
                break

        positions, velocities, yaws, yaw_rates = self._robot_states()
        desired = self.formation.desired_positions(positions[:self.n_leaders])
        distances = np.linalg.norm(positions[self.n_leaders:] - desired[self.n_leaders:], axis=1)
        tracking_error = float(np.sqrt(np.mean(np.square(distances))))
        goal_distance = float(np.linalg.norm(self._goal - positions.mean(axis=0)))
        leader_offsets = (positions[:self.n_leaders] - desired.mean(axis=0)) @ self._frame.T
        target_offsets = self.formation.nominal[:self.n_leaders]
        shape_error = float(np.sqrt(np.mean(np.sum(np.square(leader_offsets - target_offsets), axis=1))))

        i, j = self._pair_indices
        if collision is None and np.any(np.linalg.norm(positions[i] - positions[j], axis=1) < 0.7):
            collision = "Crash_Team"
        if collision is None and self.enable_walls and np.any(np.abs(positions) >= self._arena):
            collision = "Wall_Out"

        reward = (10.0 * (self._goal_distance - goal_distance) - 0.05
                  - 0.05 * action_change - 0.5 * max(0.0, tracking_error - 0.25))
        self._goal_distance = goal_distance
        if goal_distance < 5.0:
            reward -= 0.2 * shape_error
        success = collision is None and goal_distance < 2.0 and shape_error <= 1.0 and tracking_error <= 0.5
        if success:
            reward += 150.0
        if collision:
            reward -= 50.0

        terminated = bool(success or collision)
        truncated = not terminated and self._steps >= self.max_steps
        self._needs_reset = terminated or truncated
        info = {
            "is_success": bool(success),
            "goal_distance": goal_distance,
            "formation_error": tracking_error,
            "shape_error": shape_error,
            "reason": collision or ("Success" if success else "Timeout" if truncated else ""),
        }
        self._draw_formation(positions)
        obs = self._observation(positions, velocities, yaws, yaw_rates)
        return obs, float(reward), terminated, truncated, info

    @staticmethod
    def _limit_speed(velocities, limit):
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        return velocities * (limit / np.maximum(norms, limit))

    def _robot_states(self):
        return tuple(np.asarray(values) for values in zip(*(robot.state() for robot in self._robots)))

    def _contact_reason(self):
        for contact in p.getContactPoints(physicsClientId=self._client):
            a, b = contact[1:3]
            if a in self._robot_ids:
                other = b
            elif b in self._robot_ids:
                other = a
            else:
                continue
            if other in self._robot_ids and a != b:
                return "Crash_Team"
            if other in self._wall_ids:
                return "Wall_Out"
            if other in self._obstacle_ids:
                return "Crash_Obstacle"
        return None

    def _observation(self, positions, velocities, yaws, yaw_rates):
        leaders = positions[:self.n_leaders]
        center = self.formation.desired_positions(leaders).mean(axis=0)
        goal = (self._goal - center) @ self._frame.T / self._arena
        offsets = (leaders - center) @ self._frame.T / self.formation.scale
        leader_velocities = velocities[:self.n_leaders] @ self._frame.T / self._leader_speed
        yaw_error = yaws[:self.n_leaders] - self._heading
        attitudes = np.column_stack((np.sin(yaw_error), np.cos(yaw_error), yaw_rates[:self.n_leaders]))
        kinematics = np.concatenate((goal, offsets.ravel(), leader_velocities.ravel(),
                                     attitudes.ravel(), self._previous_action)).astype(np.float32)

        scan = self._scan_lidar(leaders)
        if not self._lidar_history:
            self._lidar_history.extend([scan] * self._lidar_history.maxlen)
        else:
            self._lidar_history.append(scan)
        lidar = np.stack([self._lidar_history[i] for i in self._lidar_history_indices], axis=1)
        return {"kinematics": kinematics, "lidar": lidar}

    def _scan_lidar(self, leaders):
        origins = np.repeat(leaders, len(self._ray_vectors), axis=0)
        rays = self._ray_vectors @ self._frame
        ends = origins + np.tile(rays, (self.n_leaders, 1))
        height = np.full(len(origins), 0.32)
        hits = p.rayTestBatch(np.column_stack((origins, height)).tolist(),
                             np.column_stack((ends, height)).tolist(),
                             collisionFilterMask=2, physicsClientId=self._client)
        fractions = np.asarray([hit[2] for hit in hits], dtype=np.float32)
        return fractions.reshape(self.n_leaders, self._lidar_bins, 3).min(axis=2)

    def _build_scene(self):
        client = self._client

        def add_wall(position, half_extents):
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=(0.6, 0.6, 0.6, 1),
                                         physicsClientId=client)
            body = p.createMultiBody(baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
                                     basePosition=position, physicsClientId=client)
            p.setCollisionFilterGroupMask(body, -1, 2, 3, physicsClientId=client)
            return body

        x_limit, y_limit = self._arena
        if self.enable_walls:
            for sign in (-1, 1):
                self._wall_ids.add(add_wall((sign * (x_limit + 0.15), 0, 0.5), (0.15, y_limit + 0.3, 0.5)))
                self._wall_ids.add(add_wall((0, sign * (y_limit + 0.15), 0.5), (x_limit, 0.15, 0.5)))
        if self.enable_obstacles:
            obstacle = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.5, height=1.5, physicsClientId=client)
            visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, length=1.5,
                                         rgbaColor=(0.2, 0.2, 0.2, 1), physicsClientId=client)
            for x in np.linspace(-4.0, 4.0, 4):
                for y in (-y_limit / 2, 0.0, y_limit / 2):
                    body = p.createMultiBody(baseCollisionShapeIndex=obstacle, baseVisualShapeIndex=visual,
                                             basePosition=(x, y, 0.75), physicsClientId=client)
                    p.setCollisionFilterGroupMask(body, -1, 2, 3, physicsClientId=client)
                    self._obstacle_ids.add(body)
        if self.render_mode == "human":
            marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=(0, 1, 0, 1),
                                         physicsClientId=client)
            p.createMultiBody(baseVisualShapeIndex=marker, basePosition=(*self._goal, 0.2), physicsClientId=client)
            p.resetDebugVisualizerCamera(36, 0, -75, (0, 0, 0), physicsClientId=client)

    def _draw_formation(self, positions):
        if self.render_mode == "human":
            for k, (i, j) in enumerate(self.formation.edges):
                self._debug_lines[k] = p.addUserDebugLine(
                    (*positions[i], 0.35), (*positions[j], 0.35), (0.2, 0.6, 1.0),
                    replaceItemUniqueId=self._debug_lines[k], physicsClientId=self._client,
                )

    def close(self):
        if self._client is not None:
            p.disconnect(physicsClientId=self._client)
            self._client = None
        self._needs_reset = True
