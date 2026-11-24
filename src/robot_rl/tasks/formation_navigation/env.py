"""Gymnasium interface and episode state for affine formation navigation."""

from dataclasses import asdict, dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from robot_rl.controllers.formation import AffineFormation, FollowerController, limit_speed
from robot_rl.maps.layouts import DEFAULT_OBSTACLE_MAP, OBSTACLE_COURSES, OPEN_COURSE
from robot_rl.sim.world import BulletWorld

from .config import DEFAULT_TASK, EnvironmentConfig
from .observations import ObservationEncoder
from .rewards import RewardInputs, reward_terms


@dataclass(frozen=True, slots=True)
class FormationMeasurements:
    goal_vector: np.ndarray
    goal_distance: float
    follower_errors: np.ndarray
    tracking_error: float
    shape_error: float
    triangle_error: float


class FormationPyBulletEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None, enable_obstacles=True, max_steps=DEFAULT_TASK.max_steps, enable_walls=True,
                 *, follower_control_mode="centralized", neighbor_velocity_source="measured",
                 obstacle_map=DEFAULT_OBSTACLE_MAP):
        super().__init__()
        if isinstance(render_mode, bool):
            render_mode = "human" if render_mode else None
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'.")
        if not isinstance(max_steps, (int, np.integer)) or isinstance(max_steps, bool):
            raise ValueError("max_steps must be a positive integer.")
        settings = EnvironmentConfig(
            stage=2 if enable_obstacles else 1, map=obstacle_map, walls=bool(enable_walls),
            follower_control=follower_control_mode, neighbor_velocity=neighbor_velocity_source, max_steps=int(max_steps),
        )
        settings.validate()
        self.task = DEFAULT_TASK
        self.render_mode = render_mode
        self.enable_obstacles = enable_obstacles
        self.obstacle_map = obstacle_map if enable_obstacles else None
        self.course = OBSTACLE_COURSES[obstacle_map] if enable_obstacles else OPEN_COURSE
        self.enable_walls = enable_walls
        self.max_steps = settings.max_steps
        self.follower_control_mode = follower_control_mode
        self.neighbor_velocity_source = neighbor_velocity_source

        self.formation = AffineFormation()
        self.n_leaders = self.formation.n_leaders
        self.num_robots = self.formation.n_agents
        self._follower_control = FollowerController(
            self.formation, follower_control_mode, neighbor_velocity_source, self.task.follower_speed,
        )
        self._control_stride = (self.task.distributed_substeps if follower_control_mode == "distributed"
                                else self.task.physics_substeps)
        self._pair_indices = np.triu_indices(self.num_robots, k=1)
        self._leader_pairs = np.triu_indices(self.n_leaders, k=1)
        self.world = BulletWorld(render_mode, self.task.physics_time_step)
        self.observations = ObservationEncoder(
            self.formation, self.world, self.course.half_extents, self.task.leader_speed,
        )
        self.action_space = spaces.Box(-1.0, 1.0, (2 * self.n_leaders,), dtype=np.float32)
        self.observation_space = self.observations.space
        self._needs_reset = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._needs_reset = True
        start, goal = self.course.sample_endpoints(self.np_random)
        start, self._goal = np.asarray(start), np.asarray(goal)
        direction = self._goal - start
        self._heading = np.arctan2(direction[1], direction[0])
        forward = direction / np.linalg.norm(direction)
        self._frame = np.array([forward, [-forward[1], forward[0]]])
        initial = self.formation.nominal @ self._frame + start

        obstacle_seed = int(self.np_random.integers(2**32)) if self.enable_obstacles else None
        self.world.reset_scene(self.course, self._goal, walls=self.enable_walls, obstacle_seed=obstacle_seed)
        self.world.spawn_robots(initial, self._heading, self.n_leaders)
        self._steps = 0
        self._previous_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._follower_control.reset()
        self.observations.reset()
        states = self.world.robot_states()
        center = states[0].mean(axis=0)
        self._goal_distance = float(np.linalg.norm(self._goal - center))
        self.world.focus(center)
        obs = self._observe(states)
        self._needs_reset = False
        return obs, {}

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset() before stepping the environment.")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError("Expected six finite action components in leader order.")
        action = np.clip(action, -1, 1)
        self._previous_action = action
        self._steps += 1

        velocities = limit_speed(action.reshape(self.n_leaders, 2) @ self._frame, self.task.leader_speed)
        contact = self._advance_control(velocities)
        states = self.world.robot_states()
        positions = states[0]
        formation = self._measure_formation(positions)
        collision = contact or self._geometry_violation(positions)
        success = (
            collision is None
            and formation.goal_distance < self.task.goal_tolerance
            and formation.shape_error <= self.task.success_shape_error
            and formation.tracking_error <= self.task.success_tracking_error
        )
        terms = self._reward_terms(states, formation, success, bool(collision))
        self._goal_distance = formation.goal_distance
        terminated = bool(success or collision)
        truncated = not terminated and self._steps >= self.max_steps
        self._needs_reset = terminated or truncated
        info = {
            "is_success": bool(success),
            "goal_distance": formation.goal_distance,
            "formation_error": formation.tracking_error,
            "shape_error": formation.shape_error,
            "reward_terms": asdict(terms),
            "reason": collision or ("Success" if success else "Timeout" if truncated else ""),
        }
        return self._observe(states), terms.total, terminated, truncated, info

    def _advance_control(self, leader_velocities):
        initial = self.world.robot_states()
        self.world.command_robots(0, leader_velocities, initial[2], self._heading)

        def update_followers(substep):
            positions, measured, yaws, _ = self.world.robot_states() if substep else initial
            commands = self._follower_control.velocities(positions, measured, leader_velocities)
            self.world.command_robots(self.n_leaders, commands, yaws, self._heading)

        return self.world.advance(self.task.physics_substeps, self._control_stride, update_followers)

    def _measure_formation(self, positions):
        leaders = positions[:self.n_leaders]
        desired = self.formation.desired_positions(leaders)
        distances = np.linalg.norm(positions[self.n_leaders:] - desired[self.n_leaders:], axis=1)
        goal_vector = self._goal - positions.mean(axis=0)
        offsets = (leaders - desired.mean(axis=0)) @ self._frame.T
        errors = offsets - self.formation.nominal[:self.n_leaders]
        i, j = self._leader_pairs
        return FormationMeasurements(
            goal_vector=goal_vector,
            goal_distance=float(np.linalg.norm(goal_vector)),
            follower_errors=distances,
            tracking_error=float(np.sqrt(np.mean(np.square(distances)))),
            shape_error=float(np.sqrt(np.mean(np.sum(np.square(errors), axis=1)))),
            triangle_error=float(np.sqrt(np.mean(np.sum(np.square(errors[i] - errors[j]), axis=1)))),
        )

    def _reward_terms(self, states, formation, success, collided):
        positions, velocities, _, _ = states
        leaders = velocities[:self.n_leaders]
        weights = self.task.reward
        empty = np.empty(0)
        predicted_clearances = surface_distances = empty
        if self.enable_obstacles:
            predicted_leaders = positions[:self.n_leaders] + weights.prediction_horizon * leaders
            predicted_followers = self.formation.follower_map @ predicted_leaders
            predicted_clearances = self.world.point_clearances(predicted_followers)
            if not collided:
                surface_distances = self.world.obstacle_surface_distances(weights.surface_warning_distance)
        return reward_terms(RewardInputs(
            previous_goal_distance=self._goal_distance,
            goal_vector=formation.goal_vector,
            leader_velocities=leaders,
            triangle_error=formation.triangle_error,
            follower_errors=formation.follower_errors if self.enable_obstacles else empty,
            predicted_clearances=predicted_clearances,
            obstacle_surface_distances=surface_distances,
            success=success,
            collided=collided,
        ), weights)

    def _geometry_violation(self, positions):
        i, j = self._pair_indices
        if np.any(np.linalg.norm(positions[i] - positions[j], axis=1) < self.task.minimum_team_distance):
            return "Crash_Team"
        if self.enable_walls and np.any(np.abs(positions) >= self.course.half_extents):
            return "Wall_Out"
        return None

    def _observe(self, states):
        self.world.draw_formation(states[0], self.formation.edges)
        return self.observations.encode(states, goal=self._goal, frame=self._frame, heading=self._heading,
                                        previous_action=self._previous_action)

    @property
    def physics_client_id(self):
        return self.world.client_id

    def close(self):
        self.world.close()
        self._needs_reset = True
