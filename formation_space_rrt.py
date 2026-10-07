"""RRT feasibility pilot in a four-dimensional formation space.

The planner state is not an individual robot position.  It is

    z = [s_x, s_y, tau_x, tau_y],

where ``s_x`` and ``s_y`` non-uniformly scale a selectable nominal formation
and ``tau_x``, ``tau_y`` translate it.  Attitudes are not used in this pilot.
Every candidate RRT edge is interpolated in formation space and every
intermediate formation is collision checked after mapping it back to all
robot positions.

Run from the repository root:

    python formation_space_rrt.py

Outputs are written to ``outputs/formation_space_rrt`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from project_paths import PROJECT_ROOT
from utils.formation_utils import (
    get_five_agent_configuration,
    get_five_agent_topology,
    get_nominal_configuration,
    get_square_configuration,
    get_square_topology,
    get_topology,
)


Z_LABELS = ("s_x", "s_y", "tau_x", "tau_y")
FORMATION_CONFIG_NAMES = ("square4", "five5", "paper9")
FORMATION_CONFIG_ALIASES = {
    "4": "square4",
    "4-agent": "square4",
    "square": "square4",
    "5": "five5",
    "5-agent": "five5",
    "five": "five5",
    "9": "paper9",
    "9-agent": "paper9",
    "nine": "paper9",
    "paper": "paper9",
}


@dataclass(frozen=True)
class AxisAlignedObstacle:
    """Closed axis-aligned rectangular obstacle in the world plane."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def __post_init__(self) -> None:
        if self.min_x >= self.max_x or self.min_y >= self.max_y:
            raise ValueError(f"Invalid obstacle bounds: {self}")

    def distance_to_point(self, point: np.ndarray) -> float:
        """Euclidean distance from a point to this rectangle (zero inside)."""

        dx = max(self.min_x - point[0], 0.0, point[0] - self.max_x)
        dy = max(self.min_y - point[1], 0.0, point[1] - self.max_y)
        return float(math.hypot(dx, dy))


@dataclass(frozen=True)
class CircularObstacle:
    """Circular footprint matching a PyBullet cylindrical obstacle."""

    center_x: float
    center_y: float
    radius: float = 0.5

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"Obstacle radius must be positive: {self}")

    @property
    def center(self) -> np.ndarray:
        return np.array([self.center_x, self.center_y], dtype=float)

    def distance_to_point(self, point: np.ndarray) -> float:
        return max(float(np.linalg.norm(point - self.center)) - self.radius, 0.0)


Obstacle = AxisAlignedObstacle | CircularObstacle


@dataclass
class RRTNode:
    z: np.ndarray
    parent: int | None


@dataclass(frozen=True)
class PlanningScenario:
    config_id: int
    config_name: str
    obstacle_seed: int
    start: np.ndarray
    goal: np.ndarray
    z_bounds: np.ndarray
    workspace_bounds: tuple[float, float, float, float]
    obstacles: tuple[Obstacle, ...]
    robot_radius: float
    minimum_agent_gap: float


class FormationModel:
    """Map four maneuver parameters to all configured agent positions."""

    def __init__(self, nominal_positions: np.ndarray, theta: float = 0.0):
        nominal_positions = np.asarray(nominal_positions, dtype=float)
        if nominal_positions.ndim != 2 or nominal_positions.shape[1] != 2:
            raise ValueError("nominal_positions must have shape (n_agents, 2).")
        if not np.all(np.isfinite(nominal_positions)):
            raise ValueError("nominal_positions must be finite.")
        self.nominal_positions = nominal_positions.copy()
        cosine, sine = math.cos(theta), math.sin(theta)
        self.rotation = np.array([[cosine, -sine], [sine, cosine]])

    @property
    def n_agents(self) -> int:
        return len(self.nominal_positions)

    def positions(self, z: Sequence[float]) -> np.ndarray:
        """Return all world positions for ``z=[s_x,s_y,tau_x,tau_y]``."""

        z = np.asarray(z, dtype=float)
        if z.shape != (4,) or not np.all(np.isfinite(z)):
            raise ValueError("z must be a finite vector [s_x, s_y, tau_x, tau_y].")
        scale = self.rotation @ np.diag(z[:2]) @ self.rotation.T
        return self.nominal_positions @ scale.T + z[2:]


class FormationCollisionChecker:
    """Collision checks performed after formation-to-agent position mapping."""

    def __init__(
        self,
        model: FormationModel,
        scenario: PlanningScenario,
        edge_resolution: float,
    ):
        if edge_resolution <= 0:
            raise ValueError("edge_resolution must be positive.")
        self.model = model
        self.scenario = scenario
        self.edge_resolution = float(edge_resolution)
        self.checked_states = 0
        self.continuous_edge_checks = 0

    @staticmethod
    def _segment_intersects_aabb(
        start: np.ndarray,
        end: np.ndarray,
        bounds: tuple[float, float, float, float],
    ) -> bool:
        """Return whether a closed line segment intersects a closed AABB."""

        min_x, max_x, min_y, max_y = bounds
        lower = np.array([min_x, min_y], dtype=float)
        upper = np.array([max_x, max_y], dtype=float)
        direction = end - start
        interval_start = 0.0
        interval_end = 1.0
        for axis in range(2):
            if abs(direction[axis]) < 1e-12:
                if start[axis] < lower[axis] or start[axis] > upper[axis]:
                    return False
                continue
            first = (lower[axis] - start[axis]) / direction[axis]
            second = (upper[axis] - start[axis]) / direction[axis]
            entry, exit_ = sorted((first, second))
            interval_start = max(interval_start, entry)
            interval_end = min(interval_end, exit_)
            if interval_start > interval_end:
                return False
        return True

    def _continuous_edge_is_free(
        self, start: np.ndarray, end: np.ndarray
    ) -> bool:
        """Conservative continuous sweep for the affine mapped agent paths."""

        self.continuous_edge_checks += 1
        start_positions = self.model.positions(start)
        end_positions = self.model.positions(end)
        radius = self.scenario.robot_radius

        # Each mapped agent follows a line segment because positions are affine in z.
        for start_position, end_position in zip(start_positions, end_positions):
            for obstacle in self.scenario.obstacles:
                if isinstance(obstacle, AxisAlignedObstacle):
                    # Radius-inflated AABB is conservative around rectangle corners.
                    inflated = (
                        obstacle.min_x - radius,
                        obstacle.max_x + radius,
                        obstacle.min_y - radius,
                        obstacle.max_y + radius,
                    )
                    if self._segment_intersects_aabb(
                        start_position, end_position, inflated
                    ):
                        return False
                else:
                    segment = end_position - start_position
                    denominator = float(segment @ segment)
                    if denominator <= 1e-15:
                        ratio = 0.0
                    else:
                        ratio = float(
                            np.clip(
                                ((obstacle.center - start_position) @ segment)
                                / denominator,
                                0.0,
                                1.0,
                            )
                        )
                    closest = start_position + ratio * segment
                    if (
                        np.linalg.norm(closest - obstacle.center)
                        <= obstacle.radius + radius
                    ):
                        return False

        # Relative position is also affine along an edge, so its minimum norm is
        # obtained exactly by projection onto the relative-motion segment.
        minimum_center_distance = 2.0 * radius + self.scenario.minimum_agent_gap
        for first in range(self.model.n_agents):
            for second in range(first + 1, self.model.n_agents):
                relative_start = start_positions[first] - start_positions[second]
                relative_end = end_positions[first] - end_positions[second]
                relative_delta = relative_end - relative_start
                denominator = float(relative_delta @ relative_delta)
                if denominator <= 1e-15:
                    ratio = 0.0
                else:
                    ratio = float(
                        np.clip(
                            -(relative_start @ relative_delta) / denominator,
                            0.0,
                            1.0,
                        )
                    )
                minimum_relative = relative_start + ratio * relative_delta
                if np.linalg.norm(minimum_relative) <= minimum_center_distance:
                    return False
        return True

    def state_is_free(self, z: Sequence[float]) -> bool:
        """Check workspace, robot-obstacle, and robot-robot clearances."""

        self.checked_states += 1
        z = np.asarray(z, dtype=float)
        if z.shape != (4,) or np.any(z < self.scenario.z_bounds[:, 0]) or np.any(
            z > self.scenario.z_bounds[:, 1]
        ):
            return False

        positions = self.model.positions(z)
        min_x, max_x, min_y, max_y = self.scenario.workspace_bounds
        radius = self.scenario.robot_radius
        if np.any(positions[:, 0] < min_x + radius) or np.any(
            positions[:, 0] > max_x - radius
        ):
            return False
        if np.any(positions[:, 1] < min_y + radius) or np.any(
            positions[:, 1] > max_y - radius
        ):
            return False

        for position in positions:
            for obstacle in self.scenario.obstacles:
                if obstacle.distance_to_point(position) <= radius:
                    return False

        minimum_center_distance = 2.0 * radius + self.scenario.minimum_agent_gap
        for first in range(self.model.n_agents):
            for second in range(first + 1, self.model.n_agents):
                if (
                    np.linalg.norm(positions[first] - positions[second])
                    <= minimum_center_distance
                ):
                    return False
        return True

    def interpolation_count(self, start: np.ndarray, end: np.ndarray) -> int:
        """Choose samples so no mapped agent moves more than edge_resolution."""

        start_positions = self.model.positions(start)
        end_positions = self.model.positions(end)
        max_displacement = float(
            np.max(np.linalg.norm(end_positions - start_positions, axis=1))
        )
        return max(1, int(math.ceil(max_displacement / self.edge_resolution)))

    def edge_is_free(self, start: Sequence[float], end: Sequence[float]) -> bool:
        """Interpolate a complete RRT edge and check every formation state."""

        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        if not self._continuous_edge_is_free(start, end):
            return False
        segments = self.interpolation_count(start, end)
        for ratio in np.linspace(0.0, 1.0, segments + 1):
            if not self.state_is_free(start + ratio * (end - start)):
                return False
        return True

    def densify_path(self, path: Sequence[np.ndarray]) -> np.ndarray:
        """Return the same interpolation states used for edge validation."""

        dense = [np.asarray(path[0], dtype=float)]
        for start, end in zip(path[:-1], path[1:]):
            start = np.asarray(start, dtype=float)
            end = np.asarray(end, dtype=float)
            segments = self.interpolation_count(start, end)
            for ratio in np.linspace(0.0, 1.0, segments + 1)[1:]:
                dense.append(start + ratio * (end - start))
        return np.asarray(dense)

    def path_is_free(self, path: Sequence[np.ndarray]) -> bool:
        return all(
            self.edge_is_free(start, end)
            for start, end in zip(path[:-1], path[1:])
        )

    def path_clearances(self, dense_path: np.ndarray) -> tuple[float, float]:
        """Return minimum obstacle/workspace and inter-agent surface clearance."""

        obstacle_clearance = math.inf
        interagent_clearance = math.inf
        radius = self.scenario.robot_radius
        min_x, max_x, min_y, max_y = self.scenario.workspace_bounds

        for z in dense_path:
            positions = self.model.positions(z)
            for position in positions:
                boundary_clearance = min(
                    position[0] - min_x,
                    max_x - position[0],
                    position[1] - min_y,
                    max_y - position[1],
                ) - radius
                obstacle_clearance = min(obstacle_clearance, boundary_clearance)
                for obstacle in self.scenario.obstacles:
                    obstacle_clearance = min(
                        obstacle_clearance,
                        obstacle.distance_to_point(position) - radius,
                    )
            for first in range(self.model.n_agents):
                for second in range(first + 1, self.model.n_agents):
                    surface_distance = (
                        np.linalg.norm(positions[first] - positions[second])
                        - 2.0 * radius
                    )
                    interagent_clearance = min(
                        interagent_clearance, float(surface_distance)
                    )
        return float(obstacle_clearance), float(interagent_clearance)


class FormationSpaceRRT:
    """Basic single-tree RRT operating on four-dimensional formation states."""

    def __init__(
        self,
        scenario: PlanningScenario,
        checker: FormationCollisionChecker,
        rng: np.random.Generator,
        step_size: float = 0.8,
        goal_bias: float = 0.12,
        max_iterations: int = 20_000,
        metric_weights: Sequence[float] = (3.0, 3.0, 1.0, 1.0),
    ):
        if step_size <= 0 or max_iterations < 1:
            raise ValueError("step_size and max_iterations must be positive.")
        if not 0.0 <= goal_bias <= 1.0:
            raise ValueError("goal_bias must lie in [0, 1].")
        self.scenario = scenario
        self.checker = checker
        self.rng = rng
        self.step_size = float(step_size)
        self.goal_bias = float(goal_bias)
        self.max_iterations = int(max_iterations)
        self.metric_weights = np.asarray(metric_weights, dtype=float)
        if self.metric_weights.shape != (4,) or np.any(self.metric_weights <= 0):
            raise ValueError("metric_weights must contain four positive values.")
        self.nodes: list[RRTNode] = []

    def _distance(self, first: np.ndarray, second: np.ndarray) -> float:
        return float(np.linalg.norm((first - second) * self.metric_weights))

    def _sample(self) -> np.ndarray:
        if self.rng.random() < self.goal_bias:
            return self.scenario.goal.copy()
        return self.rng.uniform(
            self.scenario.z_bounds[:, 0], self.scenario.z_bounds[:, 1]
        )

    def _nearest_index(self, sample: np.ndarray) -> int:
        states = np.stack([node.z for node in self.nodes])
        distances = np.linalg.norm(
            (states - sample) * self.metric_weights,
            axis=1,
        )
        return int(np.argmin(distances))

    def _steer(self, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        weighted_delta = (target - start) * self.metric_weights
        distance = float(np.linalg.norm(weighted_delta))
        if distance <= self.step_size:
            candidate = target.copy()
        else:
            candidate = start + (
                weighted_delta / distance * self.step_size / self.metric_weights
            )
        return np.clip(
            candidate,
            self.scenario.z_bounds[:, 0],
            self.scenario.z_bounds[:, 1],
        )

    def _extract_path(self, goal_index: int) -> list[np.ndarray]:
        reversed_path: list[np.ndarray] = []
        index: int | None = goal_index
        while index is not None:
            reversed_path.append(self.nodes[index].z.copy())
            index = self.nodes[index].parent
        reversed_path.reverse()
        return reversed_path

    def plan(self) -> tuple[list[np.ndarray], int]:
        if not self.checker.state_is_free(self.scenario.start):
            raise RuntimeError("The configured start formation is in collision.")
        if not self.checker.state_is_free(self.scenario.goal):
            raise RuntimeError("The configured goal formation is in collision.")

        self.nodes = [RRTNode(self.scenario.start.copy(), None)]
        for iteration in range(1, self.max_iterations + 1):
            sample = self._sample()
            nearest_index = self._nearest_index(sample)
            nearest = self.nodes[nearest_index].z
            candidate = self._steer(nearest, sample)
            if np.allclose(candidate, nearest) or not self.checker.edge_is_free(
                nearest, candidate
            ):
                continue

            self.nodes.append(RRTNode(candidate, nearest_index))
            new_index = len(self.nodes) - 1
            if self._distance(candidate, self.scenario.goal) <= 1.5 * self.step_size:
                if self.checker.edge_is_free(candidate, self.scenario.goal):
                    self.nodes.append(RRTNode(self.scenario.goal.copy(), new_index))
                    return self._extract_path(len(self.nodes) - 1), iteration

        raise RuntimeError(
            f"RRT did not reach the goal after {self.max_iterations} iterations "
            f"and {len(self.nodes)} accepted nodes."
        )


def shortcut_path(
    path: Sequence[np.ndarray], checker: FormationCollisionChecker
) -> list[np.ndarray]:
    """Greedily remove intermediate waypoints without weakening edge checks."""

    shortened = [np.asarray(path[0], dtype=float)]
    current = 0
    while current < len(path) - 1:
        next_index = current + 1
        for candidate in range(len(path) - 1, current, -1):
            if checker.edge_is_free(path[current], path[candidate]):
                next_index = candidate
                break
        shortened.append(np.asarray(path[next_index], dtype=float))
        current = next_index
    return shortened


def _poisson_circles(
    *,
    count: int,
    minimum_spacing: float,
    rng: np.random.Generator,
    x_bounds: tuple[float, float] = (-20.0 / 3.0, 20.0 / 3.0),
    y_bounds: tuple[float, float] = (-9.0, 9.0),
) -> list[CircularObstacle]:
    """Reproduce the environment's seeded rejection-sampled forest layouts."""

    points: list[np.ndarray] = []
    for _ in range(100_000):
        if len(points) == count:
            break
        candidate = rng.uniform(
            [x_bounds[0], y_bounds[0]],
            [x_bounds[1], y_bounds[1]],
        )
        if all(np.linalg.norm(candidate - point) >= minimum_spacing for point in points):
            points.append(candidate)
    if len(points) != count:
        raise RuntimeError(
            f"Could not place {count} forest obstacles with spacing {minimum_spacing}."
        )
    return [CircularObstacle(point[0], point[1], 0.5) for point in points]


def environment_obstacles(config_id: int, obstacle_seed: int) -> tuple[Obstacle, ...]:
    """Return one of the five obstacle configurations from ``MDP_env.py``."""

    if config_id not in range(5):
        raise ValueError("config_id must be one of 0, 1, 2, 3, 4.")

    if config_id == 0:
        positions = [
            (x, y)
            for x in (-4.0, -1.33333333, 1.33333333, 4.0)
            for y in (0.0, 5.0, -5.0)
        ]
        return tuple(CircularObstacle(x, y, 0.5) for x, y in positions)

    if config_id == 1:
        doors_y = (-5.0, 0.0, 5.0)
        gap_half_width = 0.8
        wall_half_width = 0.5
        current_y = -10.0 + 0.5
        rectangles: list[AxisAlignedObstacle] = []
        for door_y in doors_y:
            segment_end_y = door_y - gap_half_width
            if segment_end_y > current_y:
                rectangles.append(
                    AxisAlignedObstacle(
                        -wall_half_width,
                        wall_half_width,
                        current_y,
                        segment_end_y,
                    )
                )
            current_y = door_y + gap_half_width
        final_y = 10.0 - 0.5
        if final_y > current_y:
            rectangles.append(
                AxisAlignedObstacle(
                    -wall_half_width,
                    wall_half_width,
                    current_y,
                    final_y,
                )
            )
        return tuple(rectangles)

    if config_id == 2:
        positions = []
        for ratio in np.linspace(0.0, 1.0, 10):
            x = -5.0 + ratio * 8.0
            half_gap = 8.0 + ratio * (1.2 - 8.0)
            positions.extend(((x, half_gap), (x, -half_gap)))
        return tuple(CircularObstacle(x, y, 0.5) for x, y in positions)

    rng = np.random.default_rng(obstacle_seed)
    if config_id == 3:
        return tuple(
            _poisson_circles(
                count=6,
                minimum_spacing=4.5,
                rng=rng,
            )
        )
    return tuple(
        _poisson_circles(
            count=10,
            minimum_spacing=4.0,
            rng=rng,
        )
    )


CONFIG_NAMES = {
    0: "uniform_pillar_rows",
    1: "three_door_wall",
    2: "funnel",
    3: "sparse_forest",
    4: "dense_forest",
}


def parse_formation_config(value: str) -> str:
    """Normalize a formation name or a convenient agent-count alias."""

    normalized = value.strip().lower()
    normalized = FORMATION_CONFIG_ALIASES.get(normalized, normalized)
    if normalized not in FORMATION_CONFIG_NAMES:
        valid = ", ".join(FORMATION_CONFIG_NAMES)
        raise argparse.ArgumentTypeError(
            f"unknown formation configuration {value!r}; choose one of {valid} "
            "(agent-count aliases 4, 5, and 9 are also accepted)"
        )
    return normalized


def load_formation_config(
    config_name: str,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Load one repository formation and its visualization topology."""

    loaders = {
        "square4": (get_square_configuration, get_square_topology),
        "five5": (get_five_agent_configuration, get_five_agent_topology),
        "paper9": (get_nominal_configuration, get_topology),
    }
    if config_name not in loaders:
        raise ValueError(
            f"formation config must be one of {FORMATION_CONFIG_NAMES}, "
            f"got {config_name!r}"
        )
    configuration_loader, topology_loader = loaders[config_name]
    configuration = np.asarray(configuration_loader(), dtype=float)
    topology = tuple(tuple(map(int, edge)) for edge in topology_loader())
    if (
        configuration.ndim != 2
        or configuration.shape[1] < 2
        or len(configuration) < 2
        or not np.all(np.isfinite(configuration[:, :2]))
    ):
        raise ValueError(
            f"Invalid formation {config_name!r}: expected finite shape "
            "(n_agents, >=2)."
        )
    for first, second in topology:
        if not (0 <= first < len(configuration) and 0 <= second < len(configuration)):
            raise ValueError(
                f"Topology edge {(first, second)} is outside formation "
                f"{config_name!r} with {len(configuration)} agents."
            )
    return configuration, topology


def default_scenario(config_id: int = 0, obstacle_seed: int = 0) -> PlanningScenario:
    """Build a formation-space scenario using an environment obstacle map."""

    return PlanningScenario(
        config_id=config_id,
        config_name=CONFIG_NAMES[config_id],
        obstacle_seed=obstacle_seed,
        start=np.array([1.0, 1.0, -14.0, 0.0]),
        goal=np.array([1.0, 1.0, 15.5, 0.0]),
        z_bounds=np.array(
            [
                [0.12, 1.15],
                [0.12, 1.15],
                [-14.0, 15.5],
                [-6.0, 6.0],
            ],
            dtype=float,
        ),
        workspace_bounds=(-20.0, 20.0, -10.0, 10.0),
        obstacles=environment_obstacles(config_id, obstacle_seed),
        robot_radius=0.28,
        minimum_agent_gap=0.05,
    )


def save_parameter_path(path: Sequence[np.ndarray], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("waypoint", *Z_LABELS))
        for index, z in enumerate(path):
            writer.writerow((index, *[f"{value:.10f}" for value in z]))


def save_agent_trajectory(
    dense_path: np.ndarray,
    model: FormationModel,
    output_path: Path,
) -> None:
    header = ["sample", *Z_LABELS]
    for agent in range(model.n_agents):
        header.extend((f"agent_{agent}_x", f"agent_{agent}_y"))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, z in enumerate(dense_path):
            positions = model.positions(z).reshape(-1)
            writer.writerow(
                (
                    index,
                    *[f"{value:.10f}" for value in z],
                    *[f"{value:.10f}" for value in positions],
                )
            )


def draw_result(
    dense_path: np.ndarray,
    model: FormationModel,
    scenario: PlanningScenario,
    topology: Iterable[tuple[int, int]],
    metrics: dict[str, object],
    output_path: Path,
) -> None:
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, model.n_agents))
    figure, (world_axis, parameter_axis) = plt.subplots(
        1, 2, figsize=(13.0, 5.6), constrained_layout=True
    )

    for obstacle in scenario.obstacles:
        if isinstance(obstacle, AxisAlignedObstacle):
            patch = Rectangle(
                (obstacle.min_x, obstacle.min_y),
                obstacle.max_x - obstacle.min_x,
                obstacle.max_y - obstacle.min_y,
                facecolor="#777777",
                edgecolor="#333333",
                alpha=0.85,
                label="Obstacle",
            )
        else:
            patch = Circle(
                obstacle.center,
                obstacle.radius,
                facecolor="#777777",
                edgecolor="#333333",
                alpha=0.85,
                label="Obstacle",
            )
        world_axis.add_patch(patch)

    all_positions = np.stack([model.positions(z) for z in dense_path])
    for agent in range(model.n_agents):
        world_axis.plot(
            all_positions[:, agent, 0],
            all_positions[:, agent, 1],
            color=colors[agent],
            linewidth=2.0,
            label=f"Agent {agent}",
        )

    snapshot_indices = np.unique(
        np.linspace(0, len(dense_path) - 1, 8, dtype=int)
    )
    for snapshot_index in snapshot_indices:
        positions = all_positions[snapshot_index]
        for first, second in topology:
            world_axis.plot(
                positions[[first, second], 0],
                positions[[first, second], 1],
                color="#7C7C7C",
                linewidth=0.8,
                alpha=0.35,
            )
        for agent, position in enumerate(positions):
            circle = plt.Circle(
                position,
                scenario.robot_radius,
                facecolor=colors[agent],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.30,
            )
            world_axis.add_patch(circle)

    min_x, max_x, min_y, max_y = scenario.workspace_bounds
    world_axis.set_xlim(min_x, max_x)
    world_axis.set_ylim(min_y, max_y)
    world_axis.set_aspect("equal", adjustable="box")
    world_axis.set_xlabel("world x [m]")
    world_axis.set_ylabel("world y [m]")
    world_axis.set_title(
        f"{metrics['formation_config_name']} ({model.n_agents} agents), "
        f"map {scenario.config_id} ({scenario.config_name})"
    )
    world_axis.grid(color="#DDDDDD", linewidth=0.7)
    handles, labels = world_axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    world_axis.legend(unique.values(), unique.keys(), loc="upper left", fontsize=8)

    progress = np.linspace(0.0, 1.0, len(dense_path))
    for dimension, label in enumerate(Z_LABELS):
        parameter_axis.plot(
            progress,
            dense_path[:, dimension],
            linewidth=2.0,
            label=label,
        )
    parameter_axis.set_xlabel("normalized path progress")
    parameter_axis.set_ylabel("formation parameter value")
    parameter_axis.set_title("Collision-free trajectory in formation space")
    parameter_axis.grid(color="#DDDDDD", linewidth=0.7)
    parameter_axis.legend()

    figure.suptitle(
        "Formation-space RRT feasibility pilot\n"
        f"seed={metrics['seed']}, nodes={metrics['accepted_nodes']}, "
        f"min obstacle clearance={metrics['min_obstacle_clearance']:.3f} m"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan a collision-free path in 4-D formation space with RRT."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--formation-config",
        type=parse_formation_config,
        choices=FORMATION_CONFIG_NAMES,
        default="square4",
        metavar="{square4,five5,paper9}",
        help=(
            "Nominal agent formation. Agent-count aliases 4, 5, and 9 are "
            "accepted; paper9 is the nine-agent configuration from the paper."
        ),
    )
    parser.add_argument(
        "--obstacle-config",
        type=int,
        choices=range(5),
        default=0,
        metavar="{0,1,2,3,4}",
        help="Obstacle configuration id copied from MDP_env.py.",
    )
    parser.add_argument(
        "--obstacle-seed",
        type=int,
        default=0,
        help="Seed used only for sparse/dense forest placement.",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Run configurations 0 through 4 and write one folder per map.",
    )
    parser.add_argument("--max-iterations", type=int, default=20_000)
    parser.add_argument("--step-size", type=float, default=0.8)
    parser.add_argument("--goal-bias", type=float, default=0.12)
    parser.add_argument(
        "--edge-resolution",
        type=float,
        default=0.10,
        help="Maximum mapped displacement [m] of any agent between edge samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "formation_space_rrt",
    )
    parser.add_argument(
        "--no-shortcut",
        action="store_true",
        help="Keep the raw RRT waypoint chain instead of collision-checked shortcutting.",
    )
    return parser.parse_args()


def _serialize_obstacle(obstacle: Obstacle) -> dict[str, object]:
    if isinstance(obstacle, AxisAlignedObstacle):
        return {
            "type": "rectangle",
            "bounds": [
                obstacle.min_x,
                obstacle.max_x,
                obstacle.min_y,
                obstacle.max_y,
            ],
        }
    return {
        "type": "circle",
        "center": [obstacle.center_x, obstacle.center_y],
        "radius": obstacle.radius,
    }


def run_scenario(
    args: argparse.Namespace,
    config_id: int,
    output_dir: Path,
) -> dict[str, object]:
    scenario = default_scenario(config_id, args.obstacle_seed)
    configured, topology = load_formation_config(args.formation_config)
    model = FormationModel(configured[:, :2], theta=0.0)
    checker = FormationCollisionChecker(model, scenario, args.edge_resolution)
    planner = FormationSpaceRRT(
        scenario=scenario,
        checker=checker,
        rng=np.random.default_rng(args.seed + config_id),
        step_size=args.step_size,
        goal_bias=args.goal_bias,
        max_iterations=args.max_iterations,
    )

    direct_start_goal_free = checker.edge_is_free(scenario.start, scenario.goal)
    raw_path, iterations = planner.plan()
    final_path = raw_path if args.no_shortcut else shortcut_path(raw_path, checker)
    if not checker.path_is_free(final_path):
        raise RuntimeError("Internal validation failed: final path is not collision-free.")

    dense_path = checker.densify_path(final_path)
    obstacle_clearance, interagent_clearance = checker.path_clearances(dense_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, object] = {
        "seed": int(args.seed + config_id),
        "base_seed": int(args.seed),
        "obstacle_config_id": int(config_id),
        "obstacle_config_name": scenario.config_name,
        "obstacle_seed": int(args.obstacle_seed),
        "formation_config_name": args.formation_config,
        "n_agents": int(model.n_agents),
        "formation_topology": [list(edge) for edge in topology],
        "status": "success",
        "iterations": int(iterations),
        "accepted_nodes": int(len(planner.nodes)),
        "raw_waypoints": int(len(raw_path)),
        "shortcut_waypoints": int(len(final_path)),
        "dense_collision_checked_states": int(len(dense_path)),
        "total_collision_checks": int(checker.checked_states),
        "continuous_edge_checks": int(checker.continuous_edge_checks),
        "direct_start_goal_edge_free": bool(direct_start_goal_free),
        "continuous_sweep_model": "disc agents against exact circles and radius-inflated AABBs",
        "edge_resolution_m": float(args.edge_resolution),
        "robot_radius_m": float(scenario.robot_radius),
        "minimum_agent_gap_m": float(scenario.minimum_agent_gap),
        "min_obstacle_clearance": float(obstacle_clearance),
        "min_interagent_clearance": float(interagent_clearance),
        "min_s_x": float(np.min(dense_path[:, 0])),
        "max_s_x": float(np.max(dense_path[:, 0])),
        "min_s_y": float(np.min(dense_path[:, 1])),
        "max_s_y": float(np.max(dense_path[:, 1])),
        "tau_x_span": float(np.ptp(dense_path[:, 2])),
        "tau_y_span": float(np.ptp(dense_path[:, 3])),
        "start_z": scenario.start.tolist(),
        "goal_z": scenario.goal.tolist(),
        "nominal_positions": model.nominal_positions.tolist(),
        "z_bounds": scenario.z_bounds.tolist(),
        "workspace_bounds": list(scenario.workspace_bounds),
        "obstacles": [_serialize_obstacle(obstacle) for obstacle in scenario.obstacles],
    }

    save_parameter_path(final_path, output_dir / "formation_parameter_path.csv")
    save_agent_trajectory(dense_path, model, output_dir / "agent_trajectories.csv")
    np.savez_compressed(
        output_dir / "trajectory.npz",
        z_waypoints=np.asarray(final_path),
        z_dense=dense_path,
        agent_positions=np.stack([model.positions(z) for z in dense_path]),
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    draw_result(
        dense_path=dense_path,
        model=model,
        scenario=scenario,
        topology=topology,
        metrics=metrics,
        output_path=output_dir / "formation_space_rrt.png",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Artifacts written to: {output_dir}")
    return metrics


def main() -> int:
    args = parse_args()
    output_root = args.output_dir.resolve() / args.formation_config
    if args.all_configs:
        summaries: list[dict[str, object]] = []
        configs_succeeded = 0
        n_agents = len(load_formation_config(args.formation_config)[0])
        for config_id in range(5):
            config_output = output_root / f"config_{config_id}"
            try:
                result = run_scenario(
                    args,
                    config_id,
                    config_output,
                )
                configs_succeeded += 1
            except RuntimeError as error:
                result = {
                    "status": "failure",
                    "error": str(error),
                    "seed": int(args.seed + config_id),
                    "base_seed": int(args.seed),
                    "obstacle_config_id": int(config_id),
                    "obstacle_config_name": CONFIG_NAMES[config_id],
                    "obstacle_seed": int(args.obstacle_seed),
                    "formation_config_name": args.formation_config,
                    "n_agents": int(n_agents),
                    "max_iterations": int(args.max_iterations),
                }
                config_output.mkdir(parents=True, exist_ok=True)
                with (config_output / "metrics.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(result, handle, ensure_ascii=False, indent=2)
                print(json.dumps(result, ensure_ascii=False, indent=2))
            summaries.append(result)
        if configs_succeeded == 5:
            aggregate_status = "success"
        elif configs_succeeded:
            aggregate_status = "partial"
        else:
            aggregate_status = "failure"
        aggregate = {
            "status": aggregate_status,
            "configs_requested": 5,
            "configs_succeeded": configs_succeeded,
            "configs_failed": 5 - configs_succeeded,
            "base_seed": int(args.seed),
            "obstacle_seed": int(args.obstacle_seed),
            "formation_config_name": args.formation_config,
            "n_agents": int(n_agents),
            "results": summaries,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        with (output_root / "all_configs_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(aggregate, handle, ensure_ascii=False, indent=2)
        print(f"All-config summary written to: {output_root}")
        if configs_succeeded != 5:
            return 1
    else:
        run_scenario(
            args,
            args.obstacle_config,
            output_root / f"config_{args.obstacle_config}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
