import heapq
import math

import numpy as np


class PathPlanningError(RuntimeError):
    pass


class ThetaStarPlanner:
    """Reusable 2-D Theta* planner over inflated axis-aligned obstacles."""

    _NEIGHBORS = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )

    def __init__(self, bounds, resolution=0.25, inflation_radius=0.45):
        self.bounds = tuple(float(value) for value in bounds)
        if len(self.bounds) != 4:
            raise ValueError("bounds must be (min_x, max_x, min_y, max_y).")
        self.min_x, self.max_x, self.min_y, self.max_y = self.bounds
        if self.min_x >= self.max_x or self.min_y >= self.max_y:
            raise ValueError("Planner bounds are invalid.")
        if resolution <= 0 or inflation_radius < 0:
            raise ValueError("resolution must be positive and inflation non-negative.")

        self.resolution = float(resolution)
        self.inflation_radius = float(inflation_radius)
        self.width = int(math.floor((self.max_x - self.min_x) / self.resolution)) + 1
        self.height = int(math.floor((self.max_y - self.min_y) / self.resolution)) + 1
        self.occupied = np.zeros((self.width, self.height), dtype=bool)

    def update_obstacles(self, obstacle_aabbs):
        """Rasterize `(min_x, min_y, max_x, max_y)` obstacle bounds."""
        self.occupied.fill(False)
        for obstacle in obstacle_aabbs:
            min_x, min_y, max_x, max_y = map(float, obstacle)
            min_index = self._world_to_grid((
                min_x - self.inflation_radius,
                min_y - self.inflation_radius,
            ), clip=True)
            max_index = self._world_to_grid((
                max_x + self.inflation_radius,
                max_y + self.inflation_radius,
            ), clip=True)
            self.occupied[
                min_index[0]:max_index[0] + 1,
                min_index[1]:max_index[1] + 1,
            ] = True

    def _world_to_grid(self, point, clip=False):
        point = np.asarray(point, dtype=np.float64)
        index = np.rint((point - (self.min_x, self.min_y)) / self.resolution).astype(int)
        if clip:
            index[0] = np.clip(index[0], 0, self.width - 1)
            index[1] = np.clip(index[1], 0, self.height - 1)
        return int(index[0]), int(index[1])

    def _grid_to_world(self, index):
        return np.array([
            self.min_x + index[0] * self.resolution,
            self.min_y + index[1] * self.resolution,
        ], dtype=np.float32)

    def _in_bounds(self, index):
        return 0 <= index[0] < self.width and 0 <= index[1] < self.height

    def _is_free(self, index):
        return self._in_bounds(index) and not self.occupied[index]

    def _line_of_sight(self, start, end):
        delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        samples = max(int(np.ceil(np.max(np.abs(delta)) * 2.0)), 1)
        for ratio in np.linspace(0.0, 1.0, samples + 1):
            index = tuple(np.rint(np.asarray(start) + ratio * delta).astype(int))
            if not self._is_free(index):
                return False
        return True

    @staticmethod
    def _distance(first, second):
        return math.hypot(first[0] - second[0], first[1] - second[1])

    def plan(self, start, goal):
        start_world = np.asarray(start, dtype=np.float32)
        goal_world = np.asarray(goal, dtype=np.float32)
        start_index = self._world_to_grid(start_world, clip=True)
        goal_index = self._world_to_grid(goal_world, clip=True)
        if not self._is_free(start_index):
            raise PathPlanningError("Start lies inside an inflated obstacle.")
        if not self._is_free(goal_index):
            raise PathPlanningError("Goal lies inside an inflated obstacle.")

        if self._line_of_sight(start_index, goal_index):
            return np.stack((start_world, goal_world))

        parents = {start_index: start_index}
        costs = {start_index: 0.0}
        open_heap = [(self._distance(start_index, goal_index), 0.0, start_index)]
        closed = set()

        while open_heap:
            _, current_cost, current = heapq.heappop(open_heap)
            if current in closed or current_cost > costs.get(current, math.inf):
                continue
            if current == goal_index:
                break
            closed.add(current)

            for dx, dy in self._NEIGHBORS:
                neighbor = current[0] + dx, current[1] + dy
                if not self._is_free(neighbor) or neighbor in closed:
                    continue

                parent = parents[current]
                if self._line_of_sight(parent, neighbor):
                    candidate_parent = parent
                    candidate_cost = costs[parent] + self._distance(parent, neighbor)
                else:
                    candidate_parent = current
                    candidate_cost = costs[current] + self._distance(current, neighbor)

                if candidate_cost + 1e-12 < costs.get(neighbor, math.inf):
                    costs[neighbor] = candidate_cost
                    parents[neighbor] = candidate_parent
                    estimate = candidate_cost + self._distance(neighbor, goal_index)
                    heapq.heappush(open_heap, (estimate, candidate_cost, neighbor))

        if goal_index not in parents:
            raise PathPlanningError(
                f"No path found from {start_world.tolist()} to {goal_world.tolist()}."
            )

        indices = [goal_index]
        while indices[-1] != start_index:
            indices.append(parents[indices[-1]])
        indices.reverse()

        points = [start_world]
        points.extend(self._grid_to_world(index) for index in indices[1:-1])
        points.append(goal_world)
        return np.asarray(points, dtype=np.float32)
