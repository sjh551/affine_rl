"""Obstacle geometry and course layouts from two public ZJU FAST Lab demos.

Car-like-Robotic-swarm @ eabbf7c, map_generator_easy.py and swarm.launch.
Swarm-Formation @ 967a4bd, normal_hexagon.launch and random_forest_sensing.cpp.

All positions and lengths are in metres.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import floor
from random import Random


@dataclass(frozen=True)
class Box:
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class Cylinder:
    center: tuple[float, float, float]
    radius: float
    height: float


def car_like_obstacles() -> tuple[Box | Cylinder, ...]:
    """The 23 solid primitives behind the fixed 90 × 90 m demo point cloud."""
    cylinder_xy = (
        (-20, 30), (2, 25), (23, 28), (-16, 10), (20, 4), (-20, -20),
        (4, -25), (25, -10),
    )
    cube_xy = (
        (-40, 30), (10, 32), (40, 28), (-32, 10), (8, 8), (30, 10),
        (-35, -10), (-18, -3), (38, -4), (-20, -30), (20, -22),
        (-8, 18), (-8, -18), (10, -10),
    )
    # The point-cloud generator offsets each declared center by -0.5 m in z.
    pillars = tuple(Cylinder((x, y, 1.0), 1.5, 2.5) for x, y in cylinder_xy)
    cubes = tuple(Box((x, y, 1.0), (3.0, 3.0, 3.0)) for x, y in cube_xy)
    return (*pillars, *cubes, Box((0.0, -3.0, 1.0), (4.0, 6.0, 3.0)))


def swarm_formation_forest(seed: int = 1) -> tuple[Cylinder, ...]:
    """Reproducible ground adaptation of the 34 × 15 m hexagon-demo forest."""
    rng = Random(seed)
    resolution = 0.1
    count = 60
    minimum_spacing = 0.8
    obstacles = []
    for _ in range(10_000):
        x = (floor(rng.uniform(-17.0, 17.0) / resolution) + 0.5) * resolution
        y = (floor(rng.uniform(-7.5, 7.5) / resolution) + 0.5) * resolution
        # Enforce spacing on the final grid-aligned centers.
        if any((x - item.center[0]) ** 2 + (y - item.center[1]) ** 2
               < minimum_spacing ** 2 - 1e-12 for item in obstacles):
            continue

        # Upstream draws a width of 0.5 m and multiplies it by a factor in [0.5, 1.5].
        radius = 0.25 * rng.uniform(0.5, 1.5)
        height = rng.uniform(2.0, 3.0)
        obstacles.append(Cylinder((x, y, height / 2), radius, height))
        if len(obstacles) == count:
            return tuple(obstacles)
    raise RuntimeError(f"Could not place {count} forest obstacles for seed {seed}.")


@dataclass(frozen=True)
class Course:
    half_extents: tuple[float, float]
    start_x: tuple[float, float]
    goal_x: tuple[float, float]
    obstacles: Callable[[int], tuple[Box | Cylinder, ...]] | None = None
    scale: float = 1.0

    start_y: tuple[float, float] = (-2.0, 2.0)
    goal_y: tuple[float, float] = (-4.0, 4.0)

    def sample_endpoints(self, rng):
        start = (rng.uniform(*self.start_x), rng.uniform(*self.start_y))
        goal = (rng.uniform(*self.goal_x), rng.uniform(*self.goal_y))
        return start, goal

DEFAULT_OBSTACLE_MAP = "car-like"
OPEN_COURSE = Course((22.0, 10.0), (-14.0, -13.8), (15.5, 16.0))
OBSTACLE_COURSES = {
    "car-like": Course((30.0, 17.0), (-24.1, -24.0), (23.5, 24.0),
                       obstacles=lambda _: car_like_obstacles(), scale=0.5),
    "forest": Course((27.0, 10.0), (-21.0, -20.8), (20.4, 20.6), obstacles=swarm_formation_forest),
}
