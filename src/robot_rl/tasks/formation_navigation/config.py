"""Task defaults and selectable environment settings."""

from dataclasses import asdict, dataclass

from robot_rl.controllers.formation import FOLLOWER_CONTROL_MODES, NEIGHBOR_VELOCITY_SOURCES
from robot_rl.maps.layouts import DEFAULT_OBSTACLE_MAP, OBSTACLE_COURSES


@dataclass(frozen=True, slots=True)
class RewardWeights:
    progress: float = 10.0
    step_cost: float = 0.1
    velocity_alignment: float = 0.1
    alignment_goal_distance: float = 2.0
    alignment_min_speed: float = 0.1
    tracking_error: float = 5.0
    tracking_tolerance: float = 0.5
    shape_error: float = 1.0    # the weight of adjusting formation shape
    shape_penalty_radius: float = 8.0
    shape_tolerance: float = 0.5
    prediction_horizon: float = 0.5
    predicted_clearance: float = 10.0
    predicted_warning_distance: float = 0.7
    obstacle_proximity: float = 0.1
    surface_warning_distance: float = 0.5
    surface_distance_floor: float = 0.03
    success_bonus: float = 150.0
    collision_cost: float = 50.0


@dataclass(frozen=True, slots=True)
class TaskConfig:
    max_steps: int = 1500
    goal_tolerance: float = 2.0
    success_shape_error: float = 1.0
    success_tracking_error: float = 0.5
    minimum_team_distance: float = 0.7
    leader_speed: float = 1.0
    follower_speed: float = 1.3
    physics_time_step: float = 1 / 240
    physics_substeps: int = 24
    distributed_substeps: int = 4
    reward: RewardWeights = RewardWeights()


DEFAULT_TASK = TaskConfig()


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    stage: int = 2
    map: str = DEFAULT_OBSTACLE_MAP
    walls: bool = True
    follower_control: str = "centralized"
    neighbor_velocity: str = "measured"
    max_steps: int = DEFAULT_TASK.max_steps

    def validate(self):
        if type(self.stage) is not int or self.stage not in (1, 2):
            raise ValueError("environment.stage must be 1 or 2.")
        if not isinstance(self.map, str) or self.map not in OBSTACLE_COURSES:
            raise ValueError(f"Unknown obstacle map {self.map!r}.")
        if type(self.walls) is not bool:
            raise ValueError("environment.walls must be true or false.")
        if self.follower_control not in FOLLOWER_CONTROL_MODES:
            raise ValueError(f"Unknown follower control mode {self.follower_control!r}.")
        if self.neighbor_velocity not in NEIGHBOR_VELOCITY_SOURCES:
            raise ValueError(f"Unknown neighbor velocity source {self.neighbor_velocity!r}.")
        if self.neighbor_velocity == "command" and self.follower_control != "distributed":
            raise ValueError("Neighbor velocity commands require distributed follower control.")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("environment.max_steps must be a positive integer.")

    @property
    def map_name(self):
        return self.map if self.stage == 2 else "open"

    def env_kwargs(self):
        return dict(enable_obstacles=self.stage == 2, obstacle_map=self.map, enable_walls=self.walls,
                    follower_control_mode=self.follower_control, neighbor_velocity_source=self.neighbor_velocity,
                    max_steps=self.max_steps)

    def run_settings(self):
        settings = dict(stage=self.stage, map=self.map_name, walls=self.walls,
                        follower_control=self.follower_control, neighbor_velocity=self.neighbor_velocity,
                        reward_weights=asdict(DEFAULT_TASK.reward))
        if self.max_steps != DEFAULT_TASK.max_steps:
            settings["max_steps"] = self.max_steps
        return settings
