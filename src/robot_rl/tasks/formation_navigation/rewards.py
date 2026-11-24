"""Reward contributions for one formation navigation step."""

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_TASK, RewardWeights


@dataclass(frozen=True, slots=True)
class RewardInputs:
    previous_goal_distance: float
    goal_vector: np.ndarray
    leader_velocities: np.ndarray
    triangle_error: float
    follower_errors: np.ndarray
    predicted_clearances: np.ndarray
    obstacle_surface_distances: np.ndarray
    success: bool
    collided: bool


@dataclass(frozen=True, slots=True)
class RewardTerms:
    progress: float
    step_cost: float
    velocity_alignment: float
    tracking_error: float
    shape_error: float
    predicted_clearance: float
    obstacle_proximity: float
    success: float
    collision: float

    @property
    def total(self):
        return (self.progress + self.step_cost + self.velocity_alignment + self.tracking_error + self.shape_error
                + self.predicted_clearance + self.obstacle_proximity + self.success + self.collision)


def reward_terms(step: RewardInputs, weights: RewardWeights = DEFAULT_TASK.reward) -> RewardTerms:
    """Return the signed, weighted contribution of each reward term."""
    distance = float(np.linalg.norm(step.goal_vector))
    alignment = 0.0
    if distance > weights.alignment_goal_distance:
        speeds = np.linalg.norm(step.leader_velocities, axis=1)
        moving = speeds > weights.alignment_min_speed
        direction_cosines = (step.leader_velocities[moving] @ (step.goal_vector / distance)) / speeds[moving]
        alignment = float(np.sum(1.0 - np.clip(direction_cosines, -1.0, 1.0)))
    tracking_excess = np.maximum(step.follower_errors - weights.tracking_tolerance, 0.0)
    predicted_shortfall = np.maximum(weights.predicted_warning_distance - step.predicted_clearances, 0.0)
    surface_distances = np.clip(step.obstacle_surface_distances,
                                weights.surface_distance_floor, weights.surface_warning_distance)
    proximity = 1.0 / surface_distances - 1.0 / weights.surface_warning_distance
    near_goal_deformation = distance < weights.shape_penalty_radius and step.triangle_error > weights.shape_tolerance
    return RewardTerms(
        progress=weights.progress * (step.previous_goal_distance - distance),
        step_cost=-weights.step_cost,
        velocity_alignment=-weights.velocity_alignment * alignment,
        tracking_error=-weights.tracking_error * float(np.sum(tracking_excess)),
        shape_error=-weights.shape_error * step.triangle_error if near_goal_deformation else 0.0,
        predicted_clearance=-weights.predicted_clearance * float(np.sum(predicted_shortfall)),
        obstacle_proximity=-weights.obstacle_proximity * float(np.sum(proximity)),
        success=weights.success_bonus if step.success else 0.0,
        collision=-weights.collision_cost if step.collided else 0.0,
    )


def compute_reward(step: RewardInputs, weights: RewardWeights = DEFAULT_TASK.reward) -> float:
    return reward_terms(step, weights).total
