"""Reward contributions for one formation navigation step."""

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_TASK, RewardWeights


@dataclass(frozen=True, slots=True)
class RewardInputs:
    previous_goal_distance: float
    goal_distance: float
    tracking_error: float
    shape_error: float
    action: np.ndarray
    previous_action: np.ndarray
    success: bool
    collided: bool


@dataclass(frozen=True, slots=True)
class RewardTerms:
    progress: float
    step_cost: float
    action_change: float
    tracking_error: float
    shape_error: float
    success: float
    collision: float

    @property
    def total(self):
        return (self.progress + self.step_cost + self.action_change + self.tracking_error
                + self.shape_error + self.success + self.collision)


def reward_terms(step: RewardInputs, weights: RewardWeights = DEFAULT_TASK.reward) -> RewardTerms:
    """Return the signed, weighted contribution of each reward term."""
    action_change = float(np.mean(np.square(step.action - step.previous_action)))
    tracking_excess = max(0.0, step.tracking_error - weights.tracking_tolerance)
    return RewardTerms(
        progress=weights.progress * (step.previous_goal_distance - step.goal_distance),
        step_cost=-weights.step_cost,
        action_change=-weights.action_change * action_change,
        tracking_error=-weights.tracking_error * tracking_excess,
        shape_error=-weights.shape_error * step.shape_error if step.goal_distance < weights.shape_penalty_radius else 0.0,
        success=weights.success_bonus if step.success else 0.0,
        collision=-weights.collision_cost if step.collided else 0.0,
    )


def compute_reward(step: RewardInputs, weights: RewardWeights = DEFAULT_TASK.reward) -> float:
    return reward_terms(step, weights).total
