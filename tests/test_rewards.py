import unittest
from dataclasses import replace

import numpy as np

from robot_rl.tasks.formation_navigation.rewards import RewardInputs, compute_reward, reward_terms


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.step = RewardInputs(
            previous_goal_distance=7.5, goal_vector=np.array([7.0, 0.0]),
            leader_velocities=np.array([[0.5, 0.0], [0.0, 0.5], [-0.05, 0.0]]),
            triangle_error=1.0, follower_errors=np.array([0.75, 0.5, 0.4, 0.6]),
            predicted_clearances=np.array([0.7, 0.5, 1.0, np.inf]),
            obstacle_surface_distances=np.array([0.5, 0.25, 1.0]),
            success=False, collided=False,
        )

    def test_progress_alignment_tracking_and_clearance(self):
        terms = reward_terms(self.step)
        self.assertEqual(terms.progress, 5.0)
        self.assertEqual(terms.step_cost, -0.1)
        self.assertEqual(terms.velocity_alignment, -1.0)
        self.assertAlmostEqual(terms.tracking_error, -1.75)
        self.assertAlmostEqual(terms.predicted_clearance, -2.0)
        self.assertAlmostEqual(terms.obstacle_proximity, -0.2)
        self.assertEqual(terms.shape_error, 0.0)
        self.assertAlmostEqual(compute_reward(self.step), -0.05)

    def test_near_goal_shape_and_tracking_deadband(self):
        step = replace(self.step, goal_vector=np.array([5.0, 0.0]), triangle_error=3.0,
                       follower_errors=np.full(4, 0.5))
        self.assertEqual(reward_terms(step).shape_error, 0.0)
        step = replace(step, goal_vector=np.array([4.99, 0.0]))
        self.assertEqual(reward_terms(replace(step, triangle_error=2.0)).shape_error, 0.0)
        terms = reward_terms(step)
        self.assertEqual(terms.shape_error, -0.03)
        self.assertEqual(terms.tracking_error, 0.0)

    def test_alignment_is_disabled_at_the_goal_and_repulsion_is_bounded(self):
        step = replace(self.step, goal_vector=np.zeros(2), obstacle_surface_distances=np.array([0.0]))
        terms = reward_terms(step)
        self.assertEqual(terms.velocity_alignment, 0.0)
        self.assertAlmostEqual(terms.obstacle_proximity, -0.1 * (1 / 0.03 - 1 / 0.5))
        self.assertTrue(np.isfinite(terms.total))

    def test_terminal_bonus_and_penalty(self):
        reward = compute_reward(self.step)
        self.assertAlmostEqual(compute_reward(replace(self.step, success=True)) - reward, 150.0)
        self.assertAlmostEqual(compute_reward(replace(self.step, collided=True)) - reward, -50.0)
