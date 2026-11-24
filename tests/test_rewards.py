import unittest
from dataclasses import replace

import numpy as np

from robot_rl.tasks.formation_navigation.rewards import RewardInputs, compute_reward, reward_terms


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.step = RewardInputs(
            previous_goal_distance=7.5, goal_distance=7.0, tracking_error=0.75, shape_error=1.0,
            action=np.full(6, 0.5, dtype=np.float32), previous_action=np.zeros(6, dtype=np.float32),
            success=False, collided=False,
        )

    def test_progress_action_and_tracking(self):
        terms = reward_terms(self.step)
        self.assertEqual(terms.progress, 5.0)
        self.assertEqual(terms.step_cost, -0.05)
        self.assertEqual(terms.action_change, -0.0125)
        self.assertEqual(terms.tracking_error, -0.25)
        self.assertEqual(terms.shape_error, 0.0)
        self.assertAlmostEqual(compute_reward(self.step), 4.6875)

    def test_near_goal_shape_and_tracking_deadband(self):
        step = replace(self.step, previous_goal_distance=5.0, goal_distance=5.0, tracking_error=0.25)
        self.assertEqual(reward_terms(step).shape_error, 0.0)
        terms = reward_terms(replace(step, goal_distance=4.99))
        self.assertEqual(terms.shape_error, -0.2)
        self.assertEqual(terms.tracking_error, 0.0)

    def test_terminal_bonus_and_penalty(self):
        reward = compute_reward(self.step)
        self.assertAlmostEqual(compute_reward(replace(self.step, success=True)) - reward, 150.0)
        self.assertAlmostEqual(compute_reward(replace(self.step, collided=True)) - reward, -50.0)
