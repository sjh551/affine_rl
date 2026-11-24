import unittest
from unittest.mock import patch

from gymnasium.error import ResetNeeded
import numpy as np
import pybullet as p

from robot_rl.tasks.formation_navigation.env import FormationPyBulletEnv


class EnvironmentTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = FormationPyBulletEnv(**kwargs)
        self.addCleanup(env.close)
        return env

    def test_observations_and_seeded_reset(self):
        env = self.make_env(obstacle_map="forest", follower_control_mode="distributed",
                            neighbor_velocity_source="command")
        first, info = env.reset(seed=13)
        self.assertEqual(info, {})
        self.assertEqual(env.action_space.shape, (6,))
        self.assertEqual(first["kinematics"].shape, (29,))
        self.assertEqual(first["lidar"].shape, (3, 5, 60))
        for key, value in env.reset(seed=13)[0].items():
            np.testing.assert_array_equal(first[key], value)
        action = np.full(6, 0.15, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(terminated or truncated)
        self.assertEqual(info["reason"], "")
        np.testing.assert_array_equal(obs["kinematics"][-6:], action)

    def test_requires_reset_and_truncates_at_horizon(self):
        env = self.make_env(enable_obstacles=False, max_steps=1)
        with self.assertRaises(ResetNeeded):
            env.step(np.zeros(6))
        env.reset(seed=2)
        _, _, terminated, truncated, info = env.step(np.zeros(6))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["reason"], "Timeout")
        with self.assertRaises(ResetNeeded):
            env.step(np.zeros(6))

    def test_success_terminates(self):
        env = self.make_env(enable_obstacles=False, max_steps=1)
        env.reset(seed=2)
        with patch.object(env, "_measure_formation", return_value=(1.0, 0.0, 0.0)):
            _, _, terminated, truncated, info = env.step(np.zeros(6))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["is_success"])
        self.assertEqual(info["reason"], "Success")

    def test_obstacle_contact_terminates(self):
        env = self.make_env()
        env.reset(seed=0)
        client = env.physics_client_id
        obstacle = min(env.world.obstacle_ids)
        center, _ = p.getBasePositionAndOrientation(obstacle, physicsClientId=client)
        body = env.world.robots[0].body
        robot_position, rotation = p.getBasePositionAndOrientation(body, physicsClientId=client)
        p.resetBasePositionAndOrientation(body, (*center[:2], robot_position[2]), rotation, physicsClientId=client)
        _, _, terminated, truncated, info = env.step(np.zeros(6))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["reason"], "Crash_Obstacle")
