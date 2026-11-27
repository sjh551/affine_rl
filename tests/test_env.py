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

    def test_leaders_only_keeps_initial_leader_positions_and_uses_their_centroid(self):
        env = self.make_env(enable_obstacles=False, enable_followers=False)
        full_team = self.make_env(enable_obstacles=False)
        obs, _ = env.reset(seed=13)
        full_team.reset(seed=13)
        leaders = env.world.robot_states()[0]
        self.assertEqual(len(env.world.robot_ids), 3)
        np.testing.assert_allclose(leaders, full_team.world.robot_states()[0][:3], atol=2e-5)
        center = leaders.mean(axis=0)
        expected_goal = (env._goal - center) @ env._frame.T / env.course.half_extents
        np.testing.assert_allclose(obs["kinematics"][:2], expected_goal, atol=1e-6)
        self.assertEqual(env.action_space, full_team.action_space)
        self.assertEqual(env.observation_space, full_team.observation_space)

        obs, _, terminated, truncated, info = env.step(np.tile([0.4, 0.0], 3))
        self.assertTrue(env.observation_space.contains(obs))
        self.assertFalse(terminated or truncated)
        self.assertEqual(info["formation_error"], 0.0)
        self.assertLess(info["shape_error"], 0.01)

    def test_requires_reset_and_truncates_at_horizon(self):
        env = self.make_env(enable_obstacles=False, max_steps=1)
        with self.assertRaises(ResetNeeded):
            env.step(np.zeros(6))
        env.reset(seed=2)
        _, _, terminated, truncated, info = env.step(np.zeros(6))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["reason"], "Timeout")
        for key in ("tracking_error", "predicted_clearance", "obstacle_proximity"):
            self.assertEqual(info["reward_terms"][key], 0.0)
        with self.assertRaises(ResetNeeded):
            env.step(np.zeros(6))

    def test_success_terminates(self):
        env = self.make_env(enable_obstacles=False, max_steps=1)
        env.reset(seed=2)
        goal = env.world.robot_states()[0].mean(axis=0) + [1.0, 0.0]
        with patch.object(env, "_goal", goal):
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
        self.assertEqual(info["reward_terms"]["collision"], -50.0)

    def test_approaching_obstacle_costs_reward_without_ending_episode(self):
        env = self.make_env(enable_followers=False)
        env.reset(seed=0)
        client = env.physics_client_id
        p.resetBasePositionAndOrientation(env.world.robots[0].body, (1.6, -1.5, 0.315), (0, 0, 0, 1),
                                          physicsClientId=client)
        p.performCollisionDetection(physicsClientId=client)
        with (
            patch.object(env, "_advance_control", return_value=None),
            patch.object(env.world, "point_clearances", side_effect=AssertionError("Unexpected follower prediction")),
        ):
            _, reward, terminated, truncated, info = env.step(np.zeros(6))
        self.assertFalse(terminated or truncated)
        self.assertLess(info["reward_terms"]["obstacle_proximity"], 0.0)
        self.assertEqual(info["reward_terms"]["predicted_clearance"], 0.0)
        self.assertEqual(info["reward_terms"]["tracking_error"], 0.0)
        self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))

    def test_prediction_uses_affine_map_and_obstacle_footprints(self):
        env = self.make_env()
        env.reset(seed=0)
        points = np.array([[0.0, -1.5], [1.2, -1.5], [2.0, -1.5], [env.course.half_extents[0] - 0.2, 0.0]])
        np.testing.assert_allclose(env.world.point_clearances(points), [0.0, 0.2, 1.0, 0.2], atol=1e-10)
        positions, velocities, yaws, yaw_rates = env.world.robot_states()
        velocities[:3] = [[0.5, 0.1], [0.1, -0.3], [0.7, 0.1]]
        expected = env.formation.follower_map @ (positions[:3] + 0.5 * velocities[:3])
        with patch.object(env.world, "point_clearances", return_value=np.zeros(4)) as clearance:
            terms = env._reward_terms((positions, velocities, yaws, yaw_rates), env._measure_formation(positions),
                                      success=False, collided=True)
        np.testing.assert_allclose(clearance.call_args.args[0], expected)
        self.assertAlmostEqual(terms.predicted_clearance, -28.0)
