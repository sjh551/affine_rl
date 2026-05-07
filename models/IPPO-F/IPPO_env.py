import os
import sys
import time
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.formation_utils import get_square_configuration, get_square_topology


PROJECT_ROOT = ROOT_DIR
RANGER_URDF_PATH = os.path.join(PROJECT_ROOT, "assets", "ranger_mini_v3", "urdf", "ranger_mini.urdf").replace("\\", "/")


class FormationIPPOEnv(gym.Env):
    """Homogeneous multi-agent formation environment for IPPO.

    Every robot shares the same policy class, action interface, observation
    structure, and team reward. There is no leader/follower split here.
    """

    metadata = {"render_modes": ["human", None]}

    def __init__(
        self,
        render_mode=False,
        enable_obstacles=True,
        hide_robots=False,
        draw_line=True,
        k_vel_align=1.0,
        k_progress=10.0,
        k_shape_err=0.01,
        k_disconnect=0.2,
        k_smooth_linear=0.05,
        k_omega_limit=0.2,
        k_path_cost=0.2,
    ):
        super().__init__()

        self.k_vel_align = k_vel_align
        self.k_progress = k_progress
        self.k_shape_err = k_shape_err
        self.k_disconnect = k_disconnect
        self.k_smooth_linear = k_smooth_linear
        self.k_path_cost = k_path_cost

        self.render_mode = render_mode
        self.hide_robots = hide_robots
        self.draw_line = draw_line
        self.enable_obstacles = enable_obstacles

        if self.render_mode:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.g_nom = np.array(get_square_configuration(), dtype=np.float32)
        self.edges = get_square_topology()
        self.num_agents = len(self.g_nom)
        self.num_robots = self.num_agents
        self.robot_ids = []
        self.agent_names = [f"agent_{i}" for i in range(self.num_agents)]

        self.action_space = spaces.Dict(
            {
                name: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
                for name in self.agent_names
            }
        )

        self.n_lidar_rays = 180
        self.lidar_range = 7.0
        self.frame_len = 20
        self.in_channels = 5

        obs_dim = 2 + 2 + 2 + 2 + 2 + (self.num_agents - 1) * 2 + (self.num_agents - 1) * 2 # 目标相对位置2 速度2 
        agent_obs_space = spaces.Dict(
            {
                "kinematics": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
                "lidar": spaces.Box(low=0.0, high=1.0, shape=(self.in_channels, 60), dtype=np.float32),
            }
        )
        self.observation_space = spaces.Dict({name: agent_obs_space for name in self.agent_names})

        self.target_pos = np.array([8.5, 0.0, 0.0], dtype=np.float32)
        self.wheel_indices = [3, 1, 5, 7]
        self.steer_indices = [2, 0, 4, 6]
        self.max_vel = 1.0
        self.max_omega = 1.0
        self.wheel_radius = 0.09
        self.safe_margin = 0.7
        self.max_steps = 1500
        self.wall_length = 20.0
        self.wall_width = 10.0

        self.obstacles = []
        self.wall_ids = []
        self.debug_line_ids = []

        self.nominal_center = np.mean(self.g_nom[:, :2], axis=0)
        self.nominal_offsets = self.g_nom[:, :2] - self.nominal_center

        self.lidar_buffers = {name: deque(maxlen=self.frame_len) for name in self.agent_names}
        self.last_actions = {name: np.zeros(2, dtype=np.float32) for name in self.agent_names}
        self.last_agent_vels = {name: np.zeros(2, dtype=np.float32) for name in self.agent_names}

        self.last_center_dist = None
        self.current_step = 0

        self.ep_rew_progress = 0.0
        self.ep_rew_shape = 0.0
        self.ep_rew_collision = 0.0
        self.ep_rew_time = 0.0
        self.ep_rew_smooth = 0.0
        self.ep_formation_error_sum = 0.0
        self.ep_formation_error_max = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        self._create_walls()

        tx = np.random.uniform(self.wall_length - 4.5, self.wall_length - 4.0)
        ty = np.random.uniform(-self.wall_width + 6.0, self.wall_width - 6.0)
        self.target_pos = np.array([tx, ty, 0.0], dtype=np.float32)
        self._create_visual_sphere(self.target_pos, [0, 1, 0, 0.5])

        self.robot_ids = []
        self.start_center = np.array(
            [
                np.random.uniform(-self.wall_length + 6.0, -self.wall_length + 6.2),
                np.random.uniform(-2.0, 2.0),
            ],
            dtype=np.float32,
        )

        self.L = self.target_pos[:2] - self.start_center[:2]
        self.L_norm = np.linalg.norm(self.L)
        dir_start_to_goal = self.L / (self.L_norm + 1e-8)
        self.virtual_target_pos = self.target_pos[:2] + dir_start_to_goal * 4.0

        for i in range(self.num_agents):
            rx = self.start_center[0] + self.g_nom[i][0]
            ry = self.start_center[1] + self.g_nom[i][1]
            yaw = self.g_nom[i][2]

            spawn_pos = [rx, ry, 0.15]
            spawn_quat = p.getQuaternionFromEuler([0, 0, yaw])
            rid = p.loadURDF(RANGER_URDF_PATH, basePosition=spawn_pos, baseOrientation=spawn_quat)

            if self.hide_robots:
                p.changeVisualShape(objectUniqueId=rid, linkIndex=-1, rgbaColor=[1.0, 0.0, 0.0, 0.0])
                for link_idx in range(p.getNumJoints(rid)):
                    p.changeVisualShape(objectUniqueId=rid, linkIndex=link_idx, rgbaColor=[1.0, 0.0, 0.0, 0.0])

            for joint_idx in range(p.getNumJoints(rid)):
                if joint_idx in self.steer_indices:
                    p.changeDynamics(rid, joint_idx, lateralFriction=0.1)
                    p.setJointMotorControl2(rid, joint_idx, p.POSITION_CONTROL, targetPosition=0)
                elif joint_idx in self.wheel_indices:
                    p.changeDynamics(rid, joint_idx, lateralFriction=0.5, rollingFriction=0.02)
                    p.setJointMotorControl2(rid, joint_idx, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

            p.changeDynamics(rid, -1, lateralFriction=0.5, rollingFriction=0.01)
            self.robot_ids.append(rid)

        if self.enable_obstacles:
            self._generate_obstacles()
        else:
            self.obstacles = []

        for _ in range(100):
            for rid in self.robot_ids:
                p.setJointMotorControlArray(
                    rid,
                    self.steer_indices,
                    p.POSITION_CONTROL,
                    targetPositions=[0] * 4,
                    forces=[80.0] * 4,
                )
                p.setJointMotorControlArray(
                    rid,
                    self.wheel_indices,
                    p.VELOCITY_CONTROL,
                    targetVelocities=[0] * 4,
                    forces=[20.0] * 4,
                )
            p.stepSimulation()

        self.current_step = 0
        self.last_actions = {name: np.zeros(2, dtype=np.float32) for name in self.agent_names}
        self.last_agent_vels = {name: np.zeros(2, dtype=np.float32) for name in self.agent_names}

        center_pos = self._get_center_pos()
        vec_to_goal = self.target_pos[:2] - center_pos[:2]
        self.last_center_dist = np.linalg.norm(vec_to_goal)
        target_yaw = np.arctan2(vec_to_goal[1], vec_to_goal[0])

        for name, rid in zip(self.agent_names, self.robot_ids):
            pos, _ = p.getBasePositionAndOrientation(rid)
            init_lidar = self._get_lidar(pos[:2], target_yaw, self.n_lidar_rays, self.lidar_range)
            self.lidar_buffers[name].clear()
            for _ in range(self.frame_len):
                self.lidar_buffers[name].append(init_lidar) # 初始化所有帧队列

        p.removeAllUserDebugItems()
        self.debug_line_ids = []

        self.ep_rew_progress = 0.0
        self.ep_rew_shape = 0.0
        self.ep_rew_collision = 0.0
        self.ep_rew_time = 0.0
        self.ep_rew_smooth = 0.0
        self.ep_formation_error_sum = 0.0
        self.ep_formation_error_max = 0.0

        obs = self._get_obs()
        return obs, {}

    def step(self, actions):
        self.current_step += 1

        action_dict = self._normalize_actions(actions)
        action_diff_dict = {}

        for name in self.agent_names:
            current_action = np.asarray(action_dict[name], dtype=np.float32)
            action_diff_dict[name] = current_action - self.last_actions[name]
            self.last_actions[name] = current_action.copy()

        for name, rid in zip(self.agent_names, self.robot_ids):
            action_vec = np.asarray(action_dict[name], dtype=np.float32)
            v_x = float(np.clip(action_vec[0], -1.0, 1.0) * self.max_vel)
            v_y = float(np.clip(action_vec[1], -1.0, 1.0) * self.max_vel)
            _, quat = p.getBasePositionAndOrientation(rid)
            yaw = p.getEulerFromQuaternion(quat)[2]
            body_vx = v_x * np.cos(yaw) + v_y * np.sin(yaw)
            body_vy = -v_x * np.sin(yaw) + v_y * np.cos(yaw)
            self._drive_robot(rid, body_vx, body_vy, 0.0)

        for _ in range(24):
            p.stepSimulation()

        if self.render_mode and self.draw_line:
            self._draw_formation_lines()
            time.sleep(0.01)

        center_pos = self._get_center_pos()
        obs = self._get_obs()
        reward, reward_info = self._compute_reward(center_pos, action_dict, action_diff_dict)

        self.ep_rew_progress += reward_info["r_progress"]
        self.ep_rew_shape += reward_info["r_shape"]
        self.ep_rew_collision += reward_info["r_collision"]
        self.ep_rew_time += reward_info["r_time"]
        self.ep_rew_smooth += reward_info["r_smooth"]
        self.ep_formation_error_sum += reward_info["formation_error_step"]
        self.ep_formation_error_max = max(self.ep_formation_error_max, reward_info["formation_error_step"])

        done = reward_info["done"]
        terminated = {name: False for name in self.agent_names}
        truncated = {name: False for name in self.agent_names}
        infos = {name: {} for name in self.agent_names}

        if done:
            if reward_info["reason"] == "Timeout":
                truncated = {name: True for name in self.agent_names}
            else:
                terminated = {name: True for name in self.agent_names}

            episode_info = {
                "team_reward": reward,
                "ep_rew_progress": self.ep_rew_progress,
                "ep_rew_shape": self.ep_rew_shape,
                "ep_rew_collision": self.ep_rew_collision,
                "ep_rew_smooth": self.ep_rew_smooth,
                "ep_rew_time": self.ep_rew_time,
                "ep_formation_error_avg": self.ep_formation_error_sum / max(self.current_step, 1),
                "ep_formation_error_max": self.ep_formation_error_max,
                "reason": reward_info.get("reason", "Unknown"),
            }
            for name in self.agent_names:
                infos[name].update(episode_info)
            infos["__all__"] = episode_info

        return obs, {name: reward for name in self.agent_names}, terminated, truncated, infos

    def close(self):
        if hasattr(self, "physicsClient"):
            try:
                p.disconnect(self.physicsClient)
            except Exception:
                pass

    def _normalize_actions(self, actions):
        if isinstance(actions, dict):
            return {name: np.asarray(actions[name], dtype=np.float32) for name in self.agent_names}

        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.ndim == 1 and action_array.size == self.num_agents * 2:
            action_array = action_array.reshape(self.num_agents, 2)

        if action_array.shape != (self.num_agents, 2):
            raise ValueError(f"Expected actions with shape ({self.num_agents}, 2), got {action_array.shape}")

        return {name: action_array[i] for i, name in enumerate(self.agent_names)}

    def _get_obs(self):
        center_pos = self._get_center_pos()
        obs = {}

        all_states = self._get_all_robot_states()
        goal_vec = self.target_pos[:2] - center_pos[:2]
        target_yaw = np.arctan2(goal_vec[1], goal_vec[0])

        for idx, name in enumerate(self.agent_names):
            obs[name] = self._get_agent_obs(idx, name, center_pos, all_states, target_yaw)

        return obs

    def _get_agent_obs(self, agent_idx, agent_name, center_pos, all_states, target_yaw):
        state = all_states[agent_idx]
        pos = state["pos"]
        vel = state["vel"]
        yaw = state["yaw"]

        c, s = np.cos(yaw), np.sin(yaw)
        rot_world_to_body = np.array([[c, s], [-s, c]], dtype=np.float32)

        goal_vec = self.target_pos[:2] - pos[:2]
        goal_body = rot_world_to_body @ goal_vec
        center_rel_body = rot_world_to_body @ (pos[:2] - center_pos[:2])
        vel_body = rot_world_to_body @ vel[:2]

        teammate_rel_pos = []
        teammate_rel_vel = []
        for other_idx, other_state in enumerate(all_states):
            if other_idx == agent_idx:
                continue
            rel_pos_body = rot_world_to_body @ (other_state["pos"][:2] - pos[:2])
            rel_vel_body = rot_world_to_body @ (other_state["vel"][:2] - vel[:2])
            teammate_rel_pos.extend(rel_pos_body.tolist())
            teammate_rel_vel.extend(rel_vel_body.tolist())

        lid = self._get_lidar(pos[:2], yaw, self.n_lidar_rays, self.lidar_range)
        self.lidar_buffers[agent_name].append(lid)
        sparse_frames = [
            self.lidar_buffers[agent_name][0],
            self.lidar_buffers[agent_name][4],
            self.lidar_buffers[agent_name][9],
            self.lidar_buffers[agent_name][14],
            self.lidar_buffers[agent_name][-1],
        ]
        stacked_lidar = np.asarray(sparse_frames, dtype=np.float32)

        kinematics_obs = np.asarray(
            [
                goal_body[0] / self.wall_length,
                goal_body[1] / self.wall_width,
                center_rel_body[0] / self.wall_length,
                center_rel_body[1] / self.wall_width,
                vel_body[0],
                vel_body[1],
                np.sin(yaw),
                np.cos(yaw),
                self.last_actions[agent_name][0],
                self.last_actions[agent_name][1],
                *teammate_rel_pos,
                *teammate_rel_vel,
            ],
            dtype=np.float32,
        )

        return {"kinematics": kinematics_obs, "lidar": stacked_lidar}

    def _compute_reward(self, center_pos, action_dict, action_diff_dict):
        reward = 0.0
        done = False
        reason = ""
        formation_error_step = 0.0

        vec_to_goal = self.target_pos[:2] - center_pos[:2]
        dist_to_goal = np.linalg.norm(vec_to_goal)
        dir_to_goal = vec_to_goal / (dist_to_goal + 1e-8)

        if self.last_center_dist is None:
            self.last_center_dist = dist_to_goal

        all_states = self._get_all_robot_states()
        all_pos = [state["pos"][:2] for state in all_states]

        r_time = -0.1
        r_progress = (self.last_center_dist - dist_to_goal) * self.k_progress
        self.last_center_dist = dist_to_goal

        actual_offsets = np.asarray(all_pos, dtype=np.float32) - center_pos[:2]
        shape_errors = np.linalg.norm(actual_offsets - self.nominal_offsets, axis=1)
        formation_error_step = float(np.mean(shape_errors))

        r_shape = 0.0
        if dist_to_goal < 5.0 and formation_error_step > 0.5:
            r_shape -= self.k_shape_err * formation_error_step

        r_smooth = 0.0
        if dist_to_goal > 2.0:
            for state, name in zip(all_states, self.agent_names):
                speed = np.linalg.norm(state["vel"][:2])
                if speed > 0.1:
                    r_smooth -= self.k_vel_align * (1.0 - np.dot(state["vel"][:2] / (speed + 1e-8), dir_to_goal))

        action_change_cost = np.mean([np.linalg.norm(action_diff_dict[name]) for name in self.agent_names])
        r_smooth -= self.k_smooth_linear * action_change_cost

        R_vec = center_pos[:2] - self.start_center[:2]
        cross_product = abs(self.L[0] * R_vec[1] - self.L[1] * R_vec[0])
        cte = cross_product / (self.L_norm + 1e-8)
        r_smooth -= self.k_path_cost * cte

        r_collision = 0.0
        k_wall_crash = 50.0
        k_team_crash = 50.0
        k_obs_crash = 50.0
        k_repulsion_base = 0.1

        if self.enable_obstacles:
            for state in all_states:
                est_pos = np.array(state["pos"][:2], dtype=np.float32)
                min_dist, _ = self._get_obstacle_repulsion(est_pos)
                if min_dist < self.safe_margin:
                    r_collision -= self.safe_margin - min_dist

        limit_x = self.wall_length - 0.5 - 0.6
        limit_y = self.wall_width - 0.5 - 0.6
        for i, pos_2d in enumerate(all_pos):
            if abs(pos_2d[0]) > limit_x or abs(pos_2d[1]) > limit_y:
                r_collision -= k_wall_crash
                done = True
                reason = f"Wall_Out (Agent {i})"
                break

        if not done:
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    if np.linalg.norm(all_pos[i] - all_pos[j]) < self.safe_margin:
                        r_collision -= k_team_crash
                        done = True
                        reason = "Crash_Team"
                        break
                if done:
                    break

        if self.enable_obstacles and not done:
            crash_threshold = 0.05
            warn_threshold = 0.35
            for agent_idx, robot_id in enumerate(self.robot_ids):
                for obs_id in self.obstacles:
                    closest_points = p.getClosestPoints(bodyA=robot_id, bodyB=obs_id, distance=warn_threshold)
                    if not closest_points:
                        continue

                    min_surface_dist = min(pt[8] for pt in closest_points)
                    if min_surface_dist <= crash_threshold:
                        r_collision -= k_obs_crash
                        done = True
                        reason = f"Crash_Obstacle (Agent {agent_idx})"
                        break
                    if min_surface_dist < warn_threshold:
                        r_collision -= k_repulsion_base * (1.0 / max(min_surface_dist, 0.01) - 1.0 / warn_threshold)
                if done:
                    break

        k_success_bonus = 150.0
        if not done and dist_to_goal < 2.0 and formation_error_step <= 1.0:
            r_progress += k_success_bonus
            done = True
            reason = "Success"
        elif self.current_step >= self.max_steps:
            done = True
            reason = "Timeout"

        reward = r_progress + r_shape + r_collision + r_time + 0.1 * r_smooth

        if done and reason == "Success":
            print(f"Episode End: {reason}")

        reward_info = {
            "done": done,
            "reason": reason,
            "r_progress": r_progress,
            "r_shape": r_shape,
            "r_collision": r_collision,
            "r_time": r_time,
            "r_smooth": 0.1 * r_smooth,
            "formation_error_step": formation_error_step,
        }
        return reward, reward_info

    def _drive_robot(self, robot_id, v_x, v_y, w):
        wheel_positions = [(0.25, 0.19), (0.25, -0.19), (-0.25, 0.19), (-0.25, -0.19)]
        target_steering_angles = []
        target_wheel_speeds = []
        current_joint_states = p.getJointStates(robot_id, self.steer_indices)
        wheel_radius = self.wheel_radius

        for i, (wx, wy) in enumerate(wheel_positions):
            wheel_vx = v_x - w * wy
            wheel_vy = v_y + w * wx
            speed_2d = np.hypot(wheel_vx, wheel_vy)

            if speed_2d < 2e-2:
                current_angle = current_joint_states[i][0]
                target_steering_angles.append(current_angle)
                target_wheel_speeds.append(0.0)
                continue

            ideal_angle = np.arctan2(wheel_vy, wheel_vx)
            cmd_angle = -ideal_angle
            ideal_speed = speed_2d / wheel_radius
            current_angle = current_joint_states[i][0]

            angle1 = cmd_angle
            speed1 = ideal_speed
            angle2 = angle1 - np.pi if angle1 > 0 else angle1 + np.pi
            speed2 = -ideal_speed

            valid_configs = []
            limit = 2.1
            if -limit <= angle1 <= limit:
                valid_configs.append((angle1, speed1))
            if -limit <= angle2 <= limit:
                valid_configs.append((angle2, speed2))

            if valid_configs:
                best_angle, best_speed = min(valid_configs, key=lambda x: abs(x[0] - current_angle))
            else:
                best_angle, best_speed = current_angle, 0.0

            angle_diff = abs(best_angle - current_angle)
            speed_penalty = max(0.0, np.cos(angle_diff))
            final_speed = best_speed * speed_penalty

            target_steering_angles.append(best_angle)
            target_wheel_speeds.append(final_speed)

        p.setJointMotorControlArray(
            robot_id,
            self.steer_indices,
            p.POSITION_CONTROL,
            targetPositions=target_steering_angles,
            forces=[80.0] * 4,
        )

        p.setJointMotorControlArray(
            robot_id,
            self.wheel_indices,
            p.VELOCITY_CONTROL,
            targetVelocities=target_wheel_speeds,
            forces=[20.0] * 4,
        )

    def _draw_formation_lines(self):
        if len(self.robot_ids) < 2:
            return

        points = []
        for rid in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(rid)
            points.append(pos)

        line_color = [1, 1, 0]
        for u, v in self.edges:
            if u < len(points) and v < len(points):
                p.addUserDebugLine(
                    lineFromXYZ=points[u],
                    lineToXYZ=points[v],
                    lineColorRGB=line_color,
                    lineWidth=2,
                    lifeTime=0.1,
                )

    def _create_walls(self):
        self.wall_ids = []
        wall_thickness = 0.5
        wall_height = 1.0
        wall_length = self.wall_length
        wall_width = self.wall_width

        walls_config = [
            ([wall_length, 0, 0.5], [wall_thickness, wall_width, wall_height]),
            ([-wall_length, 0, 0.5], [wall_thickness, wall_width, wall_height]),
            ([0, wall_width, 0.5], [wall_length, wall_thickness, wall_height]),
            ([0, -wall_width, 0.5], [wall_length, wall_thickness, wall_height]),
        ]

        for pos, extents in walls_config:
            v_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=extents, rgbaColor=[0.8, 0.8, 0.8, 1])
            c_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=extents)
            wall_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_shape, baseVisualShapeIndex=v_shape, basePosition=pos)
            self.wall_ids.append(wall_id)

    def _create_visual_sphere(self, pos, color):
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=color)
        body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[pos[0] - 2.0, pos[1], 0])
        return body_id

    def _generate_obstacles(self):
        self.obstacles = []
        config_id = np.random.randint(0, 5)

        v_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, length=1.5, rgbaColor=[0.2, 0.2, 0.2, 1])
        c_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.5, height=1.5)

        pos_list = []
        mid_x_min, mid_x_max = -self.wall_length / 3, self.wall_length / 3
        mid_y_range = self.wall_width - 1.0

        if config_id == 0:
            for x in [-4, -1.33333333, 1.33333333, 4]:
                pos_list.append([x, 0, 0.5])
                pos_list.append([x, 0.5 * self.wall_width, 0.5])
                pos_list.append([x, -0.5 * self.wall_width, 0.5])
        elif config_id == 1:
            doors_y = [-5.0, 0.0, 5.0]
            gap_half_width = 0.8
            wall_thickness = 1.0
            wall_height = 1.5
            wall_x = 0.0
            current_y = -self.wall_width + 0.5

            for door_y in doors_y:
                segment_end_y = door_y - gap_half_width
                segment_length = segment_end_y - current_y
                if segment_length > 0:
                    center_y = current_y + segment_length / 2.0
                    half_extents = [wall_thickness / 2.0, segment_length / 2.0, wall_height / 2.0]
                    v_shape_seg = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.2, 0.2, 0.2, 1])
                    c_shape_seg = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
                    obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_shape_seg, baseVisualShapeIndex=v_shape_seg, basePosition=[wall_x, center_y, wall_height / 2.0])
                    self.obstacles.append(obs_id)
                current_y = door_y + gap_half_width

            segment_end_y = self.wall_width - 0.5
            segment_length = segment_end_y - current_y
            if segment_length > 0:
                center_y = current_y + segment_length / 2.0
                half_extents = [wall_thickness / 2.0, segment_length / 2.0, wall_height / 2.0]
                v_shape_seg = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.2, 0.2, 0.2, 1])
                c_shape_seg = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
                obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_shape_seg, baseVisualShapeIndex=v_shape_seg, basePosition=[wall_x, center_y, wall_height / 2.0])
                self.obstacles.append(obs_id)
        elif config_id == 2:
            x_start = -5.0
            x_end = 3.0
            y_gap_start = 8.0
            y_gap_end = 1.2
            num_pillars = 10
            for i in range(num_pillars):
                t = i / (num_pillars - 1)
                cx = x_start + t * (x_end - x_start)
                cy = y_gap_start + t * (y_gap_end - y_gap_start)
                pos_list.append([cx, cy, 0.5])
                pos_list.append([cx, -cy, 0.5])
        elif config_id == 3:
            while len(pos_list) < 6:
                ox = np.random.uniform(mid_x_min, mid_x_max)
                oy = np.random.uniform(-mid_y_range, mid_y_range)
                if all(np.linalg.norm([ox - p_exist[0], oy - p_exist[1]]) >= 4.5 for p_exist in pos_list):
                    pos_list.append([ox, oy, 0.5])
        else:
            while len(pos_list) < 10:
                ox = np.random.uniform(mid_x_min, mid_x_max)
                oy = np.random.uniform(-mid_y_range, mid_y_range)
                if all(np.linalg.norm([ox - p_exist[0], oy - p_exist[1]]) >= 4.0 for p_exist in pos_list):
                    pos_list.append([ox, oy, 0.5])

        for pos in pos_list:
            obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_shape, baseVisualShapeIndex=v_shape, basePosition=pos)
            self.obstacles.append(obs_id)

    def _get_all_robot_states(self):
        states = []
        for rid in self.robot_ids:
            pos, quat = p.getBasePositionAndOrientation(rid)
            vel, angular_vel = p.getBaseVelocity(rid)
            yaw = p.getEulerFromQuaternion(quat)[2]
            states.append(
                {
                    "pos": np.array(pos, dtype=np.float32),
                    "yaw": float(yaw),
                    "vel": np.array(vel, dtype=np.float32),
                    "angular_vel": float(angular_vel[2]),
                }
            )
        return states

    def _get_center_pos(self):
        pos_list = []
        for robot_id in self.robot_ids:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            pos_list.append(np.array(pos, dtype=np.float32))
        return np.mean(pos_list, axis=0)

    def _get_lidar(self, pos_2d, reference_yaw, num_rays=180, ray_length=7.0):
        ray_from = []
        ray_to = []
        z_height = 0.23

        for i in range(num_rays):
            angle = reference_yaw + (i * 2 * np.pi / num_rays)
            ray_from.append([pos_2d[0], pos_2d[1], z_height])
            ray_to.append([pos_2d[0] + ray_length * np.cos(angle), pos_2d[1] + ray_length * np.sin(angle), z_height])

        results = p.rayTestBatch(ray_from, ray_to)
        features = np.array([1.0 - res[2] for res in results], dtype=np.float32).reshape(60, 3).max(axis=1)
        return features

    def _get_obstacle_repulsion(self, pos_2d, safe_margin=1.0, num_rays=90):
        ray_from = []
        ray_to = []
        z_height = 0.23
        ray_dirs = []

        for i in range(num_rays):
            angle = i * (2 * np.pi / num_rays)
            dir_vec = np.array([np.cos(angle), np.sin(angle)])
            ray_dirs.append(dir_vec)
            ray_from.append([pos_2d[0], pos_2d[1], z_height])
            ray_to.append([pos_2d[0] + safe_margin * dir_vec[0], pos_2d[1] + safe_margin * dir_vec[1], z_height])

        results = p.rayTestBatch(ray_from, ray_to)
        min_dist = safe_margin
        repulsion_vec = np.zeros(2, dtype=np.float32)
        k_repulsion = 1.5

        for i, res in enumerate(results):
            hit_object_id = res[0]
            hit_fraction = res[2]
            if hit_fraction < 1.0 and hit_object_id not in self.robot_ids:
                dist = hit_fraction * safe_margin
                if dist < min_dist:
                    min_dist = dist
                force_magnitude = k_repulsion * (safe_margin - dist) ** 2
                repulsion_vec -= force_magnitude * ray_dirs[i]

        return min_dist, repulsion_vec



