import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import sys
import itertools

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.formation_utils import get_square_configuration, get_square_topology

RANGER_URDF_PATH = os.path.join(PROJECT_ROOT, "assets", "ranger_mini_v3", "urdf", "ranger_mini.urdf").replace("\\", "/")

class FormationPyBulletEnv(gym.Env):
    """纯粹的 APF 编队控制基准环境 (无 RL 冗余代码)"""
    def __init__(self, render_mode=False, enable_obstacles=True, hide_robots=False, draw_line=True):
        super(FormationPyBulletEnv, self).__init__()
        
        self.render_mode = render_mode
        self.hide_robots = hide_robots
        self.draw_line = draw_line
        self.enable_obstacles = enable_obstacles

        if self.render_mode:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # --- 参数配置 ---
        self.n_leaders = 2
        self.n_followers = 2
        self.num_robots = self.n_leaders + self.n_followers 
        self.robot_ids = []
        
        # --- 初始化标称构型和权重 ---
        self.g_nom, self.edges = get_square_configuration(), get_square_topology()

        # 占位用的 Gym 空间 (保持对外部测试脚本的兼容性)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.target_pos = np.array([8.5, 0, 0])
        self.wheel_indices = [3, 1, 5, 7] 
        self.steer_indices = [2, 0, 4, 6] 

        self.max_vel = 1.0  
        self.safe_margin = 0.7 
        self.obstacles = []
        self.max_steps = 1500
        self.wall_length = 20.0 
        self.wall_width = 10.0 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        
        self._create_walls()

        # 随机生成目标点
        tx = np.random.uniform(self.wall_length-4.5, self.wall_length-4.0)
        ty = np.random.uniform(-self.wall_width+6.0, self.wall_width-6.0)
        self.target_pos = np.array([tx, ty, 0])
        self._create_visual_sphere(self.target_pos, [0, 1, 0, 0.5])

        # 随机生成起点中心
        self.start_center = np.array([
            np.random.uniform(-self.wall_length + 6.0, -self.wall_length + 6.2),
            np.random.uniform(-2.0, 2.0)
        ])

        self.robot_ids = []
        for i in range(self.num_robots):
            rx = self.start_center[0] + self.g_nom[i][0]
            ry = self.start_center[1] + self.g_nom[i][1]
            yaw = self.g_nom[i][2]
            
            spawn_pos = [rx, ry, 0.15]
            spawn_quat = p.getQuaternionFromEuler([0, 0, yaw])
            rid = p.loadURDF(RANGER_URDF_PATH, basePosition=spawn_pos, baseOrientation=spawn_quat)

            if i < self.n_leaders:
                light_green = [0.65, 0.9, 0.65, 1.0]
                p.changeVisualShape(objectUniqueId=rid, linkIndex=-1, rgbaColor=light_green)
                for link_idx in range(p.getNumJoints(rid)):
                    if link_idx not in self.wheel_indices and link_idx not in self.steer_indices:
                        p.changeVisualShape(objectUniqueId=rid, linkIndex=link_idx, rgbaColor=light_green)
            
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

        # 让物理引擎稳定
        for _ in range(50):
            p.stepSimulation()

        self.current_step = 0
        # 回合统计初始化
        self.ep_formation_error_sum = 0.0
        self.ep_formation_error_max = 0.0
        return np.zeros(1), {}

    def step(self, action=None):
        self.current_step += 1
        all_pos = []
        all_yaw = []

        # 1. 采集当前位置和朝向
        for i, rid in enumerate(self.robot_ids):
            pos, quat = p.getBasePositionAndOrientation(rid)
            yaw = p.getEulerFromQuaternion(quat)[2]
            all_pos.append(np.array(pos[:2]))
            all_yaw.append(yaw)

        # 2. APF 核心控制循环
        for i, robot_id in enumerate(self.robot_ids):
            
            # 调用你写好的全功能 APF 函数
            v_world = self._apf_control_robot(i, all_pos)
            yaw = all_yaw[i]

            # 世界坐标系 -> 体坐标系
            v_body_x = v_world[0] * np.cos(yaw) + v_world[1] * np.sin(yaw)
            v_body_y = -v_world[0] * np.sin(yaw) + v_world[1] * np.cos(yaw)

            # 车头平滑转向运动方向
            w = 0.0
            speed = np.linalg.norm(v_world)
            if speed > 0.1:
                target_yaw = np.arctan2(v_world[1], v_world[0])
                yaw_diff = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
                w = np.clip(yaw_diff * 2.0, -1.0, 1.0) 

            # 驱动小车
            self._drive_robot(robot_id, v_body_x, v_body_y, w)

        # 3. 物理仿真推进
        for _ in range(24): 
            p.stepSimulation()

        # 4. 可视化队形连线
        if self.render_mode and self.draw_line:
            self._draw_formation_lines()
            time.sleep(0.01)
            
        # 5. 碰撞检测与回合终止判定
        done = False
        reason = "Unknown"
        center_pos = self._get_center_pos()
        dist_to_goal = np.linalg.norm(self.target_pos[:2] - center_pos[:2])

        # A. 撞墙检测
        limit_x = self.wall_length - 1.1
        limit_y = self.wall_width - 1.1
        for i, pos_2d in enumerate(all_pos):
            if abs(pos_2d[0]) > limit_x or abs(pos_2d[1]) > limit_y:
                done = True
                reason = f"Wall_Out"
                break

        # B. 团队互撞检测
        if not done:
            for p1, p2 in itertools.combinations(all_pos, 2):
                if np.linalg.norm(p1 - p2) < self.safe_margin:
                    done = True
                    reason = "Crash_Team"
                    break

        # C. 障碍物碰撞
        if self.enable_obstacles and not done:
            for i, robot_id in enumerate(self.robot_ids):
                for obs_id in self.obstacles:
                    pts = p.getClosestPoints(bodyA=robot_id, bodyB=obs_id, distance=0.05)
                    if len(pts) > 0:
                        done = True
                        reason = "Crash_Obstacle"
                        break
                if done: break

        # D. 到达终点或超时
        if not done and dist_to_goal < 2.0:
            done = True
            reason = "Success"
        elif self.current_step >= self.max_steps:
            done = True
            reason = "Timeout"

        # if done and reason == "Success":
        #     print(f"Episode End: {reason}")

        # --- 计算本步编队误差（只统计 followers），并累积到回合统计中 ---
        formation_error_step = 0.0
        try:
            g_offsets = np.array([self.g_nom[i][:2] for i in range(self.num_robots)])
            g_mean = np.mean(g_offsets, axis=0)
            team_center = np.mean(all_pos, axis=0)
            for i in range(self.n_leaders, self.num_robots):
                expected_pos = team_center + (g_offsets[i] - g_mean)
                actual_pos = all_pos[i]
                formation_error_step += float(np.linalg.norm(actual_pos - expected_pos))
        except Exception:
            formation_error_step = 0.0

        # 更新回合累计统计
        self.ep_formation_error_sum += formation_error_step
        self.ep_formation_error_max = max(self.ep_formation_error_max, formation_error_step)

        info = {'reason': reason, 'formation_error_step': formation_error_step}
        if done:
            info['ep_formation_error_avg'] = self.ep_formation_error_sum / max(self.current_step, 1)
            info['ep_formation_error_max'] = self.ep_formation_error_max

        terminated = done if reason != "Timeout" else False
        truncated = done if reason == "Timeout" else False

        return np.zeros(1), 0.0, terminated, truncated, info

    # ============================================================
    # 以下为你保留的核心控制函数和几何生成函数，未作更改
    # ============================================================

    def _apf_control_robot(self, robot_idx, all_pos):
        """
        真正的 APF + 一致性协议编队控制
        """
        pos = all_pos[robot_idx]

        # ===============================
        # 1. 全局导航吸引 (Navigation)
        # ===============================
        # 每个小车的终极目标是全局目标点加上它在阵型中的相对位置
        goal = self.target_pos[:2] + self.g_nom[robot_idx][:2]  
        vec_goal = goal - pos
        dist_goal = np.linalg.norm(vec_goal)
        
        if dist_goal > 1e-6:
            v_goal = (vec_goal / dist_goal) * self.max_vel
        else:
            v_goal = np.zeros(2)
            
        k_goal = 0.8  # 导航引力系数
        v_goal = k_goal * v_goal

        # ===============================
        # 2. 分布式位置一致性协议 (Consensus)
        # ===============================
        # 公式：v_cons = Σ A_ij * ((P_j - P_i) - (D_j - D_i))
        v_cons = np.zeros(2)
        Kp_consensus = 1.2  # 一致性增益系数（也就是拉扯弹簧的硬度）

        for j in range(self.num_robots):
            if j != robot_idx:
                actual_diff = all_pos[j] - pos
                # D_j - D_i：标称构型（g_nom）中的期望相对距离
                expected_diff = self.g_nom[j][:2] - self.g_nom[robot_idx][:2]
                
                # 误差：如果队友比期望离我更远，我就朝他靠近
                error = actual_diff - expected_diff
                v_cons += error
                
        # 取平均并乘上增益系数
        v_cons = (v_cons / (self.num_robots - 1)) * Kp_consensus

        # ===============================
        # 3. 环境障碍物静态斥力 (Obstacles)
        # ===============================
        _, v_obs = self._get_obstacle_repulsion(pos, safe_margin=2.0, num_rays=120)
        k_obs = 1.0
        v_obs = k_obs * v_obs

        # ===============================
        # 4. 智能体间防穿模斥力 (Inter-Agent APF)
        # ===============================
        #  一致性协议虽然能保持队形，但在遇到柱子队形被挤压时，仍需要短距离强斥力防止互撞
        v_team = np.zeros(2)
        k_team = 0.8
        d_safe = self.safe_margin + 0.5

        for j in range(self.num_robots):
            if j == robot_idx: 
                continue
            
            other_pos = all_pos[j]
            diff = pos - other_pos
            dist = np.linalg.norm(diff)

            if 1e-6 < dist < d_safe:
                direction = diff / dist
                # 标准 APF 斥力梯度公式
                strength = k_team * (1.0 / dist - 1.0 / d_safe) / (dist ** 2)
                v_team += strength * direction

        # ===============================
        # 5. 合成最终速度向量
        # ===============================
        v_world = v_goal + v_cons + v_obs + v_team

        # 速度限幅，防止计算出的合力使速度爆炸
        norm = np.linalg.norm(v_world)
        if norm > self.max_vel:
            v_world = (v_world / norm) * self.max_vel

        return v_world

    def _get_obstacle_repulsion(self, pos_2d, safe_margin=1.0, num_rays=90):
        ray_from, ray_to, ray_dirs = [], [], []
        z_height = 0.23
        for i in range(num_rays):
            angle = i * (2 * np.pi / num_rays)
            dir_vec = np.array([np.cos(angle), np.sin(angle)])
            ray_dirs.append(dir_vec)
            ray_from.append([pos_2d[0], pos_2d[1], z_height])
            ray_to.append([pos_2d[0] + safe_margin * dir_vec[0], pos_2d[1] + safe_margin * dir_vec[1], z_height])

        results = p.rayTestBatch(ray_from, ray_to)
        min_dist = safe_margin
        repulsion_vec = np.zeros(2) 
        k_repulsion = 1.5 
        
        for i, res in enumerate(results):
            hit_object_id, hit_fraction = res[0], res[2]
            if hit_fraction < 1.0 and hit_object_id not in self.robot_ids:
                dist = hit_fraction * safe_margin
                if dist < min_dist: min_dist = dist
                force_magnitude = k_repulsion * (safe_margin - dist)**2 
                repulsion_vec -= force_magnitude * ray_dirs[i] 
                    
        return min_dist, repulsion_vec

    def _drive_robot(self, robot_id, v_x, v_y, w):
        R = 0.09  
        wheel_positions = [(0.25, 0.19), (0.25, -0.19), (-0.25, 0.19), (-0.25, -0.19)]
        target_steering_angles, target_wheel_speeds = [], []
        current_joint_states = p.getJointStates(robot_id, self.steer_indices)
        
        for i, (wx, wy) in enumerate(wheel_positions):
            wheel_vx = v_x - w * wy
            wheel_vy = v_y + w * wx
            speed_2d = np.hypot(wheel_vx, wheel_vy)
            if speed_2d < 2e-2:
                target_steering_angles.append(current_joint_states[i][0]) 
                target_wheel_speeds.append(0.0)
                continue
                
            ideal_angle = np.arctan2(wheel_vy, wheel_vx)
            cmd_angle = -ideal_angle 
            ideal_speed = speed_2d / R
            current_angle = current_joint_states[i][0]
            
            angle1, speed1 = cmd_angle, ideal_speed
            angle2 = angle1 - np.pi if angle1 > 0 else angle1 + np.pi
            speed2 = -ideal_speed
            
            valid_configs = []
            if -2.1 <= angle1 <= 2.1: valid_configs.append((angle1, speed1))
            if -2.1 <= angle2 <= 2.1: valid_configs.append((angle2, speed2))
                
            if len(valid_configs) > 0:
                best_angle, best_speed = min(valid_configs, key=lambda x: abs(x[0] - current_angle))
            else:
                best_angle, best_speed = current_angle, 0.0
                
            angle_diff = abs(best_angle - current_angle)
            final_speed = best_speed * max(0, np.cos(angle_diff))
            target_steering_angles.append(best_angle)
            target_wheel_speeds.append(final_speed)
            
        p.setJointMotorControlArray(
            robot_id, self.steer_indices, p.POSITION_CONTROL,
            targetPositions=target_steering_angles, forces=[80.0] * 4
        )
        p.setJointMotorControlArray(
            robot_id, self.wheel_indices, p.VELOCITY_CONTROL,
            targetVelocities=target_wheel_speeds, forces=[20.0] * 4
        )

    def _get_center_pos(self):
        pos_list = [np.array(p.getBasePositionAndOrientation(rid)[0]) for rid in self.robot_ids[:self.num_robots]]
        return np.mean(pos_list, axis=0)

    def _draw_formation_lines(self):
        if len(self.robot_ids) < 2: return
        points = [p.getBasePositionAndOrientation(rid)[0] for rid in self.robot_ids]
        edges = [(0, 1)] if self.n_followers == 0 else self.edges
        for u, v in edges:
            if u < len(points) and v < len(points):
                p.addUserDebugLine(points[u], points[v], [1, 1, 0], 2, 0.1)

    def _create_walls(self):
        self.wall_ids = []
        walls_config = [
            ([self.wall_length, 0, 0.5], [0.5, self.wall_width, 1.0]),  
            ([-self.wall_length, 0, 0.5], [0.5, self.wall_width, 1.0]), 
            ([0, self.wall_width, 0.5], [self.wall_length, 0.5, 1.0]),  
            ([0, -self.wall_width, 0.5], [self.wall_length, 0.5, 1.0])  
        ]
        for pos, extents in walls_config:
            v_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=extents, rgbaColor=[0.8, 0.8, 0.8, 1])
            c_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=extents)
            self.wall_ids.append(p.createMultiBody(0, c_shape, v_shape, basePosition=pos))

    def _create_visual_sphere(self, pos, color):
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=color)
        return p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[pos[0]-2.0, pos[1], 0])

    def _generate_obstacles(self):
        self.obstacles = []
        config_id = np.random.randint(0, 5) 
        
        v_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, length=1.5, rgbaColor=[0.2, 0.2, 0.2, 1])
        c_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.5, height=1.5)

        pos_list = []
        
        mid_x_min, mid_x_max = -self.wall_length / 3, self.wall_length / 3
        mid_y_range = self.wall_width - 1.0

        if config_id == 0:  # 均匀4排
            for x in [-4, -1.33333333, 1.33333333, 4]:
                pos_list.append([x, 0, 0.5])
                pos_list.append([x, 0.5 * self.wall_width, 0.5])
                pos_list.append([x, -0.5 * self.wall_width, 0.5])


        elif config_id == 1:
            # 目标：在 x=0 处建一堵墙，支持多开几个门降低探索难度
            
            # --- 在这里自由定义门的位置和大小 ---
            doors_y = [-5.0, 0.0, 5.0]  # 定义三个门的中心 Y 坐标
            gap_half_width = 0.8        # 每个门留 1.6 米的宽缝
            
            wall_thickness = 1.0
            wall_height = 1.5
            wall_x = 0.0                # 墙在 X 轴的位置
            
            # 墙体的起始 Y 坐标 (从场地的最右侧边界开始)
            current_y = -self.wall_width + 0.5
            
            # 遍历所有的门，在门与门之间“砌墙”
            for door_y in doors_y:
                # 计算当前墙段的终点 (门的下边缘)
                segment_end_y = door_y - gap_half_width 
                segment_length = segment_end_y - current_y
                
                # 如果墙段长度大于0，则生成这块墙
                if segment_length > 0:
                    center_y = current_y + segment_length / 2.0
                    half_extents = [wall_thickness / 2.0, segment_length / 2.0, wall_height / 2.0]
                    
                    v_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.2, 0.2, 0.2, 1])
                    c_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
                    
                    obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_shape, 
                                               baseVisualShapeIndex=v_shape, 
                                               basePosition=[wall_x, center_y, wall_height / 2.0])
                    self.obstacles.append(obs_id)
                
                # 更新 current_y 到门的另一侧上边缘
                current_y = door_y + gap_half_width
                
            # --- 收尾：最后一道门到左侧边界的墙段 ---
            segment_end_y = self.wall_width - 0.5
            segment_length = segment_end_y - current_y
            if segment_length > 0:
                center_y = current_y + segment_length / 2.0
                half_extents = [wall_thickness / 2.0, segment_length / 2.0, wall_height / 2.0]
                
                v_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.2, 0.2, 0.2, 1])
                c_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
                
                obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_shape, 
                                           baseVisualShapeIndex=v_shape, 
                                           basePosition=[wall_x, center_y, wall_height / 2.0])
                self.obstacles.append(obs_id)
            

        elif config_id == 2:    
            # ==========================================
            # 八字形逐渐缩小的袋口 (Funnel / Pocket)
            # ==========================================
            # 1. 定义袋口的 X 轴跨度 (假设小车从 X 负半轴往正半轴开)
            x_start = -5.0       # 袋口的最外端 (宽口)
            x_end = 3.0          # 袋口的最里端 (窄口)
            
            # 2. 定义开口大小 (Y轴的正负一半)
            y_gap_start = 8.0    # 宽口一半宽度 (总宽 12 米)
            y_gap_end = 1.2      # 窄口一半宽度 (总宽 2.4 米，极度考验编队收缩)
            
            # 3. 定义单侧墙壁的柱子密集程度
            num_pillars = 10    
            
            # 4. 利用线性插值 (Linear Interpolation) 生成两条斜线
            for i in range(num_pillars):
                # t 是插值系数，从 0.0 逐渐变到 1.0
                t = i / (num_pillars - 1)
                
                # 计算当前的 X 坐标
                cx = x_start + t * (x_end - x_start)
                
                # 计算当前 X 对应的 Y 边界
                cy = y_gap_start + t * (y_gap_end - y_gap_start)
                
                # 上半边的斜墙 (y > 0)
                pos_list.append([cx, cy, 0.5])
                
                # 下半边的斜墙 (y < 0，对称)
                pos_list.append([cx, -cy, 0.5])

        elif config_id == 3:
            # --- 配置 5：动态泊松盘（稀疏森林） ---
            while len(pos_list) < 6: # 生成 6 根柱子
                ox = np.random.uniform(mid_x_min, mid_x_max)
                oy = np.random.uniform(-mid_y_range, mid_y_range)
                
                is_far_enough = True
                for p_exist in pos_list:
                    if np.linalg.norm([ox - p_exist[0], oy - p_exist[1]]) < 4.5:
                        is_far_enough = False
                        break
                
                if is_far_enough:
                    pos_list.append([ox, oy, 0.5])

        else:
            # --- 配置 5：动态泊松盘（稀疏森林） ---
            while len(pos_list) < 10: # 生成 10 根柱子
                ox = np.random.uniform(mid_x_min, mid_x_max)
                oy = np.random.uniform(-mid_y_range, mid_y_range)
                
                is_far_enough = True
                for p_exist in pos_list:
                    if np.linalg.norm([ox - p_exist[0], oy - p_exist[1]]) < 4.0:
                        is_far_enough = False
                        break
                
                if is_far_enough:
                    pos_list.append([ox, oy, 0.5])

        # 统一生成障碍物
        for pos in pos_list:
            obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_shape, 
                                       baseVisualShapeIndex=v_shape, basePosition=pos)
            self.obstacles.append(obs_id)

    def close(self):
        if self.physicsClient >= 0:
            p.disconnect(self.physicsClient)
            self.physicsClient = -1